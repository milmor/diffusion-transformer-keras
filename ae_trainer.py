'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2023
'''
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *


class AutoencoderKL():
    def __init__(self, augmenter, ae, discriminator, 
                 ae_opt, d_opt, config):
        self.augmenter = augmenter
        self.ae = ae
        self.discriminator = discriminator
        self.ae_opt = ae_opt
        self.d_opt = d_opt
        self.rec_weight = config['rec_weight']
        self.kl_weight = config['kl_weight']
        self.d_start = config['d_start']
        self.n_images = tf.Variable(0, dtype=tf.int64)
        # metrics
        self.train_metrics = {}
        self._build_metrics()
        # loss
        self.mae = tf.keras.losses.MeanAbsoluteError()
        
    def _build_metrics(self):
        metric_names = [
        'rec_loss',
        'kl_loss',
        'd_loss',
        'ae_total_loss',
        'g_loss',
        ]

        for metric_name in metric_names:
            self.train_metrics[metric_name] = tf.keras.metrics.Mean()
        
    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        return tf.clip_by_value(images, 0.0, 1.0)
    
    def discriminator_loss(self, real_img, fake_img):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real_img))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_img))
        return 0.5 * (fake_loss + real_loss)

    def generator_loss(self, fake_img):
        return -tf.reduce_mean(fake_img)
    
    @tf.function
    def train_step(self, train_img):
        train_img = self.augmenter(train_img, training=True)
        disc_factor = 0.0
        if self.n_images > self.d_start:
            disc_factor = 1.0
        with tf.GradientTape() as ae_tape, tf.GradientTape() as disc_tape:

            rec_img, z_mean, z_log_var = self.ae(train_img, training=True)
            kl_loss = (-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))) 
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=[1, 2, 3])) * self.kl_weight

            fake_logits = self.discriminator(rec_img, training=True)[0]
            real_logits = self.discriminator(train_img, training=True)[0]
            d_loss = self.discriminator_loss(real_logits, fake_logits) * disc_factor
            g_loss = self.generator_loss(fake_logits)

            rec_loss = self.mae(rec_img, train_img) * self.rec_weight
            ae_total_loss = rec_loss + kl_loss + g_loss

        ae_grad = ae_tape.gradient(ae_total_loss, self.ae.trainable_weights)
        self.ae_opt.apply_gradients(zip(ae_grad, self.ae.trainable_weights))

        d_grad = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))
        
        update_metrics(
         self.train_metrics,
         rec_loss=rec_loss,
         kl_loss=kl_loss,
         d_loss=d_loss,
         ae_total_loss=ae_total_loss,
         g_loss=g_loss, 
      )
        
    def plot_images(self, img, step=None, num_rows=5, num_cols=5, plot_image_size=128,
                   is_jupyter=False):
        img_dir = os.path.join(self.model_dir, 'ae-log-img')
        os.makedirs(img_dir, exist_ok=True)        
        generated_images, _, _ = self.ae(img[:num_rows*num_cols], training=False)
        generated_images = self.denormalize(generated_images)
        
        # organize generated images into a grid
        generated_images = tf.image.resize(
            generated_images, (plot_image_size, plot_image_size), method="nearest"
        )
        generated_images = tf.reshape(
            generated_images,
            (num_rows, num_cols, plot_image_size, plot_image_size, 3),
        )
        generated_images = tf.transpose(generated_images, (0, 2, 1, 3, 4))
        generated_images = tf.reshape(
            generated_images,
            (num_rows * plot_image_size, num_cols * plot_image_size, 3),
        )
        
        if is_jupyter:
            plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
            plt.imshow(generated_images.numpy())
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            plt.imsave(os.path.join(
                img_dir, f'{step}.png'), generated_images.numpy()
            )
        
    def create_ckpt(self, model_dir, max_ckpt_to_keep, restore_best=True):
        # log dir
        self.model_dir = model_dir
        ae_log_dir = os.path.join(self.model_dir, 'ae-log-dir')
        self.writer = tf.summary.create_file_writer(ae_log_dir)
        # checkpoint dir
        checkpoint_dir = os.path.join(model_dir, 'ae-ckpt')
        best_checkpoint_dir = os.path.join(model_dir, 'ae-best-ckpt')

        self.ckpt = tf.train.Checkpoint(
            ae=self.ae, ae_opt=self.ae_opt, discriminator=self.discriminator,
            d_opt=self.d_opt, n_images=self.n_images,
            best_loss=tf.Variable(10000.0) # initialize with big value
        ) 

        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=checkpoint_dir, max_to_keep=max_ckpt_to_keep
        )
        self.best_ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, directory=best_checkpoint_dir, max_to_keep=max_ckpt_to_keep
        )
               
        if restore_best == True and self.best_ckpt_manager.latest_checkpoint:    
            last_ckpt = self.best_ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Best checkpoint restored from {last_ckpt}')
        elif restore_best == False and self.ckpt_manager.latest_checkpoint:
            last_ckpt = self.ckpt_manager.latest_checkpoint
            self.ckpt.restore(last_ckpt)
            print(f'Checkpoint restored from {last_ckpt}')     
        else:
            print(f'Checkpoint created at {self.model_dir} dir')
        
    def save_ckpt(self, n_images, verbose=1, reset_states=True):
        # tensorboard
        with self.writer.as_default():
            for name, metric in self.train_metrics.items():
                print(f'{name}: {metric.result():.4f} -', end=" ")
                tf.summary.scalar(name, metric.result(), step=n_images)
        print(f'n_images: {n_images}') 
        self.n_images.assign(n_images)
        # checkpoint
        if self.train_metrics['rec_loss'].result() < self.ckpt.best_loss:  
            self.ckpt.best_loss.assign(self.train_metrics['rec_loss'].result())
            self.best_ckpt_manager.save(n_images)
            print(f'Best checkpoint saved at {n_images} images')
        else:
            self.ckpt_manager.save(n_images)
            print(f'Checkpoint saved at {n_images} images')
        
        # reset metrics    
        reset_metrics(self.train_metrics)

    def restore_ae(self, model_dir, max_ckpt_to_keep=1):
        best_checkpoint_dir = os.path.join(model_dir, 'ae-best-ckpt')
        ckpt = tf.train.Checkpoint(ae=self.ae) 
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=best_checkpoint_dir, max_to_keep=max_ckpt_to_keep
        )
        last_ckpt = ckpt_manager.latest_checkpoint
        ckpt.restore(last_ckpt).expect_partial()
        print(f'AutoencoderKL checkpoint restored from {last_ckpt}')
