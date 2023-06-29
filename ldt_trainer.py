'''
MIT License

Copyright (c) 2023 milmor

Adapted functions from https://github.com/beresandras/clear-diffusion-keras 

Copyright (c) 2022 beresandras

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from PIL import Image
from fid import *
from utils import *


class LDT():
    def __init__(self, network, ae_kl, opt,
               diffusion_schedule, config):
        self.network = network
        self.encoder = ae_kl.ae.encoder
        self.encoder.trainable = False
        self.decoder = ae_kl.ae.decoder
        self.decoder.trainable = False
        self.ema_network = tf.keras.models.clone_model(network)

        self.image_size = config['img_size']
        self.latent_size = config['latent_size']
        self.cuant_dim = config['cuant_dim']
        self.optimizer = opt
        self.prediction_type = config['prediction_type']
        self.loss_type = config['loss_type']
        self.batch_size = config['batch_size']
        self.ema = config['ema']
        self.diffusion_schedule = diffusion_schedule
        
        # metrics
        self.fid_avg = tf.keras.metrics.Mean()
        self.train_metrics = {}
        self._build_metrics()
        
        # loss
        self.loss = tf.keras.losses.mean_squared_error

    def _build_metrics(self):
        metric_names = [
        'velocity_loss',
        'latent_loss',
        'noise_loss',
        ]

        for metric_name in metric_names:
            self.train_metrics[metric_name] = tf.keras.metrics.Mean()

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        return tf.clip_by_value(images, 0.0, 1.0)

    def get_components(self, noisy_latents, predictions, signal_rates, 
                       noise_rates, prediction_type=None):
        if prediction_type is None:
            prediction_type = self.prediction_type

        # calculate the other signal components using the network prediction
        if prediction_type == "velocity":
            pred_velocities = predictions
            pred_latents = signal_rates * noisy_latents - noise_rates * pred_velocities
            pred_noises = noise_rates * noisy_latents + signal_rates * pred_velocities
        elif prediction_type == "signal":
            pred_latents = predictions
            pred_noises = (noisy_latents - signal_rates * pred_latents) / noise_rates
            pred_velocities = (signal_rates * noisy_latents - pred_latents) / noise_rates
        elif prediction_type == "noise":
            pred_noises = predictions
            pred_latents = (noisy_latents - noise_rates * pred_noises) / signal_rates
            pred_velocities = (pred_noises - noise_rates * noisy_latents) / signal_rates
        else:
            raise NotImplementedError

        return pred_velocities, pred_latents, pred_noises

    def generate(self, num_images, diffusion_steps, variance_preserving, seed):
        if seed is not None:
            tf.random.set_seed(seed)

        # noise -> latents -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, self.latent_size, self.latent_size, self.cuant_dim), 
            seed=seed
        )
        generated_latents = self.diffusion_process(
            initial_noise,
            diffusion_steps,
            variance_preserving,
            seed,
        )
        generated_images = self.decoder(generated_latents, training=False)
        return self.denormalize(generated_images)
    
    @tf.function
    def diffusion_process(self, initial_noise, diffusion_steps, 
                          variance_preserving, seed):
        batch_size = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        # at the first sampling step, the "noisy latent" is pure noise
        noisy_latents = initial_noise
        prev_pred_noises = []  # only required for multistep sampling
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((batch_size, 1, 1, 1)) - step * step_size

            signal_rates, noise_rates = self.diffusion_schedule(diffusion_times)
            # predict one component of the noisy latents with the network
            # exponential moving average weights are used for inference
            predictions = self.ema_network(
                [noisy_latents, noise_rates**2], training=False
            )
            # calculate the other components using it
            _, pred_latents, pred_noises = self.get_components(
                noisy_latents, predictions, signal_rates, noise_rates
            )

            next_signal_rates, next_noise_rates = self.diffusion_schedule(
                diffusion_times - step_size
            )
            # remix the predicted components using the next signal and noise rates
            noisy_latents = (
                next_signal_rates * pred_latents + next_noise_rates * pred_noises
            )
            # this new noisy latent will be used in the next step
        return pred_latents

    @tf.function
    def train_step(self, latents):
        noises = tf.random.normal(
            shape=(self.batch_size, self.latent_size, self.latent_size, self.cuant_dim)
        )

        # sample uniform random diffusion powers
        noise_powers = tf.random.uniform(
            shape=(self.batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        signal_powers = 1.0 - noise_powers
        noise_rates = noise_powers**0.5
        signal_rates = signal_powers**0.5

        # mix the latents with noises accordingly
        latents, _, _ = self.encoder(latents, training=False)
        noisy_latents = signal_rates * latents + noise_rates * noises
        velocities = -noise_rates * latents + signal_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy latents to their components
            predictions = self.network([noisy_latents, noise_rates**2], training=True)
            pred_velocities, pred_latents, pred_noises = self.get_components(
                noisy_latents, predictions, signal_rates, noise_rates
            )
            # one of the losses is used for training, the others are tracked as metrics
            velocity_loss = self.loss(velocities, pred_velocities)
            latent_loss = self.loss(latents, pred_latents)
            noise_loss = self.loss(noises, pred_noises)

        if self.loss_type == "velocity":
            loss = velocity_loss
        elif self.loss_type == "signal":
            loss = latent_loss
        elif self.loss_type == "noise":
            loss = noise_loss
        else:
            raise NotImplementedError

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        update_metrics(
         self.train_metrics,
         velocity_loss=velocity_loss,
         latent_loss=latent_loss,
         noise_loss=noise_loss,  
      )

    def plot_images(self, step=None, num_rows=5, num_cols=5, diffusion_steps=40, 
                        variance_preserving=False, seed=None, plot_image_size=128,
                        is_jupyter=False, img_dir=None):
        if img_dir == None:
            img_dir = os.path.join(self.model_dir, 'ldt-log-img')
        os.makedirs(img_dir, exist_ok=True)
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(num_rows * num_cols, diffusion_steps, 
                                         variance_preserving, seed
        )
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
        log_dir = os.path.join(model_dir, 'ldt-log-dir')
        self.writer = tf.summary.create_file_writer(log_dir)
        
        # checkpoint dir
        checkpoint_dir = os.path.join(model_dir, 'ldt-ckpt')
        best_checkpoint_dir = os.path.join(model_dir, 'ldt-best-ckpt')

        self.ckpt = tf.train.Checkpoint(
            optimizer=self.optimizer, network=self.network,
            ema_network=self.ema_network, n_images=tf.Variable(0),
            fid=tf.Variable(10000.0), 
            best_fid=tf.Variable(10000.0)# initialize with big value
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
            print(f'Checkpoint created at {model_dir} dir')
            
    def save_ckpt(self, n_images, n_fid_images, fid_diffusion_steps, 
                  fid_batch_size, val_ds):
        # tensorboard
        with self.writer.as_default():
            for name, metric in self.train_metrics.items():
                print(f'{name}: {metric.result():.4f} -', end=" ")
                tf.summary.scalar(name, metric.result(), step=n_images)
        
        # fid
        fid = self.fid(n_fid_images, fid_diffusion_steps, fid_batch_size, val_ds)
        self.fid_avg.update_state(fid)
        with self.writer.as_default():
            tf.summary.scalar('fid', self.fid_avg.result(), step=n_images)
            
        # checkpoint
        self.ckpt.n_images.assign(n_images)
        self.ckpt.fid.assign(fid)
        
        if fid < self.ckpt.best_fid:
            self.ckpt.best_fid.assign(fid)
            self.best_ckpt_manager.save(n_images)
            print(f'FID improved. Best checkpoint saved at {n_images} images') 
        else:
            self.ckpt_manager.save(n_images)
            print(f'Checkpoint saved at {n_images} images')  
            
        self.fid_avg.reset_states()   
        # reset metrics    
        reset_metrics(self.train_metrics)
        
    def gen_batches(self, n_images, batch_size, diffusion_steps, dir_path):
        n_batches = n_images // batch_size
        n_used_imgs = n_batches * batch_size

        for i in range(n_batches):
            start = i * batch_size
            gen_batch = self.generate(
                batch_size, diffusion_steps, variance_preserving=False, seed=None
            )
            gen_batch = np.clip(gen_batch * 255, 0.0, 255)

            img_index = start
            for img in gen_batch:
                img = Image.fromarray(img.astype('uint8'))
                file_name = os.path.join(dir_path, f'{str(img_index)}.png')
                img.save(file_name,"PNG")
                img_index += 1
                
    def fid(self, n_fid_images, fid_diffusion_steps, batch_size, val_dataset):
        inception = Inception()
        fid_dir = os.path.join(self.model_dir, 'fid')
        os.makedirs(fid_dir, exist_ok=True)
        # fid
        start = time.time()
        print('\nGenerating FID images...') 
        self.gen_batches(n_fid_images, batch_size, 
                         fid_diffusion_steps, fid_dir
        )
        gen_fid_ds = create_fid_ds(
            fid_dir + '/*.png', batch_size, self.image_size, n_fid_images
        )
        m_gen, s_gen = calculate_activation_statistics(
            gen_fid_ds, inception, batch_size
        )
        m_real, s_real = calculate_activation_statistics(
            val_dataset, inception, batch_size
        )        
        fid = calculate_frechet_distance(m_real, s_real, m_gen, s_gen)
        print(f'FID: {fid:.4f} - Time for FID score is {time.time()-start:.4f} sec')            
        return fid         
