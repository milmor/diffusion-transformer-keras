'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2023
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
import time
from autoencoder import Autoencoder
from discriminator import Discriminator
from ae_trainer import AutoencoderKL
from utils import * 


def train(args):
    print('\n###################')
    print('AutoencoderKL Train')
    print('###################\n')
    file_pattern = args.file_pattern
    model_dir = args.model_dir
    ae_name = args.ae_name
    max_ckpt_to_keep = args.max_ckpt_to_keep
    interval = args.interval
    restore_best = args.restore_best
    total_images = args.total_images
    
    # config file
    model_dir = os.path.join(model_dir, ae_name)
    os.makedirs(model_dir, exist_ok=True)
    config_file = os.path.join(model_dir, f"{ae_name}_config.json")

    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
            print(f'{config_file} loaded')
    else:
        from ae_config import config
        with open(config_file, 'w') as file:
            json.dump(config, file)
            print(f'{config_file} saved')
    print(config)
    
    # dataset
    train_ds = create_train_ds(
        file_pattern, config['batch_size'], config['img_size']
    )
    train_batch = next(iter(train_ds))
    
    # model
    autoencoder = Autoencoder(
        config['encoder_dim'], config['decoder_dim'],
        cuant_dim=config['cuant_dim']
    )
    autoencoder(train_batch) # init model
    print(autoencoder.summary())

    discriminator = Discriminator(config['d_dim'])
    discriminator(train_batch) # init model
    print(discriminator.summary())
    
    # optimizers
    ae_opt = tf.keras.optimizers.Adam(learning_rate=config['ae_lr'])
    d_opt = tf.keras.optimizers.Adam(learning_rate=config['d_lr'])
    
    ae_kl = AutoencoderKL(
        get_augmenter(image_size=config['img_size']), autoencoder, 
        discriminator, ae_opt, d_opt, config
    )
    ae_kl.create_ckpt(model_dir, max_ckpt_to_keep, restore_best)
    
    # train
    n_images = int(ae_kl.ckpt.n_images)
    for _ in range(total_images):
        start = time.time()
        for batch in train_ds.take(interval):
            ae_kl.train_step(batch)
        n_images += interval * config['batch_size']
        print(f'\nTime for interval is {time.time()-start:.4f} sec')
        ae_kl.plot_images(train_batch, n_images)
        ae_kl.save_ckpt(n_images)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_pattern', type=str)
    parser.add_argument('--model_dir', type=str, default='autoencoder')
    parser.add_argument('--ae_name', type=str, default='model_1')
    parser.add_argument('--max_ckpt_to_keep', type=int, default=2)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--restore_best', type=bool, default=True)    
    parser.add_argument('--total_images', type=int, default=100000000)  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
