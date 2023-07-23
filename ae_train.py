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
    train_file_pattern = args.train_file_pattern
    test_file_pattern = args.test_file_pattern
    model_dir = args.model_dir
    ae_name = args.ae_name
    max_ckpt_to_keep = args.max_ckpt_to_keep
    interval = args.interval
    restore_best = args.restore_best
    total_batches = args.total_batches
    
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
    train_ds = create_train_ds(train_file_pattern, 
                        config['batch_size'], 
                        config['img_size'])

    test_ds = create_train_ds(test_file_pattern, 
                        config['batch_size'], 
                        config['img_size'])

    train_ds = iter(train_ds.repeat())
    test_batch = next(iter(test_ds))
    
    # model
    autoencoder = Autoencoder(
        config['encoder_dim'], config['decoder_dim'],
        cuant_dim=config['cuant_dim']
    )
    autoencoder(test_batch) # init model
    print(autoencoder.summary())

    discriminator = Discriminator(config['d_dim'])
    discriminator(test_batch) # init model
    print(discriminator.summary())
    
    # optimizers
    ae_opt = tf.keras.optimizers.Adam(learning_rate=config['ae_lr'])
    d_opt = tf.keras.optimizers.Adam(learning_rate=config['d_lr'])
    
    # trainer
    ae_kl = AutoencoderKL(
        get_augmenter(image_size=config['img_size']), autoencoder, 
        discriminator, ae_opt, d_opt, config
    )
    ae_kl.create_ckpt(model_dir, max_ckpt_to_keep, restore_best)
    
    # train
    start_batch = int((ae_kl.ckpt.n_images / ae_kl.batch_size) + 1)
    n_images = int(ae_kl.ckpt.n_images)
    start = time.time()

    for n_batch in range(start_batch, total_batches):
        batch = train_ds.get_next()
        ae_kl.train_step(batch)
        
        if n_batch % interval == 0:
            print(f'\nTime for interval is {time.time()-start:.4f} sec')
            start = time.time()
            # val loop
            for batch in test_ds:
                ae_kl.test_step(batch)
            print(f'Time for val is {time.time()-start:.4f} sec')
            
            n_images = n_batch * ae_kl.batch_size
            ae_kl.plot_images(test_batch, n_images)
            ae_kl.save_ckpt(n_images)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file_pattern', type=str)
    parser.add_argument('--test_file_pattern', type=str)
    parser.add_argument('--model_dir', type=str, default='autoencoder')
    parser.add_argument('--ae_name', type=str, default='model_1')
    parser.add_argument('--max_ckpt_to_keep', type=int, default=2)
    parser.add_argument('--interval', type=int, default=100)
    parser.add_argument('--restore_best', type=bool, default=False)    
    parser.add_argument('--total_batches', type=int, default=100000000)  
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
