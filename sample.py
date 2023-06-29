'''
Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2023
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable tensorflow debugging logs
import tensorflow as tf
import time
from autoencoder import Autoencoder
from ae_trainer import AutoencoderKL
from dit import DiT
from ldt_trainer import LDT
from utils import * 
from schedule import CosineSchedule


def sample(args):
    print('\n##########')
    print('LDT Sample')
    print('##########\n')
    num_rows = args.num_rows
    num_cols = args.num_cols
    diffusion_steps = args.diffusion_steps
    model_dir = args.model_dir
    ldt_name = args.ldt_name
    restore_best = args.restore_best
    out_dir = args.out_dir

    # set batch size
    batch_size = num_rows * num_cols 
    
    # ldt config file
    model_dir = os.path.join(model_dir, ldt_name)
    os.makedirs(model_dir, exist_ok=True)
    config_file = os.path.join(model_dir, f"{ldt_name}_config.json")

    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            config = json.load(file)
            print(f'{config_file} loaded')
    else:
        from ldt_config import config
        with open(config_file, 'w') as file:
            json.dump(config, file)
            print(f'{config_file} saved')
    print(config)
    
    # ae config file
    ae_dir = os.path.join(config['ae_dir'], config['ae_name'])
    ae_config_file = os.path.join(ae_dir, f"{config['ae_name']}_config.json")
    with open(ae_config_file, 'r') as file:
        ae_config = json.load(file)
        print(f'{ae_config_file} loaded')
    
    test_batch = tf.zeros([
        batch_size, config['img_size'], config['img_size'], 3,
    ])
    
    # ae model
    autoencoder = Autoencoder(
        ae_config['encoder_dim'], ae_config['decoder_dim'], 
        cuant_dim=ae_config['cuant_dim']
    )
    autoencoder.trainable = False
    autoencoder(test_batch) # init model
    print(autoencoder.summary())

    ae_kl = AutoencoderKL(
        None, autoencoder, None, None, None, ae_config
    )
    
    # ae ckpt
    ae_kl.restore_ae(ae_dir)
    test_latent = ae_kl.ae.encoder(test_batch)[0]
    
    # ldt model
    dit = DiT(
        config['latent_size'], config['patch_size'], config['ldt_dim'], 
        heads=config['heads'], k=config['k'], mlp_dim=config['mlp_dim'], 
        depth=config['depth'], cuant_dim=config['cuant_dim']
    )

    test_noise = tf.ones([batch_size, 1, 1, 1]) * -1e9
    outputs = dit([test_latent, test_noise]) # init dit
    print(dit.summary())
    
    # schedule
    diffusion_schedule = CosineSchedule(
    start_log_snr=config['start_log_snr'],
    end_log_snr=config['end_log_snr'],
    )
    
    # ldt optimizer
    opt = tf.keras.optimizers.Adam(
            learning_rate=config['learning_rate']
    )
    ldt = LDT(
        network=dit, ae_kl=ae_kl, opt=opt, diffusion_schedule=diffusion_schedule,
        config=config
    )
    ldt.create_ckpt(
        model_dir, max_ckpt_to_keep=1, restore_best=restore_best
    )
    ldt.plot_images(
        'sample_img', diffusion_steps=diffusion_steps,
        img_dir=out_dir # init ldt
    ) 
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='ldt')
    parser.add_argument('--ldt_name', type=str, default='model_1')
    parser.add_argument('--num_rows', type=int, default='5')
    parser.add_argument('--num_cols', type=int, default='5')
    parser.add_argument('--diffusion_steps', type=int, default='40')
    parser.add_argument('--restore_best', type=bool, default=True)   
    parser.add_argument('--out_dir', type=str, default='./')   
    args = parser.parse_args()

    sample(args)


if __name__ == '__main__':
    main()
