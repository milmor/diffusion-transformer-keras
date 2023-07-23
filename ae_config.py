config = {
    'batch_size': 64,
    'img_size': 128,
    # Autoencoder hparams
    'cuant_dim': 4,
    'ae_lr': 0.0001,
    'd_lr': 0.0001,
    'd_dim': [16, 32, 64, 128, 256],
    'kl_weight': 0.000001,
    'rec_weight': 1.0,
    'adv_weight': 2.0,
    'encoder_dim': [128, 256],
    'decoder_dim': [256, 128],
    'd_start': 50000, # start training the discriminator after N samples
}
