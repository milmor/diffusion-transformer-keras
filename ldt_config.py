config = {
    'batch_size': 64,
    'img_size': 128,
    'ds_val_seed': 22,
    'fid_batch_size': 50,
    'n_fid_images': 600,
    'fid_diffusion_steps': 40,
    'ae_dir': 'autoencoder',
    'ae_name': 'model_1',
    # LDT hparams
    'img_size': 128,
    'latent_size': 32,
    'cuant_dim': 4,
    'prediction_type': "velocity", # can be ('velocity', 'signal' or 'noise')
    'loss_type': "velocity", # can be ('velocity', 'signal' or 'noise')
    'start_log_snr': 3.0, # cosine schedule
    'end_log_snr': -10.0, # cosine schedule
    'ema': 0.999,
    'learning_rate': 2e-4,
    'ldt_dim': 256, 
    'k': 64, # k linformer projection dim
    'patch_size': 2,
    'depth': 8,
    'heads': 4,
    'mlp_dim': 256,
}
