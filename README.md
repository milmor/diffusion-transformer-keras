# Latent Diffusion Transformer
Implementation of the Diffusion Transformer model in the paper:

> [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). 

![LDT architecture](./images/ldt.png)

See [here](https://github.com/facebookresearch/DiT) for the official Pytorch implementation.


## Dependencies
- Python 3.8
- Tensorfow 2.12

## Usage
### Training AutoencoderKL
Use `--file_pattern=<file_pattern>` to specify the dataset path and file pattern.
```
python ae_train.py --file_pattern=./dataset_path/*.png
```

### Training Diffusion Transformer
Use `--file_pattern=<file_pattern>` to specify the dataset path and file pattern.
```
python ldt_train.py --file_pattern=./dataset_path/*.png
```

### Hparams setting

Adjust hyperparameters in the `ae_config.py` and `ldm_config.py` files.

Implementation notes:
- LDT is designed to offer reasonable performance using a single GPU (RTX 3080 TI).
- LDT largely follows the original DiT model.
- AutoencoderKL with PatchGAN discriminator and hinge loss.
- Diffusion Transformer with [Linformer](https://arxiv.org/abs/2006.04768) attention.
- FID evaluation.
- This implementation uses code from the [beresandras](https://github.com/beresandras/clear-diffusion-keras/tree/master) repo. Under MIT Licence.

## Licence
MIT
