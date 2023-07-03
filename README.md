# Latent Diffusion Transformer
Implementation of the Diffusion Transformer model in the paper:

> [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). 

<img src="./images/ldt.png" width="450px"></img>

See [here](https://github.com/facebookresearch/DiT) for the official Pytorch implementation.


## Dependencies
- Python 3.8
- Tensorfow 2.12

## Training AutoencoderKL
Use `--file_pattern=<file_pattern>` to specify the dataset path and file pattern.
```
python ae_train.py --file_pattern=./dataset_path/*.png
```

## Training Diffusion Transformer
Use `--file_pattern=<file_pattern>` to specify the dataset path and file pattern.
```
python ldt_train.py --file_pattern=./dataset_path/*.png
```

## Sampling
Use `--model_dir=<model_dir>` and `--ldt_name=<ldt_name>` to specify the pre-trained model. For example:
```
python sample.py --model_dir=ldt --ldt_name=model_1 --diffusion_steps=40
```


## Hparams setting
Adjust hyperparameters in the `ae_config.py` and `ldt_config.py` files.

Implementation notes:
- LDT is designed to offer reasonable performance using a single GPU (RTX 3080 TI).
- LDT largely follows the original DiT model.
- Modulated layer normalization.
- Diffusion Transformer with [Linformer](https://arxiv.org/abs/2006.04768) attention.
- Cosine schedule.
- [DDIM](https://arxiv.org/abs/2010.02502) sampler.
- [FID](https://arxiv.org/abs/1706.08500) evaluation.
- AutoencoderKL with PatchGAN discriminator and hinge loss.
- This implementation uses code from the [beresandras](https://github.com/beresandras/clear-diffusion-keras/tree/master) repo. Under MIT Licence.


## Licence
MIT