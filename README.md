# Latent Diffusion Transformer
Implementation of the Diffusion Transformer model in the paper:

> [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). 

<img src="./images/ldt.png" width="450px"></img>

See [here](https://github.com/facebookresearch/DiT) for the official Pytorch implementation.

## Dependencies
- Python 3.8
- Tensorfow 2.12

## Training AutoencoderKL
Use `--train_file_pattern=<file_pattern>` and `--test_file_pattern=<file_pattern>` to specify the train and test dataset path.
```
python ae_train.py --train_file_pattern='./train_dataset_path/*.png' --test_file_pattern='./test_dataset_path/*.png' 
```

## Training Diffusion Transformer
Use `--file_pattern=<file_pattern>` to specify the dataset path.
```
python ldt_train.py --file_pattern='./dataset_path/*.png'
```
*Training DiT requires the pretrained AutoencoderKL. Use `ae_dir` and `ae_name` to specify the AutoencoderKL path in the `ldt_config.py` file.

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


## Samples
Curated samples from FFHQ

<img src="./images/generate_200.gif" width="700px"></img>

## Licence
MIT