# LiSenNet: Lightweight Sub-band and Dual-Path Modeling for Real-Time Speech Enhancement

This is the official implementation of [LiSenNet](https://arxiv.org/abs/2409.13285).



## Installation

1. Prepare a virtual environment with python, pytorch, and pytorch_lightning. (We use python==3.10.14, pytorch==2.0.0, and pytorch_lightning==2.0.7, but other versions probably also work.)
2. Install the package dependencies via `pip install -r requirements.txt`.



## Training

Before training, please check `config.yaml` to set hyperparameters, including **devices**, **logdir**, **dataset path**, ...

Then you can train the model by:

```
python train.py --config ./config.yaml
```



## Testing

```
python test.py --config ./config.yaml --ckpt_path <your-ckpt-path>
```

Or if you want to save enhanced audios, you can use `--save_enhanced` option like:

```
python test.py --config ./config.yaml --ckpt_path <path-to-ckpt> --save_enhanced <path-to-savedir>
```



## Reference

[CMGAN](https://github.com/ruizhecao96/CMGAN)
