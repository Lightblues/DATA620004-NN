# Vision-Transformer on cifar100

Vision-Transformer for cifar100 with pytorch

## Environment setup 
```bash
pip3 install -r requirements.txt
```
## Train the model
training without data augmentation
```bash
python train.py -net vit_small -gpu
```
training with data augmentation (mixup as example, cutout,cutmix available)
```bash
python train.py -net vit_small -gpu -data_aug mixup
```

## Test the model
```bash
python test.py -net vit_small -weights path/to/vit_weights -gpu
```

## Tensorboard
```bash
tensorboard --logdir='runs' --port=6006 --host=localhost
```

## Reference
["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy, A., et. al, (ICLR'21)](https://openreview.net/forum?id=YicbFdNTTy)
