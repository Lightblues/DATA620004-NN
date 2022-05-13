
- 采用包: <https://github.com/facebookresearch/detectron2>
- 训练日志保存于 `logs`
- 模型 <https://github.com/Lightblues/NN-pj/releases/tag/v0.0.1>

### 环境配置

安装 detectron2

```python
pip install pyyaml==5.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

也可以下载编译好的

```python
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html
```

### 训练

准备VOC数据, 放在 `DETECTRON2_DATASETS` 目录下.

```sh
export DETECTRON2_DATASETS=../datasets/
export CONFIG_PATH=./configs/PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml
export MODEL_PATH=./tools/output/model_final.pth

# train
python ./train_net.py --num-gpus 4 \
    --config-file $CONFIG_PATH --resume
# tensorboard
tensorboard --logdir ./output

# eval
python ./train_net.py \
    --config-file $CONFIG_PATH \
    --eval-only \
    MODEL.WEIGHTS $MODEL_PATH

# help
python ./train_net.py -h
```

### 绘图

```sh
# 绘制 proposals
python get_proposals.py --config-file $CONFIG_PATH \
    --input ../sample-fig/000001.jpg ../sample-fig/000010.jpg ../sample-fig/000040.jpg ../sample-fig/000080.jpg \
    --output output_proposals \
    --opts MODEL.WEIGHTS $MODEL_PATH MODEL.DEVICE cpu

# 绘制 bbox
python get_boundingbox.py --config-file $CONFIG_PATH \
    --input ../sample-fig/cat.jpg ../sample-fig/furniture.jpg ../sample-fig/persons.jpg \
    --output ./output_boundingbox \
    --opts MODEL.WEIGHTS $MODEL_PATH MODEL.DEVICE cpu
```

### 实验设置

- 数据集: [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html).
    - 训练集: 2007trainval + 2012trainval
    - 测试集: 2007test
- batch size: 16
- learning rate: 0.02
- optimizer: SGD with momentum
- loss: total loss with 4 parts: cls, box reg, rpn cls, rpn reg.
- metric: mAP
