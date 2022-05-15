截止 6月15日

## 任务说明

- 使用在 Cityscapes 数据集上开源的任意一个语义分割模型，网络下载一段驾驶视频（类似行车记录仪视频），对视频每一帧进行测试并可视化，结果视频上传至网盘；
- 对Faster R-CNN模型，分别进行以下训练：
    - a) 随机初始化训练VOC；
    - b) ImageNet预训练backbone网络，然后使用VOC进行fine tune；
    - c)使用coco训练的Mask R-CNN的backbone网络参数初始化Faster R-CNN的backbone网络，然后使用VOC进行fine tune；
- 设计与期中作业1模型相同参数量的Transformer网络模型，进行CIFAR-100的训练，并与期中作业1的模型结果进行比较，可使用data aug

注意：

作业1无需训练，只进行测试；
作业2无需自己训练ImageNet，无需自己训练COCO，下载开源模型；c任务中，Faster R-CNN的head为随机初始化；利用Tensorboard可视化abc三个任务的训练和测试的loss曲线、测试mAP曲线；
作业3中data aug可以选用任何可用的策略，包括期中作业使用的data aug；利用Tensorboard可视化期中作业1与期末作业3的训练和测试的loss曲线、测试acc曲线；

提交形式：实验报告（pdf格式），实验报告内包含github repo链接，模型网盘下载地址；

详细的实验报告包括实验设置：

- 数据集介绍，训练测试集划分，网络结构，batch size，learning rate，优化器，iteration，epoch，loss function，评价指标，data aug策略，分割结果可视化，检测对比结果可视化；
