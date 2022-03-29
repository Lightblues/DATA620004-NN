截止 5月8日

## 任务说明

- 使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。
- 在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3；在四张测试图像上可视化Faster R-CNN第一阶段的proposal box；两个训练好后的模型分别可视化三张不在VOC数据集内，但是包含有VOC中类别物体的图像的检测结果（类别标签，得分，boundingbox），并进行对比，一共show六张图像；

完成以上实验1与实验2。

- 组队形式提交源代码(Github public repo地址)、训练模型、实验报告。
- 编辑组队邮件：组员姓名+学号，发送给助教；组队人数少于或等于3人（同等质量工作，1人或者2人有加分）。
- 可以使用pytorch或者tensorflow等python package；
- 代码上传到自己的public github repo，repo的readme文件中编辑好训练和测试步骤；
- 训练好的模型上传到百度云/google drive等网盘。

提交

- 提交形式：实验报告（pdf格式），实验报告内包含github repo链接，模型网盘下载地址；
- 详细的实验报告包括实验设置：
- 数据集介绍，训练测试集划分，网络结构，batch size，learning rate，优化器，iteration，epoch，loss function，评价指标，检测/分割结果可视化；
- 利用Tensorboard可视化训练和测试的loss曲线、测试AP/Acc/mIoU 曲线。
