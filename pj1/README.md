作业：构建两层神经网络分类器

## 训练步骤

```bash
# packages
pip install -r requirements.txt

# 下载, 预处理数据
python preprocess.py
# 训练/验证模型
python main.py
```

## 作业要求

至少包含以下三个代码文件/部分

- 训练：
    - 激活函数
    - 反向传播，loss以及梯度的计算
    - 学习率下降策略
    - L2正则化
    - 优化器SGD
    - 保存模型
- 参数查找：学习率，隐藏层大小，正则化强度
- 测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度

数据集：MINIST；

- **不可使用pytorch，tensorflow等python package**，可以使用numpy；
- 代码上传到自己的public github repo，repo的readme文件中编辑好训练和测试步骤；
- 训练好的模型上传到百度云/google drive等网盘。
- 每位同学提交各自的作业（注意：本次作业不可组队，期中作业可以组队）；
- 提交形式：实验报告（pdf格式），实验报告内包含github repo链接，模型网盘下载地址；
- 可视化 **训练和测试的loss曲线，测试的accuracy曲线，以及可视化每层的网络参数**。

Deadline: 4月10日晚23:59
