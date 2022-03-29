import numpy as np
import pickle
import random
import matplotlib.pyplot as plt

""" @220329
训练、验证 模型.
超参数搜索
训练过程可视化
网络参数可视化

模型: 单隐层网络
数据: MNIST
 """

""" 激活函数 """
def relu(z):
    return np.maximum(z, 0)

def relu_prime(z):
    return z > 0

def softmax(z):
    # print(np.mean(z))
    return np.exp(z) / np.sum(np.exp(z))

def vectorize(y, classes=10):
    e = np.zeros((classes, 1))
    e[y] = 1.0
    return e

""" 交叉熵损失 """
def cross_entropy_loss(y_pred, y_true):
    loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    return np.mean(loss)

class Net(object):
    def __init__(
        self,
        d_hidden=50,
        learning_rate=1e-2,
        mini_batch_size=16,
        weight_decay=0.001,
        activation_fn="relu"
    ):
        # 输入图像为第0层
        sizes = [784, 50, 10]
        sizes[1] = d_hidden
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.activation_fn = eval(activation_fn)
        self.activation_fn_prime = eval(f"{activation_fn}_prime")
        self.weight_decay = weight_decay

        # W和b 序列, 注意 0层即为输入 redundant
        self.weights = [np.array([0])] + [np.random.randn(y, x)/np.sqrt(x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]
        self.biases = [np.array([0])] + [np.random.randn(y, 1) for y in sizes[1:]]

        # FC层输出, 第0层无意义
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        # 激活层输出, 第0层即为输入
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.lr = learning_rate

    def fit(self, training_data, validation_data, epochs=10):
        """ 训练模型 """
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []

        lr = self.lr
        step = 0 # 记录迭代次数, 进行学习率衰减
        for epoch in range(epochs):
            random.shuffle(training_data) # 60000
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)
            ]

            for i, mini_batch in enumerate(mini_batches):
                step += 1
                # 累计 batch梯度值
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                losses = []
                for x, y in mini_batch:
                    # 对于 batch 中的样本累计梯度
                    # x = np.reshape(x, (-1, 1))
                    logits = self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y) # 注意这里的 y 是向量化的
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                    losses.append(cross_entropy_loss(logits.squeeze(), y.squeeze()))
                train_loss.append(np.mean(losses))

                # lr 策略
                # (1) StepLR. 
                step_size=500
                gamma=0.99
                if step%step_size==0:
                    lr *= gamma

                # 优化器SGD
                self.weights = [
                    # L2正则化
                    w - lr * (dw/self.mini_batch_size + self.weight_decay * w) for w, dw in
                    zip(self.weights, nabla_w)
                ]
                self.biases = [
                    b - lr * (db/self.mini_batch_size) for b, db in
                    zip(self.biases, nabla_b)
                ]

                # 检查参数大小
                # if i%500 == 0:
                #     mess = f"Epoch {epoch} {i}: "
                #     for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                #         mess += f"({i}) {np.linalg.norm(w)} {np.linalg.norm(b)} "
                #     print(mess)

            loss, accuracy = self.validate(validation_data)
            print(f"Epoch {epoch + 1}, valid loss {loss: .5f}, accuracy {accuracy*100:.5f} %.")
            test_loss.append(loss)
            test_acc.append(accuracy)

        return train_loss, test_loss, test_acc

    def _forward_prop(self, x):
        """ 前向传播 """
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            if i == self.num_layers - 1:
                self._activations[i] = softmax(self._zs[i])
            else:
                self._activations[i] = self.activation_fn(self._zs[i])
        return self._activations[-1]

    def _back_prop(self, x, y):
        """ 反向传播 """
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y)
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                self.activation_fn_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def predict(self, x):
        return np.argmax(self._forward_prop(x))

    def validate(self, validation_data):
        """ 模型评估 """
        labels = [y for _, y in validation_data]
        logits = np.array([self._forward_prop(x).squeeze() for x, _ in validation_data])
        labels_onehot = np.array([vectorize(y).squeeze() for y in labels])
        loss = cross_entropy_loss(logits, labels_onehot)

        results = [y==ypred for y,ypred in zip(labels, np.argmax(logits, axis=1))]
        acc = sum(results) / len(results)
        return loss, acc


    def save(self, filename):
        """ 保存模型 """
        np.savez(
            file = f"./saved/{filename}",
            weights = self.weights,
            biases = self.biases,
            mini_batch_size = self.mini_batch_size,
            lr = self.lr,
        )
    def load(self, filename):
        data = np.load(f"./saved/{filename}.npz", allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])

        # 恢复参数
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = int(data['mini_batch_size'])
        self.lr = float(data['lr'])

def load_data(dpath="./data"):
    def load_mnist():
        with open(f"{dpath}/mnist.pkl",'rb') as f:
            mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    def normalize(X):
        # return X/255.0

        # print(np.mean(X), np.std(X))
        mean, std = 33.318, 78.567
        # X = (X - np.mean(X)) / np.std(X)
        X = (X - mean) / std
        return X.reshape(-1, 784, 1)

    X_train, y_train, X_test, y_test = load_mnist()
    X_train = normalize(X_train)
    X_test = normalize(X_test)

    np.random.seed(116)
    _num_split = int(len(X_train) * 0.9)
    _perm = np.random.permutation(len(X_train))
    X_train, X_valid = X_train[_perm][:_num_split], X_train[_perm][_num_split:]
    y_train, y_valid = y_train[_perm][:_num_split], y_train[_perm][_num_split:]
    
    y_train = [vectorize(y) for y in y_train]

    train_data = list(zip(X_train, y_train))
    valid_data = list(zip(X_valid, y_valid))
    test_data = list(zip(X_test, y_test))

    return train_data, valid_data, test_data

def grid_search(p="lr", values=[0, .001, .01, .1, 1, 10, 100]):
    """ 参数查找：学习率，隐藏层大小，正则化强度 """
    paras = {
        "lr": 0.01,
        "wd": 0.1,
        "d": 30
    }

    for v in values:
        paras[p] = v
        print(f"[grid search] {p}: {v}")
        net = Net(d_hidden=paras['d'], learning_rate=paras['lr'], mini_batch_size=mini_batch_size, weight_decay=paras['wd'])
        net.fit(train_data, validation_data=test_data, epochs=epochs)
        loss, accuracy = net.validate(test_data)
        print(f"Test Accuracy: {accuracy*100:.4f}%.")
        net.save(f"lr={paras['lr']}+d={paras['d']}_wd={paras['wd']}.model")

def plot_loss(losses):
    plt.plot(losses)
    plt.show()
    plt.savefig("loss.png")

def test_model(modelname):
    """ 测试：导入模型，用经过参数查找后的模型进行测试，输出分类精度 """
    net = Net(d_hidden, learning_rate, mini_batch_size, weight_decay, "relu")
    net.load(modelname)
    loss, accuracy = net.validate(test_data)
    print(f"Test Accuracy: {accuracy*100:.4f}%.")

def train(modelname):
    net = Net(d_hidden, learning_rate, mini_batch_size, weight_decay, "relu")
    train_loss, test_loss, test_acc = net.fit(train_data, validation_data=valid_data, epochs=epochs)

    fig1 = plt.figure(1)
    plt.plot(train_loss)
    plt.title("Training loss")
    plt.savefig(f"figs/{modelname}_train_loss.png")

    fig2 = plt.figure(2)
    ax1 = fig2.add_subplot(111)
    ax1.plot(test_loss)
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(test_acc, 'r')
    ax2.set_ylabel("Accuracy", color='r')
    plt.title("Vaild loss and accuracy")
    plt.savefig(f"figs/{modelname}_valid_loss_acc.png")

    loss, accuracy = net.validate(test_data)
    print(f"Test Accuracy: {accuracy*100:.4f}%.")

    net.save(modelname)

def plot_weights(modelname):
    """ 可视化全连接层权重 """
    net = Net(d_hidden, learning_rate, mini_batch_size, weight_decay, "relu")
    net.load(modelname)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = [ax]
    weights = net.weights[1:]
    for column in range(len(weights)):
        # 第一层输入 784 太大了, 仅展示前100个
        # ax[0][column].imshow(weights[column][:, :50], origin="lower", vmin=0)
        heatmap = ax[0][column].pcolor(weights[column][:, :100])
        ax[0][column].set_title("Layer %i" % (column + 1))
    plt.colorbar(heatmap)
    fig.subplots_adjust(hspace=0.5)
    # plt.show()
    plt.savefig(f"figs/weights.png")

if __name__ == "__main__":
    np.random.seed(116)

    train_data, valid_data, test_data = load_data(dpath="./data")

    # 超参数设置
    d_hidden = 50
    learning_rate = 0.01
    mini_batch_size = 16
    weight_decay = 0.001
    epochs = 10

    # 参数搜索
    # grid_search('lr', values=[1e-5, 1e-4, 1e-3, 1e-2, .1, 1])
    # grid_search('d', values=[10, 30, 50, 100, 300, 500, 1000])

    # modelname = 'scheduler_None.model'
    # modelname = 'scheduler_StepLR_500_0.99.model'
    modelname = "final_model" # scheduler_None
    
    # 训练
    # train(modelname)
    # 测试
    test_model(modelname)

    # plot_weights(modelname)

