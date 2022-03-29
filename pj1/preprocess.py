""" @220329
下载、转换、保存数据
数据: MNIST http://yann.lecun.com/exdb/mnist/
 """


import os
import numpy as np
from urllib import request
import gzip
import pickle

dpath = "./data"
# 源文件名到数据集名字
filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]

def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        path = f"{dpath}/{name[1]}"
        if os.path.exists(path):
            print(f"{name[1]} already exists.")
            continue
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], path)
    print("Download complete.")

def save_mnist():
    """ 解压, 读取, 保存数据 """
    mnist = {}
    for name in filename[:2]:
        with gzip.open(f"{dpath}/{name[1]}", 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(f"{dpath}/{name[1]}", 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open(f"{dpath}/mnist.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def load_mnist():
    with open(f"{dpath}/mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

download_mnist()
save_mnist()
X_train, y_train, X_test, y_test = load_mnist()
print(X_train.shape, X_test.shape)

