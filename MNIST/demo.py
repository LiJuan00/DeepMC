from __future__ import print_function
import argparse
from time import time
import random

import numpy as np
from keras.datasets import mnist
from keras.layers import Input
# from scipy.misc import imsave
from imageio import imsave
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

# 输入图像尺寸28*28
img_rows, img_cols = 28, 28
# 加载数据，数据在训练和测试集中被打乱和分割
(_, _), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# input_shape为28*28*1
input_shape = (img_rows, img_cols, 1)
# 规定x_test数组数据类型为float32
x_test = x_test.astype('float32')
# 数据归一化
x_test /= 255

# 定义输入的占位符
input_tensor = Input(shape=input_shape)

#加载共享相同输入的多个模型
model3 = Model3(input_tensor=input_tensor)

# 初始化覆盖表，除了flatten和input的所有神经元
model_layer_dict1 = init_coverage_tables(model3)

#随机选择测试样本
i = random.randint(0,9999)
gen_img = np.expand_dims(x_test[i],axis=0)
# def update_coverage(input_data, model, model_layer_dict):
# 去掉flatten和input层的模型各层名称
# layer_names = [layer.name for layer in model1.layers if 'flatten' not in layer.name and 'input' not in layer.name]
# print(layer_names)
# for i in range(len(layer_names)):
#     dense1_layer_model = Model(inputs=model1.input, outputs=model1.get_layer(layer_names[i]).output)
#     dense1_output = dense1_layer_model.predict(gen_img)
#
#     # # print("[get output by layers index]")
#     x = dense1_output.shape
#     print(x)
update_coverage(gen_img,model3,model_layer_dict1,4,2)
# # print(scale(dense1_output[0]))
