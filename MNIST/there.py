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
model1 = Model1(input_tensor=input_tensor)

#随机选择测试样本
i = random.randint(0,9999)
gen_img = np.expand_dims(x_test[i],axis=0)
#
# def getNerons_output3(input_data, model):
#     nerons_output = []
#     layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
#     for layer_name in range(0,len(layer_names) - 4):
#         t_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[layer_name]).output)
#         t_output = t_layer_model.predict(input_data)
#         ttype = t_output.shape
#         for i in range(ttype[3]):
#             neron = []
#             for j in range(ttype[0]):
#                 for k in range(ttype[1]):
#                     for m in range(ttype[2]):
#                         neron.append(t_output[j][k][m][i])
#             nerons_output.append(np.mean(scale(np.array(neron))))
#     for i in range(len(layer_names) - 4,len(layer_names)):
#         tt_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[i]).output)
#         tt_output = tt_layer_model.predict(input_data)
#         print(scale(tt_output[0]))
#         for j in scale(tt_output[0]):
#             nerons_output.append(j)
#     return nerons_output
xx = getNerons_output(gen_img,model1,2)
print(xx)
# i = len(xx)
# xx.sort()
# print(xx)
# id = i // 8
# print(id)
# threshold = xx[id]
# print(threshold)