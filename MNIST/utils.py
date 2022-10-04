import random
from collections import defaultdict

import xlwt
from PIL import Image
import numpy as np
from keras import backend as K
from keras.models import Model
import os
import re

# 将张量转换为有效图像
def deprocess_image(x):
    x *= 255
    # np.clip(a,a_min,a_max)函数的作用是将数组a中的所有数限定到范围a_min和a_max中。并将数据转换为8位无符号整型
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm用L2范数规范化张量的效用函数
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


# 初始化覆盖率表（每层每个神经元的覆盖状态均为false）
def init_coverage_tables(model1):
    # 默认字典，key值可自定义，value的类型与defaultdict()括号中设置类型的相同。
    model_layer_dict1 = defaultdict(bool)
    # 初始化字典
    init_dict(model1, model_layer_dict1)
    return model_layer_dict1

# 将除去flatten和input层外的所有层的神经元状态均记为false
def init_dict(model, model_layer_dict):
    for layer in model.layers:
        # 对于flatten和input层不做处理
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # 将本层的每一个输出张量的值设为false
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False

# 选择未激活的神经元，返回所在层及索引
def neuron_to_cover(model_layer_dict):
    # 未激活的神经元
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    # 如果存在未激活的神经元则从未激活中随机选择一个神经元，如果全部已激活，则随机选择一个神经元
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


# 计算神经元覆盖率，返回被覆盖神经元个数、神经元总数、神经元覆盖率
def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

# 更新覆盖率，将输入数据激活的神经元记为true,threshold为用户自定义设置
def update_coverage(input_data, model, model_layer_dict, flag, t):
    #计算threshold
    neron_output = getNerons_output(input_data,model,flag)
    l = len(neron_output)
    index = l // t
    threshold = neron_output[index - 1]
    print(threshold)
    # 去掉flatten和input层的模型各层名称
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True

# x_scaled=((神经元输出-层神经元最小输出)/(层神经元最大输出-层神经元最小输出))
def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

# # 修改了threshold
# def fired(model, layer_name, index, input_data):
#     intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
#     intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
#     scaled = scale(intermediate_layer_output)
#     threshold = np.mean(scaled)
#     if np.mean(scaled[..., index]) > threshold:
#         return True
#     return False


# def diverged(predictions1, predictions2, predictions3, target):
#     #     if predictions2 == predictions3 == target and predictions1 != target:
#     if not predictions1 == predictions2 == predictions3:
#         return True
#     return False

def getlabel(name):
    test_labels = []
    filenameArray = []
    for filename in os.listdir('H:\pythonProject\DeepGiniDeepXplore\MNIST\deepgini_inputs_new\\'+ name):
        if filename != 'already':
            filenameArray.append(filename)
            test_labels.append(int(re.findall("\d+", filename)[0]))
    test_labels = np.array(test_labels)
    return filenameArray,test_labels

# 获取新生成的差异数据
def getnewData(name):
    filename = getlabel(name)[0]
    img_new = []
    for i in range(len(filename)):
        img = Image.open('H:\pythonProject\DeepGiniDeepXplore\MNIST\deepgini_inputs_new\{}\{}'.format(name,filename[i]))
        img_new.append(np.array(img))
    img_new = np.array(img_new)
    return img_new

#deepgini，计算针对一个图像t类的纯度metric
def deep_metric(pred_test_prob):
    metrics = 1-np.sum(pred_test_prob**2)
    return metrics


#  将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    j = 0
    i = 0
    for data in datas:
        sheet1.write(i,j,data)
        i = i + 1
    f.save(file_path)  # 保存文件

# model的所有神经元输出
def getNerons_output(input_data, model, flag):
    nerons_output = []
    layer_names = [layer.name for layer in model.layers if 'flatten' not in layer.name and 'input' not in layer.name]
    for layer_name in range(0,len(layer_names) - flag):
        t_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[layer_name]).output)
        t_output = t_layer_model.predict(input_data)
        ttype = t_output.shape
        for i in range(ttype[3]):
            neron = []
            for j in range(ttype[0]):
                for k in range(ttype[1]):
                    for m in range(ttype[2]):
                        neron.append(t_output[j][k][m][i])
            nerons_output.append(np.mean(scale(np.array(neron))))
    for i in range(len(layer_names) - flag,len(layer_names)):
        tt_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_names[i]).output)
        tt_output = tt_layer_model.predict(input_data)
        # print(scale(tt_output[0]))
        for j in scale(tt_output[0]):
            nerons_output.append(j)
    nerons_output.sort()
    return nerons_output