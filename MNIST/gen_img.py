'''
usage: python gen_diff.py -h
'''

from __future__ import print_function
import argparse
from time import time
import random
from keras.datasets import mnist
from keras.layers import Input
# from scipy.misc import imsave
from imageio import imsave
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

# read the parameter
# argument parsing 输入参数
# MNIST数据集中差异诱导输入生成的主要功能
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
#transformation，现实转换类型，可选light、occl、blackout
parser.add_argument('--transformation', help="realistic transformation type", default='light', choices=['light', 'occl', 'blackout'])
# weight_nc，控制神经元覆盖的超参数
parser.add_argument('--weight_nc', help="weight hyperparm to control neuron coverage", default=1, type=float)
#step，梯度下降的步长
parser.add_argument('--step', help="step size of gradient descent", default=10, type=float)
# seeds，输入种子数
parser.add_argument('--seeds', help="number of seeds of input", default=100, type=int)
#grad_iterations，梯度下降的迭代次数
parser.add_argument('--grad_iterations', help="number of iterations of gradient descent", default=50, type=int)
# threshold，确定神经元激活的阈值(选取分布的十分之一、八分之一、六分之一、中值（10，8，6，5）)7
parser.add_argument('--threshold', help="threshold for determining neuron activated", default=10, type=float)
#遮挡左上角坐标
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
#遮挡大小
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)
#目标模型
parser.add_argument('--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=1, type=int)
args = parser.parse_args()


if args.target_model == 0:
    k = 2
elif args.target_model == 1:
    k = 3
elif args.target_model == 2:
    k = 4
num = 0
gen_time = []
orig_cov= []
neron_coverge = []
start_neuron = []
neuron_high = []
# 输入图像尺寸28*28
img_rows, img_cols = 28, 28
# 加载数据，数据在训练和测试集中被打乱和分割
(_, _), (x_test, y_test) = mnist.load_data()

# shape是查看数据有多少行多少列
# reshape()是数组array中的方法，作用是将数据重新组织。reshape新生成数组和原数组公用一个内存，不管改变哪个都会互相影响。
# x_test.shape[0]=10000，表示测试集数量；原本x_test为10000*28*28，转化为10000*28*28*1
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
model1 = Model2(input_tensor=input_tensor)
# model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)

# 初始化覆盖表，除了flatten和input的所有神经元
model_layer_dict1 = init_coverage_tables(model1)
# ==============================================================================================
# 开始迭代输入
for _ in range(args.seeds):
    print('第',_, '次迭代')
    # 从测试集中随机选择一个样本，并在0位置添加数据。（random.choice(x_test)为28*28*1，gen_img为1*28*28*1）
    # gen_img = np.expand_dims(random.choice(x_test), axis=0)
    i = random.randint(0,9999)
    gen_img = np.expand_dims(x_test[i],axis=0)
    gen_img_label = y_test[i]
    # 复制gen_img
    orig_img = gen_img.copy()
    # 初始图片神经元覆盖率
    # 更新覆盖率
    update_coverage(gen_img, model1, model_layer_dict1, k, args.threshold)
    orig_cov.append(neuron_covered(model_layer_dict1)[2])
    # 输出当前神经元的覆盖信息
    print(bcolors.HEADER + 'covered neurons percentage %d neurons %.3f'
          % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)
    # argmax(a, axis=None, out=None) 返回axis维度的最大值的索引,axis参数为None时默认比较整个数组，参数为0按列比较，参数为1按行比较。
    label1 = np.argmax(model1.predict(gen_img)[0])
    # 计算纯度
    deepgini1 = deep_metric(model1.predict(gen_img))
    print("初始deepgini值：", deepgini1)
    # 如果输入导致模型的deepgini值大于等于0.5输入存储在gnerated_inputs中
    if (deepgini1 >= 0.45):
        print(bcolors.OKGREEN + 'input already causes different outputs: {}'.format(label1) + bcolors.ENDC)
        # 更新覆盖率
        update_coverage(gen_img, model1, model_layer_dict1, k, args.threshold)
        # 将张量转换为有效图片
        gen_img_deprocessed = deprocess_image(gen_img)

        # 将已经存在差异的测试输入结果保存到磁盘上
        imsave('./deepgini_inputs_new/model1/already/' + 'already_differ_' + str(gen_img_label) + '_' + str(_) + '.png', gen_img_deprocessed)
        continue
    print(bcolors.OKGREEN + 'deepgini小于阈值，更新图片...... ' + bcolors.ENDC)
    start_n = neuron_covered(model_layer_dict1)[2]
    start_neuron.append(start_n)
    start = time()
    # 如果deepgini值小于1
    orig_label = label1
    # 选择未被激活的神经元
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    print('选择激活的神经元为：', layer_name1, index1 )
    # 构造联合损失函数
    # 神经元覆盖率损失
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    # 类纯度损失
    loss1_purity = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
    # 联合优化问题
    layer_output = args.weight_nc * (loss1_neuron) + loss1_purity
    # for adversarial image generation用于对抗性图像生成
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss我们计算了输入图像的梯度wrt这个损失
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # 此函数返回给定输入图片的损失和梯度
    iterate = K.function([input_tensor], [loss1_neuron,loss1_purity,grads])
    # we run gradient ascent for 20 steps梯度上升迭代10次
    for iters in range(args.grad_iterations):
        loss_neuron1,loss_purity1, grads_value = iterate([gen_img])
        # 自定义的一些约束
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value
        # 每一次迭代生成的图片
        gen_img += grads_value * args.step
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        # print(predictions1)
        NGini1 = deep_metric(model1.predict(gen_img))
        last_deepgini = NGini1
        print(last_deepgini)
        # if (last_deepgini < pro_deepgini):
        if NGini1 >= 0.45:
            end = time()
            ttime = end - start
            gen_time.append(ttime)
            print(bcolors.OKBLUE + "更新后的deepgini:",last_deepgini)
            num = num + 1
            # 更新覆盖率
            update_coverage(gen_img, model1, model_layer_dict1, k, args.threshold)
            # 输出当前神经元的覆盖信息
            print(bcolors.HEADER + 'covered neurons percentage %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2]) + bcolors.ENDC)
            end_n = neuron_covered(model_layer_dict1)[2]
            neron_coverge.append(end_n)
            # print(bcolors.OKGREEN + 'averaged covered neurons'+ bcolors.OKGREEN)
            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)
            neuron_high.append(end_n - start_n)
            # save the result to disk
            imsave('./deepgini_inputs_new/model1/' + args.transformation + '_' + str(orig_label) + '_' + str(predictions1) + '_' + str(_) +'.png',
                   gen_img_deprocessed)
            imsave('./deepgini_inputs_orig/model1/' + args.transformation + '_' + str(orig_label) + '_' + str(_) + 'orig.png', orig_img_deprocessed)

            print(bcolors.WARNING + "生成图片所需时间为："+str(end-start))
            break
print(num)
# data_write('time.csv',gen_time)
# print(gen_time)
# print(neron_coverge)
# print(max(neron_coverge))
# sum = 0
# for i in neron_coverge:
#     sum = sum + i
# print(sum/len(neron_coverge))
# print("high:")
# print(neuron_high)
# print(max(neuron_high))
print(orig_cov)
print(max(orig_cov))
