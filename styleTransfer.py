import numpy as np
import tensorflow as tf
import os
# import tensorflow.contrib.slim as slim
import time

import utils
from utils import LIB_PATH, load_vgg19_image
from vgg19 import VGG19

# os.path.dirname(os.path.abspath(__file__)) #命令行不能用这个
#os.path.realpath(__file__)是脚本所在的绝对路径，os.getcwd()是工作目录，默认情况下是一样的

# The picture needed converting
CONTENT_IMG = os.path.join(LIB_PATH, 'MM800.jpg')
# The style picture
STYLE_IMG = os.path.join(LIB_PATH, 'fangao.jpg')

OUTPUT_DIR = LIB_PATH

NOISE_RATIO = 0.7  # The noise ratio, which is used to generate the initial picture
# IMAGE_W = 800
# IMAGE_H = 600
# COLOR_C = 3
#需要提取数据的层
CONTENT_LAYERS = ("conv4_2", "conv5_2")
STYLE_LAYERS = ("conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1")
#存放风格图形的特征（VGG不同层的特征值）
STYLE_FEATURES = {}
CONTENT_FEATURES = {}

#预处理好的图形
content_img_pre = load_vgg19_image(CONTENT_IMG)
style_img_pre = load_vgg19_image(STYLE_IMG)

def transfer():
    global STYLE_FEATURES
#     print(content_image)
    input_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "Input" )
    style_img_pre = load_vgg19_image(STYLE_IMG)
    vgg19 = VGG19()
    vgg19.forward(input_img) #搭建前向图
    
    if len(STYLE_FEATURES) != len(STYLE_LAYERS): #风格特征未提取
        with tf.Session() as sess:
    #     with tf.Graph.as_default(), tf.Graph.device('/Gpu:0'), tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    #         tf.summary.FileWriter('d:/mywork/vgg/log', sess.graph)
            #加载预训练数据
            vgg19.load_pre_para(sess, without=['fc6','fc7','fc8'])
    
            #需要提取风格的特征层
            conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 = sess.run([vgg19.conv1_1, vgg19.conv2_1, vgg19.conv3_1, 
                                            vgg19.conv4_1, vgg19.conv5_1], {input_img:style_img_pre})
            #把风格特征加入全局变量，免得重复计算
            for layer in STYLE_LAYERS:
                features = eval(layer)
                features = np.reshape(features, (-1, features.shape[3])) #把每一层的长*宽拉成一维
                #求各层（特征）之间的两两相关性 gram矩阵可以看做feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵）
                gram = np.matmul(features.T, features) / features.size #不除？
                STYLE_FEATURES[layer] = gram

        
if __name__ == '__main__':
    transfer()
    
    