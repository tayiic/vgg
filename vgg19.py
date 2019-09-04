########################################################################################
# Tayii, 2019                                                                  #
# VGG19 implementation in TensorFlow                                                   #
# Details:                                                                             #
#                                                                                      #
#                                                                                      #
# Model from https://      #
#       #
########################################################################################
import os
import tensorflow as tf
import numpy as np
import time

import tools  #自定义
from tools import WORK_PATH, LIB_PATH

VGG19_MODEL_DATA = None  #模型预训练参数 
NPY_PATH = os.path.join(LIB_PATH, 'vgg19.npy')  #模型预训练参数文件 

class VGG19():
    def __init__(self, vgg19_npy_path=None):
        global VGG19_MODEL_DATA
        if VGG19_MODEL_DATA is None:
            if vgg19_npy_path: #有传入npy文件的地址
                path = vgg19_npy_path
        else: 
            path = NPY_PATH
        #读取参数文件  
        try:
            VGG19_MODEL_DATA = np.load(NPY_PATH, encoding='latin1')
        except FileNotFoundError:
            print("""VGG16 weights were not found in the project directory
                  Please download the numpy weights file and place it in the '***/***' directory
                  Download link: https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
                  Exiting the program..""")
            exit(1)
        else:
            print("VGG net weights loaded")

        self.data_dict = VGG19_MODEL_DATA.item()  #预训练数据（字典）

    def forward(self, inputs):
        """前向传播"""
        start_time = time.time()        # 获取前向传播的开始时间
        with tf.variable_scope('vgg19'):
            self.conv1_1 = self.__conv_layer(inputs, 'conv1_1')
            self.conv1_2 = self.__conv_layer(self.conv1_1, 'conv1_2')
            self.maxpool1 = self.__maxpool_layer(self.conv1_2, 'maxpool1')
            
            self.conv2_1 = self.__conv_layer(self.maxpool1, 'conv2_1')
            self.conv2_2 = self.__conv_layer(self.conv2_1, 'conv2_2')
            self.maxpool2 = self.__maxpool_layer(self.conv2_2, 'maxpool2')
            
            self.conv3_1 = self.__conv_layer(self.maxpool2, 'conv3_1')
            self.conv3_2 = self.__conv_layer(self.conv3_1, 'conv3_2')
            self.conv3_3 = self.__conv_layer(self.conv3_2, 'conv3_3')
            self.conv3_4 = self.__conv_layer(self.conv3_3, 'conv3_4')
            self.maxpool3 = self.__maxpool_layer(self.conv3_4, 'maxpool3')        
            
            self.conv4_1 = self.__conv_layer(self.maxpool3, 'conv4_1')
            self.conv4_2 = self.__conv_layer(self.conv4_1, 'conv4_2')
            self.conv4_3 = self.__conv_layer(self.conv4_2, 'conv4_3')
            self.conv4_4 = self.__conv_layer(self.conv4_3, 'conv4_4')
            self.maxpool4 = self.__maxpool_layer(self.conv4_4, 'maxpool4')  
            
            self.conv5_1 = self.__conv_layer(self.maxpool4, 'conv5_1')
            self.conv5_2 = self.__conv_layer(self.conv5_1, 'conv5_2')
            self.conv5_3 = self.__conv_layer(self.conv5_2, 'conv5_3')
            self.conv5_4 = self.__conv_layer(self.conv5_3, 'conv5_4')
            self.maxpool5 = self.__maxpool_layer(self.conv5_4, 'maxpool5') 
    
            n = self.data_dict['fc6'][0].shape #拉平用的长度
            self.flatten = tf.reshape(self.maxpool5, [-1,n[0]], 'flatten')
            
            self.fc6 = self.__fc_layer(self.flatten, 'fc6', activation='relu')
            self.fc7 = self.__fc_layer(self.fc6, 'fc7', activation='relu')
            self.fc8 = self.__fc_layer(self.fc7, 'fc8', activation='softmax')
            
        end_time = time.time()
        print('vgg forward: time consuming: %f'% (end_time - start_time))
            
    def load_pre_para(self, sess, without=None):
        """
        "加载预训练数据
        "without: 不加载预训练参数的层
        """
        total = []
        for key in sorted(self.data_dict.keys())[:]:
            if (not without) or (key not in without): 
#                 print('覆盖...', key)
                with tf.variable_scope('vgg19/'+key, reuse=True):
                    for subkey, value in zip(('weights', 'biases'), self.data_dict[key]):
                        sess.run(tf.get_variable(subkey).assign(value))
                        total.append(key)
        print(total, '预训练参数覆盖完成.')

    def __fc_layer(self, net, name, activation=None, trainable=False):
        with tf.variable_scope(name):
            z = tf.matmul(net, self.__get_weights(name,  trainable=trainable)) \
                              + self.__get_biases(name, trainable=trainable)
            if activation == None: return z
            elif activation == 'relu': return tf.nn.relu(z)
            elif activation == 'softmax': return tf.nn.softmax(z)
            else: 
                print('输入activation={}错误'.format(activation))
                return None
            
    def __maxpool_layer(self, net, name):
        return tf.nn.max_pool(net,
                              ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                              padding = 'SAME',
                              name=name
                              )

    def __conv_layer(self, net, name, trainable=False):
        with tf.variable_scope(name):
            return tf.nn.relu(tf.nn.conv2d(net,
                                          filter = self.__get_conv_filter(name, trainable=trainable),
                                          strides = [1,1,1,1],
                                          padding = 'SAME'
                                         ) + self.__get_biases(name, trainable=trainable)
                              )

    def __get_conv_filter(self, name, trainable=False):
        return self.__get_weights(name, trainable)
        
    def __get_weights(self, name, trainable=False):
        data = self.data_dict[name][0]
        return tf.get_variable('weights',
                               shape = data.shape,
                               dtype = tf.float32,
                               initializer = tf.contrib.layers.xavier_initializer(),
                               trainable = trainable
                               )
            
    def __get_biases(self, name, trainable=False):    
        data = self.data_dict[name][1]
        return tf.get_variable('biases',        
                               shape = data.shape,
                               dtype = tf.float32,
                               initializer = tf.contrib.layers.xavier_initializer(),
                                trainable = trainable
                               )


if __name__ == '__main__':
    test_image = tools.load_vgg19_test_image()
    vgg19 = VGG19()
    vgg19.model(test_image)    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tools.print_all_variable()


 