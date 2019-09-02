#！/usr/bin/env python

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from cv2 import imread, imwrite
import time
import os

CONTENT_IMG = 'MM800.jpg'
STYLE_IMG = 'fangao.jpg'
OUTPUT_DIR = 'neural_style_transfer_tensorflow/'

NOISE_RATIO = 0.7
IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

def the_current_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

class VGG19():
    def __init__(self, weights_file=None):
        pass
        
    def model(self, sess, input_image):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],  #在arg_scope中被指定的参数值，可以局部共享，也可以在局部位置进行覆盖。
    #                         activation_fn=tf.nn.relu,  #默认了
                            weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1), 
                            weights_regularizer = slim.l2_regularizer(0.0005, ),
                            ):
            #conv1 - 1,2 循环2次 卷积层
            net = slim.repeat(input_image, 2, slim.conv2d, num_outputs=64, kernel_size=[3,3], scope='conv1' ) #会智能展开成 conv1_1,conv1_2
            #maxpool1
            net = slim.max_pool2d(net, [2,2], scope='pool1')
            #conv2 - 1,2
            net = slim.repeat(net, 2, slim.conv2d, 128, [3,3], scope='conv2')
            print('pool1', net)
            #maxpool2
            net = slim.max_pool2d(net, [2,2], scope='pool2')        
            #conv3 - 1,2,3,4
            net = slim.repeat(net, 4, slim.conv2d, 256, [3,3], scope='conv3')
            #maxpool3
            net = slim.max_pool2d(net, [2,2], scope='pool3')        
            #conv4 - 1,2,3,4
            net = slim.repeat(net, 4, slim.conv2d, 512, [3,3], scope='conv4')
            #maxpool4
            net = slim.max_pool2d(net, [2,2], scope='pool4')     
            #conv5 - 1,2,3,4
            net = slim.repeat(net, 4, slim.conv2d, 512, [3,3], scope='conv5')
            #maxpool5
            net = slim.max_pool2d(net, [2,2], scope='pool5')           
            #fully_connected1
            net = slim.fully_connected(net, 4096, scope='fc1')
            net = slim.dropout(net, keep_prob=0.5, scope='dropout1')
            #fully_connected2
            net = slim.fully_connected(net, 4096, scope='fc2')
            net = slim.dropout(net, scope='dropout2')        
#             net = slim.batch_norm(net) #效果不错？
            net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc3') #设置为None以跳过激活函数 保持线性激活。
            
        return net

def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
    noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image
 
#加载图片
def load_image(path):
    image = imread(path)
#     print(image, image.shape)
    return image

#保存图片
def save_image(path, image):
    imsave(path, image)
    
def load_pre_weights():
    """加载预训练参数"""
    PRE_WEIGHTS_FILE = os.getcwd() + '\\vgg19.npy'
    print(PRE_WEIGHTS_FILE)       
    data_dict = np.load(PRE_WEIGHTS_FILE,  encoding="latin1").item()  
    for key in sorted(data_dict.keys())[:]:
          with tf.variable_scope(k, reuse=True):
#               w, b = data_dict[k][0], data_dict[k][1]
              for w, b in zip(('weights', 'biases'), data_dict[key]):
                  pass
              
def print_all_variable(is_train_only = False):
    if is_train_only: #只可训练参数
        t_vars = tf.trainable_variables()
        print(' [*] printing trainable variables...')
    else:
        try: #TF1.0
            t_vars = tf.global_variables()
        except: #TF0.12
            t_vars = tf.all_variables()
        print(' [*] printing global variables...')
    for idx, v in enumerate(t_vars):
        print('  var {:3}: {:5}  {}'.format(idx, str(v.get_shape()), v.name))

def main():
 
    
    content_image = load_image(CONTENT_IMG)
    print('content_image', content_image)
    style_image = load_image(STYLE_IMG)
    inputs_image = generate_noise_image(content_image)
     
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('/log', sess.graph)
        
        vgg19 = VGG19(PRE_WEIGHTS_FILE)
        
        sess.run(vgg19.model(sess, inputs_image))
         
        print('ok')    

        
if __name__ == '__main__':
    sess = tf.Session()
    load_pre_weights(sess)

    main()
    
    
    
    
    