import numpy as np
import tensorflow as tf
import os
# import tensorflow.contrib.slim as slim
import time

import tools
from tools import WORK_PATH, LIB_PATH, print_all_variable
from vgg19 import VGG19

# os.path.dirname(os.path.abspath(__file__)) #命令行不能用这个
#os.path.realpath(__file__)是脚本所在的绝对路径，os.getcwd()是工作目录，默认情况下是一样的

# The picture needed converting
CONTENT_IMG = os.path.join(LIB_PATH, 'MM800.jpg')
# The style picture
STYLE_IMG = os.path.join(LIB_PATH, 'fangao.jpg')

OUTPUT_DIR = LIB_PATH

NOISE_RATIO = 0.7  # The noise ratio, which is used to generate the initial picture
IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

def what_is(out):
    from classification import labels
    index = np.argsort(out)[::-1] #降序
    for i in index[:5]:
        print('label{} is {}  prob:{:.1%}'.format(i, labels[i], out[i]))

def main():
#     print(content_image)
    input_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "Input" )
    vgg19 = VGG19()
    vgg19.forward(input_img)    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('d:/mywork/vgg/log', sess.graph)

        vgg19.load_pre_para(sess, without=['fc6','fc7','fc8']) #加载预训练参数 可选择忽略的 ['fc7',fc8'...]
#         print_all_variable()
        
        for key in sorted(vgg19.data_dict.keys()):
            with tf.variable_scope('vgg19/'+key, reuse=True):    
#                 print(key, '参数:')
#                 print('加载预训练参数后：')
                a = sess.run(tf.get_variable('weights'))
#                 print(a[0])
#                 print('原来的预训练参数：')
                b = vgg19.data_dict[key][0] 
#                 print(b[0])
                if not (a == b).all():  #判断矩阵是否相等
                    print(key + "没有加载预训练参数" )

        img_in = tools.load_vgg19_image('../lib/unbr.jpg')
        _out = sess.run(vgg19.fc8, {input_img:img_in})
        
        what_is(_out[0]) #分类结果
        
if __name__ == '__main__':
    main()
    
    