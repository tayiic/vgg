import numpy as np
import tensorflow as tf
import os
 
import utils
from vgg19 import VGG19


# The picture needed converting
C_IMG_PATH = os.path.join(utils.LIB_PATH, 'MM800.jpg')
# The style picture
S_IMG_PATH = os.path.join(utils.LIB_PATH, 'fangao.jpg')
#图片输出目录
OUTPUT_DIR = utils.LIB_PATH

#需要提取特征数据的层
C_LAYERS = {"conv4_2":0.7, "conv5_2":0.3}  # name weight
S_LAYERS = {"conv1_1":0.2, "conv2_1":0.2, "conv3_1":0.2, "conv4_1":0.2, "conv5_1":0.2}

#存放风格图形的特征（VGG不同层的特征值）
S_FEATURES = {}
C_FEATURES = {}


def main():
C_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "content_image" ) #内容原图
S_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "style_image" ) #风格原图

sess = tf.Session()

vgg19 = VGG19() #
 
#需要提取内容层的特征值
vgg19.forward(C_img, reuse=True)    
img = vgg19.image_pre(C_IMG_PATH)
for layer in C_LAYERS.keys():
    _F = eval('vgg19.'+layer) #层输出
    C_FEATURES[layer] = sess.run(_F, {C_img:img}) 
    
#需要提取风格层的特征值
vgg19.forward(S_img, reuse=True)    
img = vgg19.image_pre(S_IMG_PATH)
for layer in S_LAYERS.keys():
    _F = eval('vgg19.'+layer) #层输出
    _gram = utils.gram_matrix(_F)
    S_FEATURES[layer] = sess.run(_gram, {S_img:img})
 

"""compute loss"""
#目标图片 层的特征值
print('开始产生目标图片...')    
vgg19.forward(X_img, reuse=True) 

loss_style = 0.0 
loss_content = 0.0 
use_layers = tuple(C_LAYERS.keys()) + tuple(S_LAYERS.keys()) #目标图片要获取特征值的层

#先一次性获取完全部需要的层的值 节约计算时间
X_img.initializer.run(session=sess)
u, v = '', ''
for i in ['vgg19.'+x for x in use_layers]: v += i + ','
for i in [x for x in use_layers]: u += i + ','
for i in use_layers: locals()[i] = ''
exec(u[:-1] + '=sess.run([' + v[:-1] + '])') #获取各层的值

# 计算loss
for layer in use_layers:
    if layer in S_LAYERS.keys(): # 
        #风格图片的layer层特征值（gram）
        S_gram = S_FEATURES[layer]  
        #目标图片的layer层特征值
        _F = eval(layer)
        X_gram = utils.gram_matrix(_F)
        _shape = _F.shape #层维度
        _size = np.prod(_shape) #累乘各维度
        _loss = tf.nn.l2_loss(S_gram - X_gram) / pow(int(_size)*2.0, 2) 
        loss_style += tf.cast(_loss * S_LAYERS[layer], tf.float32) #乘以各层权重
    
    elif layer in C_LAYERS.keys(): #
        C = C_FEATURES[layer] #内容图片的layer层特征值
        #目标图片的layer层特征值
        F = eval(layer)
        _shape = F.shape #层维度
        _size = tf.cast(np.prod(_shape), tf.float32)
        _loss = tf.nn.l2_loss(C - F) / _size
        loss_content += tf.cast(_loss * C_LAYERS[layer], tf.float32) #乘以各层权重

# total variation denoising
# weights_tv = 1e2
# _h = tf.nn.l2_loss(X_img[:,1:,:,:] - X_img[:,:-1,:,:])
# _w = tf.nn.l2_loss(X_img[:,:,1:,:] - X_img[:,:,:-1,:])
# _s1 = tf.cast((X_img.shape[1]-1), tf.float32)
# _s2 = tf.cast((X_img.shape[2]-1), tf.float32)
# loss_total_tv = weights_tv * (_h/_s1 + _w/_s2)
# total loss
alpha = 0.5 #风格和内容loss调节的权重
loss = alpha*loss_style + loss_content #+ loss_total_tv 

learning_rate = 0.01
iterations = 111
tf.summary.FileWriter('e:/nnLog', sess.graph)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

initialize_uninit_tensor(sess) #初始化还未初始化的参数（VGG已经初始化）
for i in range(iterations):
    _, _img, _loss = sess.run([train_step, X_img, loss])
    if i%1==0: 
        print(_loss)
        # show_4D_image(img)
             
    sess.close()     
           
if __name__ == '__main__':
    main()
    
    