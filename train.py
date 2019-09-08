import numpy as np
import tensorflow as tf
import os
 
import utils
from vgg19 import VGG19


# The picture needed converting
C_IMG_PATH = os.path.join(utils.LIB_PATH, 'MM800.jpg')
# The style picture
S_IMG_PATH = os.path.join(utils.LIB_PATH, 'fangao.jpg')

#图片输出路径
OUTPUT_PATH = os.path.join(utils.LIB_PATH, 'save.jpg')
OUTPUT_NP_PATH = os.path.join(utils.LIB_PATH, 'save.npy')
TENSORBOARD_LOGDIR = 'e:/nnLog'

#需要提取特征数据的层
C_LAYERS = {"conv4_2":0.7, "conv5_2":0.3}  # name weight
S_LAYERS = {"conv1_1":0.2, "conv2_1":0.2, "conv3_1":0.2, "conv4_1":0.2, "conv5_1":0.2}

#存放风格图形的特征（VGG不同层的特征值）
S_FEATURES = {}
C_FEATURES = {}

CONTINUE = True #是否是中继训练

C_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "C_image" ) #内容原图
S_img = tf.placeholder(tf.float32, [None, 224, 224, 3], "S_image" ) #风格原图
if CONTINUE: #加载训练过的图片
    _img = np.load(OUTPUT_NP_PATH)
    print("中继训练 加载最后保存的 训练过的图片数据")
    X_img = tf.Variable(_img, name='X_image')
else:
    X_img = tf.Variable(utils.generate_image(C_IMG_PATH,0.3), name='X_image')

C_vgg = VGG19(C_img) 
S_vgg = VGG19(S_img, reuse=True) 
X_vgg = VGG19(X_img, reuse=True)  

"""compute loss"""
loss_style = 0.0 
loss_content = 0.0 
use_layers = tuple(C_LAYERS.keys()) + tuple(S_LAYERS.keys()) #目标图片要获取特征值的层
for layer in use_layers:
    X = eval('X_vgg.'+layer)#目标图片的layer层输出
    shape = X.get_shape() #输出维度
    size = tf.cast(np.prod(shape[1:]), tf.float32) #累乘各维度

    if layer in S_LAYERS.keys(): # 
        #风格图片的layer层特征值（gram）
        # assert len(S_FEATURES)!=len(S_LAYERS), "S_FEATURES未初始化"
        if(len(S_FEATURES)!=len(S_LAYERS)):
            print(layer, " 在S_FEATURES中未存，直接计算")
            S_gram = utils.gram_matrix(eval('S_vgg.'+layer))
            S_FEATURES[layer] = S_gram
        else:
            print(layer, " 在S_FEATURES中已存...")
            S_gram = S_FEATURES[layer]
        X_gram = utils.gram_matrix(X) #目标图片gram
        print('loss', tf.nn.l2_loss(S_gram - X_gram) ,size)
        loss_style += loss * S_LAYERS[layer] # loss*weight
    elif layer in C_LAYERS.keys(): #
        #内容源图的layer层输出  
        if(len(C_FEATURES)!=len(C_LAYERS)):
            print(layer, " 在C_FEATURES中未存，直接计算")
            C = eval('C_vgg.'+layer)
            C_FEATURES[layer] = C
        else:
            print(layer, " 在C_FEATURES中已存...")
            C = C_FEATURES[layer]
        loss = tf.nn.l2_loss(C - X) / size
        loss_content += loss * C_LAYERS[layer] # loss*weight

# total variation denoising
# weights_tv = 1e2
# _h = tf.nn.l2_loss(X_img[:,1:,:,:] - X_img[:,:-1,:,:])
# _w = tf.nn.l2_loss(X_img[:,:,1:,:] - X_img[:,:,:-1,:])
# _s1 = tf.cast((X_img.shape[1]-1), tf.float32)
# _s2 = tf.cast((X_img.shape[2]-1), tf.float32)
# loss_total_tv = weights_tv * (_h/_s1 + _w/_s2)
# total loss
alpha = 0.5 #风格和内容loss调节的权重
loss = alpha*loss_style #+ loss_content #+ loss_total_tv 
tf.summary.scalar('Loss_total', loss)

global_steps = tf.Variable(0, trainable=False) 
starter_learning_rate  = 22.0
#实现指数衰减学习率 
#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
learning_rate = tf.train.exponential_decay(starter_learning_rate,  #初始学习率
                                           global_steps, #当前迭代次数
                                           50, #衰减间隔（在迭代到该次数时学习率衰减为learning_rate * decay_rate）
                                           0.9, #学习率衰减系数，通常介于0-1之间
                                           staircase=True)#(默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
iterations = 10000
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_steps)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    tf.summary.FileWriter(TENSORBOARD_LOGDIR, sess.graph) #tensorboard
    C_vgg.load_pre_para(sess, without=['fc6','fc7','fc8']) #加载预参数 一次就够 共享
    # #风格图片的layer层特征值
    # u, v = '', ''
    # for i in ['S_vgg.'+x for x in S_LAYERS.keys()]: v += i + ','
    # for i in [x for x in S_LAYERS.keys()]: u += i + ','
    # for i in S_LAYERS.keys(): locals()[i] = ''
    # #获取需要的各层的输出
    # exec(u[:-1] + '=sess.run([' + v[:-1] + '],{S_img:S_vgg.image_pre(S_IMG_PATH)})') 
    # #计算风格源图的gram并存放到公共变量
    # for layer in S_LAYERS.keys(): 
    #     S_FEATURES[layer] = utils.gram_matrix(eval(layer))
        
    #内容图片的layer层特征值

    for i in range(iterations):
        _, _img, _loss, _step = sess.run([train_step, X_img, loss, global_steps], {C_img:utils.image_pre(C_IMG_PATH), S_img:utils.image_pre(S_IMG_PATH)})
        if i%5==0: 
            print('{}: loss={}'.format(_step, _loss))
        if i%55==1: 
            utils.show_4D_image(utils.undo_preprocess_img(_img))
        if i%200==111:
            utils.save_4D_image(OUTPUT_PATH[:-4]+str(i)+'.jpg', utils.undo_preprocess_img(_img))   
            np.save(OUTPUT_NP_PATH, _img)    
           
# if __name__ == '__main__':
#     print('dddddddddd')
#     img = utils.image_pre(C_IMG_PATH)
#     print(img.shape)
#     utils.show_4D_image(img)
#     