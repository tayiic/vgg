import numpy as np
import tensorflow as tf
from cv2 import imwrite, imread, resize, imshow, addWeighted
import os
import matplotlib.pyplot as plt

# WORK_PATH = os.path.abspath(os.path.dirname(__file__)) + '\\'
WORK_PATH = os.getcwd()
LIB_PATH = os.path.join(WORK_PATH, '..', 'lib')

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

def find_uninitialized_tensor(sess):
    """查看 未被初始化的tensor"""
    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)    
    print("未初始化: ", uninitialized_vars)
    return uninitialized_vars        

def initialize_uninit_tensor(sess):
    """对新的未被初始化的tensor进行初始化"""
    un_vars = find_uninitialized_tensor(sess) 
    sess.run(tf.variables_initializer(un_vars))   


def gram_matrix(data):
    """求各层（特征）之间的两两相关性 gram矩阵"""
    if isinstance(data, np.ndarray):
        channels = int(data.shape[3]) #最后一维 通道数
    else: 
        channels = int(data.get_shape()[3]) #最后一维 通道数
    _F = tf.reshape(data, shape=[-1, channels]) #把每一层的长*宽拉成一维
    # 各层（特征）之间的两两相关性 gram矩阵可以看做feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵）
    gram = tf.matmul(tf.transpose(_F), _F)
    return gram

#加载图片 
def load_image(path):
    ##判断地址
    return imread(path)
#保存图片
def save_4D_image(path, image):
    data = np.squeeze(image)
#     print((data[:,1,1]))
#     show_image(data)
#     info = np.iinfo(data.dtype) # Get the information of the incoming image type
#     data = data.astype(np.float64) / info.max # normalize the data to 0 - 1
#     data = 255 * data # Now scale by 255
    img = data.astype(np.uint8)
#     show_image(img)
#     print((data[:,1,1]))
    save_image(path, img)

def save_image(path, image):
    imwrite(path, image)

def show_4D_image(img):
    image = np.squeeze(img)
    show_image(image)

def show_image(img):
    """显示图片"""
    from matplotlib import pyplot as plt
#     import cv2
    if not isinstance((img[0,0,0]), np.uint8):
        image = img.astype(np.uint8)
    else: image = img
    # # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    # cv2.waitKey (0)  #不添这句，在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过。
    # cv2.destroyAllWindows() #最后释放窗口是个好习惯！
    image = image[:,:,::-1] # 必须为 ::-1
    plt.imshow(image)
    # load image using cv2....and do processing.
    # plt.imshow(cv2.cvtColor(img2, cv2.BGR2RGB))
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()

def generate_image(content_img_path, ratio=0.7):
    #初始化 生成随机噪声图，与content图以一定比率融合
    content_img = load_image(content_img_path).astype('float32')
    img = np.random.uniform(-20, 20, (600, 800, 3)).astype('float32')
    #混合
    image = addWeighted(img, ratio, content_img, 1-ratio, 0)
#     image = resize(np.squeeze(image), (224, 224)).astype(np.float32)
#     show_image(image)
    image = preprocess_img(image)
#     show_4D_image(image)
    return image

def preprocess_img(img):
    """对输入的3维图像进行处理"""
    image = resize(img, (224, 224)).astype(np.float32)
    image[:,:,:] -= [103.939, 116.779, 123.68] #GRB三个通道        
    # image = image.transpose((2,0,1))
    # print(image.shape)
#         utils.show_image(image)
    image = np.expand_dims(image, 0)  
    return image 

def image_pre(path):
    """对图像进行预处理  使其符合vgg19输入格式 输出4维"""
    image = load_image(path)
    image = preprocess_img(image)
    return image   # TF模型的输入(“image”)应该是[批次，高度，宽度，通道]

def undo_preprocess_img(img):
    return img + [103.939, 116.779, 123.68] #GRB三个通道  
    
if __name__ == '__main__':
#     print_all_variable()
    img = generate_image(os.path.join(LIB_PATH, 'MM800.jpg'))
    print(img.shape)
# #     show_4D_image(img)
#     path = os.path.join(LIB_PATH, 'save.jpg')
#     save_4D_image(path,img)
