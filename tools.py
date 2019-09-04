import numpy as np
import tensorflow as tf
from cv2 import imwrite, imread, resize, imshow
import os

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

# def generate_noise_image(content_image, noise_ratio=NOISE_RATIO):
#     noise_image = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, COLOR_C)).astype('float32')
#     input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
#     return input_image

#加载图片 
def load_vgg19_image(path):
    """加载vgg19输入格式的图片"""
    image = imread(path)
    print(LIB_PATH)
    image = resize(image, (224, 224)).astype(np.float32)
    image[:,:,:] -= [103.939, 116.779, 123.68] #GRB三个通道
#     print(image)
#     show_image(image)
    # image = image.transpose((2,0,1))
    # image = resize(image, (IMAGE_H, IMAGE_W))
    # print(image.shape)
    # show_image(image)
    image = np.expand_dims(image, 0)
    print(image.shape)
    return image   # TF模型的输入(“image”)应该是[批次，高度，宽度，通道]

def load_vgg19_test_image():
    """一张测试图片"""
    path =  os.path.join(LIB_PATH, 'MM800.jpg')
    print(path)
    return load_vgg19_image(path)

#保存图片
def save_image(path, image):
    imwrite(path, image)

def show_image(img):
    """显示图片"""
    from matplotlib import pyplot as plt
    import cv2
    if isinstance((img[0,0,0]), np.float32):
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
    
    
if __name__ == '__main__':
#     print_all_variable()
    img = load_vgg19_test_image()
    print(img)
