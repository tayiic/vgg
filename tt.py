import cv2
import numpy as np
from cv2 import imread, imwrite, resize

def nothing(x):
    pass

#创建一个黑色图像
img = np.zeros((300,512,3),np.uint8)
cv2.namedWindow('image')

 
cv2.destroyAllWindows()