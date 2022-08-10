import cv2
import numpy as np

class DRG():
    def __init__(self):
        pass

    def ellipse(self):
        blank_image = np.zeros((40, 40, 3), np.uint8)
        cv2.imshow('win',blank_image)
        cv2.waitKey(0)


blank_image = np.zeros((720, 720, 3), np.uint8)
blank_image.fill(255)

mat = cv2.imread(filename='./models/000.png').astype('uint8')
mat = cv2.resize(mat,(720,720))
cv2.ellipse(img=blank_image,center=(200,400),axes=(100,200),angle=40,startAngle=0,endAngle=360,
                    color=(0,0,0),thickness=-1)
mat = cv2.bitwise_and(mat,blank_image)
cv2.imshow('win', mat)

cv2.waitKey(0)