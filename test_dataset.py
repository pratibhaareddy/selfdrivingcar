import cv2
import math
import tensorflow as tf
import scipy.misc
from subprocess import call
import model
import os


tf.train.Saver().restore(tf.InteractiveSession(), "saved/model.ckpt")
xs = []
ys = []
initial_angle = 0
#read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        ys.append(float(line.split()[1]) * scipy.pi / 180)
#Find total number of images
num_images = len(xs)
#x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, 1])
i = math.ceil(num_images*0.8)
print("Starting frameofvideo:" +str(i))
img = cv2.imread('steering.jpg',0)
rows,cols = img.shape

while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    print("Steering angle: " + str(degrees) + " (predicted)\t" + str(ys[i]*180/scipy.pi) + " (actual)")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    initial_angle += 0.2 * pow(abs((degrees - initial_angle)), 2.0 / 3.0) * (degrees - initial_angle) / abs(degrees - initial_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-initial_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
