#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


bridge = CvBridge()

image = cv2.imread('src/linefollow_group/scripts/red.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
low_mask = cv2.inRange(image, np.array([0, 50, 50]), np.array([50, 255, 255])) > 0.0
high_mask = ~(cv2.inRange(image, np.array([200, 50, 50]), np.array([255, 255, 255])) > 0.0
mask = low_mask
# mask = low_mask | high_mask
image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
print(image.shape)
print(mask.shape)
print(mask.dtype)
print(np.mean(mask, 0))
image = image*np.expand_dims(mask, -1)
cv2.imshow('red', image)
cv2.waitKey(0)

import sys
sys.exit(0)



def image_callback(msg):
    image = bridge.imgmsg_to_cv2(msg, 'bgr8')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_mask = cv2.inRange(image, np.array([0, 128, 128]), np.array([50, 255, 255]))
    high_mask = cv2.inRange(image, np.array([200, 0, 0]), np.array([255, 128, 128]))
    mask = high_mask
    # mask = low_mask | high_mask
    image = cv2.bitwise_and(image, image, mask=~mask)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imshow('red', image)
    cv2.waitKey(3)


rospy.init_node('follower')
image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, image_callback)
rospy.spin()
