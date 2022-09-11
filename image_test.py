import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

image = cv2.imread("sample_0002.jpeg")
box, label, count = cv.detect_common_objects(image)
output = draw_bbox(image, box, label, count)
plt.imshow(output)
print("Number of cars in this image are " + str(label.count('car')))
print("Number of trucks in this image are " + str(label.count('truck')))
print("Number of buses in this image are " + str(label.count('bus')))
plt.show()

