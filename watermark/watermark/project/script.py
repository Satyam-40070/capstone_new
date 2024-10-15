import cv2
import numpy as np
#import matplotlib.pyplot as plt
import time
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
# from google.colab.patches import cv2_imshow

# Load images
def resize2(image1, image2):
    # image1 = cv2.imread('/home/himanshu/Development/django-tut/watermark/watermark/project/BGT.jpeg')
    # image2 = cv2.imread('/home/himanshu/Development/django-tut/watermark/watermark/project/JN.jpeg')

    if image1 is None:
        print("Error: Could not load 'BGT.jpeg'. Please check the file path.")
        exit()

    if image2 is None:
        print("Error: Could not load 'JN.jpeg'. Please check the file path.")
        exit()

    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    stacked_image = np.hstack((image1, image2))

    cv2.imwrite('stacked_image.jpg', stacked_image)
    cv2.imshow('Stacked Image', stacked_image)
    return stacked_image
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



