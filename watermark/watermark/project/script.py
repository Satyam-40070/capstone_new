from custom_layers import SliceLayer , IRDWTLayer , RDWTLayer , FullModel , FullModel1
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.layers import Input, Conv2D, concatenate, Layer
import numpy as np
import os
from tensorflow.keras.models import load_model

def load_data1(image1):
    # secret_path = ""
    # imagename = "secretimage.jpg"
    # img_i = image.load_img(os.path.join(secret_path, imagename))
    # file_content = image1.read()
    
    # # Create a BytesIO object from the file content
    # bytes_io = BytesIO(file_content)



    img_i = image.load_img(image1)
    img_i = img_i.resize((256, 256))
    x = image.img_to_array(img_i)
    secret_image = x

    return secret_image

def run_model1(image1):
    print("hitesh brooo....")
    cover_image = load_data1(image1)

    print(cover_image.shape)

    # Load the weights
    path = r"E:\Semester7\Capstone\capsite\watermark\watermark\project\model_512.weights.h5"
    fullModel1 = load_model('main_model_downloaded_xray.keras' , custom_objects={'RDWTLayer': RDWTLayer,'IRDWTLayer': IRDWTLayer,'SliceLayer': SliceLayer , 'FullModel' : FullModel1})
    print(f"Weights loaded from /model_weights.weights.h5")


    X_test_cover = np.expand_dims(cover_image / 255.0, axis=0)
    print(X_test_cover.shape)
    print("are we....")
    decoded = fullModel1.predict(X_test_cover)
    print("decoded image is here....")
    if decoded.any():
        print("yes sir...")
    decoded_C = decoded

    # -------------------------------------------------
    # display_images(decoded_S, decoded_C)
    
    print("Images saved as decoded_secret_image.jpg and decoded_cover_image.jpg")
    return decoded_C