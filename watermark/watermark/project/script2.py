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


# Secret_Image_Data = ""
# Cover_Image_Data = ""


def load_data(image1, image2):
    # secret_path = ""
    # imagename = "secretimage.jpg"
    # img_i = image.load_img(os.path.join(secret_path, imagename))
    # file_content = image1.read()
    
    # # Create a BytesIO object from the file content
    # bytes_io = BytesIO(file_content)



    img_i = image.load_img(image1)

    img_i = img_i.resize((128, 128))
    x = image.img_to_array(img_i)
    secret_image = x

    # cover image code
    c = "imagecover1.jpg"
    c_dir = os.path.join("", c)

    img_i = image.load_img(image2)
    img_i = img_i.resize((256, 256))
    x = image.img_to_array(img_i)
    cover_image = x

    return [cover_image, secret_image]


def display_images(decoded_S, decoded_C):
    plt.figure(figsize=(12, 6))

    # Display decoded secret image
    plt.subplot(1, 2, 1)
    plt.title("Decoded Secret Image")
    plt.imshow(decoded_S[0])  # Remove the batch dimension for display
    plt.axis("off")

    # Display encoded cover image
    plt.subplot(1, 2, 2)
    plt.title("Encoded Cover Image")
    plt.imshow(decoded_C[0])  # Remove the batch dimension for display
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_images(decoded_S, decoded_C):
    # Save decoded secret image
    secret_image = (decoded_S[0] * 255).astype(np.uint8)  # Scale back to [0, 255]
    secret_image_pil = Image.fromarray(secret_image)
    secret_image_pil.save("decoded_cover_image.jpg")

    # Save encoded cover image
    cover_image = (decoded_C[0] * 255).astype(np.uint8)  # Scale back to [0, 255]
    cover_image_pil = Image.fromarray(cover_image)
    cover_image_pil.save("decoded_server_image.jpg")


# Call the function to save the images

from custom_layers import SliceLayer , IRDWTLayer , RDWTLayer , FullModel
from tensorflow.keras.models import load_model

def run_model(image1, image2):
    cover_image, secret_image = load_data(image1, image2)

    print(cover_image.shape)
    print(secret_image.shape)
    

    

    # Load the weights
    path = r"E:\Semester7\Capstone\capsite\watermark\watermark\project\model_512.weights.h5"
    fullModel = load_model('main_model_downloaded_xray.keras' , custom_objects={'RDWTLayer': RDWTLayer, 'IRDWTLayer': IRDWTLayer, 'SliceLayer': SliceLayer , 'FullModel' : FullModel})
    print(f"Weights loaded from /model_weights.weights.h5")
    # -----------------------------------------------
    X_test_secret = np.expand_dims(secret_image / 255.0, axis=0)
    X_test_cover = np.expand_dims(cover_image / 255.0, axis=0)

    print("are we....")
    decoded = fullModel.predict([X_test_secret, X_test_cover])
    decoded_S, decoded_C = decoded

    # -------------------------------------------------
    # display_images(decoded_S, decoded_C)

    save_images(decoded_S, decoded_C)
    print("Images saved as decoded_secret_image.jpg and decoded_cover_image.jpg")
    return decoded_S
