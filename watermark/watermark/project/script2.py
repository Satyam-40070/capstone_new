from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
)
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.layers import Input, Conv2D, concatenate, Layer
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from tqdm import tqdm
from imageio import imread

import imageio


def build_encoder(secret_image_shape, cover_image_shape):

    print(cover_image_shape)

    # Input layers for secret and cover images
    secret_input = Input(shape=secret_image_shape)
    cover_input = Input(shape=cover_image_shape)

    print(secret_image_shape)

    print(cover_input)

    # Apply Discrete Wavelet Transform (DWT) to the cover image
    dwt_layer = DWTLayer()
    [coeff_approx, coeff_horiz, coeff_vert, coeff_diag] = dwt_layer(cover_input)

    # Concatenate the approximation coefficients with the secret image

    prep_input = concatenate([coeff_approx, secret_input])

    # Preparation network for processing the secret image
    prep_conv_3x3 = Conv2D(50, (3, 3), padding="same", activation="relu")(prep_input)
    prep_conv_4x4 = Conv2D(50, (4, 4), padding="same", activation="relu")(prep_input)
    prep_conv_5x5 = Conv2D(50, (5, 5), padding="same", activation="relu")(prep_input)
    prep_output = concatenate([prep_conv_3x3, prep_conv_4x4, prep_conv_5x5])

    # Hiding network for embedding the secret image into the cover image
    hide_conv_3x3 = Conv2D(50, (3, 3), padding="same", activation="relu")(prep_output)
    hide_conv_4x4 = Conv2D(50, (4, 4), padding="same", activation="relu")(prep_output)
    hide_conv_5x5 = Conv2D(50, (5, 5), padding="same", activation="relu")(prep_output)
    hide_output = concatenate([hide_conv_3x3, hide_conv_4x4, hide_conv_5x5])

    # Final Conv2D layer to reduce the channels to 3
    final_output = Conv2D(3, (3, 3), padding="same", activation="relu")(hide_output)

    # Apply inverse DWT to reconstruct the modified cover image
    idwt_layer = IDWTLayer()
    reconstructed_cover = idwt_layer(
        [final_output, coeff_horiz, coeff_vert, coeff_diag]
    )

    # Return the encoder model
    return Model(
        inputs=[secret_input, cover_input], outputs=reconstructed_cover, name="Encoder"
    )


def build_decoder(cover_image_shape, secret_image_shape):

    # Input layer for the encoded cover image
    encoded_cover_input = Input(shape=cover_image_shape)

    # Apply Discrete Wavelet Transform (DWT) to the encoded image
    dwt_layer = DWTLayer()
    approx_coeffs, horiz_coeffs, vert_coeffs, diag_coeffs = dwt_layer(
        encoded_cover_input
    )

    # Reveal network to extract the secret image from the approximation coefficients
    reveal_conv_3x3 = Conv2D(50, (3, 3), padding="same", activation="relu")(
        approx_coeffs
    )
    reveal_conv_4x4 = Conv2D(50, (4, 4), padding="same", activation="relu")(
        reveal_conv_3x3
    )
    reveal_conv_5x5 = Conv2D(50, (5, 5), padding="same", activation="relu")(
        reveal_conv_4x4
    )
    revealed_secret_output = Conv2D(3, (3, 3), padding="same", activation="sigmoid")(
        reveal_conv_5x5
    )

    # Return the decoder model
    return Model(
        inputs=encoded_cover_input, outputs=revealed_secret_output, name="Decoder"
    )


class FullModel(tf.keras.Model):
    def __init__(self, encoder_model, decoder_model):
        super(FullModel, self).__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def call(self, inputs):
        secret_input, cover_input = inputs
        encoded_cover = self.encoder_model([secret_input, cover_input])
        decoded_secret = self.decoder_model(encoded_cover)
        return [decoded_secret, encoded_cover]

    @tf.function
    def train_step(self, data):
        (secret_input, cover_input), _ = data

        with tf.GradientTape() as tape:
            # Forward pass
            encoded_cover = self.encoder_model([secret_input, cover_input])
            decoded_secret = self.decoder_model(encoded_cover)

            # Compute the loss for both secret and cover outputs
            secret_loss = tf.reduce_sum(tf.square(secret_input - decoded_secret))
            cover_loss = tf.reduce_sum(tf.square(cover_input - encoded_cover))
            total_loss = secret_loss + cover_loss

        # Compute gradients and apply them
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Return a dictionary mapping metric names to current values
        return {
            "total_loss": total_loss,
            "secret_loss": secret_loss,
            "cover_loss": cover_loss,
        }


# ---------------------------------------------------------------------------


# Haar wavelet filters for DWT
def haar_wavelet_filters():
    low_pass_filter = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]
    high_pass_filter = [[1 / 2, -1 / 2], [1 / 2, -1 / 2]]

    low_pass_filter = tf.convert_to_tensor(low_pass_filter, dtype=tf.float32)
    high_pass_filter = tf.convert_to_tensor(high_pass_filter, dtype=tf.float32)

    # Reshape filters to match TensorFlow's convolutional filter shape: [filter_height, filter_width, in_channels, out_channels]
    low_pass_filter = tf.reshape(low_pass_filter, [2, 2, 1, 1])
    high_pass_filter = tf.reshape(high_pass_filter, [2, 2, 1, 1])

    return low_pass_filter, high_pass_filter


# Custom layer to apply DWT using convolution
class DWTLayer(Layer):
    def __init__(self):
        super(DWTLayer, self).__init__()
        self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

    def call(self, inputs):
        channels = tf.split(inputs, num_or_size_splits=inputs.shape[-1], axis=-1)
        cA_list, cH_list, cV_list, cD_list = [], [], [], []

        for channel in channels:
            cA = tf.nn.conv2d(
                channel, self.low_pass_filter, strides=[1, 2, 2, 1], padding="SAME"
            )
            cH = tf.nn.conv2d(
                channel, self.high_pass_filter, strides=[1, 2, 2, 1], padding="SAME"
            )
            cV = tf.nn.conv2d(
                channel,
                tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]),
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            cD = tf.nn.conv2d(
                channel, -self.high_pass_filter, strides=[1, 2, 2, 1], padding="SAME"
            )

            cA_list.append(cA)
            cH_list.append(cH)
            cV_list.append(cV)
            cD_list.append(cD)

        # Concatenate the results along the channel axis
        cA = tf.concat(cA_list, axis=-1)
        cH = tf.concat(cH_list, axis=-1)
        cV = tf.concat(cV_list, axis=-1)
        cD = tf.concat(cD_list, axis=-1)

        return cA, cH, cV, cD


# Custom layer to apply inverse DWT using transpose convolution
class IDWTLayer(Layer):
    def __init__(self):
        super(IDWTLayer, self).__init__()
        self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

    def call(self, inputs):
        cA, cH, cV, cD = inputs
        channels_cA = tf.split(cA, num_or_size_splits=cA.shape[-1], axis=-1)
        channels_cH = tf.split(cH, num_or_size_splits=cH.shape[-1], axis=-1)
        channels_cV = tf.split(cV, num_or_size_splits=cV.shape[-1], axis=-1)
        channels_cD = tf.split(cD, num_or_size_splits=cD.shape[-1], axis=-1)

        reconstructed_channels = []

        for cA_channel, cH_channel, cV_channel, cD_channel in zip(
            channels_cA, channels_cH, channels_cV, channels_cD
        ):
            cA_upsampled = tf.nn.conv2d_transpose(
                cA_channel,
                self.low_pass_filter,
                output_shape=[
                    tf.shape(cA_channel)[0],
                    cA_channel.shape[1] * 2,
                    cA_channel.shape[2] * 2,
                    1,
                ],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            cH_upsampled = tf.nn.conv2d_transpose(
                cH_channel,
                self.high_pass_filter,
                output_shape=[
                    tf.shape(cH_channel)[0],
                    cH_channel.shape[1] * 2,
                    cH_channel.shape[2] * 2,
                    1,
                ],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            cV_upsampled = tf.nn.conv2d_transpose(
                cV_channel,
                tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]),
                output_shape=[
                    tf.shape(cV_channel)[0],
                    cV_channel.shape[1] * 2,
                    cV_channel.shape[2] * 2,
                    1,
                ],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
            cD_upsampled = tf.nn.conv2d_transpose(
                cD_channel,
                -self.high_pass_filter,
                output_shape=[
                    tf.shape(cD_channel)[0],
                    cD_channel.shape[1] * 2,
                    cD_channel.shape[2] * 2,
                    1,
                ],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )

            # Combine the results to reconstruct the original image channel
            reconstructed_channel = (
                cA_upsampled + cH_upsampled + cV_upsampled + cD_upsampled
            )
            reconstructed_channels.append(reconstructed_channel)

        # Concatenate all the reconstructed channels
        output = tf.concat(reconstructed_channels, axis=-1)

        return output


# Rest of the code remains the same...


# Loss for reveal network
def rev_loss(s_true, s_pred):
    beta = 1.0  # Variable used to weight the losses
    return beta * tf.reduce_sum(tf.square(s_true - s_pred))


# Loss for the full model, used for both the secret and cover images
def full_loss(y_true, y_pred):
    # Split the tensors back into secret and cover images
    s_true, c_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    s_pred, c_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    print(s_true)
    print(s_pred)
    print(c_true)
    print(c_pred)

    # Compute the sum of squared errors for the secret image
    s_loss = tf.reduce_sum(tf.square(s_true - s_pred))

    # Compute the sum of squared errors for the cover image
    c_loss = tf.reduce_sum(tf.square(c_true - c_pred))

    # Return the total loss as the sum of both losses
    return s_loss + c_loss


# ----------------------------------------------------------------------


# Secret_Image_Data = ""
# Cover_Image_Data = ""


def load_data():
    secret_path = ""
    imagename = "secretimage.jpg"
    img_i = image.load_img(os.path.join(secret_path, imagename))
    img_i = img_i.resize((256, 256))
    x = image.img_to_array(img_i)
    secret_image = x

    # cover image code
    c = "imagecover1.jpg"
    c_dir = os.path.join("", c)

    img_i = image.load_img(c_dir)
    img_i = img_i.resize((512, 512))
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


def display_images(decoded_S, decoded_C):
    plt.figure(figsize=(12, 12))

    # Display decoded secret image
    plt.subplot(1, 2, 1)
    plt.title("Decoded Secret Image")
    plt.imshow(decoded_S[0])  # Remove the batch dimension for display
    plt.axis("off")

    # Display encoded cover image
    plt.subplot(2, 2, 2)
    plt.title("Encoded Cover Image")
    plt.imshow(decoded_C[0])  # Remove the batch dimension for display
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_images(decoded_S, decoded_C):
    # Save decoded secret image
    secret_image = (decoded_S[0] * 255).astype(np.uint8)  # Scale back to [0, 255]
    secret_image_pil = Image.fromarray(secret_image)
    secret_image_pil.save("decoded_secret_image.jpg")

    # Save encoded cover image
    cover_image = (decoded_C[0] * 255).astype(np.uint8)  # Scale back to [0, 255]
    cover_image_pil = Image.fromarray(cover_image)
    cover_image_pil.save("decoded_cover_image.jpg")


# Call the function to save the images


def run_model():
    cover_image, secret_image = load_data()

    print(cover_image.shape)
    print(secret_image.shape)
    secret_size = (256, 256, 3)
    cover_size = (512, 512, 3)

    encoderModel = build_encoder(secret_size, cover_size)
    decoderModel = build_decoder(cover_size, secret_size)
    fullModel = FullModel(encoderModel, decoderModel)

    # Load the weights
    fullModel.load_weights("model_512.weights.h5")
    print(f"Weights loaded from /model_weights.weights.h5")

    # -----------------------------------------------

    X_test_secret = np.expand_dims(secret_image / 255.0, axis=0)
    X_test_cover = np.expand_dims(cover_image / 255.0, axis=0)
    decoded = fullModel.predict([X_test_secret, X_test_cover])
    decoded_S, decoded_C = decoded

    # -------------------------------------------------
    display_images(decoded_S, decoded_C)

    save_images(decoded_S, decoded_C)
    print("Images saved as decoded_secret_image.jpg and decoded_cover_image.jpg")
    return decoded_S
