import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, UpSampling2D, concatenate, Conv2D, BatchNormalization, GaussianNoise, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def haar_wavelet_filters():

    low_pass_filter = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]

    high_pass_filter = [[1 / 2, -1 / 2], [1 / 2, -1 / 2]]

    

    low_pass_filter = tf.convert_to_tensor(low_pass_filter, dtype=tf.float32)

    high_pass_filter = tf.convert_to_tensor(high_pass_filter, dtype=tf.float32)

    

    low_pass_filter = tf.reshape(low_pass_filter, [2, 2, 1, 1])

    high_pass_filter = tf.reshape(high_pass_filter, [2, 2, 1, 1])



    return low_pass_filter, high_pass_filter



# Modify RDWTLayer to add serialization
class RDWTLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RDWTLayer, self).__init__(**kwargs)
        self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

    def call(self, inputs):
        channels = tf.split(inputs, num_or_size_splits=inputs.shape[-1], axis=-1)
        cA_list, cH_list, cV_list, cD_list = [], [], [], []

        for channel in channels:
            cA = tf.nn.conv2d(channel, self.low_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
            cH = tf.nn.conv2d(channel, self.high_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
            cV = tf.nn.conv2d(channel, tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
            cD = tf.nn.conv2d(channel, self.high_pass_filter[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding='SAME')

            cA_list.append(cA)
            cH_list.append(cH)
            cV_list.append(cV)
            cD_list.append(cD)

        cA = tf.concat(cA_list, axis=-1)
        cH = tf.concat(cH_list, axis=-1)
        cV = tf.concat(cV_list, axis=-1)
        cD = tf.concat(cD_list, axis=-1)

        return cA, cH, cV, cD

    def get_config(self):
        config = super(RDWTLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Modify IRDWTLayer to add serialization
class IRDWTLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IRDWTLayer, self).__init__(**kwargs)
        self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

    def call(self, inputs):
        cA, cH, cV, cD = inputs
        channels_cA = tf.split(cA, num_or_size_splits=cA.shape[-1], axis=-1)
        channels_cH = tf.split(cH, num_or_size_splits=cH.shape[-1], axis=-1)
        channels_cV = tf.split(cV, num_or_size_splits=cV.shape[-1], axis=-1)
        channels_cD = tf.split(cD, num_or_size_splits=cD.shape[-1], axis=-1)

        reconstructed_channels = []

        for cA_channel, cH_channel, cV_channel, cD_channel in zip(channels_cA, channels_cH, channels_cV, channels_cD):
            output_shape = tf.shape(cA_channel)

            cA_reconstructed = tf.nn.conv2d(cA_channel, self.low_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
            cH_reconstructed = tf.nn.conv2d(cH_channel, self.high_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
            cV_reconstructed = tf.nn.conv2d(cV_channel, tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
            cD_reconstructed = tf.nn.conv2d(cD_channel, self.high_pass_filter[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding='SAME')

            reconstructed_channel = cA_reconstructed + cH_reconstructed + cV_reconstructed + cD_reconstructed
            reconstructed_channels.append(reconstructed_channel)

        output = tf.concat(reconstructed_channels, axis=-1)
        return output

    def get_config(self):
        config = super(IRDWTLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

import tensorflow as tf

class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, start_index, end_index, **kwargs):
        super().__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index
    
    def call(self, inputs):
        return inputs[:,:,:,self.start_index:self.end_index]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'start_index': self.start_index,
            'end_index': self.end_index
        })
        return config
    

def build_encoder(secret_image_shape, cover_image_shape):
    secret_input = Input(shape=secret_image_shape)  # 256x256
    cover_input = Input(shape=cover_image_shape)  

    # Apply RDWT to the cover image
    rdwt_layer = RDWTLayer()
    coeff_approx, coeff_horiz, coeff_vert, coeff_diag = rdwt_layer(cover_input)

    # Upscale secret image to 512x512 to match RDWT bands
    secret_upsampled = UpSampling2D(size=(2, 2))(secret_input)

    # Preparation network for processing the secret image with GaussianNoise, BatchNormalization, and Dropout
    prep_conv_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(secret_upsampled)
    prep_conv_3x3 = BatchNormalization()(prep_conv_3x3)
    prep_conv_3x3 = Dropout(0.3)(prep_conv_3x3)

    prep_conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(secret_upsampled)
    prep_conv_5x5 = BatchNormalization()(prep_conv_5x5)
    prep_conv_5x5 = Dropout(0.3)(prep_conv_5x5)

    prep_conv_7x7 = Conv2D(64, (7, 7), padding='same', activation='relu')(secret_upsampled)
    prep_conv_7x7 = BatchNormalization()(prep_conv_7x7)
    prep_conv_7x7 = Dropout(0.3)(prep_conv_7x7)

    # Concatenate augmented feature maps
    prep_output = concatenate([prep_conv_3x3, prep_conv_5x5, prep_conv_7x7])

    # Concatenate prep_output with RDWT bands and add additional layers for robustness
    combined_input = concatenate([coeff_approx, coeff_horiz, coeff_vert, coeff_diag, prep_output])

    hide_conv_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(combined_input)
    hide_conv_3x3 = BatchNormalization()(hide_conv_3x3)
    hide_conv_3x3 = Dropout(0.4)(hide_conv_3x3)

    hide_conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(hide_conv_3x3)
    hide_conv_5x5 = BatchNormalization()(hide_conv_5x5)
    hide_conv_5x5 = Dropout(0.4)(hide_conv_5x5)

    hide_conv_7x7 = Conv2D(64, (7, 7), padding='same', activation='relu')(hide_conv_5x5)
    hide_conv_7x7 = BatchNormalization()(hide_conv_7x7)
    hide_conv_7x7 = Dropout(0.4)(hide_conv_7x7)

    # Final output layer
    final_output = Conv2D(12, (3, 3), padding='same', activation='sigmoid')(hide_conv_7x7)
    
    # Use SliceLayer instead of direct slicing
    slice1 = SliceLayer(0, 3)(final_output)
    slice2 = SliceLayer(3, 6)(final_output)
    slice3 = SliceLayer(6, 9)(final_output)
    slice4 = SliceLayer(9, 12)(final_output)
    
    # Apply Inverse RDWT
    irdwt_layer = IRDWTLayer()
    reconstructed_cover = irdwt_layer([slice1, slice2, slice3, slice4])
    
    return Model(inputs=[secret_input, cover_input], outputs=reconstructed_cover, name='Encoder')


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Dropout, concatenate
from tensorflow.keras.models import Model

def build_decoder(cover_image_shape, secret_image_shape):
    encoded_cover_input = Input(shape=cover_image_shape)
    
    # Multi-branch preprocessing for different noise types (simplified)
    x = encoded_cover_input
    
    # Apply RDWT (assuming RDWTLayer is pre-defined)
    rdwt_layer = RDWTLayer()
    approx_coeffs, horiz_coeffs, vert_coeffs, diag_coeffs = rdwt_layer(x)
    
    def denoising_block(x, filters):
        skip = Conv2D(filters, (1, 1), padding='same')(x)

        # Main processing block
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

        # Add residual connection
        x = Add()([x, skip])
        return LeakyReLU(0.1)(x)
    
    # Process each coefficient band (single denoising block per band)
    approx_clean = denoising_block(approx_coeffs, 128)
    horiz_clean = denoising_block(horiz_coeffs, 128)
    vert_clean = denoising_block(vert_coeffs, 128)
    diag_clean = denoising_block(diag_coeffs, 128)
    
    # Combine cleaned coefficients
    reveal_input = concatenate([approx_clean, horiz_clean, vert_clean, diag_clean])
    
    # Additional processing layers (reduced stages and dropout)
    x = reveal_input
    for _ in range(3):  # Reduced to 2 denoising stages
        x = denoising_block(x, 128)
        x = Dropout(0.1)(x)
    
    # Final secret reconstruction
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Downscale to match secret image size
    current_height = x.shape[1]  # Get current height
    target_height = secret_image_shape[0]  # Get target height
    
    while current_height > target_height:
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        current_height //= 2
    
    secret_output = Conv2D(secret_image_shape[-1], (3, 3), padding='same', activation='sigmoid')(x)
    
    return Model(inputs=encoded_cover_input, outputs=secret_output, name='Simplified_Robust_Decoder')



import tensorflow as tf
import numpy as np

# Combined loss function
def combined_loss(secret_input, decoded_secret, cover_input, encoded_cover):
    secret_loss = tf.reduce_sum(tf.square(secret_input - decoded_secret))
    cover_loss = tf.reduce_sum(tf.square(cover_input - encoded_cover))
    total_loss = secret_loss + cover_loss
    return [total_loss, secret_loss, cover_loss]

# FullModel class
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
            encoded_cover = self.encoder_model([secret_input, cover_input])
            decoded_secret = self.decoder_model(encoded_cover)
            [total_loss, secret_loss, cover_loss] = combined_loss(secret_input, decoded_secret, cover_input, encoded_cover)
        
        # Compute gradients and apply them
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return the losses
        return {
            "total_loss": total_loss,
            "secret_loss": secret_loss,
            "cover_loss": cover_loss
        }

    def get_config(self):
        # Serialize model configuration and weights
        encoder_weights = [w.tolist() for w in self.encoder_model.get_weights()]
        decoder_weights = [w.tolist() for w in self.decoder_model.get_weights()]
        
        return {
            'encoder_config': self.encoder_model.get_config(),
            'decoder_config': self.decoder_model.get_config(),
            'encoder_weights': encoder_weights,
            'decoder_weights': decoder_weights
        }

    @classmethod
    def from_config(cls, config):
        # Recreate encoder model
        encoder = tf.keras.Model.from_config(config['encoder_config'])
        encoder_weights = [np.array(w) for w in config['encoder_weights']]
        encoder.set_weights(encoder_weights)
        
        # Recreate decoder model
        decoder = tf.keras.Model.from_config(config['decoder_config'])
        decoder_weights = [np.array(w) for w in config['decoder_weights']]
        decoder.set_weights(decoder_weights)
        
        return cls(encoder_model=encoder, decoder_model=decoder)




# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
# from tensorflow.keras.layers import *
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Input, UpSampling2D, concatenate, Conv2D, BatchNormalization, GaussianNoise, Dropout
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image

# def haar_wavelet_filters():

#     low_pass_filter = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]

#     high_pass_filter = [[1 / 2, -1 / 2], [1 / 2, -1 / 2]]

    

#     low_pass_filter = tf.convert_to_tensor(low_pass_filter, dtype=tf.float32)

#     high_pass_filter = tf.convert_to_tensor(high_pass_filter, dtype=tf.float32)

    

#     low_pass_filter = tf.reshape(low_pass_filter, [2, 2, 1, 1])

#     high_pass_filter = tf.reshape(high_pass_filter, [2, 2, 1, 1])



#     return low_pass_filter, high_pass_filter



# # Modify RDWTLayer to add serialization
# class RDWTLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(RDWTLayer, self).__init__(**kwargs)
#         self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

#     def call(self, inputs):
#         channels = tf.split(inputs, num_or_size_splits=inputs.shape[-1], axis=-1)
#         cA_list, cH_list, cV_list, cD_list = [], [], [], []

#         for channel in channels:
#             cA = tf.nn.conv2d(channel, self.low_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
#             cH = tf.nn.conv2d(channel, self.high_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
#             cV = tf.nn.conv2d(channel, tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
#             cD = tf.nn.conv2d(channel, self.high_pass_filter[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding='SAME')

#             cA_list.append(cA)
#             cH_list.append(cH)
#             cV_list.append(cV)
#             cD_list.append(cD)

#         cA = tf.concat(cA_list, axis=-1)
#         cH = tf.concat(cH_list, axis=-1)
#         cV = tf.concat(cV_list, axis=-1)
#         cD = tf.concat(cD_list, axis=-1)

#         return cA, cH, cV, cD

#     def get_config(self):
#         config = super(RDWTLayer, self).get_config()
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

# # Modify IRDWTLayer to add serialization
# class IRDWTLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(IRDWTLayer, self).__init__(**kwargs)
#         self.low_pass_filter, self.high_pass_filter = haar_wavelet_filters()

#     def call(self, inputs):
#         cA, cH, cV, cD = inputs
#         channels_cA = tf.split(cA, num_or_size_splits=cA.shape[-1], axis=-1)
#         channels_cH = tf.split(cH, num_or_size_splits=cH.shape[-1], axis=-1)
#         channels_cV = tf.split(cV, num_or_size_splits=cV.shape[-1], axis=-1)
#         channels_cD = tf.split(cD, num_or_size_splits=cD.shape[-1], axis=-1)

#         reconstructed_channels = []

#         for cA_channel, cH_channel, cV_channel, cD_channel in zip(channels_cA, channels_cH, channels_cV, channels_cD):
#             output_shape = tf.shape(cA_channel)

#             cA_reconstructed = tf.nn.conv2d(cA_channel, self.low_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
#             cH_reconstructed = tf.nn.conv2d(cH_channel, self.high_pass_filter, strides=[1, 1, 1, 1], padding='SAME')
#             cV_reconstructed = tf.nn.conv2d(cV_channel, tf.transpose(self.high_pass_filter, perm=[1, 0, 2, 3]), strides=[1, 1, 1, 1], padding='SAME')
#             cD_reconstructed = tf.nn.conv2d(cD_channel, self.high_pass_filter[::-1,::-1,:,:], strides=[1, 1, 1, 1], padding='SAME')

#             reconstructed_channel = cA_reconstructed + cH_reconstructed + cV_reconstructed + cD_reconstructed
#             reconstructed_channels.append(reconstructed_channel)

#         output = tf.concat(reconstructed_channels, axis=-1)
#         return output

#     def get_config(self):
#         config = super(IRDWTLayer, self).get_config()
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

import tensorflow as tf

class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, start_index, end_index, **kwargs):
        super().__init__(**kwargs)
        self.start_index = start_index
        self.end_index = end_index
    
    def call(self, inputs):
        return inputs[:,:,:,self.start_index:self.end_index]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'start_index': self.start_index,
            'end_index': self.end_index
        })
        return config
    

def build_encoder(secret_image_shape, cover_image_shape):
    secret_input = Input(shape=secret_image_shape)  # 256x256
    cover_input = Input(shape=cover_image_shape)  

    # Apply RDWT to the cover image
    rdwt_layer = RDWTLayer()
    coeff_approx, coeff_horiz, coeff_vert, coeff_diag = rdwt_layer(cover_input)

    # Upscale secret image to 512x512 to match RDWT bands
    secret_upsampled = UpSampling2D(size=(2, 2))(secret_input)

    # Preparation network for processing the secret image with GaussianNoise, BatchNormalization, and Dropout
    prep_conv_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(secret_upsampled)
    prep_conv_3x3 = BatchNormalization()(prep_conv_3x3)
    prep_conv_3x3 = Dropout(0.3)(prep_conv_3x3)

    prep_conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(secret_upsampled)
    prep_conv_5x5 = BatchNormalization()(prep_conv_5x5)
    prep_conv_5x5 = Dropout(0.3)(prep_conv_5x5)

    prep_conv_7x7 = Conv2D(64, (7, 7), padding='same', activation='relu')(secret_upsampled)
    prep_conv_7x7 = BatchNormalization()(prep_conv_7x7)
    prep_conv_7x7 = Dropout(0.3)(prep_conv_7x7)

    # Concatenate augmented feature maps
    prep_output = concatenate([prep_conv_3x3, prep_conv_5x5, prep_conv_7x7])

    # Concatenate prep_output with RDWT bands and add additional layers for robustness
    combined_input = concatenate([coeff_approx, coeff_horiz, coeff_vert, coeff_diag, prep_output])

    hide_conv_3x3 = Conv2D(64, (3, 3), padding='same', activation='relu')(combined_input)
    hide_conv_3x3 = BatchNormalization()(hide_conv_3x3)
    hide_conv_3x3 = Dropout(0.4)(hide_conv_3x3)

    hide_conv_5x5 = Conv2D(64, (5, 5), padding='same', activation='relu')(hide_conv_3x3)
    hide_conv_5x5 = BatchNormalization()(hide_conv_5x5)
    hide_conv_5x5 = Dropout(0.4)(hide_conv_5x5)

    hide_conv_7x7 = Conv2D(64, (7, 7), padding='same', activation='relu')(hide_conv_5x5)
    hide_conv_7x7 = BatchNormalization()(hide_conv_7x7)
    hide_conv_7x7 = Dropout(0.4)(hide_conv_7x7)

    # Final output layer
    final_output = Conv2D(12, (3, 3), padding='same', activation='sigmoid')(hide_conv_7x7)
    
    # Use SliceLayer instead of direct slicing
    slice1 = SliceLayer(0, 3)(final_output)
    slice2 = SliceLayer(3, 6)(final_output)
    slice3 = SliceLayer(6, 9)(final_output)
    slice4 = SliceLayer(9, 12)(final_output)
    
    # Apply Inverse RDWT
    irdwt_layer = IRDWTLayer()
    reconstructed_cover = irdwt_layer([slice1, slice2, slice3, slice4])
    
    return Model(inputs=[secret_input, cover_input], outputs=reconstructed_cover, name='Encoder')


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, Dropout, concatenate
from tensorflow.keras.models import Model

def build_decoder(cover_image_shape, secret_image_shape):
    encoded_cover_input = Input(shape=cover_image_shape)
    
    # Multi-branch preprocessing for different noise types (simplified)
    x = encoded_cover_input
    
    # Apply RDWT (assuming RDWTLayer is pre-defined)
    rdwt_layer = RDWTLayer()
    approx_coeffs, horiz_coeffs, vert_coeffs, diag_coeffs = rdwt_layer(x)
    
    def denoising_block(x, filters):
        skip = Conv2D(filters, (1, 1), padding='same')(x)

        # Main processing block
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

        # Add residual connection
        x = Add()([x, skip])
        return LeakyReLU(0.1)(x)
    
    # Process each coefficient band (single denoising block per band)
    approx_clean = denoising_block(approx_coeffs, 128)
    horiz_clean = denoising_block(horiz_coeffs, 128)
    vert_clean = denoising_block(vert_coeffs, 128)
    diag_clean = denoising_block(diag_coeffs, 128)
    
    # Combine cleaned coefficients
    reveal_input = concatenate([approx_clean, horiz_clean, vert_clean, diag_clean])
    
    # Additional processing layers (reduced stages and dropout)
    x = reveal_input
    for _ in range(3):  # Reduced to 2 denoising stages
        x = denoising_block(x, 128)
        x = Dropout(0.1)(x)
    
    # Final secret reconstruction
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    
    # Downscale to match secret image size
    current_height = x.shape[1]  # Get current height
    target_height = secret_image_shape[0]  # Get target height
    
    while current_height > target_height:
        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        current_height //= 2
    
    secret_output = Conv2D(secret_image_shape[-1], (3, 3), padding='same', activation='sigmoid')(x)
    
    return Model(inputs=encoded_cover_input, outputs=secret_output, name='Simplified_Robust_Decoder')



import tensorflow as tf
import numpy as np

# Combined loss function
def combined_loss(secret_input, decoded_secret, cover_input, encoded_cover):
    secret_loss = tf.reduce_sum(tf.square(secret_input - decoded_secret))
    cover_loss = tf.reduce_sum(tf.square(cover_input - encoded_cover))
    total_loss = secret_loss + cover_loss
    return [total_loss, secret_loss, cover_loss]

# FullModel class
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
            encoded_cover = self.encoder_model([secret_input, cover_input])
            decoded_secret = self.decoder_model(encoded_cover)
            [total_loss, secret_loss, cover_loss] = combined_loss(secret_input, decoded_secret, cover_input, encoded_cover)
        
        # Compute gradients and apply them
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return the losses
        return {
            "total_loss": total_loss,
            "secret_loss": secret_loss,
            "cover_loss": cover_loss
        }

    def get_config(self):
        # Serialize model configuration and weights
        encoder_weights = [w.tolist() for w in self.encoder_model.get_weights()]
        decoder_weights = [w.tolist() for w in self.decoder_model.get_weights()]
        
        return {
            'encoder_config': self.encoder_model.get_config(),
            'decoder_config': self.decoder_model.get_config(),
            'encoder_weights': encoder_weights,
            'decoder_weights': decoder_weights
        }

    @classmethod
    def from_config(cls, config):
        # Recreate encoder model
        encoder = tf.keras.Model.from_config(config['encoder_config'])
        encoder_weights = [np.array(w) for w in config['encoder_weights']]
        encoder.set_weights(encoder_weights)
        
        # Recreate decoder model
        decoder = tf.keras.Model.from_config(config['decoder_config'])
        decoder_weights = [np.array(w) for w in config['decoder_weights']]
        decoder.set_weights(decoder_weights)
        
        return cls(encoder_model=encoder, decoder_model=decoder)
