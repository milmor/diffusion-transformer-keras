'''AutoencoderKL model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2023
'''
import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, z_mean, z_log_var):
        epsilon = tf.keras.backend.random_normal(shape=(z_mean.shape))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def downBlock(filters, kernel_size=3, initializer='glorot_uniform'):
    block = tf.keras.Sequential([
            layers.Conv2D(
                filters, kernel_size=kernel_size, strides=2, 
                padding='same', use_bias=False, kernel_initializer=initializer
            ),
            layers.GroupNormalization(),
            layers.Activation('swish'),
            layers.Conv2D(
                filters, kernel_size=kernel_size, strides=1, 
                padding='same', use_bias=False, kernel_initializer=initializer
            ),
            layers.GroupNormalization(),
            layers.Activation('swish'),
    ])
    return block


class Encoder(tf.keras.models.Model):
    def __init__(self, model_dim=[64, 128, 256], cuant_dim=4):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        cuant_dim = cuant_dim * 2
        self.ch_conv = layers.Conv2D(
            model_dim[0], kernel_size=3, strides=1, padding='same'
        )
        self.encoder = [downBlock(i) for i in model_dim]
        self.cuant_conv = layers.Conv2D(
            cuant_dim, kernel_size=1, strides=1, padding='same'
        )
        self.sample = Sampling()

    def call(self, x, training=True):
        B, H, W, C = x.shape
        x = self.ch_conv(x)

        for i in range(len(self.model_dim)):
            x = self.encoder[i](x)
         
        x = self.cuant_conv(x)

        z_mean, z_log_var = tf.split(x, 2, axis=-1)
        z_log_var = tf.clip_by_value(z_log_var, -30.0, 20.0)

        x = self.sample(z_mean, z_log_var)
        return x, z_mean, z_log_var


def upBlock(filters, kernel_size=3, initializer='glorot_uniform'):
    block = tf.keras.Sequential([
            layers.UpSampling2D(2, interpolation='bilinear'),
            layers.Conv2D(
                filters, kernel_size=kernel_size, 
                padding='same', use_bias=False, kernel_initializer=initializer
            ),
            layers.GroupNormalization(),
            layers.Activation('swish'),
            layers.Conv2D(
                filters, kernel_size=kernel_size, 
                padding='same', use_bias=False, kernel_initializer=initializer
            ),
            layers.GroupNormalization(),
            layers.Activation('swish'),
    ])
    return block
                 
                 
class Decoder(tf.keras.models.Model):
    def __init__(self, model_dim=[256, 128, 64]):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.post_quant_conv = layers.Conv2D(model_dim[0], 
                                  kernel_size=1, 
                                  strides=1, padding='same')
        self.decoder = [upBlock(i) for i in model_dim]
        self.ch_conv = layers.Conv2D(3, 3, strides=1, padding='same')
            
    def call(self, x, training):      
        B, H, W, C = x.shape
        x = self.post_quant_conv(x) 

        for i in range(len(self.model_dim)):
            x = self.decoder[i](x)
        x = self.ch_conv(x)
        return x
                 
        
class Autoencoder(tf.keras.models.Model):
    def __init__(self, e_dim, d_dim, cuant_dim=4):
        super().__init__()
        self.encoder =  Encoder(e_dim, cuant_dim)
        self.decoder = Decoder(d_dim)
        
    def call(self, x, training=True): 
        x, z_mean, z_log_var = self.encoder(x, training)
        x = self.decoder(x, training)
        return x, z_mean, z_log_var
