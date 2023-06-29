'''PatchGAN discriminator model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Mar 2023
'''
import tensorflow as tf
from tensorflow.keras import layers


class convBlock(tf.keras.models.Model):
    def __init__(self, filters, kernel_size=3, strides=2, 
                 initializer='glorot_uniform'):
        super(convBlock, self).__init__()
        self.main = tf.keras.Sequential([
            layers.Conv2D(
                filters, kernel_size=kernel_size, padding='same', 
                kernel_initializer=initializer, strides=strides, use_bias=False
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
        ])
        
    def call(self, x):
        return self.main(x)
    
    
class Discriminator(tf.keras.models.Model):
    def __init__(self, model_dim=[32, 64, 128, 256, 512], 
                 initializer='glorot_uniform'):
        super(Discriminator, self).__init__()
        self.down_big = tf.keras.Sequential([
            layers.Conv2D(
                model_dim[0], kernel_size=3, strides=2, use_bias=False,
                kernel_initializer=initializer, padding='same'
            ),
            layers.LeakyReLU(0.2),
            convBlock(
                model_dim[1], kernel_size=3, strides=2, initializer=initializer
            ),
            convBlock(
                model_dim[2], kernel_size=3, strides=2, initializer=initializer
            ),
            convBlock(
                model_dim[3], kernel_size=3, strides=2, initializer=initializer
            ),
        ])    
        
        self.down_small = tf.keras.Sequential([
            layers.Conv2D(
                model_dim[4], kernel_size=1, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid'
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2D(
                1, kernel_size=4, strides=1, use_bias=False,
                kernel_initializer=initializer, padding='valid'
            )
        ])

        '''Logits'''
        self.logits = tf.keras.Sequential([
            layers.Flatten(),
            layers.Activation('linear', dtype='float32')    
        ])
    
    def call(self, img):
        x = self.down_big(img)  
        x = self.down_small(x)
        return [self.logits(x)]
