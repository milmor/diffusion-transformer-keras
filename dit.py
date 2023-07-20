'''DiT Linformer model for Tensorflow.

Author: Emilio Morales (mil.mor.mor@gmail.com)
        Jun 2023
'''
import tensorflow as tf
from tensorflow.keras import layers
import math


class LinformerAttention(layers.Layer):
    def __init__(self, model_dim, n_heads, k, rate=0.0, 
            initializer='glorot_uniform', **kwargs):
        super(LinformerAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.head_dim = model_dim // self.n_heads

        self.wq = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wk = layers.Dense(model_dim, kernel_initializer=initializer)
        self.wv = layers.Dense(model_dim, kernel_initializer=initializer)
        
        self.E = layers.Dense(k)
        self.F = layers.Dense(k)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
        self.wo = layers.Dense(model_dim, kernel_initializer=initializer)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        q = self.split_heads(q, batch_size) 
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size) 
        
        k = tf.transpose(self.E(tf.transpose(k, [0, 1, 3, 2])), [0, 1, 3, 2])
        v = tf.transpose(self.F(tf.transpose(v, [0, 1, 3, 2])), [0, 1, 3, 2])

        dh = tf.cast(self.head_dim, tf.float32)
        qk = tf.matmul(q, k, transpose_b=True)
        scaled_qk =  qk / tf.math.sqrt(dh)

        attn = self.dropout1(tf.nn.softmax(scaled_qk, axis=-1))
        attn = tf.matmul(attn, v) 

        attn = tf.transpose(attn, perm=[0, 2, 1, 3]) 
        original_size_attention = tf.reshape(attn, (batch_size, -1, self.model_dim)) 

        output = self.dropout2(self.wo(original_size_attention))
        return output


class PositionalEmbedding(layers.Layer):
    def __init__(self, n_patches, model_dim, initializer='glorot_uniform', **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.n_patches = n_patches
        self.position_embedding = layers.Embedding(
            input_dim=n_patches, output_dim=model_dim, 
            embeddings_initializer=initializer
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)
    

class adaLN(layers.Layer):
    def __init__(self, epsilon=1e-3, initializer='glorot_uniform'):
        super(adaLN, self).__init__()
        self.epsilon = epsilon
        self.initializer = initializer
        self.norm = layers.LayerNormalization(epsilon=epsilon,    
            center=False, scale=False
        )

    def build(self, input_shape):
        self.gamma = layers.Dense(input_shape[2], use_bias=True, 
            kernel_initializer=self.initializer, 
        )
        self.beta = layers.Dense(input_shape[2], use_bias=True, 
            kernel_initializer=self.initializer
        )

    def call(self, inputs, z):
        x = self.norm(inputs)
        scale = self.gamma(z)
        shift = self.beta(z)

        x = x * (1 + tf.expand_dims(scale, 1)) + tf.expand_dims(shift, 1)
        return x
    
    
class Scale(layers.Layer):
    def __init__(self, initializer='glorot_uniform'):
        super(Scale, self).__init__()
        self.initializer = initializer

    def build(self, input_shape):
        self.alpha = layers.Dense(input_shape[2], use_bias=True, 
            kernel_initializer=self.initializer,
        )

    def call(self, x, z):
        scale = self.alpha(z)
        x *= tf.expand_dims(scale, 1)
        return x


class DiTBlock(layers.Layer):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, rate=0.0,
                 eps=1e-6, initializer='glorot_uniform', 
                 mod_init='glorot_uniform', k=64, **kwargs):
        super(DiTBlock, self).__init__(**kwargs)
        self.attn = LinformerAttention(model_dim, n_heads, k=k, 
                                       initializer=initializer
        )
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu', 
                         kernel_initializer=initializer
            ), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        self.sm1 = adaLN(epsilon=eps, initializer=mod_init)
        self.sm2 = adaLN(epsilon=eps, initializer=mod_init)
        self.scale1 = Scale(initializer=mod_init)
        self.scale2 = Scale(initializer=mod_init)

    def call(self, inputs, z, training):
        x_norm1 = self.sm1(inputs, z)
        attn_output = self.scale1(self.attn(x_norm1, x_norm1, x_norm1), z)
        attn_output = inputs + attn_output
        
        x_norm2 = self.sm2(attn_output, z)
        mlp_output = self.scale2(self.mlp(x_norm2), z)
        return mlp_output + attn_output 
    
    
class FinalLayer(layers.Layer):
    def __init__(self, patch_size, out_channels,
                 eps=1e-6, initializer='glorot_uniform', **kwargs):
        super(FinalLayer, self).__init__(**kwargs)
        self.linear = tf.keras.Sequential([
            layers.Dense(patch_size * patch_size * out_channels,
                         kernel_initializer=initializer, 
            ), 
        ])
        self.sm = adaLN(epsilon=eps, initializer=initializer)

    def call(self, inputs, z, training):
        x = self.sm(inputs, z)
        x = self.linear(x)
        return x


class TimestepEmbedder(layers.Layer):
    def __init__(self, model_dim, initializer='glorot_uniform', **kwargs):
        super(TimestepEmbedder, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.mlp = tf.keras.Sequential([
            layers.Dense(model_dim, activation='silu', 
                         kernel_initializer=initializer
            ), 
            layers.Dense(model_dim, kernel_initializer=initializer),
        ])
        
    def sinusoidal_embedding(self, x):
        embedding_min_frequency = 1.0
        noise_embedding_max_frequency = 1000.0
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(embedding_min_frequency),
                tf.math.log(noise_embedding_max_frequency),
                self.model_dim  // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [
                tf.sin(angular_speeds * x),
                tf.cos(angular_speeds * x),
            ],
            axis=1,
        )
        return embeddings

    def call(self, x):
        x = layers.Lambda(self.sinusoidal_embedding)(x)
        x = self.mlp(x)
        return x


class DiT(tf.keras.models.Model):
    def __init__(self, img_size, patch_size, model_dim=256, k=64, 
                    heads=4, mlp_dim=512, depth=3, cuant_dim=4):
        super(DiT, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size)**2 
        self.depth = depth
        self.patches = tf.keras.Sequential([
            layers.Conv2D(model_dim, 
                          kernel_size=patch_size,
                          strides=patch_size, padding='same'),   
        ])
        
        self.pos = PositionalEmbedding(self.n_patches, model_dim)
        self.sin_emb = TimestepEmbedder(model_dim)
        self.transformer = [DiTBlock(model_dim, 
                            heads, mlp_dim, mod_init='zeros', k=k) for _ in range(depth)]
        self.final_layer = FinalLayer(patch_size, cuant_dim,
                                     initializer='zeros'
        )

    def call(self, x):
        noisy_latent, noise_variances = x
        B = noise_variances.shape[0]
        noise_variances = tf.reshape(noise_variances, [B, -1])
        t = self.sin_emb(noise_variances)
        x = self.patches(noisy_latent)

        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H * W, C])

        x = self.pos(x)
        for i in range(self.depth):
            x = self.transformer[i](x, t)
            
        x = self.final_layer(x, t)
        x = tf.reshape(x, [B, H, W, -1]) 
        x = tf.nn.depth_to_space(x, self.patch_size, data_format='NHWC') 
        return x
