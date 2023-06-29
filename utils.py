import os
import json
import tensorflow as tf
from keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

def train_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [img_size, img_size], method="bicubic", antialias=True
    )
    return tf.clip_by_value(img / 255.0, 0.0, 1.0)

def create_train_ds(train_dir, batch_size, img_size):
    img_paths = tf.data.Dataset.list_files(str(train_dir))
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)

    img_paths = img_paths.cache().shuffle(BUFFER_SIZE)
    ds = img_paths.map(
            lambda img: train_convert(img, img_size), 
            num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(
            batch_size, drop_remainder=True, 
            num_parallel_calls=AUTOTUNE
    )
    print(f'Train dataset size: {BUFFER_SIZE}')
    print(f'Train batches: {tf.data.experimental.cardinality(ds)}')
    ds = ds.prefetch(AUTOTUNE)
    return ds

def test_convert(file_path, img_size):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [img_size, img_size], method="bicubic", antialias=True
    )
    return img

def create_test_ds(train_dir, batch_size, img_size, n_images, seed=None):
    img_paths = tf.data.Dataset.list_files(
        str(train_dir), shuffle=True, seed=seed).take(n_images
    )
    BUFFER_SIZE = tf.data.experimental.cardinality(img_paths)

    img_paths = img_paths.cache()
    ds = img_paths.map(
            lambda img: test_convert(img, img_size), 
            num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(
            batch_size, drop_remainder=False, 
            num_parallel_calls=AUTOTUNE
    )
    print(f'Test dataset size: {BUFFER_SIZE}')
    print(f'Test batches: {tf.data.experimental.cardinality(ds)}')
    ds = ds.prefetch(AUTOTUNE)
    return ds

def get_augmenter(image_size):
    return tf.keras.Sequential([
            tf.keras.Input(shape=(image_size, image_size, 3)),
            layers.RandomFlip(mode="horizontal"),
        ])

def reset_metrics(metrics):
    for _, metric in metrics.items():
        metric.reset_states()

def update_metrics(metrics, **kwargs):
    for metric_name, metric_value in kwargs.items():
        metrics[metric_name].update_state(metric_value)
