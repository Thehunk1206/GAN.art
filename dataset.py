import os
import sys
import glob

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.image_ops_impl import ResizeMethod


class TfdataPipeline:
    def __init__(
        self, IMG_H: int = 128,
        IMG_W: int = 128,
        IMG_C: int = 3,
        batch_size: int = 16
    ) -> None:
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.IMG_C = IMG_C
        self.batch_size = batch_size

    def _load_image(self, image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image,channels=3)
        image = tf.image.resize(
            image, (self.IMG_H, self.IMG_W), method=ResizeMethod.BICUBIC)
        image = tf.cast(image, tf.float32)
        # scale the pixel value between -1 to 1
        image = (image-127.5)/127.5
        return image

    def tf_dataset(self, images_path: str):
        dataset = tf.data.Dataset.from_tensor_slices(images_path)
        dataset = dataset.shuffle(buffer_size=self.batch_size*2)
        dataset = dataset.map(
            self._load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset
