import os
from time import time
from tqdm import tqdm
import numpy as np

import scipy
import imageio

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow import keras


model_dir = "trained_model/latest_g_model_StyleGan.h5"


def load_g_model(model_path: str) -> keras.Model:
    return load_model(model_path)


def generate_z_from_seed(seeds: np.ndarray, latent_dim: int = 256):
    zs = []
    for seed_id, seed in enumerate(seeds):
        rand_state = np.random.RandomState(seed)
        z = rand_state.randn(1, latent_dim)
        zs.append(z)
    return zs


def generate_image(zs, inoise, g_model: keras.Model):
    imgs = []

    for z, noise in tqdm(zip(zs, inoise)):
        image = g_model.predict([z, noise])
        image = np.squeeze(image, axis=0)
        #image = preprocessing.image.array_to_img(image)
        imgs.append(image)

    return imgs


if __name__ == "__main__":
    #number_of_images = int(input("Number of images to be generated: "))
    #seeds = np.random.randint(99999,size = 30*5)
    #zs = generate_z_from_seed(seeds)

    duration_sec = 5
    fps = 30
    smoothing_sec = 0.1
    num_frames = int(np.rint(duration_sec * fps))

    g_model = load_g_model(model_dir)

    random_seed = np.random.randint(0, 999)
    shape = [num_frames, 1] + [256]  # [frames,1,256]

    random_state = np.random.RandomState(random_seed)
    noise_in = np.random.randn(num_frames, 1, 320, 192, 1)
    all_latents = random_state.randn(*shape)
    all_latents = scipy.ndimage.gaussian_filter(
        all_latents, [smoothing_sec * fps] + [0] * 2, mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    images = generate_image(all_latents, noise_in, g_model)
    imageio.mimwrite('myGif.mp4', images)
    # for i in range(len(images)):
    #    images[i].save(f"generated_image/generated_art_{i}.png")
