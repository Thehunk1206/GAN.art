import scipy
import moviepy.editor
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras

duration_sec = 5
image_zoom = 1.0
fps = 30
smoothing_sec = 2.0
grid_size = [1, 1]
latent_dim = 256
model_dir = "trained_model/g_model_WGAN_GP_epoch_10.h5"
g_model = load_model(model_dir)

random_seed = np.random.randint(0, 999)
num_frames = int(np.rint(duration_sec * fps))
random_state = np.random.RandomState(random_seed)


# Generate latent vectors
shape = [num_frames, 1] + list(g_model.input_shape[1:])
print(shape)
all_latents = random_state.randn(*shape)
all_latents = scipy.ndimage.gaussian_filter(
    all_latents, [smoothing_sec * fps] + [0] * len(g_model.input_shape), mode='reflect')
all_latents /= np.sqrt(np.mean(np.square(all_latents)))


def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_h, img_w, channels = images.shape

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros([grid_h * img_h, grid_w * img_w,
                     channels], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y: y + img_h, x: x + img_w] = images[idx]
    return grid


def make_frame(t):
    frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
    latents = all_latents[frame_idx]

    images = g_model.predict(latents)
    images = (images*127.5)/127.5

    grid = create_image_grid(images, grid_size)
    if image_zoom > 1:
        grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
    if grid.shape[2] == 1:
        grid = grid.repeat(3, 2)  # grayscale => RGB
    return grid


video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)


video_clip.write_videofile('random_grid_%s.mp4' %random_seed, fps=fps, codec='libx264', bitrate='2M')
