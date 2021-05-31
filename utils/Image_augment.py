import os
from tqdm import tqdm
from PIL import Image
import numpy as np


import imgaug as ia
from imgaug import augmenters as iaa

# set up the file paths
from_path = 'Abstract_art/'
to_path = 'resized_art/'

if not os.path.exists(to_path):
    os.mkdir(to_path)

# set up some parameters
img_h = 1280
img_w = 768
num_augmentations = 6

# set up the image augmenter
seq = iaa.Sequential([
    iaa.Rot90((0, 3)),
    iaa.Fliplr(0.5),
    iaa.PerspectiveTransform(scale=(0.0, 0.05), mode='replicate'),
    iaa.AddToHueAndSaturation((-20, 20))
])

# loop through the images, resizing and augementing
for file in tqdm(os.listdir(from_path)):
    try:
        image = Image.open(from_path + file)
        image_resized = image.resize((img_w, img_h), resample=Image.BICUBIC)
        image_resized.save(to_path + file)
        image_np = np.array(image_resized)
        images = [image_np] * num_augmentations
        images_aug = seq(images=images)
        for i in range(0, num_augmentations):
            im = Image.fromarray(np.uint8(images_aug[i]))
            to_file = to_path + file[:-4] + '_' + str(i).zfill(2) + '.jpg'
            im.save(to_file)  # , quality=95)
    except:
        pass
