'''
1.Smooth the image using EPF(Edge preserving Filter) DONE
2.Upsample by the factor of x8 using LapSRN model DONE
3.Adjust the brightness and contrast
'''

import sys
import os
from tqdm import tqdm
from datetime import datetime
import argparse

import cv2
from cv2 import dnn_superres


def init_super_res(model_path: str):
    '''
    Make sure that model name in the path should be in following format: modelname_xScale.pb
    '''
    global sr, modelname, modelscale
    sr = dnn_superres.DnnSuperResImpl_create()

    modelname = model_path.split(os.path.sep)[-1].split("_")[0].lower()
    modelscale = model_path.split("_x")[-1]
    modelscale = int(modelscale[:modelscale.find(".")])

    print(f"[info] Loading super resolution model {modelname}...")
    print(f"[info] Model name {modelname}")
    print(f"[info] Model scale {modelscale}")

    sr.readModel(model_path)
    sr.setModel(modelname, modelscale)


def post_process_image(image_path: str, post_process_image_path: str):
    now = datetime.now()
    str_date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    image = cv2.imread(image_path)
    image = cv2.edgePreservingFilter(image, flags=1, sigma_s=30, sigma_r=0.3)
    upscaled_image = sr.upsample(image)
    cv2.imwrite(
        f"{post_process_image_path}upsampled_art_{str_date_time}.jpg", upscaled_image)
    print(f"Upscaled {image_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to super resolution model")
    ap.add_argument("-i", "--image_folder", required=True,
                    help="path to input image folder")

    ap.add_argument("-o", "--output_folder", required=True,
                    help="path to output image folder")
    args = vars(ap.parse_args())

    model_path = args["model"]
    if not os.path.exists(model_path):
        print("File does not exist")
        sys.exit()

    init_super_res(model_path)

    images_folder = args["image_folder"]
    if not os.path.exists(images_folder):
        print(f"{images_folder} does not exist!")
        sys.exit()

    output_folder = args["output_folder"]
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for img_file in tqdm(os.listdir(images_folder)):
        post_process_image(image_path=f"{images_folder}{img_file}",post_process_image_path=output_folder)
