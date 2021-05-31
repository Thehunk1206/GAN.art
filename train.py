import os
import sys
import time
import matplotlib.pyplot as plt


from model.critic import build_critic, build_critic_for_nonsquare
from model.generator import build_generator, build_generator_for_nonsquare
from model.stylegan import Stylegan
from utils.losses import critic_loss, generator_loss, hinge_loss
from utils.dataset import TfdataPipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing


def check_dir(*args):
    for arg in args:
        if not os.path.exists(arg):
            os.mkdir(arg)


def save_sample_image(step: int, sample_image_dir: str, images: list):
    for i in range(len(images)):
        img = preprocessing.image.array_to_img(images[i])
        img.save(f"{sample_image_dir}art_{step}_{i}")


def plotResults(
    generator: keras.Model,
    latent_dim: int,
    IMG_H: int,
    IMG_W: int,
    step: int,
    sample_image_dir: str,
    number_of_sample: int = 16,
    result_dir: str = "results/",
    save_image: bool = True
):
    latent_in = tf.random.normal(
        shape=(number_of_sample, latent_dim))

    noise_in = tf.random.normal(
        shape=(1, IMG_H, IMG_W, 1))

    generated_images = generator([latent_in, noise_in])
    generated_images = (generated_images+1.0)/2.0
    if save_image:
        save_sample_image(step, sample_image_dir, generated_images)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.axis('off')
        plt.imshow(generated_images[i])
    filename = f"{result_dir}genrated_at_step_00{step}.png"
    plt.savefig(filename)
    plt.close()


def saveModel(
    generator: keras.Model,
    critic: keras.Model,
    step: int,
    save_model_dir: str,
    format: str = 'h5'
):
    generator.save(f"{save_model_dir}g_model_{step}", save_format=format)
    critic.save(f"{save_model_dir}c_model_{step}", save_format=format)


def plotLoss(
    c_loss: list,
    g_loss: list,
    gp: list,
    step: int,
    plots_dir: str
):
    plt.title(f'model losses at {step}')
    plt.plot(c_loss)
    plt.plot(g_loss)
    plt.plot(gp)
    plt.legend(['Critic', 'Generator', 'GP'], loc='upper right')
    plt.xlabel('Stepss')
    plt.ylabel('loss')
    filename = f"{plots_dir}loss_plot_at_step_{step}.png"
    plt.savefig(filename)
    plt.close()


def train(
    pre_trained_c_model_path: str = None,
    pre_trained_g_model_path: str = None,
    data_dir: str = "abstract_art/",
    result_dir: str = "results/",
    save_model_dir: str = "trained_model/",
    plots_dir: str = "loss_graph_dir/",
    sample_image_dir: str = "sample_art/",
    batch_size: int = 4,
    image_h: int = 640,
    image_w: int = 768,
    image_c: int = 3,
    latent_dim: int = 256,
    epoch: int = 300,
    LR=0.0001,
    beta_1: float = 0.5,
    beta_2: float = 0.9,
    gp_weight=10,

):
    if not os.path.exists(data_dir):
        print(f"{data_dir} directory does not exist ")
        sys.exit()

    check_dir(
        result_dir,
        save_model_dir,
        plots_dir,
        sample_image_dir
    )

    # start pipiline for datset
    images_path = [data_dir+x for x in os.listdir(data_dir)]
    tfpipeline = TfdataPipeline(
        IMG_H=image_h,
        IMG_W=image_w,
        IMG_C=image_c,
        batch_size=batch_size,
    )
    image_dataset = tfpipeline.tf_dataset(images_path=images_path)

    # Instantiate both Generator and Critic optimizer
    generator_optimizer = Adam(
        learning_rate=LR, beta_1=beta_1, beta_2=beta_2
    )
    critic_optimizer = Adam(
        learning_rate=LR*4, beta_1=beta_1, beta_2=beta_2
    )

    # Build mddels
    # look for pretrained model
    if pre_trained_c_model_path and pre_trained_g_model_path:
        g_model = load_model(pre_trained_g_model_path)
        c_model = load_model(pre_trained_c_model_path)
    else:

        if image_h == image_w:
            # Buil models for square image
            g_model = build_generator(
                latent_dim=latent_dim,
                image_size=(image_h, image_w)
            )

            c_model = build_critic(
                input_shape=(image_h, image_w, image_c)
            )
        else:
            # build for non square image
            g_model = build_generator_for_nonsquare(
                latent_dim=latent_dim,
                image_size=(image_h, image_w),
                h_factor=5,
                w_factor=6
            )

            c_model = build_critic_for_nonsquare(
                input_shape=(image_h, image_w, image_c)
            )

    # show summary of the model
    g_model.summary()
    c_model.summary()

    # plot model
    plot_model(g_model, to_file='generator.png', show_shapes=True)
    plot_model(c_model, to_file='critic.png', show_shapes=True)

    # instantiating styleGAN
    stylegan = Stylegan(
        critic=c_model,
        generator=g_model,
        latent_dim=latent_dim,
        gp_weight=gp_weight
    )

    stylegan.compile(
        c_optimizer=critic_optimizer,
        g_optimizer=generator_optimizer,
        c_loss_fn=hinge_loss,
        g_loss_fn=generator_loss
    )

    # start training
    start_time = time.time()
    generator_losses = []
    critic_losses = []
    gradient_panelties = []
    step = 0

    for e in range(epoch):
        for data in image_dataset:
            closs, gloss, gp = stylegan.train_step(data)
            step = step + 1
            if step % 50 == 0:
                critic_losses.append(closs)
                generator_losses.append(gloss)
                gradient_panelties.append(gp)

                print(f"\n\nStep No.: {step}")
                print(f"C Loss: {closs}")
                print(f"G loss: {gloss}")
                print(f"GP: {gp}")
                print(f"Number images shown: {batch_size*step}")
                print("="*30)
                s = round((time.time() - start_time), 4)
                time_per_step = (s/50)*1000
                print(f"{time_per_step}ms/step")
                start_time = time.time()

                steps_per_second = 50 / s
                steps_per_min = steps_per_second * 60
                print(f"steps/sec: {steps_per_second}")
                print(f"steps/min: {steps_per_min}")

            if step % 200 == 0:
                print("[Info] Plotting loss, Plotting results")
                plotResults(g_model, latent_dim, image_h,
                            image_w, step, sample_image_dir)
                plotLoss(critic_losses, generator_losses,
                         gradient_panelties, step, plots_dir)
            if step % 10000 == 0 and step > 0:
                print("[info] Saving model")
                saveModel(g_model, c_model, step, save_model_dir)


if __name__ == "__main__":
    train()
