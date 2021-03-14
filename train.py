import os
import sys
import matplotlib.pyplot as plt

from model.critic import build_critic, build_critic_for_nonsquare
from model.generator import build_generator, build_generator_for_nonsquare
from model.stylegan import Stylegan
from losses import critic_loss, generator_loss
from dataset import TfdataPipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class MonitorGan(callbacks.Callback):
    def __init__(
        self,
        generator: keras.Model,
        critic: keras.Model,
        sample_img: int = 16,
        latent_dim: int = 128,
        IMG_H: int = 128,
        IMG_W: int = 128,
        result_dir: str = 'results/',
        trained_model_dir: str = 'trained_model/'

    ):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.sample_img = sample_img
        self.latent_dim = latent_dim
        self.IMG_H = IMG_H
        self.IMG_W = IMG_W
        self.result_dir = result_dir
        self.trained_model_dir = trained_model_dir

        if not os.path.exists(self.trained_model_dir):
            os.mkdir(self.trained_model_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            self.generator.save(
                f"{self.trained_model_dir}g_model_StyleGAN_epoch_{epoch}.h5")
            self.critic.save(
                f"{self.trained_model_dir}c_model_StyleGAN_epoch_{epoch}.h5")

        self.generator.save(
            f"{self.trained_model_dir}latest_g_model_StyleGan.h5")
        self.critic.save(
            f"{self.trained_model_dir}latest_c_model_styleGan.h5")

        latent_in = tf.random.normal(
            shape=(self.sample_img, self.latent_dim))

        noise_in = tf.random.normal(
            shape=(self.sample_img,self.IMG_H, self.IMG_W, 1))

        generated_images = self.generator([latent_in, noise_in])
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.axis('off')
            plt.imshow(generated_images[i])
        filename = f"{self.result_dir}genrated_at_epoch_00{epoch+1}.png"
        plt.savefig(filename)
        plt.close()


def train(
    pre_trained_c_model_path: str = None,
    pre_trained_g_model_path: str = None,
    data_dir: str = "abstract_art/",
    result_dir: str = "results/",
    batch_size: int = 16,
    image_h: int = 128,
    image_w: int = 128,
    image_c: int = 3,
    latent_dim: int = 256,
    epoch: int = 1,
    LR=0.0001,
    beta_1: float = 0.0,
    beta_2: float = 0.9,
    n_critic: int = 1,
    gp_weight=10,

):

    if not os.path.exists(result_dir):
        os.mkdir("results")

    if not os.path.exists(data_dir):
        print(f"{data_dir} directory does not exist ")
        sys.exit()

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
        learning_rate=LR, beta_1=beta_1, beta_2=beta_2
    )

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
                image_size=(image_h, image_w)
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

    # instantiate keras callback
    clbk = MonitorGan(
        generator=g_model, critic=c_model, latent_dim=latent_dim, result_dir=result_dir
    )

    # instantiating wgan
    stylegan = Stylegan(
        critic=c_model,
        generator=g_model,
        latent_dim=latent_dim,
        n_critic=n_critic,
        gp_weight=gp_weight
    )

    stylegan.compile(
        c_optimizer=critic_optimizer,
        g_optimizer=generator_optimizer,
        c_loss_fn=critic_loss,
        g_loss_fn=generator_loss
    )

    history = stylegan.fit(image_dataset, batch_size=batch_size, epochs=epoch, callbacks=[clbk])


if __name__ == "__main__":
    train()
