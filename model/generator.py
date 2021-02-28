import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, \
    LeakyReLU, Dense, Dropout, Input, Reshape, Activation, BatchNormalization, add
from tensorflow.keras.models import Model
from tensorflow.python import keras



def upsample_block(
        x,
        filters: int,
        activation: bool,
        kernel_size: tuple = (3, 3),
        strides: tuple = (1, 1),
        upsample_size: tuple = (2, 2),
        padding="same",
        use_bn: bool = True,
        use_bias: bool = True,
        use_dropout: bool = False,
        drop_value: float = 0.3
):
    x = UpSampling2D(size=upsample_size, interpolation = 'bilinear')(x)
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.2)(x)
    if use_dropout:
        x = Dropout(drop_value)(x)

    return x


def upsample_res_block(
        x,
        filters: int,
        activation: bool,
        kernel_size: tuple = (3, 3),
        strides: tuple = (1, 1),
        upsample_size: tuple = (2, 2),
        padding="same",
        use_bn: bool = False,
        use_bias: bool = True,
        use_dropout: bool = True,
        drop_value: float = 0.3
):
    x_up = UpSampling2D(size=upsample_size, interpolation='bilinear')(x)
    merge_layer = x_up
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=use_bias)(x_up)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=use_bias)(x)

    if use_bn:
        x = BatchNormalization()(x)
    if activation:
        x = LeakyReLU(alpha=0.2)(x)
    if use_dropout:
        x = Dropout(drop_value)(x)

    shortcut_x = Conv2D(filters, (1, 1), strides=strides,
                        padding=padding, use_bias=use_bias)(merge_layer)
    shortcut_x = Activation('linear')(shortcut_x)

    layer_out = add([x, shortcut_x])

    return layer_out


def build_generator(latent_dim, image_size=(128, 128)) -> keras.Model:
    if image_size[0] != image_size[1]:
        raise "Image must be of square shape"

    #TODO change the divisor dynamically
    filters = image_size[0]//4

    f = [2**i for i in range(int(np.math.log2(filters)))][::-1]
    output_strides = filters//2

    h_output = image_size[0]//output_strides
    w_output = image_size[1]//output_strides

    noise = Input(shape=(latent_dim,), name="gen_noise")
    x = Dense(f[1]*filters*h_output*w_output, use_bias=False)(noise)
    x = LeakyReLU(0.2)(x)
    x = Reshape((h_output, w_output, f[1]*filters))(x)

    for i in range(1,int(np.math.log2(filters))):
        x = upsample_res_block(
            x,
            filters=f[i]*filters,
            activation=True,
        )

    x = Conv2D(filters=3, kernel_size=(3, 3),
               strides=(1, 1), padding='same', use_bias=True)(x)
    x = Dropout(0.3)(x)
    fake_out = Activation('tanh')(x)

    return Model(noise, fake_out, name="Generator1")


def build_generator_for_nonsquare(latent_dim, image_size=(320, 192)) -> keras.Model:
    if image_size[0] == image_size[1]:
        raise "input shape cannot be 1:1"
    
    #TODO change this divisor dynamically according to hieght of an image
    filters = image_size[0]//80
    h_factor = 5
    w_factor = 3

    f = [image_size[0]//2**x for x in range(1,int(np.math.log2(image_size[0]//h_factor))+1)]
    
    noise = Input(shape=(latent_dim,), name="gen_noise")
    x = Dense(f[1]*filters*(h_factor*2)*(w_factor*2), use_bias=False)(noise)
    x = LeakyReLU(0.2)(x)
    x = Reshape((h_factor*2, w_factor*2, f[1]*filters))(x)

    for i in range(1, len(f)):
        x = upsample_res_block(
            x,
            filters=f[i]*filters,
            activation=True,
        )

    x = Conv2D(filters=3, kernel_size=(3, 3),
               strides=(1, 1), padding='same', use_bias=True)(x)
    x = Dropout(0.2)(x)
    fake_out = Activation('tanh')(x)

    return Model(noise, fake_out, name="Generator_nonsquare")