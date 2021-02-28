import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, \
    LeakyReLU, Dense, Dropout, Input, Activation, Flatten, add
from tensorflow.keras.models import Model


def conv_block(
    x,
    filters: int,
    kernel_size: tuple = (3, 3),
    strides: tuple = (2, 2),
    padding: str = 'same',
    use_bias: bool = True,
    use_dropout: bool = True,
    drop_value: float = 0.4
):
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=use_bias)(x)
    x = LeakyReLU(alpha=0.2)(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def conv_res_block(
    x,
    filters: int,
    kernel_size: tuple = (3, 3),
    strides: tuple = (2, 2),
    padding: str = 'same',
    use_bias: bool = True,
    use_dropout: bool = True,
    drop_value: float = 0.4
):
    merge_layer = x
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=use_bias)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters, kernel_size, strides=(1, 1),
               padding=padding, use_bias=use_bias)(x)
    x = LeakyReLU(alpha=0.2)(x)

    if use_dropout:
        x = Dropout(drop_value)(x)

    shortcut_x = Conv2D(filters, (1, 1), strides=strides,
                        padding=padding, use_bias=use_bias)(merge_layer)
    shortcut_x = Activation('linear')(shortcut_x)

    layer_out = add([x, shortcut_x])

    return layer_out


def build_critic(input_shape: tuple = (128, 128, 3)):
    if input_shape[0] != input_shape[1]:
        raise "Image must be of square shape"

    filters = input_shape[0]//4
    f = [2**i for i in range(int(np.math.log2(filters)))]

    img_input = Input(shape=input_shape, name='image_in')
    x = img_input
    for i in range(int(np.math.log2(filters))):
        x = conv_res_block(
            x,
            filters=f[i] * filters,
        )
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)

    return Model(img_input, out, name="Critic1")


def build_critic_for_nonsquare(input_shape: tuple = (320, 192, 3)):
    if input_shape[0] == input_shape[1]:
        raise "input shape cannot be 1:1"

    filters = input_shape[0]//80

    f = [input_shape[0]//2**x for x in range(1,int(np.math.log2(input_shape[0]//5)))][::-1]

    img_input = Input(shape=input_shape, name='image_in')
    x = img_input
    for i in range(1,len(f)):
        x = conv_res_block(
            x,
            filters=f[i] * filters,
        )
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    out = Dense(1)(x)

    return Model(img_input, out, name="Critic_nonsquare")
