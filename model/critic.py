import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, \
    LeakyReLU, Dense, Dropout, Input, Activation, Flatten, AveragePooling2D
from tensorflow.keras.models import Model

channel_factor = 32

def d_block(
    x,
    filters: int,
    kernel_size: tuple = (3, 3),
    padding: str = 'same',
    use_dropout: bool = True,
    drop_value: float = 0.2,
    use_pool: bool = True
):
    x = Conv2D(filters, kernel_size,
               padding=padding,kernel_initializer='he_normal')(x)
    x = LeakyReLU(alpha=0.2)(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    
    if use_pool:
        x = AveragePooling2D()(x)

    return x



def build_critic(input_shape: tuple = (128, 128, 3)):
    if input_shape[0] != input_shape[1]:
        raise "Image must be of square shape"

    # TODO change the divisor dynamically
    filters = input_shape[0]//4
    f = [2**i for i in range(int(np.math.log2(filters)))]

    image_input = Input(input_shape, name = "Image_in")

    x = d_block(image_input, channel_factor)

    x = d_block(x, 2 * channel_factor)

    x = d_block(x, 3 * channel_factor)

    x = d_block(x, 4 * channel_factor)

    x = d_block(x, 6 * channel_factor)

    x = d_block(x, 8 * channel_factor)

    x = d_block(x, 16 * channel_factor, use_pool=False)
    
    x = Flatten()(x)

    x = Dense(channel_factor, kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)
    out = Dense(1, kernel_initializer='he_normal')(x)

    return Model(image_input, out, name="Critic1")


def build_critic_for_nonsquare(input_shape: tuple = (320, 192, 3)):
    if input_shape[0] == input_shape[1]:
        raise "input shape cannot be 1:1"

    filters_factor = 2

    f = [input_shape[0]//2 **
         x for x in range(1, int(np.math.log2(input_shape[0]//5)))][::-1]

    image_input = Input(input_shape, name = "Image_in")

    x = d_block(image_input, channel_factor)

    x = d_block(x, 2*channel_factor)

    x = d_block(x, 4*channel_factor)

    x = d_block(x, 6*channel_factor)

    x = d_block(x, 8*channel_factor)

    x = d_block(x, 16*channel_factor)

    x = d_block(x, 16*channel_factor, use_pool=False)

    x = Flatten()(x)

    x = Dense(channel_factor, kernel_initializer='he_normal')(x)

    x = Dropout(0.2)(x)

    out = Dense(1,kernel_initializer='he_normal')(x)

    

    return Model(image_input, out, name="Critic_nonsquare")
