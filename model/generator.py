import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, \
    LeakyReLU, Dense, Dropout, Input, Reshape, Activation, add, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python import keras

channel_factor = 32

# Lambda functions


def AdaIN(x: list):
    '''
    x : Its a list of [image representation, (scale)styleGamma, (bias)styleBeta]
    '''
    # Normalize image representation
    mean = K.mean(x[0], axis=[1, 2], keepdims=True)
    std = K.std(x[0], axis=[1, 2], keepdims=True) + 1e-8
    y = (x[0] - mean) / std

    # reshape (scale)styleGamma and (bias)styleBeta params
    p_shape = [-1, 1, 1, y.shape[-1]]
    scale = K.reshape(x[1], p_shape)
    bias = K.reshape(x[2], p_shape)

    return y * scale + bias


def crop_and_fit(x):
    '''
    X : Its a lit of [1Channel Noise, image representation]
    crop any piece of Noise channel as same height and width of image representation 
    '''

    h = x[1].shape[1]
    w = x[1].shape[2]

    # crop noise channel having same h and w as image representation
    return x[0][:, :h, :w, :]


def g_block(
        x,
        filters: int,
        latent_vector: any,
        noise: any,
        kernel_size: tuple = (3, 3),
        upsample_size: tuple = (2, 2),
        padding="same",
        upsample: bool = True
):

    # upsampling
    if upsample:
        x = UpSampling2D(size=upsample_size, interpolation='bilinear')(x)
    else:
        x = Activation('linear')(x)

    # Get (Scale)styleGamma and (Bias)styleBeta prams for AdaIn
    g = Dense(filters)(latent_vector)
    b = Dense(filters)(latent_vector)

    # Crop noise and scale using Dense layer
    noise = Lambda(crop_and_fit)([noise, x])
    noise = Dense(filters, kernel_initializer='zeros')(noise)

    out = Conv2D(
        filters, kernel_size,
        padding=padding,
        kernel_initializer='he_normal'
    )(x)
    # add noise
    out = add([out, noise])

    # AdaIn layer
    out = Lambda(AdaIN)([out, g, b])
    out = LeakyReLU(alpha=0.2)(out)

    return out


def build_generator(latent_dim, image_size=(128, 128)) -> keras.Model:
    '''
    G_Block arch: noise --> Upsample --> conv+noise --> adaIn --> activation --> out 
    '''

    if image_size[0] != image_size[1]:
        raise "Image must be of square shape"

    # TODO change the divisor dynamically
    f = [2**i for i in range(int(np.math.log2(image_size[0])))][::-1]

    latent_in = Input(shape=(latent_dim,), name="latentIn")
    noise_in = Input(shape=(image_size[0], image_size[1], 1), name="noiseIn")

    # 6 Latent Mapping layers(mlp)
    latent = Dense(latent_dim)(latent_in)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)

    # Using constant as input to main Generator model
    x = Dense(1)(latent_in)
    # no matter whats the latent input be, it will always be constant
    x = Lambda(lambda x: x * 0 + 1, name="constant")(x)
    x = Dense(4*4*latent_dim, kernel_initializer='he_normal')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((4, 4, latent_dim))(x)

    # 4x4
    x = g_block(x, filters=16*channel_factor, latent_vector=latent,
                noise=noise_in, upsample=False)

    # 8x8
    x = g_block(x, filters=10*channel_factor,
                latent_vector=latent, noise=noise_in)

    # 16x16
    x = g_block(x, filters=8*channel_factor,
                latent_vector=latent, noise=noise_in)

    # 32x32
    x = g_block(x, filters=6*channel_factor,
                latent_vector=latent, noise=noise_in)

    # 64x64
    x = g_block(x, filters=4*channel_factor,
                latent_vector=latent, noise=noise_in)

    # 128x128
    x = g_block(x, filters=2*channel_factor,
                latent_vector=latent, noise=noise_in)

    # 256x256
    x = g_block(x, filters=channel_factor,
                latent_vector=latent, noise=noise_in)

    # 512x512
    x = g_block(x, filters=channel_factor,
                latent_vector=latent, noise=noise_in)

    x = Conv2D(filters=3, kernel_size=(3, 3),
               strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    fake_out = Activation('tanh')(x)

    return Model(inputs=[latent_in, noise_in], outputs=fake_out, name="Generator_for_square")


def build_generator_for_nonsquare(latent_dim: int, image_size: tuple = (320, 192), h_factor: int = 5, w_factor: int = 3) -> keras.Model:
    if image_size[0] == image_size[1]:
        raise "input shape cannot be 1:1"

    f = [image_size[0]//2 **
         x for x in range(1, int(np.math.log2(image_size[0]//h_factor))+1)]

    latent_in = Input(shape=(latent_dim,), name="latentIn")
    noise_in = Input(shape=(image_size[0], image_size[1], 1), name="noiseIn")

    # 6 Latent Mapping layers(mlp)
    latent = Dense(latent_dim)(latent_in)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)
    latent = Dense(latent_dim)(latent)
    latent = LeakyReLU(alpha=0.1)(latent)

    # Using constant as input to main Generator model
    x = Dense(1)(latent_in)
    # no matter whats the latent input be, it will always be constant
    x = Lambda(lambda x: x * 0 + 1, name="constant")(x)
    x = Dense(h_factor*w_factor*latent_dim, kernel_initializer='he_normal')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((h_factor, w_factor, latent_dim))(x)

    x = g_block(x, filters=16*channel_factor, latent_vector=latent,
                noise=noise_in, upsample=False)

    x = g_block(x, filters=16*channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=8*channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=6*channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=4*channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=2*channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=channel_factor,
                latent_vector=latent, noise=noise_in)

    x = g_block(x, filters=channel_factor,
                latent_vector=latent, noise=noise_in)

    x = Conv2D(filters=3, kernel_size=(3, 3),
               strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    fake_out = Activation('tanh')(x)

    return Model(inputs=[latent_in, noise_in], outputs=fake_out, name="Generator_for_non_square")
