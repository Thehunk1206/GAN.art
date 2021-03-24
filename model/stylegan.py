import tensorflow as tf
from tensorflow.python import keras


class Stylegan(keras.Model):

    def __init__(
        self,
        critic: keras.Model,
        generator: keras.Model,
        latent_dim=128,
        gp_weight=10.0,
    ):
        super(Stylegan, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        super(Stylegan, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn

    @tf.function
    def calculate_gradient_penalty(self, batch_size, real_images, fake_images):
        '''
        This function calculates Gradient penalty in order to enforce 1-Lipschitz
        constraint over the Gradients norms of fucntion C(x). i.e it has the
        gradient norm of atmost 1 everywhere.

        To penalize the gradients we calculate C_loss on an interpolated image
        between real and fake image i.e
        x_interpolated = x_real*alpha + (1-alpha)*x_fake
        where alpha is randomly sampled from a normal dist between 0 and 1

        after calcualting c_loss of x_interpolated we add this loss to Critic loss
        '''
        # Interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = (fake_images*alpha) + ((1-alpha)*real_images)

        # watch for gradient
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # get critic's out put on interpolated image
            inter_pred = self.critic(interpolated, training=True)

        # calculate the gradient w.r.t to this interploted image
        grads = gp_tape.gradient(inter_pred, [interpolated])[0]
        # calculate the norm of the gradient
        # L2-norm of the vector is given as ||V|| = sqrt((sum(squares(v))))
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((l2norm - 1.0)**2)

        return gp

    @tf.function
    def train_step(self, real_images):

        # we'll perform following steps for each batch as laid in
        # WGAN_GP paper
        # 1. Train the generator and get the generator loss
        # 2. Train the critic and get the critic loss
        # 3. Calculate Gradient penalty
        # 4. Multiply GP with Lambda (GP_weigth)
        # 5. Add the gradient penalty to critic loss
        # 6. return g_loss and C_loss as dict

        batch_size = tf.shape(real_images)[0]
        img_h = tf.shape(real_images)[1]
        img_w = tf.shape(real_images)[2]

        # Train Generator and Critic
        # latent vector
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        # stochastic noise
        noise_in = tf.random.normal(
            shape=(batch_size, img_h, img_w, 1))

        with tf.GradientTape() as ctape, tf.GradientTape() as gtape:
            # generate fake image
            fake_images = self.generator(
                [random_latent_vectors, noise_in], training=True)

            # get output of critic on fake images
            fake_logits = self.critic(fake_images, training=True)
            # get outpit of critic on real images
            real_logits = self.critic(real_images, training=True)

            # calculate loss of Generator and Critic using values on fake and real images
            g_loss = self.g_loss_fn(fake_logits)
            c_cost = self.c_loss_fn(real_logits, fake_logits)
            gp = self.calculate_gradient_penalty(
                batch_size, real_images, fake_images)
            c_loss = c_cost + (self.gp_weight * gp)

        # Calculate gradients Of Generator and Criticx
        g_gradients = gtape.gradient(
            g_loss, self.generator.trainable_variables)
        c_gradient = ctape.gradient(
            c_loss, self.critic.trainable_variables)

        # update the weigths of generator and critic using optimizer
        self.g_optimizer.apply_gradients(
            zip(g_gradients, self.generator.trainable_variables)
        )
        self.c_optimizer.apply_gradients(
            zip(c_gradient, self.critic.trainable_variables))

        return float(c_loss), float(g_loss), float(gp)
