import tensorflow as tf
'''
W_loss_gp for Critic and 
W_loss for Generator
The total loss of WGAN_GP(Wasserstein Generative Adverserial nets Gradient panelty)
is given as min max {E[C(x)] - E[C(G(x_fake))] + lambda x E[(||Grad(C(X_interpolated))||2- 1)^2] }
'''


def critic_loss(real_image, fake_image):
    real_loss = tf.reduce_mean(real_image)
    fake_loss = tf.reduce_mean(fake_image)

    return fake_loss - real_loss


def generator_loss(fake_image):
    return -tf.reduce_mean(fake_image)
