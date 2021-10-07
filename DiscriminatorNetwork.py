import tensorflow as tf
from ops import *

d_bn1 = batch_norm(name='d_bn1')
d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
df_dim = 64

def discriminator( image, batch_size, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        h0 = lrelu(conv2d(image, df_dim, name='d_h0_conv'))

        h1 = lrelu(d_bn1(conv2d(h0, df_dim * 2, name='d_h1_conv')))
        h2 = lrelu(d_bn2(conv2d(h1, df_dim * 4, name='d_h2_conv')))
        h3 = lrelu(d_bn3(conv2d(h2, df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [batch_size, -1]), 1, 'd_h4_lin')


        return tf.nn.sigmoid(h4), h4

