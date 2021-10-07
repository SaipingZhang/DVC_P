import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
import CNN_img_NearestNeighbor
import motion
import MC_network
import load
import gc
import DiscriminatorNetwork
from VGG19 import Vgg19

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# config lambda
parser.add_argument("--l", type=int, default=512, choices=[256, 512, 1024, 2048])
# config N:Number of filters, M:The number of filters in the last layer of the MV Encoder net
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

AddAL = 1
VGGLoss = 1
if VGGLoss:
    vgg = Vgg19(vgg_path='E:/DVC/OpenDVC-master') # initialize vgg19 net

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4
iter = 0
folder = np.load('folder.npy')

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
learning_rate = tf.placeholder(tf.float32, [])

with tf.variable_scope("flow_motion"):
    # ...............................Optical Flow Estimation..................................#
    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
flow_latent = CNN_img_NearestNeighbor.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
# string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=True)

flow_hat = CNN_img_NearestNeighbor.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Encode residual
Res = Y1_raw - Y1_MC

res_latent = CNN_img_NearestNeighbor.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
# string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

Res_hat = CNN_img_NearestNeighbor.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC

if VGGLoss:
    # VGG_loss
    [w, h, d] = Y1_raw.get_shape().as_list()[1:]
    vgg_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((vgg.extract_feature(Y1_raw) - vgg.extract_feature(Y1_com))))) / (w * h * d))

if AddAL:
    D_real, D_logits_real = DiscriminatorNetwork.discriminator(Y1_raw, batch_size)
    D_fake, D_logits_fake = DiscriminatorNetwork.discriminator(Y1_com, batch_size, reuse=True)
    d_loss = 0.5 * (tf.reduce_mean((D_real - 1)**2) + tf.reduce_mean(D_fake**2))
    g_loss = 0.5 * tf.reduce_mean((D_fake - 1)**2)

train_bpp_MV = tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

total_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
warp_mse = tf.reduce_mean(tf.squared_difference(Y1_warp, Y1_raw))
MC_mse = tf.reduce_mean(tf.squared_difference(Y1_raw, Y1_MC))

psnr = 10.0*tf.log(1.0/total_mse)/tf.log(10.0)

l = args.l

if AddAL:
    if VGGLoss:
        if iter < 400000:
            total_mse_new = total_mse + 0.04 * vgg_loss
        else:
            total_mse_new = total_mse + 0.1 * g_loss + 0.04 * vgg_loss

    warp_mse_new = warp_mse
    MC_mse_new = MC_mse
    train_loss_total_new = l * total_mse_new + (train_bpp_MV + train_bpp_Res)
    train_loss_MV_new = l * warp_mse_new + train_bpp_MV
    train_loss_MC_new = l * MC_mse_new + train_bpp_MV
else:
    total_mse_new = total_mse
    warp_mse_new = warp_mse
    MC_mse_new = MC_mse
    train_loss_total_new = l * total_mse_new + (train_bpp_MV + train_bpp_Res)
    train_loss_MV_new = l * warp_mse_new + train_bpp_MV
    train_loss_MC_new = l * MC_mse_new + train_bpp_MV

# Minimize loss and auxiliary loss, and execute update op.
step = tf.train.create_global_step()

train_MV = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MV_new, global_step=step)
train_MC = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MC_new, global_step=step)
train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total_new, global_step=step)

if AddAL:
    train_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_loss, global_step=step)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

train_op_MV = tf.group(train_MV, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MC = tf.group(train_MC, aux_step, entropy_bottleneck_mv.updates[0])
train_op_all = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])

if AddAL:
    train_op_D = tf.group(train_D)

tf.summary.scalar('psnr', psnr)
tf.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
tf.summary.scalar('g_loss', g_loss)
tf.summary.scalar('d_loss', d_loss)

save_path = './OpenDVC_PSNR_' + str(l)
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
saver = tf.train.Saver(max_to_keep=None)

if iter == 0:
    sess.run(tf.global_variables_initializer())
    var_motion = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
    saver_motion = tf.train.Saver(var_list=var_motion, max_to_keep=None)
    saver_motion.restore(sess, save_path='E:/DVC/PretrainedOpticalFlow/OpenDVC_model-20200715T023521Z-001/OpenDVC_model/PSNR_512_model/model.ckpt')
else:
    saver.restore(sess, save_path='E:/DVC/OpenDVC-master/OpenDVC_PSNR_512' + '/model.ckpt-' + str(iter-1))

while(True):

    if iter <= 100000:
        frames = 2

        if iter <= 20000:
            train_op = train_op_MV
        elif iter <= 40000:
            train_op = train_op_MC
        else:
            train_op = train_op_all
    else:
        frames = 7
        train_op = train_op_all

# optional
    if iter <= 400000:
        lr = lr_init
    else:
        lr = lr_init / 100.0

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data = load.load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)

    for ff in range(frames-1):

        if ff == 0:

            F0_com = data[0]
            F1_raw = data[1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

            if AddAL:
                if iter > 400000:
                    _ = sess.run([train_op_D],
                                 feed_dict={Y0_com: F0_com / 255.0,
                                            Y1_com: F1_decoded,
                                            Y1_raw: F1_raw / 255.0,
                                            learning_rate: lr})

        else:

            F0_com = F1_decoded * 255.0
            F1_raw = data[ff+1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com / 255.0,
                                                Y1_raw: F1_raw / 255.0,
                                                learning_rate: lr})

            if AddAL:
                if iter > 400000:
                    _ = sess.run([train_op_D],
                                 feed_dict={Y0_com: F0_com / 255.0,
                                            Y1_com: F1_decoded,
                                            Y1_raw: F1_raw / 255.0,
                                            learning_rate: lr})

        print('Training_OpenDVC Iteration:', iter)

        iter = iter + 1

        if iter % 500 == 0:

            merged_summary_op = tf.summary.merge_all()
            summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/255.0,
                                                                  Y1_raw: F1_raw/255.0})

            summary_writer.add_summary(summary_str, iter)

        if iter % 10000 == 0:

             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 700000:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decoded

    gc.collect()
