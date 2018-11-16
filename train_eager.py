#!/usr/bin/env python3

import math
import multiprocessing
import os
import time
import sys

import cv2
import matplotlib
# matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from pycocotools.coco import maskUtils

import _pickle as cPickle

sys.path.append('.')

from train_config import config
from openpose_plus.models import model
from openpose_plus.utils import PoseInfo, draw_results, get_heatmap, get_vectormap, load_mscoco_dataset, tf_repeat, vis_annos

#Frizy add for eager
import matplotlib.pyplot as plt
tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)  # to save visualization results
tl.files.exists_or_mkdir(config.MODEL.model_path, verbose=False)  # to save model files

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FIXME: Don't use global variables.
# define hyper-parameters for training
batch_size = config.TRAIN.batch_size
lr_decay_every_step = config.TRAIN.lr_decay_every_step
n_step = config.TRAIN.n_step
save_interval = config.TRAIN.save_interval
weight_decay_factor = config.TRAIN.weight_decay_factor
lr_init = config.TRAIN.lr_init
lr_decay_factor = config.TRAIN.lr_decay_factor

# FIXME: Don't use global variables.
# define hyper-parameters for model
model_path = config.MODEL.model_path
n_pos = config.MODEL.n_pos
hin = config.MODEL.hin
win = config.MODEL.win
hout = config.MODEL.hout
wout = config.MODEL.wout


def get_pose_data_list(im_path, ann_path):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    print("[x] Get pose data from {}".format(im_path))
    data = PoseInfo(im_path, ann_path, False)
    imgs_file_list = data.get_image_list()
    objs_info_list = data.get_joint_list()
    mask_list = data.get_mask()
    targets = list(zip(objs_info_list, mask_list))
    if len(imgs_file_list) != len(objs_info_list):
        raise Exception("number of images and annotations do not match")
    else:
        print("{} has {} images".format(im_path, len(imgs_file_list)))
    return imgs_file_list, objs_info_list, mask_list, targets


def draw_results_one(images, heats_ground=None, heats_result=None, pafs_ground=None, pafs_result=None, masks=None, name='', index=0):
    if heats_ground is not None:
        heat_ground = heats_ground
    if heats_result is not None:
        heat_result = heats_result
    if pafs_ground is not None:
        paf_ground = pafs_ground
    if pafs_result is not None:
        paf_result = pafs_result
    if masks is not None:
        mask = masks[:, :, np.newaxis]
        mask1 = np.repeat(mask, n_pos, 2)
        mask2 = np.repeat(mask, n_pos * 2, 2)
        # print(mask1.shape, mask2.shape)
    image = images

    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(2, 3, 1)
    plt.imshow(image)

    if pafs_ground is not None:
        a = fig.add_subplot(2, 3, 2)
        a.set_title('Vectormap_ground')
        vectormap = paf_ground * mask2
        tmp2 = vectormap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

        # tmp2_odd = tmp2_odd * 255
        # tmp2_odd = tmp2_odd.astype(np.int)
        plt.imshow(tmp2_odd, alpha=0.3)

        # tmp2_even = tmp2_even * 255
        # tmp2_even = tmp2_even.astype(np.int)
        plt.colorbar()
        plt.imshow(tmp2_even, alpha=0.3)

    if pafs_result is not None:
        a = fig.add_subplot(2, 3, 3)
        a.set_title('Vectormap result')
        if masks is not None:
            vectormap = paf_result * mask2
        else:
            vectormap = paf_result
        tmp2 = vectormap.transpose((2, 0, 1))
        tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
        tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
        plt.imshow(tmp2_odd, alpha=0.3)

        plt.colorbar()
        plt.imshow(tmp2_even, alpha=0.3)

    if heats_result is not None:
        a = fig.add_subplot(2, 3, 4)
        a.set_title('Heatmap result')
        if masks is not None:
            heatmap = heat_result * mask1
        else:
            heatmap = heat_result
        tmp = heatmap
        tmp = np.amax(heatmap[:, :, :-1], axis=2)

        plt.colorbar()
        plt.imshow(tmp, alpha=0.3)

    if heats_ground is not None:
        a = fig.add_subplot(2, 3, 5)
        a.set_title('Heatmap ground truth')
        if masks is not None:
            heatmap = heat_ground * mask1
        else:
            heatmap = heat_ground
        tmp = heatmap
        tmp = np.amax(heatmap[:, :, :-1], axis=2)

        plt.colorbar()
        plt.imshow(tmp, alpha=0.3)

    if masks is not None:
        a = fig.add_subplot(2, 3, 6)
        a.set_title('Mask')
        # print(mask.shape, tmp.shape)
        plt.colorbar()
        plt.imshow(mask[:, :, 0], alpha=0.3)
    # plt.savefig(str(i)+'.png',dpi=300)
    # plt.show()
    plt.savefig(os.path.join(config.LOG.vis_path, '%s%d.png' % (name, index)), dpi=300)


def make_model(img, results, mask, is_train=True, reuse=False):
    confs = results[:, :, :, :n_pos]
    pafs = results[:, :, :, n_pos:]
    m1 = tf_repeat(mask, [1, 1, 1, n_pos])
    m2 = tf_repeat(mask, [1, 1, 1, n_pos * 2])

    cnn, b1_list, b2_list, net = model(img, n_pos, m1, m2, is_train, reuse)

    # define loss
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    stage_losses = []

    for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
        loss_l1 = tf.nn.l2_loss((l1.outputs - confs) * m1)
        loss_l2 = tf.nn.l2_loss((l2.outputs - pafs) * m2)

        losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        stage_losses.append(loss_l1 / batch_size)
        stage_losses.append(loss_l2 / batch_size)

    last_conf = b1_list[-1].outputs
    last_paf = b2_list[-1].outputs
    last_losses_l1.append(loss_l1)
    last_losses_l2.append(loss_l2)
    l2_loss = 0.0

    for p in tl.layers.get_variables_with_name('kernel', True, True):
        l2_loss += tf.contrib.layers.l2_regularizer(weight_decay_factor)(p)
    total_loss = tf.reduce_sum(losses) / batch_size + l2_loss

    log_tensors = {'total_loss': total_loss, 'stage_losses': stage_losses, 'l2_loss': l2_loss}
    net.cnn = cnn
    net.img = img  # net input
    net.last_conf = last_conf  # net output
    net.last_paf = last_paf  # net output
    net.confs = confs  # GT
    net.pafs = pafs  # GT
    net.m1 = m1  # mask1, GT
    net.m2 = m2  # mask2, GT
    net.stage_losses = stage_losses
    net.l2_loss = l2_loss
    return net, total_loss, log_tensors
    # return total_loss, last_conf, stage_losses, l2_loss, cnn, last_paf, img, confs, pafs, m1, net



def single_train(img, results, mask):
    net, total_loss, log_tensors = make_model(img, results, mask, is_train=True, reuse=False)
    x_ = net.img  # net input
    last_conf = net.last_conf  # net output
    last_paf = net.last_paf  # net output
    confs_ = net.confs  # GT
    pafs_ = net.pafs  # GT
    mask = net.m1  # mask1, GT
    # net.m2 = m2                 # mask2, GT
    stage_losses = net.stage_losses
    l2_loss = net.l2_loss

    global_step = tf.Variable(1, trainable=False)
    print('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_every_step: {}'.format(
        n_step, batch_size, lr_init, lr_decay_every_step))
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    opt = tf.train.MomentumOptimizer(lr_v, 0.9)
    train_op = opt.minimize(total_loss, global_step=global_step)

    # sess.run(tf.global_variables_initializer())
    tf.global_variables_initializer()

    # restore pre-trained weights
    try:
        # tl.files.load_and_assign_npz(sess, os.path.join(model_path, 'pose.npz'), net)
        # tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'pose.npz'))
        tl.files.load_and_assign_npz(name=os.path.join(model_path, 'pose.npz'))
    except:
        print("no pre-trained model")

    # train until the end
    # sess.run(tf.assign(lr_v, lr_init))
    tf.assign(lr_v, lr_init)
    while True:
        tic = time.time()
        # step = sess.run(global_step)
        step = global_step
        if step != 0 and (step % lr_decay_every_step == 0):
            new_lr_decay = lr_decay_factor ** (step // lr_decay_every_step)
            # sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            tf.assign(lr_v, lr_init * new_lr_decay)

        # [_, _loss, _stage_losses, _l2, conf_result, paf_result] = \
        #     sess.run([train_op, total_loss, stage_losses, l2_loss, last_conf, last_paf])
        [_, _loss, _stage_losses, _l2, conf_result, paf_result] = \
            [train_op, total_loss, stage_losses, l2_loss, last_conf, last_paf]

        # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
        # lr = sess.run(lr_v)
        lr = lr_v
        print('Total Loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
            step, n_step, _loss, lr, _l2,
            time.time() - tic))
        for ix, ll in enumerate(_stage_losses):
            print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

        # save intermediate results and model
        if (step != 0) and (step % save_interval == 0):
            # save some results
            # [img_out, confs_ground, pafs_ground, conf_result, paf_result,
            #  mask_out] = sess.run([x_, confs_, pafs_, last_conf, last_paf, mask])
            [img_out, confs_ground, pafs_ground, conf_result, paf_result, mask_out] \
                = [x_, confs_, pafs_, last_conf, last_paf, mask]
            draw_results(img_out, confs_ground, conf_result, pafs_ground, paf_result, mask_out, 'train_%d_' % step)
            # save model
            # tl.files.save_npz(
            #    net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
            # tl.files.save_npz(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
            # tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
            # tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
            tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'))
            tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'))
        if step == n_step:  # training finished
            break


if __name__ == '__main__':

    if 'coco' in config.DATA.train_data:
        # automatically download MSCOCO data to "data/mscoco..."" folder
        train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
            load_mscoco_dataset(config.DATA.data_path, config.DATA.coco_version, task='person')

        # read coco training images contains valid people
        train_imgs_file_list, train_objs_info_list, train_mask_list, train_targets = \
            get_pose_data_list(train_im_path, train_ann_path)

        # read coco validating images contains valid people (you can use it for training as well)
        val_imgs_file_list, val_objs_info_list, val_mask_list, val_targets = \
            get_pose_data_list(val_im_path, val_ann_path)

    if 'custom' in config.DATA.train_data:
        ## read your own images contains valid people
        ## 1. if you only have one folder as follow:
        ##   data/your_data
        ##           /images
        ##               0001.jpeg
        ##               0002.jpeg
        ##           /coco.json
        # your_imgs_file_list, your_objs_info_list, your_mask_list, your_targets = \
        #     get_pose_data_list(config.DATA.your_images_path, config.DATA.your_annos_path)
        ## 2. if you have a folder with many folders: (which is common in industry)
        folder_list = tl.files.load_folder_list(path='data/your_data')
        your_imgs_file_list, your_objs_info_list, your_mask_list = [], [], []
        for folder in folder_list:
            _imgs_file_list, _objs_info_list, _mask_list, _targets = \
                get_pose_data_list(os.path.join(folder, 'images'), os.path.join(folder, 'coco.json'))
            print(len(_imgs_file_list))
            your_imgs_file_list.extend(_imgs_file_list)
            your_objs_info_list.extend(_objs_info_list)
            your_mask_list.extend(_mask_list)
        print("number of own images found:", len(your_imgs_file_list))

    # choose dataset for training
    if config.DATA.train_data == 'coco':
        # 1. only coco training set
        imgs_file_list = train_imgs_file_list
        train_targets = list(zip(train_objs_info_list, train_mask_list))
    elif config.DATA.train_data == 'custom':
        # 2. only your own data
        imgs_file_list = your_imgs_file_list
        train_targets = list(zip(your_objs_info_list, your_mask_list))
    elif config.DATA.train_data == 'coco_and_custom':
        # 3. your own data and coco training set
        imgs_file_list = train_imgs_file_list + your_imgs_file_list
        train_targets = list(zip(train_objs_info_list + your_objs_info_list, train_mask_list + your_mask_list))
    else:
        raise Exception('please choose a valid config.DATA.train_data setting.')


    img_path = imgs_file_list[0]
    label_target = train_targets[0]

    image = tf.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    ############################## only for show ####################################
    # image_show = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    # plt.figure("Image")  # 图像窗口名称
    # plt.imshow(image_show)
    # plt.axis('off')  # 关掉坐标轴为 off
    # # plt.title('image')  # 图像题目
    # plt.show()
    #################################################################################
    image_tf = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image_tf.numpy()

    ground_truth = label_target

    annos = ground_truth[0]
    mask = ground_truth[1]
    h_mask, w_mask, _ = np.shape(image)
    # mask
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)

    for seg in mask:
        bin_mask = maskUtils.decode(seg)
        bin_mask = np.logical_not(bin_mask)
        mask_miss = np.bitwise_and(mask_miss, bin_mask)

    ###################################################################################
    vis_annos(image, annos, name="drawAnnos")
    ###################################################################################

    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    M_combined = M_rotate.dot(M_zoom)

    h, w, _ = image.shape
    h_int = tf.cast(h,dtype=tf.int32)
    w_int = tf.cast(w,dtype=tf.int32)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=h_int, y=w_int)
    image = tl.prepro.affine_transform_cv2(image, transform_matrix)
    mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, borderMode=cv2.BORDER_REPLICATE)
    annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)
    ############################## only for show ####################################
    vis_annos(image, annos, name="drawAnnos_afterAffine")
    #################################################################################


    image, annos, mask_miss = tl.prepro.keypoint_random_flip(image, annos, mask_miss, prob=0.5)
    ############################## only for show ####################################
    vis_annos(image, annos, name="drawAnnos_afterFlip")
    #################################################################################

    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win))  # hao add
    ############################## only for show ####################################
    vis_annos(image, annos, name="drawAnnos_afterCrop_resize")
    #################################################################################

    # generate result maps including keypoints heatmap, pafs and mask
    h, w, _ = np.shape(image)
    height, width, _ = np.shape(image)
    heatmap = get_heatmap(annos, height, width)
    vectormap = get_vectormap(annos, height, width)
    resultmap = np.concatenate((heatmap, vectormap), axis=2)

    image = np.array(image, dtype=np.float32)

    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)

    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = cv2.resize(mask_miss, (hout, wout), interpolation=cv2.INTER_AREA)
    mask_miss = np.array(mask_miss, dtype=np.float32)

    ############################## only for show ####################################
    draw_results_one(image,heatmap,pafs_ground=vectormap,masks=mask_miss,name="draw")
    #################################################################################

    image = tf.reshape(image, [hin, win, 3])
    resultmap = tf.reshape(resultmap, [hout, wout, n_pos * 3])
    mask = tf.reshape(mask_miss, [hout, wout, 1])

    image = tf.image.random_brightness(image, max_delta=45. / 255.)  # 64./255. 32./255.)  caffe -30~50
    ############################## only for show ####################################
    vis_annos(image, annos, name="drawAnnos_after_brightness")
    #################################################################################

    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # lower=0.2, upper=1.8)  caffe 0.3~1.5
    ############################## only for show ####################################
    vis_annos(image, annos, name="drawAnnos_after_contrast")
    #################################################################################
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    image = tf.expand_dims(image,0)
    resultmap = tf.expand_dims(resultmap,0)
    mask = tf.expand_dims(mask,0)


    single_train(image, resultmap, mask)



    # if config.TRAIN.train_mode == 'single':
    #     single_train(dataset)
    # elif config.TRAIN.train_mode == 'parallel':
    #     single_train(dataset)
    # else:
    #     raise Exception('Unknown training mode')
