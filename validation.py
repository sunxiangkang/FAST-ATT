import os
import cv2
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import utils
import residual_attention_network


class Validation(object):
    def __init__(self, input_size):
        self.input_size = input_size
        self.sess = tf.InteractiveSession()
        self.input_x = tf.placeholder(tf.float32, [None, input_size[0], input_size[1], 3])
        self.att_mask1, self.att_mask2, self.logits = self._load_model()

    def _load_model(self):
        net_work = residual_attention_network.ResidualAttentionNetwork(False, 2)
        att_mask1, att_mask2, logits = net_work.interface(self.input_x)
        att_mask1 = tf.cast(tf.where(
            tf.nn.sigmoid(att_mask1) > 0.5, tf.ones_like(att_mask1), tf.zeros_like(att_mask1)) * 255, tf.uint8
        )
        att_mask2 = tf.cast(tf.where(
            tf.nn.sigmoid(att_mask2) > 0.5, tf.ones_like(att_mask2), tf.zeros_like(att_mask2)) * 255, tf.uint8
        )
        logits = tf.squeeze(logits)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(r'./weights')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)

            return att_mask1, att_mask2, logits
        else:
            print('Wrong while loading trained model.')
            sys.exit(1)

    def inference(self, img_paths_list, labels_list=None, batch_size=16, save_mask=True):
        num_iter = len(img_paths_list)//batch_size+1
        pred_labels = []
        for i in range(num_iter):
            temp_batch_imgs = []
            temp_img_names = []
            temp_img_size = []
            # 取一个batch
            temp_img_paths_list = img_paths_list[i*batch_size: (i+1)*batch_size]
            for ind, temp_img_path in enumerate(temp_img_paths_list):
                # 图片名称
                img_name = os.path.basename(temp_img_path)
                temp_img = cv2.imread(temp_img_path)
                temp_img_size.append(temp_img.shape)
                temp_img, temp_mask = utils.p_resize(temp_img, self.input_size)
                temp_img_names.append(img_name)
                temp_batch_imgs.append(temp_img/128. - 1.)
            temp_att_mask1, temp_att_mask2, temp_logits = self.sess.run(
                [self.att_mask1, self.att_mask2, self.logits],
                feed_dict={self.input_x: temp_batch_imgs}
            )
            # print('-------------', time.time()-st)
            temp_labels = np.argmax(temp_logits, 1)
            pred_labels.extend(temp_labels.tolist())
            # 保存生成的mask
            if save_mask:
                if not os.path.exists(r'./datasets/val/gen_mask_att1'):
                    os.makedirs(r'./datasets/val/gen_mask_att1')
                    os.makedirs(r'./datasets/val/gen_mask_att2')
                for j in range(temp_att_mask1.shape[0]):
                    # 当前mask对应图片的原始大小
                    c_mask_shape = temp_img_size[j]
                    # 第一步，还原回输入网络图片的大小
                    c_mask_att1 = cv2.resize(temp_att_mask1[j], self.input_size[::-1])
                    c_mask_att2 = cv2.resize(temp_att_mask2[j], self.input_size[::-1])
                    # cv2.imshow('ori', ((temp_batch_imgs[j]+1)*128).astype(np.uint8))
                    # cv2.imshow('att_mask1', c_mask_att1)
                    # cv2.imshow('att_mask2', c_mask_att2)

                    # 第二步，还原回原始大小
                    c_mask_att1 = utils.p_resize_reverse(c_mask_att1, c_mask_shape)
                    c_mask_att2 = utils.p_resize_reverse(c_mask_att2, c_mask_shape)

                    mask1_path = os.path.join(r'./datasets/val/gen_mask_att1', temp_img_names[j])
                    mask2_path = os.path.join(r'./datasets/val/gen_mask_att2', temp_img_names[j])
                    cv2.imwrite(mask1_path, c_mask_att1)
                    cv2.imwrite(mask2_path, c_mask_att2)
        res = {'img_path': img_paths_list, 'label_p': pred_labels}
        if labels_list is not None:
            res.update({'label': labels_list})
        pd.DataFrame.from_dict(res).to_excel(r'./inference_val.xlsx', index=None)


if __name__ == '__main__':
    # pos_imgs_path = r'./datasets/pos/pic'
    # neg_imgs_path = r'./datasets/neg/pic'
    # pos_imgs_list = os.listdir(pos_imgs_path)
    # neg_imgs_list = os.listdir(neg_imgs_path)
    # for i, item in enumerate(pos_imgs_list):
    #     pos_imgs_list[i] = os.path.join(pos_imgs_path, item)
    # for i, item in enumerate(neg_imgs_list):
    #     neg_imgs_list[i] = os.path.join(neg_imgs_path, item)

    # imgs_list = pos_imgs_list.copy()
    # imgs_list.extend(neg_imgs_list)
    # labels_list = np.concatenate([
    #     np.ones_like(pos_imgs_list, dtype=np.int32),
    #     np.zeros_like(neg_imgs_list, dtype=np.int32)
    # ])

    val_imgs_dir = r'./datasets/val/pic'
    val_img_list = os.listdir(val_imgs_dir)
    for i, item in enumerate(val_img_list):
        val_img_list[i] = os.path.join(val_imgs_dir, item)
    val = Validation((480, 640))
    val.inference(val_img_list, None, batch_size=16, save_mask=True)
