import os
import cv2
import sys
import time
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

import utils
import generate_masks


class DataReader:
    def __init__(
            self, datasets_path, ratio_sample=2, ratio_l=1.3, ratio_u=1.8, max_delta=20,
            max_offset_ratio=0.1, max_angle=20, scale=(0.8, 1.2), input_size=(480, 640)
    ):
        self.pos_datasets_path = os.path.join(datasets_path, 'pos')
        self.neg_datasets_path = os.path.join(datasets_path, 'neg')
        self.batch_ind = 0
        self.batch_ind_val = 0
        self.epoch_ind = 0
        self.aug_params = {
            'random_scale': scale,
            'input_size': input_size,
            'max_offset_ratio': max_offset_ratio,
            'max_angle': max_angle,
            'max_delta': max_delta
        }
        self.imgs_list_train, self.imgs_list_val, self.labels_list_train, self.labels_list_val = self._get_imgsinfo(
            ratio_sample, ratio_l, ratio_u
        )

    def _get_imgsinfo(self, ratio_l, ratio_u, ratio_sample):
        pos_imgs_path = os.path.join(self.pos_datasets_path, 'pic')
        neg_imgs_path = os.path.join(self.neg_datasets_path, 'pic')
        pos_imgs_list = os.listdir(pos_imgs_path)
        neg_imgs_list = os.listdir(neg_imgs_path)
        for i, item in enumerate(pos_imgs_list):
            pos_imgs_list[i] = os.path.join(pos_imgs_path, item)
        for i, item in enumerate(neg_imgs_list):
            neg_imgs_list[i] = os.path.join(neg_imgs_path, item)

        # 如果正样本比例/负杨版本比例大于阈值，则进行下采样
        if max(len(pos_imgs_list), len(neg_imgs_list))/min(len(pos_imgs_list), len(neg_imgs_list)) > ratio_sample:
            ratio = np.random.uniform(ratio_l, ratio_u)
            if len(pos_imgs_list) > len(neg_imgs_list):
                pos_imgs_list = np.random.choice(pos_imgs_list, int(len(neg_imgs_list) * ratio), replace=False)
            else:
                neg_imgs_list = np.random.choice(neg_imgs_list, int(len(pos_imgs_list) * ratio), replace=False)

        imgs_list = pos_imgs_list.copy()
        imgs_list.extend(neg_imgs_list)
        labels_list = np.concatenate([
            np.ones_like(pos_imgs_list, dtype=np.int32),
            np.zeros_like(neg_imgs_list, dtype=np.int32)
        ])

        state_s = np.random.get_state()
        np.random.shuffle(imgs_list)
        np.random.set_state(state_s)
        np.random.shuffle(labels_list)
        imgs_list_train, imgs_list_val, labels_list_train, labels_list_val \
            = train_test_split(imgs_list, labels_list, test_size=0.1)
        with open(r'./datasets/train.txt', 'w') as fw:
            for item in zip(imgs_list_train, labels_list_train):
                fw.write('{} {} \n'.format(item[0], item[1]))
        with open(r'./datasets/val.txt', 'w') as fw:
            for item in zip(imgs_list_val, labels_list_val):
                fw.write('{} {} \n'.format(item[0], item[1]))

        return imgs_list_train, imgs_list_val, labels_list_train, labels_list_val

    def next_batch(self, batch_size, pool):
        if (self.batch_ind+batch_size) > len(self.imgs_list_train):
            self.epoch_ind = self.epoch_ind+1
            self.batch_ind = 0
            state = np.random.get_state()
            np.random.shuffle(self.imgs_list_train)
            np.random.set_state(state)
            np.random.shuffle(self.labels_list_train)

        batch_imgs_path = self.imgs_list_train[self.batch_ind:self.batch_ind + batch_size]
        batch_labels = self.labels_list_train[self.batch_ind:self.batch_ind + batch_size]
        pos_datasets_paths = [self.pos_datasets_path, ]*batch_size
        max_delta = [self.aug_params['max_delta'], ]*batch_size
        max_offset_ratio = [self.aug_params['max_offset_ratio'], ]*batch_size
        max_angle = [self.aug_params['max_angle'], ]*batch_size
        input_size = [self.aug_params['input_size'], ]*batch_size

        input_params = zip(
            batch_imgs_path, batch_labels, pos_datasets_paths, max_delta, max_offset_ratio, max_angle, input_size
        )

        batch_imgs = []
        batch_masks = []
        batch_labels = []
        # st = time.time()
        res = pool.map(utils.preprocess_one_img, input_params)
        # print('batch_time:', time.time()-st)
        for item in res:
            temp_img = item[0]
            temp_mask = item[1]
            temp_label = item[2]

            batch_imgs.append(temp_img)
            batch_masks.append(temp_mask)
            batch_labels.append(temp_label)
        self.batch_ind = self.batch_ind+batch_size
        
        return batch_imgs, batch_masks, batch_labels

        # batch_imgs = []
        # batch_masks = []
        # for i, img_path in enumerate(batch_imgs_path):
        #     temp_img_name = os.path.basename(img_path)
        #     temp_img = cv2.imread(img_path)
        #     ind_label = batch_labels[i]
        #     if ind_label:
        #         json_path = os.path.join(self.pos_datasets_path, 'json', temp_img_name.split('.')[0] + '.json')
        #         with open(json_path, 'r') as fr:
        #             labeled_data = json.load(fr)
        #         temp_mask = generate_masks.gen_mask(labeled_data)
        #     else:
        #         temp_mask = None
        #
        #     st0 = time.time()
        #     temp_contour = crop_finger.crop_finger(temp_img)
        #     print('time_crop_finger:', time.time()-st0)
        #     st1 = time.time()
        #     temp_img = utils.p_brightness(temp_img, self.aug_params['max_delta'])
        #     # print('time_birghness:', time.time()-st1)
        #     cv2.fillPoly(temp_img, temp_contour, (128, 128, 128))
        #     # print('imgB.shape:', temp_img.shape)
        #     # cv2.imshow('test1', temp_img)
        #     # cv2.waitKey()
        #     st2 = time.time()
        #     temp_img, temp_mask = utils.p_flip(temp_img, temp_mask)
        #     # print('time_filp:', time.time()-st2)
        #     # print('imgF.shape:', temp_img.shape)
        #     # cv2.imshow('test2_1', temp_img)
        #     # if temp_mask is not None:
        #     #     cv2.imshow('test2_2', temp_mask)
        #     # cv2.waitKey()
        #     st3 = time.time()
        #     temp_img, temp_mask = utils.p_crop(temp_img, self.aug_params['max_offset_ratio'], temp_mask)
        #     # print('time_crop:', time.time()-st3)
        #     # print('imgC.shape:', temp_img.shape)
        #     # cv2.imshow('test3_1', temp_img)
        #     # if temp_mask is not None:
        #     #     cv2.imshow('test3_2', temp_mask)
        #     # cv2.waitKey()
        #     st4 = time.time()
        #     temp_img, temp_mask = utils.p_rotate(temp_img, self.aug_params['max_angle'], temp_mask)
        #     # print('time_rotate:', time.time()-st4)
        #     # print('imgR.shape:', temp_img.shape)
        #     # cv2.imshow('test4_1', temp_img)
        #     # if temp_mask is not None:
        #     #     cv2.imshow('test4_2', temp_mask)
        #     # cv2.waitKey()
        #     st5 = time.time()
        #     temp_img, temp_mask = utils.p_resize(temp_img, self.aug_params['input_size'], temp_mask)
        #     # print('time_resize:', time.time()-st5)
        #     # print('imgRS.shape:', temp_img.shape)
        #     # cv2.imshow('test5_1', temp_img)
        #     # if temp_mask is not None:
        #     #     cv2.imshow('test5_2', temp_mask)
        #     # cv2.waitKey()
        #
        #     if temp_mask is None:
        #         temp_mask = np.zeros_like(temp_img[:, :, 0])
        #     else:
        #         # 数据增强后是不是把油污切掉了
        #         if np.sum(temp_mask) < 1:
        #             batch_labels[i] = 0
        #     print('---------------time_pre_img:', time.time()-st0)
        #
        #     temp_img = temp_img / 128. - 1.
        #     temp_mask = temp_mask / 255.
        #
        #     batch_imgs.append(temp_img)
        #     batch_masks.append(temp_mask)

        # batch_imgs = np.array(batch_imgs, dtype=np.float32)
        # batch_labels = np.array(batch_labels, dtype=np.int32)
        #
        # return batch_imgs, batch_masks, batch_labels

    def get_val_imgs(self):
        val_imgs = []
        val_masks = []
        val_labels = self.labels_list_val
        for i, img_path in enumerate(self.imgs_list_val):
            ind_label = self.labels_list_val[i]
            temp_img_name = os.path.basename(img_path)

            temp_img = cv2.imread(img_path)
            if ind_label:
                json_path = os.path.join(self.pos_datasets_path, 'json', temp_img_name.split('.')[0] + '.json')
                with open(json_path, 'r') as fr:
                    labeled_data = json.load(fr)
                temp_mask = generate_masks.gen_mask(labeled_data)
            else:
                temp_mask = None
            temp_img, temp_mask = utils.p_resize(temp_img, self.aug_params['input_size'], temp_mask)
            if temp_mask is None:
                temp_mask = np.zeros_like(temp_img[:, :, 0])

            temp_img = temp_img / 128. - 1.
            temp_mask = (temp_mask / 255.)[:, :, None]
            val_imgs.append(temp_img)
            val_masks.append(temp_mask)

        return val_imgs, val_masks, val_labels


if __name__ == '__main__':
    reader = DataReader(r'./datasets', )
    # for i in range(10):
    #     val_imgs, val_labels = reader.get_val_imgs()
    #     print(val_labels)
    #   reader.next_batch(32)
    batch_size = 16
    st = time.time()
    batch_images, batch_masks, batch_labels = reader.next_batch(batch_size)
    print('batch_time:', time.time()-st)
    for i in range(batch_size):
        print('label:', batch_labels[i])
        cv2.imshow('test1', batch_images[i])
        print(np.max(batch_images[i]))
        print(np.min(batch_images[i]))
        cv2.imshow('test2', batch_masks[i])
        cv2.waitKey()
