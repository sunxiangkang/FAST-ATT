import os
import cv2
import json
import numpy as np

import crop_finger
import generate_masks


# 将网络生成的mask还原回原始的图片大小
# input_img是网络生成的mask，还原到输入到网络图片的大小的mask
# ori_size是原始图片大小
def p_resize_reverse(input_img, ori_size):
    input_size = input_img.shape
    # 要把原始图片resize到输入大小的scale
    scales = np.array(input_size)[:2] / np.array(ori_size)[:2]
    scaled_shape = np.min(scales) * np.array(ori_size)[:2]
    # 在哪个轴上进行padding
    pad_ind = np.argmax(scales)
    # padding量
    pad_f = (input_size[pad_ind] - scaled_shape[pad_ind]) // 2
    pad_s = input_size[pad_ind] - scaled_shape[pad_ind] - pad_f

    # 删除padding量
    input_img = np.delete(input_img, np.arange(input_size[pad_ind]-1, input_size[pad_ind]-1-pad_s, -1), pad_ind)
    input_img = np.delete(input_img, np.arange(pad_f), pad_ind)

    input_img = cv2.resize(input_img, (int(ori_size[1]+0.0005), int(ori_size[0]+0.0005)), cv2.INTER_CUBIC)

    return input_img


# 缩放成固定大小，同时要保留图片的原始长宽比
def p_resize(input_img, input_size, input_mask=None):
    scales = np.array(input_size)[:2] / np.array(input_img.shape)[:2]
    scaled_shape = np.min(scales) * np.array(input_img.shape)[:2]
    input_img = cv2.resize(input_img, (int(scaled_shape[1] + 0.05), int(scaled_shape[0] + 0.05)), cv2.INTER_CUBIC)
    if input_mask is not None:
        input_mask = cv2.resize(input_mask, (int(scaled_shape[1] + 0.05), int(scaled_shape[0] + 0.05)), cv2.INTER_CUBIC)
    # 在哪个轴上进行padding
    pad_ind = np.argmax(scales)
    # padding量
    pad_f = (input_size[pad_ind] - input_img.shape[pad_ind]) // 2
    pad_s = input_size[pad_ind] - input_img.shape[pad_ind] - pad_f

    axis_pad = []
    n_dim = len(input_img.shape)
    for ind in range(n_dim):
        if ind == pad_ind:
            axis_pad.append((pad_f, pad_s))
        else:
            axis_pad.append((0, 0))
    input_img = np.pad(input_img, tuple(axis_pad), 'constant', constant_values=((128,) * (n_dim - 1),) * n_dim)
    if input_mask is not None:
        input_mask = np.pad(input_mask, tuple(axis_pad[:2]), 'constant', constant_values=(0, 0))

    return input_img, input_mask


# 随机旋转
def p_rotate(input_img, max_angle, input_mask=None):
    (h, w) = input_img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    angle = (np.random.random() - 0.5) * 2 * max_angle
    # 获得绕图片中心旋转angle的旋转矩阵
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 旋转后的图片大小
    # 计算方式很简单，计算矩形四个顶点绕(0,0)点旋转angle角度的坐标，相减就可以得出长宽
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 旋转矩阵加上平移量，得到仿射变换矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # 进行仿射变换，用（128，128， 128）去padding
    input_img = cv2.warpAffine(
        input_img, M, (nW, nH), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128)
    )
    if input_mask is not None:
        input_mask = cv2.warpAffine(
            input_mask, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, )
        )

    return input_img, input_mask


# 随机裁剪
def p_crop(input_img, max_offset_ratio, input_mask=None):
    input_size = np.array(input_img.shape[:2])

    dice_row = np.random.random()
    dice_col = np.random.random()
    row_offset = int((dice_row - 0.5) * 2 * (input_size[0] * max_offset_ratio))
    col_offset = int((dice_col - 0.5) * 2 * (input_size[1] * max_offset_ratio))
    center = np.array((input_size // 2).astype(np.int) + np.array([row_offset, col_offset]).astype(np.int))
    left_upper = np.max(np.vstack([center-input_size//2, [0, 0]]), axis=0)
    right_bot = np.min(np.vstack([center+input_size-input_size//2, input_size]), axis=0)

    input_img = input_img[left_upper[0]: right_bot[0], left_upper[1]: right_bot[1]]
    if input_mask is not None:
        input_mask = input_mask[left_upper[0]: right_bot[0], left_upper[1]: right_bot[1]]

    return input_img, input_mask


# 随机翻转
def p_flip(input_img, input_mask=None):
    dice = np.random.random()
    do_flip = dice >= 2 / 3
    if do_flip:
        filp_code = int(np.random.random() > 0.5)
        input_img = cv2.flip(src=input_img, flipCode=filp_code)
        if input_mask is not None:
            input_mask = cv2.flip(src=input_mask, flipCode=filp_code)

    return input_img, input_mask


# 随机缩放
def p_scale(input_img, random_scale, input_mask=None):
    scaler = np.random.uniform(random_scale[0], random_scale[1])
    input_img = cv2.resize(input_img, None, fx=scaler, fy=scaler, interpolation=cv2.INTER_CUBIC)
    if input_mask is not None:
        input_mask = cv2.resize(input_mask, None, fx=scaler, fy=scaler, interpolation=cv2.INTER_CUBIC)

    return input_img, input_mask


# 随机光照强度
def p_brightness(input_image, max_delta):
    delta = np.random.uniform(-max_delta, max_delta)
    input_image = input_image.astype(np.float)+delta
    input_image = np.clip(input_image, 0, 255)

    return input_image.astype(np.uint8)


def preprocess_one_img(input_param):
    img_path = input_param[0]
    ind_label = input_param[1]
    pos_datasets_path = input_param[2]
    max_delta = input_param[3]
    max_offset_ratio = input_param[4]
    max_angle = input_param[5]
    input_size = input_param[6]

# def preprocess_one_img(img_path, ind_label, pos_datasets_path, max_delta, max_offset_ratio, max_angle, input_size):

    temp_img_name = os.path.basename(img_path)
    temp_img = cv2.imread(img_path)
    if ind_label:
        json_path = os.path.join(pos_datasets_path, 'json', temp_img_name.split('.')[0] + '.json')
        with open(json_path, 'r') as fr:
            labeled_data = json.load(fr)
        temp_mask = generate_masks.gen_mask(labeled_data)
    else:
        temp_mask = None

    # temp_contour = crop_finger.crop_finger(temp_img)
    temp_img = p_brightness(temp_img, max_delta)
    # cv2.fillPoly(temp_img, temp_contour, (128, 128, 128))
    # print('imgB.shape:', temp_img.shape)
    # cv2.imshow('test1', temp_img)
    # cv2.waitKey()
    temp_img, temp_mask = p_flip(temp_img, temp_mask)
    # print('imgF.shape:', temp_img.shape)
    # cv2.imshow('test2_1', temp_img)
    # if temp_mask is not None:
    #     cv2.imshow('test2_2', temp_mask)
    # cv2.waitKey()
    temp_img, temp_mask = p_crop(temp_img, max_offset_ratio, temp_mask)
    # print('imgC.shape:', temp_img.shape)
    # cv2.imshow('test3_1', temp_img)
    # if temp_mask is not None:
    #     cv2.imshow('test3_2', temp_mask)
    # cv2.waitKey()
    temp_img, temp_mask = p_rotate(temp_img, max_angle, temp_mask)
    # print('imgR.shape:', temp_img.shape)
    # cv2.imshow('test4_1', temp_img)
    # if temp_mask is not None:
    #     cv2.imshow('test4_2', temp_mask)
    # cv2.waitKey()
    temp_img, temp_mask = p_resize(temp_img, input_size, temp_mask)
    # print('imgRS.shape:', temp_img.shape)
    # cv2.imshow('test5_1', temp_img)
    # if temp_mask is not None:
    #     cv2.imshow('test5_2', temp_mask)
    # cv2.waitKey()

    if temp_mask is None:
        temp_mask = np.zeros_like(temp_img[:, :, 0])
    else:
        # 数据增强后是不是把油污切掉了
        if np.sum(temp_mask) < 1:
            ind_label = 0
    # cv2.imshow('nima', temp_mask)
    # cv2.imshow('nima2', temp_img)
    # cv2.waitKey()
    temp_img = temp_img / 128. - 1.
    temp_mask = (temp_mask / 255.)[:, :, None]

    return temp_img, temp_mask, ind_label
