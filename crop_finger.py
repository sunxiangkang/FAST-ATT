import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler(feature_range=(0, 255))


def crop_finger(img, base=1.5, scaler=scaler, kernel_size=7, thres=70, min_area=2500, width_ratio=2/3):
    img_c = img.copy()
    # 转为灰度图像进行处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    mask = np.zeros_like(gray_img).astype(np.uint8)
    # 指数增强
    index_img = np.power(base, gray_img)
    index_img[np.where(index_img >= 1e+38)] = 1e+38
    # 缩放到0-255
    scaled_img = scaler.fit_transform(index_img).astype(np.uint8)
    # 二值化做选择
    thres_, scaled_img_t = cv2.threshold(scaled_img, thres, 255, cv2.THRESH_BINARY)
    # 形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img_open = cv2.morphologyEx(scaled_img_t, cv2.MORPH_OPEN, kernel)
    # 查找轮廓（有手指，还有其他）
    temp_img, contours, hierarchy = cv2.findContours(
        img_open, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    # 选择轮廓，只选择手指轮廓
    selected_contours = []
    for i, contour in enumerate(contours):
        # 计算轮廓的各阶矩,字典形式
        M = cv2.moments(contours[i])
        center_x = int(M["m10"] / M["m00"])
        area = cv2.contourArea(contours[i])
        if area > min_area and center_x > gray_img.shape[1]*width_ratio:
            contour = cv2.approxPolyDP(contour, 2, True)
            selected_contours.append(np.squeeze(np.array(contour)))

    # 对轮廓进行填充
    cv2.fillPoly(mask, selected_contours, (255, ))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size+15, kernel_size+15))
    # 对填充后的轮廓进行膨胀，放大和平滑轮廓
    mask = cv2.dilate(mask, kernel_2)
    affine_arr = np.float32([[1, 0, 5], [0, 1, 0]])
    for i in range(50):
        mask_ = cv2.warpAffine(
            mask, affine_arr, (mask.shape[1], mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, )
        )
        mask = (cv2.bitwise_or(mask/255, mask_/255)*255).astype(np.uint8)
    # 检测轮廓，这时就只剩下手指
    temp_img_, contours_, hierarchy_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res_contour = []
    for i, contour in enumerate(contours_):
        contour = cv2.approxPolyDP(contour, 2, True)
        res_contour.append(np.squeeze(np.array(contour)))

    # 手指部分填充
    # cv2.fillPoly(img_c, res_contour, (128, 128, 128))
    # cv2.imshow('test', gray_img_c)
    # cv2.waitKey()

    return res_contour
    # return img_c


def draw_contour(img, contour):
    cv2.fillPoly(img, contour, (128, 128, 128))

    return img


if __name__ == '__main__':
    import os
    import time
    import pickle
    imgs_path = r'F:\images\data_n\stage1\neg\pic'
    dst_path = r'F:\images\data_n\stage1\neg\fig_contour'
    dst_path_c = r'F:\images\data_n\stage1\neg\pic_crop'
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        img = cv2.imread(os.path.join(imgs_path, img_name))
        with open(os.path.join(dst_path, img_name.split('.')[0]+'.pkl'), 'rb') as fr:
            contour = pickle.load(fr)
        img_con = draw_contour(img, contour)
        cv2.imwrite(os.path.join(dst_path_c, img_name), img_con)
        # cv2.imshow('test2', img_con)
        # cv2.waitKey()
        # # st = time.time()
        # dst = crop_finger(img)
        # # print('time:', time.time()-st)
        # # cv2.imwrite(os.path.join(dst_path, img_name), dst)
        # with open(os.path.join(dst_path, img_name.split('.')[0]+'.pkl'), 'wb') as fw:
        #     pickle.dump(dst, fw)
