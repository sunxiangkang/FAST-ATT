import os
import cv2
import json
import numpy as np


def adjust_rect(datasets_path, datasets_path_n, margin=10):
    json_files = os.listdir(os.path.join(datasets_path, 'json'))
    for file in json_files:
        file_path = os.path.join(datasets_path, 'json', file)
        with open(file_path, 'r') as fr:
            label_data = json.load(fr)
        img_name = label_data['imagePath']
        img_ori = cv2.imread(os.path.join(datasets_path, 'pic', img_name))
        img_equ_hist = cv2.imread(os.path.join(datasets_path, 'pic_equ_histogram', img_name))

        shapes = label_data['shapes']
        for item in shapes:
            if item['label'] == 'st':
                points = np.around(np.array(item['points']), decimals=0).astype(np.int)
                left_upper = np.min(points, axis=0) - margin
                sub_img_height = np.max(points, axis=0)[1] - left_upper[1] + margin
                sub_img_ori = img_ori[left_upper[1]:left_upper[1] + sub_img_height, left_upper[0]:]
                sub_img_equ_hist = img_equ_hist[left_upper[1]:left_upper[1] + sub_img_height, left_upper[0]:]
                cv2.imwrite(os.path.join(datasets_path_n, 'pic', img_name), sub_img_ori)
                cv2.imwrite(os.path.join(datasets_path_n, 'pic_equ_histogram', img_name), sub_img_equ_hist)
                break


def adjust_rect_(datasets_path, dst_path, margin=10):
    json_files = os.listdir(os.path.join(datasets_path, 'json'))
    for file in json_files:
        file_path = os.path.join(datasets_path, 'json', file)
        with open(file_path, 'r') as fr:
            label_data = json.load(fr)
        img_name = label_data['imagePath']
        img_ori = cv2.imread(os.path.join(datasets_path, 'pic', img_name))
        # cv2.imshow('test', img_ori)
        # cv2.waitKey()
        # img_equ_hist = cv2.imread(os.path.join(datasets_path, 'pic_equ_histogram', img_name))

        shapes = label_data['shapes']
        for item in shapes:
            if item['label'] == 'st':
                points = np.around(np.array(item['points']), decimals=0).astype(np.int)
                # print(np.min(points, axis=0) - margin)
                # print(np.vstack([[0, 0], np.min(points, axis=0) - margin]))
                # print('--------------------------------------')
                left_upper = np.max(np.vstack([[0, 0], np.min(points, axis=0) - margin]), axis=0)
                sub_img_height = np.max(points, axis=0)[1] - left_upper[1] + margin
                sub_img_ori = img_ori[left_upper[1]:left_upper[1] + sub_img_height, left_upper[0]:]
                # sub_img_equ_hist = img_equ_hist[left_upper[1]:left_upper[1] + sub_img_height, left_upper[0]:]
                cv2.imwrite(os.path.join(dst_path, 'pic', img_name), sub_img_ori)
                # cv2.imwrite(os.path.join(datasets_path_n, 'pic_equ_histogram', img_name), sub_img_equ_hist)
                break


def adjust(datasets_path, datasets_path_n, margin=10):
    json_files = os.listdir(os.path.join(datasets_path, 'json'))
    for file in json_files:
        file_path = os.path.join(datasets_path, 'json', file)
        with open(file_path, 'r') as fr:
            label_data = json.load(fr)

        img_name = label_data['imagePath']
        img_ori = cv2.imread(os.path.join(datasets_path, 'pic', img_name))
        img_equ_hist = cv2.imread(os.path.join(datasets_path, 'pic_equ_histogram', img_name))

        img_height = label_data['imageHeight']
        img_width = label_data['imageWidth']
        shapes = label_data['shapes']

        sub_img_height = -1
        sub_img_width = -1
        left_upper = [-1, -1]
        for item in shapes:
            if item['label'] == 'st':
                points = np.around(np.array(item['points']), decimals=0).astype(np.int)
                left_upper = np.min(points, axis=0) - margin
                sub_img_width = img_width - left_upper[0]
                sub_img_height = np.max(points, axis=0)[1] - left_upper[1]+margin
                sub_img_ori = img_ori[left_upper[1]:left_upper[1]+sub_img_height, left_upper[0]:]
                sub_img_equ_hist = img_equ_hist[left_upper[1]:left_upper[1]+sub_img_height, left_upper[0]:]
                cv2.imwrite(os.path.join(datasets_path_n, 'pic', img_name), sub_img_ori)
                cv2.imwrite(os.path.join(datasets_path_n, 'pic_equ_histogram', img_name), sub_img_equ_hist)
                break

        label_data_ad = label_data.copy()
        if sub_img_height > 0 and sub_img_width > 0:
            label_data_ad['imageHeight'] = int(sub_img_height)
            label_data_ad['imageWidth'] = int(sub_img_width)
        else:
            print('{}------------{}--------------error'.format(img_name, 'st'))

        temp_ind = -1
        for ind, item in enumerate(label_data_ad['shapes']):
            if item['label'] == 'st':
                temp_ind = ind
            else:
                points = np.around(np.array(item['points']), decimals=0).astype(np.int)
                if np.sum(left_upper) > 0:
                    points = points-left_upper
                    points[np.where(points < 0)] = 0
                    item['points'] = points.tolist()
                else:
                    print('{}-----------------{}-------------error'.format(img_name, 'left_upper'))
        label_data_ad['shapes'].pop(temp_ind)
        with open(os.path.join(datasets_path_n, 'json', file), 'w') as fw:
            json.dump(label_data_ad, fw)


if __name__ == '__main__':
    # datasets_path = r'F:\images\data\stage1\good'
    # datasets_path_n = r'F:\images\data_n\stage1\neg'
    # adjust_rect(datasets_path, datasets_path_n)
    datasets_path = r'F:\images\data_val'
    dst_path = r'D:\pythonproj\rb\FAST-ATT\datasets\val'
    adjust_rect_(datasets_path, dst_path)
