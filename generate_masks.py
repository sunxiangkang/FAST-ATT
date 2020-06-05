import os
import cv2
import json
import numpy as np


def gen_mask(labeled_data):
    img_height = labeled_data['imageHeight']
    img_width = labeled_data['imageWidth']
    mask = np.zeros([img_height, img_width])
    mask_sz = np.ones([img_height, img_width])

    if 'shapes' in labeled_data.keys():
        shapes = labeled_data['shapes']
        points = [
            np.round(
                np.array(item['points']),
                decimals=0
            ).astype(np.int) for item in shapes if item['label'] == 'yw'
        ]

        points_st = [
            np.round(
                np.array(item['points']),
                decimals=0
            ).astype(np.int) for item in shapes if item['label'] == 'sz'
        ]
        cv2.fillPoly(mask, points, (255,))
        cv2.fillPoly(mask_sz, points_st, (0,))

        mask = mask * mask_sz

        return mask
    else:
        print('Error while generating mask.')

        return mask_sz


def gen_masks(datasets_path):
    jsons_path = os.path.join(datasets_path, 'json')
    saved_masks_path = os.path.join(datasets_path, 'masks')
    json_names = os.listdir(jsons_path)
    for json_name in json_names:
        # json_name = '18092521402919.json'
        with open(os.path.join(jsons_path, json_name), 'r') as fr:
            labeled_data = json.load(fr)

        img_name = labeled_data['imagePath']
        img_height = labeled_data['imageHeight']
        img_width = labeled_data['imageWidth']
        mask = np.zeros([img_height, img_width])
        mask_sz = np.ones([img_height, img_width])

        if 'shapes' in labeled_data.keys():
            shapes = labeled_data['shapes']
            points = [
                np.round(
                    np.array(item['points']),
                    decimals=0
                ).astype(np.int) for item in shapes if item['label'] == 'yw'
            ]

            points_st = [
                np.round(
                    np.array(item['points']),
                    decimals=0
                ).astype(np.int) for item in shapes if item['label'] == 'sz'
            ]
            cv2.fillPoly(mask, points, (255, ))
            cv2.fillPoly(mask_sz, points_st, (0,))

            mask = mask * mask_sz
            cv2.imwrite(os.path.join(saved_masks_path, img_name), mask)
        else:
            pass


if __name__ == '__main__':
    datasets_path = r'F:\images\data_n\stage1\bad'
    gen_masks(datasets_path)
