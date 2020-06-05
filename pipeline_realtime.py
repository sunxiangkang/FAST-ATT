import os
import cv2
import sys
import time
import logging
import numpy as np
from pypylon import pylon
from datetime import datetime

# 定义log
logger = logging.getLogger('pipeline_realtime')
logger.setLevel(level=logging.INFO)
if not os.path.exists(r'./log'):
    os.makedirs(r'./log')
handler = logging.FileHandler('./log/inference_realtime.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler_con = logging.StreamHandler()
handler_con.setLevel(logging.INFO)
handler_con.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(handler_con)


def main(cam_ind, save_path=None):
    # 安装相机
    tl_factory = pylon.TlFactory.GetInstance()
    if len(tl_factory.EnumerateDevices()) == 0:
        logger.critical('Find no camera.')
        sys.exit(1)
    else:
        cam_dict = {}
        for dev_info in tl_factory.EnumerateDevices():
            cam_dict.update({'cam_name': dev_info.GetModelName(), 'cam_ip': dev_info.GetIpAddress()})
            # if dev_info.GetModelName() == line_and_side:
            #     camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
        logger.info('cameras_info: {}'.format(cam_dict))
    camera = pylon.InstantCamera(tl_factory.CreateDevice(tl_factory.EnumerateDevices()[cam_ind]))

    converter = pylon.ImageFormatConverter()
    # converting to opencv bgr format
    # 转换到opencv的bgr形式
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    # 打开相机采图
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            if save_path:
                filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.bmp"
                cv2.imwrite(os.path.join(save_path, filename), img)
            cv2.namedWindow('title', cv2.WINDOW_NORMAL)
            cv2.imshow('title', img)
            k = cv2.waitKey(1)
            if k == 27:
                break
        grabResult.Release()


if __name__ == '__main__':
    main(0)
