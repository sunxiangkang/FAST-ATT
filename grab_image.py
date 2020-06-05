import cv2
import numpy as np
from pypylon import pylon


def search_get_device():
    tl_factory = pylon.TlFactory.GetInstance()
    camera_list = []
    for dev_info in tl_factory.EnumerateDevices():

        if dev_info.GetDeviceClass() == 'BaslerGigE':
            camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
            camera_list.append(camera)
    if len(camera_list) > 0:

        return camera_list
    else:
        raise EnvironmentError("no GigE device found")


def grab_img_realtime(camera):
    converter = pylon.ImageFormatConverter()
    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            # cv2.namedWindow('title', cv2.WINDOW_NORMAL)
            # cv2.imshow('title', img)
            # k = cv2.waitKey(1)
            # if k == 27:
            #     break
        grabResult.Release()