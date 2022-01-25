# -*- coding: utf-8 -*-
import cv2
import numpy as np
from  time import perf_counter as dida

di = dida()

cap = cv2.VideoCapture('./1_02_00.mp4')
p_thre = 20  # p_thre表示像素阈值
i = 0
saved = 1


def frame_diff(image_1, image_2, p_thre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    #  dst  =   cv.GaussianBlur(    src, ksize, sigmaX)
    # gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 2), 0)  #removing Gaussian noise

    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    # gray_image_2 = cv2.GaussianBlur(gray_image_2, (3, 3), 0)

    d_frame = cv2.absdiff(gray_image_1, gray_image_2)
    # _, d_frame = cv2.threshold(d_frame, p_thre, 255, cv2.THRESH_BINARY)
    _, d_frame = cv2.threshold(d_frame, p_thre, 255, cv2.THRESH_TOZERO)
    # d_frame /= 255.0
    # d_frame = cv2.resize(d_frame,cv2.bicubic)
    # print(f'{d_frame.max()=}')
    return d_frame



# 从第几帧开始
#  frm_id = 30*60*5
frm_id = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frm_id)
_, frame_1 = cap.read()  #  作为所截取片段的 第一帧
frame_2 = frame_1


while True:
    frame_0 = frame_1
    frame_1 = frame_2
    _, frame_2 = cap.read()

    if _ is False:
        print('end of vdo')
        break
    else:
        print(f'{frm_id=}')
        frm_id += 1


        f2__f1 = frame_diff(frame_2, frame_1, p_thre)
        f1__f0 = frame_diff(frame_1, frame_0, p_thre)
        cv2.imwrite(f'./bg/{frm_id}.png',f2__f1)


cap.release()

da = dida()
print(f'  {1000*(da - di)/frm_id=} ms/frame ')

