import torch
import math
import cv2


def cx_cy_in_map(h_map):
    edges, _ = cv2.findContours(h_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(edge) for edge in edges]

    # The ball location is considered as the center of the largest area in the 0-1 heatmap.
    max_area_idx = 0
    max_area = rects[max_area_idx][2] * rects[max_area_idx][3]  # rects: x,y,w,h?
    for _id in range(len(rects)):
        area = rects[_id][2] * rects[_id][3]
        if area > max_area:
            max_area_idx = _id
            max_area = area
    xywh = rects[max_area_idx]
    return int(xywh[0] + xywh[2] / 2),       int(xywh[1] + xywh[3] / 2)

def pred_vs_gt(y_pred, y_true, dist_tolerance):
    # y_pred 和y_true都是batchSize x 通道数 x 宽 x 高 的tensor, 其标量的取值为True 或 False
    batchSize = y_pred.shape[0]
    id_in_Batch = 0
    TP = TN = FP1 = FP2 = FN = 0
    while id_in_Batch < batchSize:
        for channel in range(3):
            # 每张图上，(tp, tn, fp1, fp2, fn)是个one hot向量。
            # fp1表示预测位置与gt离得太远。
            #在算acc, precision, recall时，其地位和常规的误报fp2一样
            # 逐点分类:(gt中正负样本比例为1:512x288左右,极度不平衡)
            # 对于gt中心所在的点，相当于这是一个漏检, 对于预测出的位置，相当于这是一个误报
            # 逐帧分类(这帧有没有羽毛球):
            #  fp1其实应该改为 tp。
            map_pred = y_pred[id_in_Batch][channel]
            map_true = y_true[id_in_Batch][channel]
            if torch.max(y_pred[id_in_Batch][channel]) == 0 and torch.max(map_true) == 0:
                TN += 1
            elif torch.max(map_pred) > 0 and torch.max(map_true) == 0:
                FP2 += 1
            elif torch.max(map_pred) == 0 and torch.max(map_true) > 0:
                FN += 1
            elif torch.max(map_pred) > 0 and torch.max(map_true) > 0:
                h_pred = (map_pred * 255).cpu().numpy().astype('uint8')
                h_true = (map_true * 255).cpu().numpy().astype('uint8')

                cx_pred, cy_pred = cx_cy_in_map(h_pred)
                cx_true, cy_true = cx_cy_in_map(h_true)
                dist = math.sqrt(pow(cx_pred-cx_true, 2) +
                                 pow(cy_pred-cy_true, 2))
                if dist > dist_tolerance:
                    FP1 += 1
                else:
                    TP += 1
        id_in_Batch += 1
    return (TP, TN, FP1, FP2, FN)


def evaluation(TP, TN, FP1, FP2, FN):
    #    FP1叫做FP_f 更好， f for far
    #    FP2: FP_b: b for black , gt(热图)是全黑的
    #    FP = FP1 + FP2

    #    gt中
    #        +样本总量: TP + FN + FP1
    #        -样本总量:TN + FP2

    #    训练:
    #        FN 4055  fp1  1674 fp2 2975 tn 27002  tp 198930
    #        +总量:204659
    #        -总量: 29977
    #        正样本/负样本:6.83

    #    测试:
    #        FN 5040  fp1  1926 fp2 1521 tn 4982   tp 24322
    #        +总量: 31288
    #        -总量:  6503
    #        正样本/负样本:4.81

    #    故意往自制数据集放：: 没有球的画面，让负样本:正样本=100:1 ？

    #    单帧内（不过 是否用mcc代替F1或者acc，跟这个没关系吧)
    #        正样本/负样本:3x3 : 288x512 = 16384 = 1:16k

    FP = FP1 + FP2
    try:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except Exception :
        accuracy = 0
    try:
        precision = TP / (TP + FP)
    except Exception :
        precision = 0
    try:
        recall = TP / (TP + FN)
    except Exception :
        recall = 0
    try:
        #todo : 开根号提速
        mcc = (TP * TN - FP * FN)/(math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    except Exception :
        mcc = 0
    return (mcc, accuracy, precision, recall)
