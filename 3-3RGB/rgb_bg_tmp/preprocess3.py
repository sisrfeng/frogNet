# 没文件调用本py文件
# 本文件作用: 1. outputfile.write("%s,%s,%s\n"%(train_x[i][0], train_x[i][1], train_x[i][2]))
            # 2. 根据gt, 生成data/match某/heatmap


import numpy as np
from glob import glob
import pandas as pd
import os
import cv2

HEIGHT=288
WIDTH=512
mag = 1  #  对图片的缩放尺度
sigma = 2.5

def genHeatMap(w, h, cx, cy, r, mag):
    if cx < 0 or cy < 0:
        return np.zeros((h, w))
    x, y = np.meshgrid(np.linspace(1, w, w), np.linspace(1, h, h))
    # 距离的平方
    heatmap = ((y - (cy + 1))**2) + ((x - (cx + 1))**2)
    heatmap[heatmap <= r**2] = 1
    heatmap[heatmap > r**2] = 0
    return heatmap * mag

#  vdo_path ='/data/wf/badminton_tracknet_v2/'
vdo_path ='./'
#  game_list = ['match'+str(i) for i in range(1,27) ]
game_list = ['match'+str(i) for i in range(1,2) ]
#  game_list = ['test_match'+str(i) for i in range(1,4) ]

#  ratio_W_H =cv2.imread( os.path.join(vdo_path + 'match1/frame/1_01_00/1.png')).shape[0] / HEIGHT
ratio_W_H =cv2.imread( os.path.join(vdo_path, '1_02_01/1.png') ).shape[0] / HEIGHT


train_x = []
train_y = []
for game in game_list:
    #  all_path = glob(  os.path.join(vdo_path, game, 'frame', '*')  )
    #todo: 这行只是临时的
    all_path = glob(  os.path.join(vdo_path, 'frame', '*')  )
    train_path = all_path[:int(len(all_path)*1)]
    for i in range(len(train_path)):
        train_path[i] = train_path[i].split('/')[-1]
    for p in train_path:
        print(p)
        pp = (os.path.join(vdo_path, game,'heatmap',p))
        if not os.path.exists(pp):
            os.makedirs(pp)
        labelPath = os.path.join(vdo_path, game, 'ball_trajectory', p + '_ball.csv')
        data = pd.read_csv(labelPath)
        no = data['Frame'].values
        v = data['Visibility'].values
        x = data['X'].values
        y = data['Y'].values
        num = no.shape[0]
        r  = os.path.join(vdo_path, game, 'frame', p)
        r2 = os.path.join(vdo_path, game, 'heatmap', p)
        x_data_tmp = []
        y_data_tmp = []
        # 3 in  , last 2 frame have no index i
        for i in range(num-2):
            # ------------------3 frame as 1 unit -------------
            unit = []
            for j in range(3):
                #  每张图都会考虑上 后面2张
                target=str(no[i+j])+'.png'
                png_path = os.path.join(r, target)
                unit.append(png_path)
            train_x.append(unit)
            unit = []
            for j in range(3):
                target=str(no[i+j])+'.png'
                heatmap_path = os.path.join(r2, target)
                if v[i+j] == 0:
                    heatmap_img = genHeatMap(WIDTH, HEIGHT, -1, -1, sigma, mag)
                                  # return np.zeros((h, w))
                else:
                    heatmap_img = genHeatMap(WIDTH, HEIGHT, int(x[i+j]/ratio_W_H), int(y[i+j]/ratio_W_H), sigma, mag)
                heatmap_img *= 255
                unit.append(heatmap_path)
                cv2.imwrite(heatmap_path,heatmap_img)
            train_y.append(unit)

# outputfile_name = 'tracknet_train_list_x_3.csv'
outputfile_name = 'tracknet_test_list_x_3.csv'
with open(outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        #  每unit有3张图
        outputfile.write("%s,%s,%s\n"%(train_x[i][0], train_x[i][1], train_x[i][2]))

#outputfile_name = 'tracknet_train_list_y_3.csv'
outputfile_name = 'tracknet_test_list_y_3.csv'
with open(outputfile_name,'w') as outputfile:
    for i in range(len(train_x)):
        outputfile.write("%s,%s,%s\n"%(train_y[i][0], train_y[i][1], train_y[i][2]))

print('finish')
