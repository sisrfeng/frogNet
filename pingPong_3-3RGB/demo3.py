import os
import torch
import argparse
from TrackNet3 import TrackNet3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import time
from scipy.signal import find_peaks
import csv

BATCH_SIZE = 1
HEIGHT = 288
WIDTH = 512
prefix = "pinp_pong"

parser = argparse.ArgumentParser(description='Pytorch TrackNet6')
parser.add_argument('--load_weight', default='pingpong4_49.tar')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [[================================= 没用上的函数==========================================begin
def angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def get_point_line_distance(point, line):
    point_x = point[0]
    point_y = point[1]
    line_s_x = line[0]
    line_s_y = line[1]
    line_e_x = line[2]
    line_e_y = line[3]
    if line_e_x - line_s_x == 0:
        return math.fabs(point_x - line_s_x)
    if line_e_y - line_s_y == 0:
        return math.fabs(point_y - line_s_y)
    # 斜率
    k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
    # 截距
    b = line_s_y - k * line_s_x
    # 带入公式得到距离dis
    dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
    return dis
#================================= 没用上的函数==========================================]]


def custom_time(time):
    remain = int(time / 1000)
    ms = (time / 1000) - remain
    s = remain % 60
    s += ms
    remain = int(remain / 60)
    m = remain % 60
    remain = int(remain / 60)
    h = remain
    # Generate custom time string
    cts = ''
    if len(str(h)) >= 2:
        cts += str(h)
    else:
        for i in range(2 - len(str(h))):
            cts += '0'
        cts += str(h)

    cts += ':'

    if len(str(m)) >= 2:
        cts += str(m)
    else:
        for i in range(2 - len(str(m))):
            cts += '0'
        cts += str(m)

    cts += ':'

    if len(str(int(s))) == 1:
        cts += '0'
    cts += str(s)

    return cts


csv_file = open(str(prefix)+'predict.csv', 'w')
print(f'结果在：{prefix}_predict.csv')
csv_file.write('Frame,Visibility,X,Y,Time\n')

# cap = cv2.VideoCapture(0)
vdo_cap = cv2.VideoCapture('./ytmc.mp4')  
# total_frames = vdo_cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = vdo_cap.get(cv2.CAP_PROP_FPS)
_, frame = vdo_cap.read()
if _:
    ratio_h = frame.shape[0] / HEIGHT
    ratio_w = frame.shape[1] / WIDTH
    size = (frame.shape[1], frame.shape[0])
else:
    print("open wabcam error")
    os._exit(0)

out_vdo = cv2.VideoWriter(f'{prefix}_predict.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (frame.shape[1], frame.shape[0]))

queue_ball = []
model = TrackNet3()
model.to(device)
checkpoint = torch.load(args.load_weight)
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()
count = 0
count2 = -3
time_list=[]
start1 = time.time()
while True:
    trues = []
    images = []
    frame_times = []
    for idx in range(3):
        # 3in-3out。 每张RGB图 只进入model一次
        # 3in-1out的话，很多RBG图要进入model3次？
        _, frame = vdo_cap.read()  # vdo_cap: cv2.VideoCapture类的一个instance
        t = custom_time(vdo_cap.get(cv2.CAP_PROP_POS_MSEC))
        trues.append(_)
        images.append(frame)
        frame_times.append(t)
        count += 1
        count2 += 1

    grays=[]
    if all(trues):
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            grays.append(img[:,:,0])
            grays.append(img[:,:,1])
            grays.append(img[:,:,2])
    elif count >= count:
        break
    else:
        print("read frame error. skip...")
        continue

    # TackNet prediction
    unit = np.stack(grays, axis=2)
    unit = cv2.resize(unit, (WIDTH, HEIGHT))
    unit = np.moveaxis(unit, -1, 0).astype('float32')/255
    #unit = np.asarray([unit])
    unit = torch.from_numpy(np.asarray([unit])).to(device)
    with torch.no_grad():
        #start = time.time()
        h_pred = model(unit)  # 3通道， 288x512
        #end = time.time()
        #time_list.append(end - start)
    h_pred = h_pred>0.5
    h_pred = h_pred.cpu().numpy()
    h_pred = h_pred.astype('uint8')
    h_pred = h_pred[0]*255

    for idx_frame, (image, frame_time) in enumerate(zip(images, frame_times)):
        showImg = np.copy(image)
        showImg = cv2.resize(showImg, (frame.shape[1], frame.shape[0]))

        # Ball tracking
        if np.amax(h_pred[idx_frame]) <= 0:  # no ball
            csv_file.write(str(count2 + (idx_frame))+',0,0,0,'+frame_time+'\n')
            queue_ball.insert(0,None)
            #out_vdo.write(image)
        else:  # shuttlecock detected
            (cnts, _) = cv2.findContours(h_pred[idx_frame].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = [cv2.boundingRect(ctr) for ctr in cnts]
            max_area_idx = 0
            max_area = rects[max_area_idx][2] * rects[max_area_idx][3]  # why?
            for i in range(len(rects)):
                area = rects[i][2] * rects[i][3]
                if area > max_area:
                    max_area_idx = i
                    max_area = area
            target = rects[max_area_idx]
            cx_pred = int(ratio_w*(target[0] + target[2] / 2))
            cy_pred = int(ratio_h*(target[1] + target[3] / 2))
            csv_file.write(f'{count2 + idx_frame},1,{cx_pred},{cy_pred},{frame_time}' +
                            '\n')
            # 像素点上，其实画不出circle，矩形四周挖掉一些?
            cv2.circle(image,
                        (cx_pred, cy_pred),
                        50,
                        (255,0,255),
                        -1)  # -1表示实心
            queue_ball.insert(0, (cx_pred, cy_pred))
            print('queue_ball: ', queue_ball)
            # out_vdo.write(image)

        # 没搞懂这个queue_ball队列是干啥的
        balls_in_queue = 0
        for t in range(3):   # 对于每一有检测到球的帧，画前一次、本次、和后一次的球的位置？
            try:  # 一开始时，queue_ball太短，可能index out of range
                if queue_ball[t] is not None:
                    print(queue_ball)
                    balls_in_queue += 1

                    cv2.circle(showImg,
                                queue_ball[t],
                                8-t,
                                # 100*(8-t),
                                # (175,112,224),
                                (0,255,0),
                                -1)  # 3个圆,半径越来越小
                    #out_vdo.write(showImg)
                else:
                    pass

                # 作者还检测了运动员
            except Exception:
                break

        # if balls_in_queue == 3:
        #     cv2.imshow('多球',showImg)
        #     if cv2.waitKey(0) & 0xFF == ord('q'):
        #         cv2.destroyWindow('多球')
        out_vdo.write(showImg)

        try:
            queue_ball.pop(3)
        except Exception:
            pass
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break

cv2.destroyAllWindows()
csv_file.close()
print('csv_file保存好了')


print('不知道干啥的内容:')
list1=[]
frames=[]
realx=[]
realy=[]
points=[]
vis=[]
with open(f'{prefix}predict.csv', newline='') as my_csv:
    rows = csv.reader(my_csv, delimiter=',')
    num = 0
    count=0
    for row in rows:
        list1.append(row)
    front_zeros=np.zeros(len(list1))
    for i in range(1,len(list1)):
        frames.append(int(float(list1[i][0])))  # int('11.3'): invalid literal for int() with base 10: '11.3'
        vis   .append(int(float(list1[i][1])))  # 为了避免遇到小数 导致报错，先变成float
        realx .append(int(float(list1[i][2])))
        realy .append(int(float(list1[i][3])))
        if int(float(list1[i][2])) != 0:
            front_zeros[num] = count
            x_y_frm= (     int(float(list1[i][2])),
                            int(float(list1[i][3])),
                            int(float(list1[i][0]))
                        )
            points.append(x_y_frm)
            num += 1
        else:
            count += 1


points = np.array(points)
x, y, z = points.T

Predict_hit_points = np.zeros(len(frames))
peaks, properties = find_peaks(y, prominence=10)    #distance=10) # y非0,所以index有跳

print('Predict points : ')
plt.plot(z,y*-1,'-')
predict_hit=[]
for i in range(len(peaks)):
    print(peaks[i]+int(front_zeros[peaks[i]]))
    predict_hit.append(peaks[i]+int(front_zeros[peaks[i]]))
    #if(peaks[i]+int(front_zeros[peaks[i]]) >= start_point and peaks[i]+int(front_zeros[peaks[i]]) <= end_point):
    #Predict_hit_points[peaks[i]+int(front_zeros[peaks[i]])] = 1

for i in range(len(peaks)-1):
    start = peaks[i]
    end = peaks[i+1]+1
    plt.plot(z[start:end],y[start:end]*-1,'-')

#print(predict_hit)
with open(str(prefix)+'predict_shot.csv','w', newline='') as csvfile1:
    h = csv.writer(csvfile1)
    h.writerow(['Frame','Visibility','X','Y','Hit'])
    for i in range(len(frames)):
        if i in predict_hit:
            h.writerow([frames[i], vis[i], realx[i], realy[i], 1])
        else:
            h.writerow([frames[i], vis[i], realx[i], realy[i], 0])

out_vdo.release()
plt.show()
