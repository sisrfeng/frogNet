import ttach as tta
import torch
import os
from loguru import logger
import argparse
from conv_not_pool_TrackNet3  import TrackNet3
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

BATCH_SIZE = 1
# 这里导致模型一次只读 1个unit (3帧)
#    for idx in range(3):
# ret, frame = my_vdo_cap.read()
# 指定model的输入尺寸
HEIGHT = 288
WIDTH = 512

parser = argparse.ArgumentParser(description='leo_changed')
parser.add_argument('--video_name', default='wall.mp4')
parser.add_argument('--lr', type=float, default=1e-1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--load_weight',
                    default='exp/bs_32_adadelta_go_on_10_03/best/E43_acc_82.02_Pre_96.45_Rec_80.88')
parser.add_argument('--optimizer', default='Ada',
                    help='Ada or SGD (default: Ada)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()


# log_file = os.path.expanduser('~/.t/predict.log')
# os.system('mv -f {}  {}_bk'.format(log_file, log_file))
# logger.add(log_file)
# logger.info(str(vars(args)))


def find_gpus(num_of_cards_needed=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    _p = os.path.expanduser('~/.tmp_free_gpus')
    with open(_p, 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [(idx, int(x.split()[2]))
                               for idx, x in enumerate(frees)]
    idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [str(idx_memory_pair[0]) for idx_memory_pair in
                 idx_freeMemory_pair[:num_of_cards_needed]]
    usingGPUs = ','.join(usingGPUs)
    # try:
    # logger.info('using GPUs:')
    # for pair in idx_freeMemory_pair[:num_of_cards_needed]:
    # logger.info('{}号: {} MB free'.format(*pair) )
    # except Exception:
    # pass
    return usingGPUs


os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(
    num_of_cards_needed=1)  # 必须在import torch前面
# ('cuda:号数')   号数:从0到N, N是VISIBLE显卡的数量。号数默认是0 [不是显卡的真实编号]
myXPU = torch.device('cuda')

# print('using GPU? ',torch.cuda.is_available())


def WBCE(y_pred, y_true):

    eps = 1e-7
    loss = (torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
            torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    loss = -loss

    return torch.mean(loss)


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


# -------------------------video----------------------------
my_vdo_cap = cv2.VideoCapture(args.video_name)
try:
    total_frames = my_vdo_cap.get(cv2.CAP_PROP_FRAME_COUNT)
# except:   # equivalent to except BaseException.  This will catch SystemExit and KeyboardInterrupt exceptions,
# making it harder to interrupt a program with Control-C, and can disguise other problems.
except Exception:
    total_frames = -1

height = int(my_vdo_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(my_vdo_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

ratio_h = height / HEIGHT
ratio_w = width / WIDTH

out_vdo = cv2.VideoWriter(f'{args.video_name[:-4]}_predict.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          my_vdo_cap.get(cv2.CAP_PROP_FPS),
                          (width, height)
                          )
# -------------------------video----------------------------


csv_file = open(args.video_name[:-4]+'_predict.csv', 'w')
csv_file.write('Frame,Visibility,X,Y,Time\n')
# print(f'csv结果: {args.video_name[:-4]}_predict.csv')

# [[====================================TrackNet=======================================
model = TrackNet3()
tta_model = model
#  tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(100,100))  # test
#  time augmentation  要重新训练模型？
tta_model.to(myXPU)

checkpoint = torch.load(args.load_weight)
tta_model.load_state_dict(checkpoint['state_dict'])
tta_model.eval()
multi_out_last_frm_idx = 0
time_list = []
start1 = time.time()

# cv2.namedWindow('wf_window', flags=(cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO ))  # 允许resize
# cv2.resizeWindow('wf_window', 1080, 960 )

got_img = 0
while got_img < 10:
    # while True:    # 每进一次这循环，就执行一次tta_model(), （在一张9通道的图上推断一次）

    bools = []
    frameX3_WH = []
    frame_times = []

    for idx in range(3):
        _, frame = my_vdo_cap.read()
        bools.append(_)
        frameX3_WH.append(frame)

        t = custom_time(my_vdo_cap.get(cv2.CAP_PROP_POS_MSEC))  # 当前帧是第几毫秒
        frame_times.append(t)

        multi_out_last_frm_idx += 1

    rgbX3 = []  # 3张RGB变为1张9通道的"图”
    if all(bools):
        for frame in frameX3_WH:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbX3.append(frame[:, :, 0])
            rgbX3.append(frame[:, :, 1])
            rgbX3.append(frame[:, :, 2])
    else:
        print('all(bools) 为 False, 视频读取完了?')
        break

    w_h_9 = np.stack(rgbX3, axis=2)
    w_h_9 = cv2.resize(w_h_9, (WIDTH, HEIGHT))
    in_wh_9c = np.moveaxis(w_h_9, -1, 0).astype('float32')/255
    # in_wh_9c = np.moveaxis(w_h_9, -1, 0).astype('float16')/255  # 能用半精度?
    # Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
    in_wh_9c = torch.from_numpy(np.asarray([in_wh_9c])).to(myXPU)
    with torch.no_grad():
        start = time.time()
        # h_pred_3c = tta_model(in_wh_9c)  # 输出3通道的“概率图” ,每张分别对应一帧RGB
        h_pred_3c = tta_model(in_wh_9c)[0]  # batch_size是1，只有1个元素
        #  print(f'{h_pred_3c.shape=}')  # (3, 288, 512)
        end = time.time()
        time_list.append(end - start)

    hm_01_3c = h_pred_3c > 0.5  # 全部变为True False  h_pred_3c和hm_01_3c在后面都要用到
    hm_01_3c = hm_01_3c.cpu().numpy()
    hm_01_3c = hm_01_3c.astype('uint8')  # True变1
    # hm_01_3c = hm_01_3c[0]*255 # batch_size是1，只有1个元素
    hm_01_3c = hm_01_3c*255

    #  hm_01_3c.shape:  (1, 3, 288, 512) batchsize是1
    for hm_idx_in_unit, (frame_WH, frame_time) in enumerate(zip(frameX3_WH, frame_times)):
        # for 3次。每一次for，获得csv上 的一行结果（一张rgb原图上的检测结果）
        # no ball
        current_frm = multi_out_last_frm_idx - 3 + hm_idx_in_unit
        hm_01 = hm_01_3c[hm_idx_in_unit]
        if np.amax(hm_01) <= 0:
            csv_file.write(f'{current_frm},0,0,0,{frame_time}' + '\n')
        else:
            # retrieves only the `extreme outer`` contours.
            mode = cv2.RETR_EXTERNAL
            # It sets hierarchy[i][2] = hierarchy[i][3] = -1 for all the contours
            # compresses horizontal, vertical, and diagonal segments
            method = cv2.CHAIN_APPROX_SIMPLE
            # and leaves only their end points.
            # For example, an up-right rectangular contour is encoded with 4 points.
            contours, _hierarchy = cv2.findContours(
                hm_01, mode, method)  # 应该是最简单粗暴的 找轮廓的方法

            boxes = [cv2.boundingRect(_) for _ in contours]

            if len(boxes) < 2:
                continue

            # heatmap上的最大面积  vs  最大平均置信度：

            # >>>------------------------------------------------------------------ 1. 面积最大的区域，当作是球
            max_area_id = 0
            max_area = boxes[max_area_id][2] * boxes[max_area_id][3]
            for i in range(1, len(boxes)):
                area = boxes[i][2] * boxes[i][3]
                if area > max_area:
                    max_area_id, max_area = i, area
            theBall = boxes[max_area_id]
            X = int(ratio_w * (theBall[0] + theBall[2] / 2))  # 映射/缩放/还原/到原图上
            Y = int(ratio_h * (theBall[1] + theBall[3] / 2))

            radius = int((ratio_w * ratio_h * theBall[2] * theBall[3])**0.5)

            # if radius < 1 or radius > 8:
            #     print(f'{radius=}')
            hm_box_of_bdr = h_pred_3c[hm_idx_in_unit][theBall[1]:theBall[1]+theBall[3],
                                                      theBall[0]:theBall[0]+theBall[2]]
            hm_box_of_bdr = hm_box_of_bdr.cpu().numpy()
            max_area_check_prob = hm_box_of_bdr.sum() / (np.count_nonzero(hm_box_of_bdr) + 1e-5)

            cv2.putText(frame_WH,
                        f'p: {max_area_check_prob:.2f}',
                        # 对于坐标点，先x再y，对于numpy的array或者matrix，先行后列。反过来的
                        (X-radius-60, Y-radius-60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (40, 255, 0),
                        2,
                        cv2.LINE_AA)
            cv2.putText(frame_WH,
                        'large_r:'+str(radius),
                        (X-radius-4, Y-radius-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (40, 255, 0),
                        2,
                        cv2.LINE_AA)

            pt1 = (int(ratio_w * theBall[0]),  int(ratio_h * theBall[1]))
            pt2 = (int(ratio_w * (theBall[0] + theBall[2])),
                   int(ratio_h * (theBall[1] + theBall[3])))
            print(f'{radius=}')
            print(f'{pt1=}')
            print(f'{pt2=}')
            cv2.rectangle(frame_WH,
                          pt1,
                          pt2,
                          (0, 255, 0),
                          3)

            cv2.circle(frame_WH,
                       (X, Y),
                       radius,
                       (255, 255, 255),
                       2)

            cv2.circle(frame_WH,
                       (X, Y),
                       2,
                       (0, 0, 0),
                       -1)
            # ---------------------------------------------------------------------<<< 1. 面积最大的区域，当作是球


            # >>>------------------------------------------------------------------ 1. 平均置信度 最大的区域，当作是球
            max_prob_id = 0
            max_prob = 0
            for i in range(len(boxes)):
                box = boxes[i]
                hm_box_of_bdr = h_pred_3c[hm_idx_in_unit][box[1]
                    :box[1]+box[3], box[0]:box[0]+box[2]]
                # hm_box_of_bdr = h_pred_3c[hm_idx_in_unit][box[0]:box[0]+box[2],box[1]:box[1]+box[3]]
                hm_box_of_bdr = hm_box_of_bdr.cpu().numpy()
                prob = hm_box_of_bdr.sum() / (np.count_nonzero(hm_box_of_bdr) + 1e-5)
                if prob > max_prob:
                    max_prob_id, max_prob = i, prob

            if (max_prob - max_area_check_prob) != 0:
                print(f'{max_prob - max_area_check_prob=}')

            # theBall = boxes[max_area_id]
            if theBall == boxes[max_prob_id]:  # 按最大面积 与 按最大概率 取得同一个框
                continue

            theBall = boxes[max_prob_id]
            X = int(ratio_w * (theBall[0] + theBall[2] / 2))
            Y = int(ratio_h * (theBall[1] + theBall[3] / 2))
            # 每一帧只保留一个球的中心, 但之前看到过检测出多个球的
            csv_file.write(f'{current_frm},1,{X},{Y},{frame_time}' + '\n')

            radius = int((ratio_h * ratio_w * theBall[2] * theBall[3])**0.5)
            # if radius < 1 or radius > 8:
            # print(f'{radius=}')
            cv2.putText(frame_WH,
                        # 记录平均置信度
                        f'p: {max_prob:.2f}',
                        (X-radius-150, Y-radius-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)
            cv2.putText(frame_WH,
                        'r:'+str(radius),
                        (X-radius-4, Y-radius-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (200, 0, 0),
                        2,
                        cv2.LINE_AA)

            pt1 = (int(ratio_w * theBall[0]),  int(ratio_h * theBall[1]))
            pt2 = (int(ratio_w * (theBall[0] + theBall[2])),
                   int(ratio_h * (theBall[1] + theBall[3])))
            print(f'{radius=}')
            print(f'{pt1=}')
            print(f'{pt2=}')
            cv2.rectangle(frame_WH,
                          pt1,
                          pt2,
                          (255, 0, 0),
                          3)

            cv2.circle(frame_WH,
                       (X, Y),
                       radius,
                       (255, 255, 255),
                       2)

            cv2.circle(frame_WH,
                       (X, Y),
                       0,
                       (0, 0, 0),
                       -1)

            #  cv2.circle(frame_WH,
            #             (X, Y),
            #             12*radius,
            #             (255,0,255),
            #             5)

            # ---------------------------------------------------------------------《《《 1. 平均置信度 最大的区域，当作是球

            #  只在hm_01不是一片黑时，imwrite
            got_img += 1
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            cv2.imwrite(f'./viz_3-3RGB/{current_frm}.png', frame_WH)
            cv2.imwrite(f'./viz_3-3RGB/{current_frm}_hm.png', hm_01)

        out_vdo.write(frame_WH)  # 无论frame_WH是否检测到球，都放到输出视频里
        # cv2.imwrite(f'./viz_3-3RGB/{current_frm}.png', frame_WH)

        # cv2.imshow('wf_window', frame_WH)
        # print(f'{current_frm=}')
        # my_key = cv2.waitKey(0)

        # if my_key == ord('q'):
        #     # exit()
        #     import sys
        #     sys.exit(0)
        # else:
        #     pass  #  下一帧


csv_file.close()
my_vdo_cap.release()
out_vdo.release()

end1 = time.time()
print()
print(f'{total_frames=}')
print(f'Prediction time: {end1-start1:.2f} secs')
print(f'FPS: {total_frames / (end1-start1):.1f}')
