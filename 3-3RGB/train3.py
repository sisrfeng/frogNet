from time import perf_counter as dida
dida0 = dida()
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.expanduser('~/dotF') )
from wf_snippet import vpp as print

# todo
# from numba import jit

from torch.utils.tensorboard import SummaryWriter
from TrackNet3 import TrackNet3
from dataloader3 import TrackNetLoader
from trainFunc import pred_vs_gt, evaluation
import time
import argparse

parser = argparse.ArgumentParser(description='leo_changed')
parser.add_argument("--exp_name", default='my_exp' )  # 不小心敲成--exp也行，毕竟没有重名
parser.add_argument("--gpus", type = int, default='1')
parser.add_argument("--workers", type=int, default=0)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--dist_tolerance', type=int, default=4 )
parser.add_argument('--optimizer', type=str, default='Adadelta',
                    help='先别用SGD。改好学习率衰减策略再用，否则precision和recall都是0')
parser.add_argument('--momentum', type=float, default=0.9 )
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--load_weight', type = str)
parser.add_argument("--interval", type = int, default=1, help="evaluate and save")
parser.add_argument("--confidence", type = float, default=0.5, help="sigmoid后的置信度阈值")
args = parser.parse_args()


weight_path = './exp/{}'.format(args.exp_name)
import os
if os.path.isdir(weight_path):
    print('别重复实验才好啊')
os.system('mkdir -p {}/best'.format(weight_path))  # mkdir 不会覆盖已有目录

#tb : tensorboard
tb_path = '{}/tb'.format(weight_path)
os.system('mkdir -p {}'.format(tb_path)) ; tb_write = SummaryWriter(tb_path)


import numpy as np
import random
#===================================保证可复现======================
an_int = args.seed
torch .manual_seed(an_int)
torch .cuda.manual_seed_all(an_int)
np    .random.seed(an_int)
random.seed(an_int)
torch .backends.cudnn.deterministic = True
torch .backends.cudnn.benchmark = False  # cuDNN supports many algorithms to compute convolution
                                        # autotuner runs a short benchmark and
                                        # selects the algorithm with the best performance
#===================================保证可复现======================
from loguru import logger
log_file = f'./exp/{args.exp_name}/my.log'
if not os.system(f'mv {log_file} {log_file}_bk_`date  +"%m月%d日%H:%M:%S"`'):
    print(f'旧的{log_file}改名了')
else:  #my.log不存在
    print(f'旧的{log_file}改名失败')
print(f'新建: {log_file}')
logger.add(log_file)
logger.info(str(vars(args)))


def find_gpus(num_of_cards_needed=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser ('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx, int(x.split()[2]))
                                for idx, x in enumerate(frees) ]
    idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [str(idx_memory_pair[0]) for idx_memory_pair in
                    idx_freeMemory_pair[:num_of_cards_needed] ]
    usingGPUs = ','.join(usingGPUs)
    logger.info('using GPUs:')
    for pair in idx_freeMemory_pair[:num_of_cards_needed]:
        logger.info('{}号: {} MB free'.format(*pair) )
    return usingGPUs


os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(args.gpus)  # 必须在import torch前面
myXPU =torch.device('cuda')

#26段视频全在一个文件
train_data = TrackNetLoader('', 'train')
test_data = TrackNetLoader('', 'test')


def seed_worker(worker_id):
    # DataLoader will reseed workers following Randomness  in
    # `multi-process`` data loading algorithm.不过这里好像不涉及。
    # Use worker_init_fn() and generator to preserve reproducibility:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


generator = torch.Generator()
generator.manual_seed(0)
train_loader = DataLoader(
    dataset = train_data,
    batch_size=args.batchsize,
    pin_memory=True,
    num_workers= int(args.workers),
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=generator,
)
#不确定这样对不对
generator_test = torch.Generator()
generator_test.manual_seed(2021)
test_loader = DataLoader(
    dataset = test_data,
    batch_size=args.batchsize,
    pin_memory=True,
    num_workers= int(args.workers),
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=generator_test,
)


def WBCE(y_pred, y_true):
    eps = 1e-7
    # loss = (-1)*(torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) +
    #              torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))  # 太丑，换成下面的

    # 如果是多分类问题，交叉熵要对所有类别的  “-标签概率 x log（预测概率） ” 求和。
    # 只不过如果标签是one-hot的，很多项都有0去乘，省略了
    fg_as_classA = torch.pow(1 - y_pred, 2) * y_true * torch.log(torch.clamp(y_pred, eps, 1))
    bg_as_classB = torch.pow(y_pred,2) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1))
    loss    = (-1) * (fg_as_classA + bg_as_classB)

    # todo
    # 改成torch.pow(1 - y_pred,`4`)？ 是y_pred的减函数, 且取值在0到1。
    # 或者 指数函数对原loss的变动太大，可以改成10 * (1-y_pred) + 9等其他减函数？

    return torch.mean(loss)


scaler = torch.cuda.amp.GradScaler()
interval = args.interval

#from rich.progress import track
def train(epoch):
    model.train()
    train_loss = 0
    TP = TN = FP1 = FP2 = FN = 0

    for batch_idx, (a_batch_x, y_true) in enumerate(train_loader):
        #rich_track = track(batch_idx)
        # a_batch_x.shape: 类似 torch.Size([2, 9, 288, 512],即(batch_size, 9通道, 高，宽)
        a_batch_x = a_batch_x.type(torch.FloatTensor).to(myXPU)  #  没有做data parallel吧 # todo
        y_true = y_true.type(torch.FloatTensor).to(myXPU)  #  y_true是一张二值图，球的位置 是矩形抠掉四角的一些像素
        optimizer.zero_grad()
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            y_pred = model(a_batch_x)
            loss = WBCE(y_pred, y_true)
        # scalar 控制 梯度更新计算(检查是否溢出)  #loss.backwards()被 scaler.scale(loss).backwards()取代
        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        percent_of_whole_epoch = 100.0 * (batch_idx+1) * len(a_batch_x) / len(train_loader.dataset)
        if percent_of_whole_epoch == 0.8:
            print(f'Epoch {epoch}跑到80%了，待会再打断吧')

        # print('^_', end='')

        loss_value = loss.detach().cpu().numpy()
        train_loss += loss_value

        # scalar 控制 优化器(将丢弃的batches转换为 no-op)。
        # Unscales gradients
        # If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the Unscaled gradients
        # Otherwise, ``optimizer.step()`` is skipped 丢弃这个batch， to avoid corrupting the params.
        scaler.step(optimizer)  #代替：optimizer.step()
        # Updates the scale for next iteration
        scaler.update()

        #SGD才需要？:
        #scheduler.step()

        if(epoch % interval == 0):
            #  `sigmoid` activation function outputs the heatmap of values: 0 到 1
            # a threshold to turn each value to True or False .
            #  只影响算指标，不影响训练？ 从train_loss的结果看,确实是。对比了0.6 0.7还是取0.5时MCC最高.( 再试试0.55?)
            # 要不把loss改成TP， TN, -FP1, -FP2的加权平均？ 不过貌似不太合理
            y_pred =   y_pred > args.confidence  # y_pred 和y_true都是batchSize x 通道数 x 宽 x 高 的tensor, 其标量的取值为True 或 False
            (tp, tn, fp1, fp2, fn) = pred_vs_gt(y_pred, y_true, args.dist_tolerance)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn
    train_loss /= len(train_loader)
    logger.info('len(train_loader): {}'.format(len(train_loader)))
    tb_write.add_scalar('Loss/train', train_loss, epoch)

    (mcc, accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)
    print('mcc, accuracy, precision, recall')
    print(mcc, accuracy, precision, recall)

    tb_write.add_scalar('A唯一指标/训练_MCC', mcc, epoch)
    tb_write.add_scalar('train_精度/Accuracy', accuracy, epoch)
    tb_write.add_scalar('train_精度/Precision', precision, epoch)
    tb_write.add_scalar('train_精度/Recall', recall, epoch)

    tb_write.add_scalar('Train计数/1确实有球_TP_↑', TP, epoch)
    tb_write.add_scalar('Train计数/2还真没球_TN_↑', TN, epoch)
    tb_write.add_scalar('Train计数/3太远_FP1', FP1, epoch)
    tb_write.add_scalar('Train计数/4误检_FP2', FP2, epoch)
    tb_write.add_scalar('Train计数/5漏掉_FN', FN, epoch)


def test(epoch):
    t1 = time.perf_counter()
    print('======================testing_by_leo=======================')
    model.eval()
    test_loss = 0
    TP = TN = FP1 = FP2 = FN = 0
    for _, (a_batch_x, label) in enumerate(test_loader):
        a_batch_x = a_batch_x.type(torch.FloatTensor).to(myXPU)
        label = label.type(torch.FloatTensor).to(myXPU)
        with torch.no_grad():  # with torch.inference_mode():  #更快，但小概率会出bug
            y_pred = model(a_batch_x)

        loss = WBCE(y_pred, label)
        test_loss += loss.data

        y_pred = y_pred > args.confidence
        (tp, tn, fp1, fp2, fn) = pred_vs_gt(y_pred, label, args.dist_tolerance)
        TP  += tp
        TN  += tn
        FP1 += fp1
        FP2 += fp2
        FN  += fn

    test_loss /= len(test_loader)
    tb_write.add_scalar('Loss/test', test_loss, epoch)
    (mcc, accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)

    tb_write.add_scalar('A唯一指标/测试_MCC', mcc, epoch)
    tb_write.add_scalar('test_精度/Accuracy', accuracy, epoch)
    tb_write.add_scalar('test_精度/Precision', precision, epoch)
    tb_write.add_scalar('test_精度/Recall', recall, epoch)

    tb_write.add_scalar('Test计数/1确实有球_TP_↑', TP, epoch)
    tb_write.add_scalar('Test计数/2还真没球_TN_↑', TN, epoch)
    tb_write.add_scalar('Test计数/3太远_FP1', FP1, epoch)
    tb_write.add_scalar('Test计数/4误检_FP2', FP2, epoch)
    tb_write.add_scalar('Test计数/5漏掉_FN', FN, epoch)


    savefilename = f'{weight_path}/E{epoch}_acc_{100*accuracy:.2f}_Pre_{100*precision:.2f}_Rec_{100*recall:.2f}'
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'mcc': mcc,
                'acc': accuracy,
                'pre': precision,
                'rec': recall  },
               savefilename)
    t2 = time.perf_counter()

    logger.info('len(test_loader): {}'.format(len(test_loader)))
    logger.info('len(test_loader) x batch_size: {}'.format(len(test_loader)*args.batchsize))
    logger.info('spend {} secends'.format(t2 - t1))
    # round(len(test_loader) /(t2 - t1),2)
    # 4203+4233+4222 = 12658,
    # 但print出来是12608，可能是batch_size大了，最后几张图被扔了
    logger.info(f'epoch {epoch} | inference: {(4203+4233+4222)/(t2 - t1):.2f} FPS')
    return mcc, accuracy, precision, recall


model = TrackNet3()
model.to(myXPU)
# model = model.cuda(0) # 0:cuda可识别的gpu序号,可以是1,2,3等
# model.cuda() by default will send your model to the "current device", which can be set with
# torch.cuda.set_device(device).
# An alternative way to send the model to a specific device is model.to(torch.device('cuda:0')) :

# if torch.cuda.device_count() > 1:
    # model = nn.DataParallel(model) # the default gpus here will use all available gpus.

#1、如果数据是稀疏的，用自适应方法较好，如Adagrad, Adadelta, RMSprop, Adam。
#2、RMSprop, Adadelta, Adam 在很多情况下的效果相似。 # 据说adadelta一般在分类问题上效果比较好，adam在生成问题上效果比较好。
#3、SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠。因此很多论文至今仍使用SGD。
#可以先用ada系列先跑,最后快收敛的时候,更换成sgd继续训练.同样也会有提升.
if args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    #默认：     torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    #作者考虑过用下面这个：
    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)

'''
#作者没用这个。SGD才需要？
# Learning rate scheduling should be applied after optimizer’s update
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    cycle_momentum=False,
    epochs= args.epochs,
    )
'''

last_epoch = 0
if(args.load_weight):
    print('======================Retrain=======================')
    checkpoint = torch.load(args.load_weight)
    model.load_state_dict(checkpoint['state_dict'])
    last_epoch = checkpoint['epoch']

def main():
    best_mcc = best_acc = best_pre = best_rec= 0
    if args.load_weight:
        try:
            best_mcc = checkpoint['mcc']
            best_acc = checkpoint['acc']
            best_pre = checkpoint['pre']
            best_rec = checkpoint['rec']
        except:
            print(f'上次实验室时，未在train3.py设置保存最佳指标，epoch {last_epoch+1}的指标再低，都当作best指标')

    for epoch in range(last_epoch + 1, 9999):
        sota = 0
        train(epoch)
        print(f'{epoch=}')

        mcc, accuracy, precision, recall = test(epoch)

        # 几个best不一定同时取得
        if accuracy > best_acc:
            best_acc = accuracy
            sota = 1
        if precision > best_pre:
            best_pre = precision
            sota = 1
        if recall > best_rec:
            best_rec= recall
            sota = 1
        if mcc> best_mcc:
            best_mcc= mcc
            sota = 1

        if sota:
            savefilename = f'{weight_path}/best/E{epoch}_acc_{100*accuracy:.2f}_Pre_{100*precision:.2f}_Rec_{100*recall:.2f}'
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'mcc': mcc,
                    'acc': accuracy,
                    'pre': precision,
                    'rec': recall  },
                   savefilename)
    tb_write.close()

main()
