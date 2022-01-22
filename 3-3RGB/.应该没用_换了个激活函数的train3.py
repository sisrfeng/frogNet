import argparse
import torch
#For convolutional neural networks, enable cuDNN autotuner by setting:
torch.backends.cudnn.benchmark = True
#cuDNN supports many algorithms to compute convolution
#autotuner runs a short benchmark and selects the algorithm with the best performance
from torch.utils.data import DataLoader
#from TrackNet3 import TrackNet3
from fancy_tracknet3 import *
from dataloader3 import TrackNetLoader
import cv2
import math
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('fancy_tensorboard')

parser = argparse.ArgumentParser(description = 'Pytorch TrackNet6')
parser.add_argument('--batchsize', type = int, default = 32, help = 'input batch size for training (defalut: 8)')
parser.add_argument('--epochs', type = int, default = 900, help = 'number of epochs to train (default: 30)')
parser.add_argument('--lr', type = float, default = 1, help = 'learning rate (default: 1)')
parser.add_argument('--tol', type = int, default = 4, help = 'tolerance values (defalut: 4)')
parser.add_argument('--optimizer', type = str, default = 'Adadelta', help = 'Adadelta or SGD (default: Adadelta)')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum fator (default: 0.9)')
parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'weight decay (default: 5e-4)')
parser.add_argument('--seed', type=int, default = 1, help = 'random seed (default: 1)')
parser.add_argument('--load_weight', type = str)
parser.add_argument('--save_weight', type = str, default = 'fancy_weights', help = 'the weight you want to save')
#parser.add_argument("-lp", "--log_path", default='log_train' , help="log_path,not log file name") #暂时不需要log
parser.add_argument("--interval", type = int, default=1 , help="evaluate and save")
args = parser.parse_args()

leo_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print('GPU Use : ',torch.cuda.is_available())
#26段视频全在一个文件
train_data = TrackNetLoader('' , 'train')
test_data = TrackNetLoader('' , 'test')
train_loader = DataLoader(dataset = train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(dataset = test_data, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True )

def pred_vs_gt(y_pred, y_true, tol):
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) == 0:
                TN += 1
            elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) == 0:
                FP2 += 1
            elif torch.max(y_pred[i][j]) == 0 and torch.max(y_true[i][j]) > 0:
                FN += 1
            elif torch.max(y_pred[i][j]) > 0 and torch.max(y_true[i][j]) > 0:
                h_pred = (y_pred[i][j] * 255).cpu().numpy()
                h_true = (y_true[i][j] * 255).cpu().numpy()
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')

                #h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                #h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)

def evaluation(TP, TN, FP1, FP2, FN):
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return (accuracy, precision, recall)

def WBCE(y_pred, y_true):
    eps = 1e-7
    loss = torch.square(1 - y_pred) * y_true * torch.log(torch.clamp(y_pred, eps, 1)) + (
           torch.square(y_pred) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, eps, 1)))
    return torch.mean(loss*(-1))

scaler = torch.cuda.amp.GradScaler()
interval = args.interval
def train(epoch):
    model.train()
    train_loss = 0
    TP = TN = FP1 = FP2 = FN = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(leo_device)
        label = label.type(torch.FloatTensor).to(leo_device)
        optimizer.zero_grad()
         # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            y_pred = model(data)
            loss = WBCE(y_pred, label)
        # scalar 控制 梯度更新计算(检查是否溢出)  #loss.backwards()被 scaler.scale(loss).backwards()取代
        # Scales the loss, and calls backward() to create scaled gradients
        scaler.scale(loss).backward() 

        loss_value = loss.detach().cpu().numpy()
        bar = 100.0 * (batch_idx+1) / len(train_loader)
        #if bar%5==0:
        if 1: 
            print('Train Epoch" {} [{}/{} ({:.0f}%)];  Loss : {:.0f}'.format(
                    epoch, (batch_idx+1) * len(data), len(train_loader.dataset), bar , 1000000*loss_value))
        train_loss += loss_value

        # scalar 控制 优化器(将丢弃的batches转换为 no-op)。 
        # Unscales gradients
        # If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the Unscaled gradients
        # Otherwise, ``optimizer.step()`` is skipped 丢弃这个batch， to avoid corrupting the params.
        scaler.step(optimizer) #代替：optimizer.step()
         # Updates the scale for next iteration
        scaler.update()

        #SGD才需要？:
        #scheduler.step()

        if(epoch % interval == 0):
            y_pred = y_pred > 0.5
            (tp, tn, fp1, fp2, fn) = pred_vs_gt(y_pred, label, args.tol)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn
    train_loss /= len(train_loader)
    writer.add_scalar('train/train_loss', train_loss, epoch)

    (accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)
    writer.add_scalar('train/TP', TP, epoch)
    writer.add_scalar('train/TN', TN, epoch)
    writer.add_scalar('train/FP1', FP1, epoch)
    writer.add_scalar('train/FP2', FP2, epoch)
    writer.add_scalar('train/FN', FN, epoch)

    writer.add_scalar('train/Accuracy', accuracy, epoch)
    writer.add_scalar('train/Precision',precision, epoch)
    writer.add_scalar('train/Recall',recall, epoch)


def test(epoch):
    print('======================testing_by_leo=======================')
    model.eval()
    torch.no_grad()
    test_loss = 0
    TP = TN = FP1 = FP2 = FN = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.type(torch.FloatTensor).to(leo_device)
        label = label.type(torch.FloatTensor).to(leo_device)
        with torch.no_grad():
            y_pred = model(data)
        loss = WBCE(y_pred, label)
        bar = 100.0 * (batch_idx+1) / len(test_loader)
        if bar%49==0:
            print('test Epoch" {} [{}/{} ({:.0f}%)]   Loss : {:.0f}'.format(
                    epoch, (batch_idx+1) * len(data), len(test_loader.dataset), bar , 1000000*loss.data))
        test_loss += loss.data
        y_pred = y_pred > 0.5
        (tp, tn, fp1, fp2, fn) = pred_vs_gt(y_pred, label, args.tol)
        TP += tp
        TN += tn
        FP1 += fp1
        FP2 += fp2
        FN += fn
    test_loss /= len(test_loader)
    writer.add_scalar('test/loss', test_loss, epoch)
    (accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)

    writer.add_scalar('test/TP', TP, epoch)
    writer.add_scalar('test/TN', TN, epoch)
    writer.add_scalar('test/FP1', FP1, epoch)
    writer.add_scalar('test/FP2', FP2, epoch)
    writer.add_scalar('test/FN', FN, epoch)

    writer.add_scalar('test/Accuracy', accuracy, epoch)
    writer.add_scalar('test/Precision',precision, epoch)
    writer.add_scalar('test/Recall',recall, epoch)

    savefilename = args.save_weight + '/acc_{:.2f}_Pre_{:.2f}_Rec_{:.2f}_E_{}'.format(
                                                                                round(100*accuracy,2),
                                                                                round(100*precision,2),
                                                                                round(100*recall,2),
                                                                                epoch)

    torch.save({'epoch':epoch,'state_dict':model.state_dict(),},savefilename)
    return accuracy, precision, recall

def display(TP, TN, FP1, FP2, FN):
    print('======================Evaluate=======================')
    print("Number of true positive:", TP)
    print("Number of true negative:", TN)
    print("Number of false positive FP1:", FP1)
    print("Number of false positive FP2:", FP2)
    print("Number of false negative:", FN)
    (accuracy, precision, recall) = evaluation(TP, TN, FP1, FP2, FN)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print('=====================================================')


model = TrackNet3()
model.to(leo_device)
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
    best_acc = best_pre = best_rec= 0
    sota = 0
    for epoch in range(last_epoch + 1 , args.epochs + 1):
        train(epoch)

        accuracy, precision, recall = test(epoch)
        if accuracy > best_acc:
            best_acc = accuracy ; sota = 1
        if precision > best_pre: 
            best_pre = precision ; sota = 1
        if recall > best_rec: 
            best_rec= recall ; sota = 1
        
        if sota:
            savefilename = './fancy_best_w/acc_{:.2f}_Pre_{:.2f}_Rec_{:.2f}_E_{}'.format(
                                                                                    round(100*accuracy,2),
                                                                                    round(100*precision,2),
                                                                                    round(100*recall,2),
                                                                                    epoch)
        torch.save( {
                        'epoch':     epoch,
                        'state_dict':model.state_dict(),
                    },
                    savefilename)
    writer.close()

main()
