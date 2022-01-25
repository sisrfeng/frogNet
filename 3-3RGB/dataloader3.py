import pandas as pd
from torch.utils import data

#todo  训练提速
#import jax.numpy as np  # 使用"JAX版"的numpy
import numpy as np

from PIL import Image
from preprocess3 import genHeatMap


HEIGHT     = 288
WIDTH      = 512
mag        = 1
sigma      = 2.5
n_in_n_out = 3



def getData(mode):
    img   = pd.read_csv('tracknet_{}_list_x_3.csv'.format(mode))
    label = pd.read_csv('tracknet_{}_list_y_3.csv'.format(mode))
    return np.squeeze(img.values), np.squeeze(label.values)

class TrackNetLoader(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label_name = getData(mode)
        self.mode = mode
        img = Image.open(self.img_name[0][0]).convert('LA')
        w, h = img.size
        self.ratio = h / HEIGHT
        print("> Found %d data..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        img_path = self.img_name[index]
        bg_path  = self.img_name[index] + '_bg'
        label_path = self.label_name[index]
        img_all = []
        label_all = []
        for id_of_Unit in range(n_in_n_out):
            x = Image.open(img_path[id_of_Unit]).convert('RGB')
            b = Image.open( bg_path[id_of_Unit])
            x = x.resize((WIDTH, HEIGHT))

            # 归一化，变成float64了
            x = np.asarray(x).transpose(2, 0, 1)
            x = x / 255.0
            #x = x.resize((WIDTH, HEIGHT, 3))

            img_all.append(x[0])
            img_all.append(x[1])
            #  img_all.append(x[2])
            img_all.append(b)

            y = Image.open(label_path[id_of_Unit])
            y = np.asarray(y) / 255.0
            label_all.append(y)

        img_all = np.asarray(img_all)
        label_all = np.asarray(label_all)
        '''
        if self.mode == 'train':
          if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        '''
        return img_all, label_all
