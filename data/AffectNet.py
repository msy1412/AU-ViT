import numpy as np
from scipy.fft import irfft
from torch.utils import data
import torch,os
from PIL import Image

class load_AffectNet(data.Dataset):
    def __init__(self,label_npy_path,dataRoot,transform=None):
        self.labelList=np.load(label_npy_path)
        self.tranform=transform
        self.dataRoot=dataRoot

    def __getitem__(self,index):
        all=self.labelList[index]
        imgName=all[0]
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label_raw=int(all[1])
        label=torch.IntTensor([label_raw])
        return img,label

    def __len__(self):
        return len(self.labelList)

class load_AffectNet_7(data.Dataset):
    def __init__(self,label_npy_path,dataRoot,transform=None):
        self.labelList=np.load(label_npy_path)
        self.tranform=transform
        # temp_list_7=[]
        # for item in self.labelList:
        #     if int(item[1])!=7:
        #         temp_list_7.append(item)
        # self.labelList=np.array(temp_list_7)
        self.labelList=self.remove_comtempt()
        self.dataRoot=dataRoot
    def remove_comtempt(self):
        temp_list_7=[]
        for item in self.labelList:
            if int(item[1])!=7:
                temp_list_7.append(item)
        # self.labelList=temp_list_7
        return np.array(temp_list_7)
    def __getitem__(self,index):
        all=self.labelList[index]
        imgName=all[0]
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label_raw=int(all[1])
        label=torch.IntTensor([label_raw])
        return img,label

    def __len__(self):
        return len(self.labelList)