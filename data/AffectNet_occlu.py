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
    def __init__(self,label_npy_path,dataRoot,transform=None,return_paths=False):
        self.labelList=np.load(label_npy_path)
        self.tranform=transform
        # temp_list_7=[]
        # for item in self.labelList:
        #     if int(item[1])!=7:
        #         temp_list_7.append(item)
        # self.labelList=np.array(temp_list_7)
        self.labelList=self.remove_comtempt()
        self.labelList=self.fetch_occlusion()
        self.dataRoot=dataRoot
        self.return_paths=return_paths
    def fetch_occlusion(self):
        occl_path="/home/sztu/dataset/occlusion/affectnet_occlusion_list.txt"
        with open(occl_path,'r') as f:
            labels=f.readlines()
        occlusion_list=[]
        for xx in labels:
            x =xx.split('/')
            occlusion_list.append(x[2])

        temp_list=[]
        for item in self.labelList:
            name=item[0].split('.')[0]
            name=name.split('/')[1]
            if name not in occlusion_list:
                continue
            temp_list.append(item)
        return np.array(temp_list)

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
        if self.return_paths:
            return img,label,imgPath
        else:
            return img,label

    def __len__(self):
        return len(self.labelList)