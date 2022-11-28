from operator import neg
import os, sys, shutil
# from signal import valid_signals
import csv
from PIL import Image
import numpy as np
import torch
from torch import tensor
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import transforms

def load_imgs(csv_label,img_folder_path,mode="discrete"):
    imgs = list()
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            confidence = float(row[1].replace(' ',''))
            if confidence<0.8:
                continue
            label_arr=[]
            for i in row[2:]:
                value=float(i.replace(' ',''))
                if mode=="discrete":
                    if value>0.5:
                        label_arr.append(str(1))
                    else:
                        label_arr.append(str(0))
                elif mode=="continuous":
                    # if value>1:
                    #     label_arr.append(str(1))
                    # else:
                    label_arr.append(str(value))
            img_path =os.path.join(img_folder_path,img_name)
            if os.path.exists(img_path):
                imgs.append((img_path,label_arr))
    return imgs

def gen_posweight(csv_label,img_folder_path):
    cnt=0
    pos_list=label_arr=np.array([0 for i in range(16)])
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            img_path =os.path.join(img_folder_path,img_name)
            if not os.path.exists(img_path):
                continue
            confidence = float(row[1].replace(' ',''))
            if confidence<0.8:
                continue

            label_arr=[]
            for i in row[2:]:
                value=float(i.replace(' ',''))
                if value>1:
                    label_arr.append(1)
                else:
                    label_arr.append(0)
            label_arr=np.array(label_arr)
            pos_list +=label_arr
            cnt+=1
    neg_list=[cnt for i in range(16)]-pos_list
    pos_weight=np.divide(neg_list,pos_list)
    return pos_weight

class FERPlus_AU(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_imgs(csv_label,img_folder_path,mode=mode)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def load_imgs_13AU(csv_label,img_folder_path,mode="discrete"):#drop AU7,14,23
    imgs = list()
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            confidence = float(row[1].replace(' ',''))
            if confidence<0.8:
                continue
            label_arr=[]
            cnt=1
            for i in row[2:]:
                if cnt==6 or cnt==10 or cnt==14:
                    cnt+=1
                    continue

                value=float(i.replace(' ',''))
                cnt+=1
                if mode=="discrete":
                    if value>0.5:
                        label_arr.append(str(1))
                    else:
                        label_arr.append(str(0))
                elif mode=="continuous":
                    # if value>1:
                    #     label_arr.append(str(1))
                    # else:
                    label_arr.append(str(value))
            img_path =os.path.join(img_folder_path,img_name)
            if os.path.exists(img_path):
                imgs.append((img_path,label_arr))
    return imgs

class FERPlus_13AU(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_imgs_13AU(csv_label,img_folder_path,mode=mode)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    img_folder_path="/home/ssd7T/FaceData/FERPlus/data-aligned/FER2013Train"
    train_list_file = "/home/ssd7T/FaceData/FERPlus/FERPlus_AU_label_OpenFace.csv"
    test_list_file = "/home/ssd7T/FaceData/FERPlus/FERPlus_AU_label_OpenFace_valid.csv"
    FERPlus_valid_img ="/home/ssd7T/FaceData/FERPlus/data-aligned/FER2013Valid/"
    #gen pos weight
    # print(gen_posweight(train_list_file,img_folder_path))
    # pos_=[8077, 3551, 3355, 6288, 4389, 5741, 2190, 5310, 6286, 6325, 3221, 2334, 8112, 120, 5452, 2616]
    # neg_=[16237, 20763, 20959, 18026, 19925, 18573, 22124, 19004, 18028, 17989, 21093, 21980, 16202, 24194, 18862, 21698]
    
    #gen pos weight

    mean = (0.5,0.5,0.5)
    std = (0.5, 0.5,0.5)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        normalize,
    ])
    valid_set= FERPlus_13AU(test_list_file,FERPlus_valid_img,transform=train_transform)
    print(valid_set.__len__())
    # train_dataset = FERPlus_AU(train_list_file,img_folder_path,transform=train_transform)
    # print(train_dataset.__len__())

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=2, shuffle=True)
    # for idx, (images, targets) in enumerate(train_loader):
    #     AU_target_arr = np.array(targets,dtype='int32')
    #     temp=AU_target_arr.T
    #     AU_target_tensor = torch.tensor(AU_target_arr)
