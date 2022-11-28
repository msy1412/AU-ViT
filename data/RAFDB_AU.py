import os, sys, shutil
import csv
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

def load_all_imgs(csv_label,img_folder_path,mode="discrete"):
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

class RAFDB_AU(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_all_imgs(csv_label,img_folder_path,mode="discrete")
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def load_imgs(csv_label,img_folder_path,phase,mode="discrete"):
    imgs = list()
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            img_phase=img_name.split('.')[0].split('_')[0]
            if img_phase!=phase:
                continue
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

class RAFDB_AU_trian(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_imgs_13AU(csv_label,img_folder_path,phase='train',mode=mode)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

class RAFDB_AU_test(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None,mode="discrete"):
        self.imgs= load_imgs_13AU(csv_label,img_folder_path,phase='test',mode=mode)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def load_imgs_13AU(csv_label,img_folder_path,phase,mode="discrete"):
    imgs = list()
    with open(csv_label)as in_f:
        r_in_csv = csv.reader(in_f)
        for i,row in enumerate(r_in_csv):
            if i == 0:
                continue
            img_name = row[0]
            img_phase=img_name.split('.')[0].split('_')[0]
            if img_phase!=phase:
                continue
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

if __name__ == '__main__':
    img_folder_path="/home/ssd7T/FaceData/RAF-DB/aligned/"
    train_list_file = "/home/ssd7T/FaceData/RAF-DB/RAFDB_AU_label_OpenFace.csv"

    #gen pos weight
    # print(gen_posweight(train_list_file,img_folder_path))
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
    all_dataset = RAFDB_AU(train_list_file,img_folder_path,transform=train_transform)
    print(all_dataset.__len__())
    train_set = RAFDB_AU_trian(train_list_file,img_folder_path,transform=train_transform)
    print(train_set.__len__())
    test_set = RAFDB_AU_test(train_list_file,img_folder_path,transform=train_transform)
    print(test_set.__len__())
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=2, shuffle=True)
    # for idx, (images, targets) in enumerate(train_loader):
    #     AU_target_arr = np.array(targets,dtype='int32')
    #     temp=AU_target_arr.T
    #     AU_target_tensor = torch.tensor(AU_target_arr)
