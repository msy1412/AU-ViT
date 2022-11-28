import os, sys,csv,torch
from PIL import Image
import torch.utils.data as data


def load_imgs(csv_label,img_folder_path):
    imgs = list()
    with open(csv_label, 'r') as file:
        reader = csv.reader(file)
        head=next(reader)
        for line in reader:
            img_name = line[0]
            label_arr = int(line[1])
            img_path =os.path.join(img_folder_path,img_name)
            imgs.append((img_path,label_arr))
    return imgs

class load_RAFAU_FER(data.Dataset):
    def __init__(self, csv_label,img_folder_path, transform=None):

        self.imgs= load_imgs(csv_label,img_folder_path)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        label=torch.IntTensor([target])#don't diretly return torch.IntTensor([target])
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.imgs)


import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
if __name__ == '__main__':

    transform = transforms.Compose([transforms.Resize(size=(112, 112)),transforms.ToTensor()])
    AU_img_folder_path="/home/ssd7T/FaceData/RAF-AU/aligned/"
    train_list_file = '/home/sztu/dataset/RAF-AU/RAFAU_FER-labels.csv'
    train_dataset = load_RAFAU_FER(train_list_file,AU_img_folder_path,transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64, shuffle=True,
        num_workers=2, pin_memory=True)

    print(len(train_dataset),len(train_loader))
    # for images, targets in train_loader:
    #     print(len(images))
    iter_data=iter(train_loader)
    for i in range(50):
        img,label=next(iter_data)
        print(label[0])
