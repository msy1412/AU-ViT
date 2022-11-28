from torch.utils import data
import torch,os
from PIL import Image
from torchvision import transforms
class load_RAFDB(data.Dataset):
    def __init__(self,labelPath,dataRoot,transform=None):
        with open(labelPath,'r') as f:
            labels=f.readlines()
        self.tranform=transform
        self.dataRoot=dataRoot
        # self.size=size
        # self.index=index
        self.imgList=[]
        self.labelList=[]
        self.img_exist_list=os.listdir(dataRoot)
        for xx in labels:
            x =xx.split()
            x[0]+='.jpg'#just for occlusion label list
            imgPath=os.path.join(self.dataRoot,x[0])
            if x[0] in self.img_exist_list:
                self.imgList.append(imgPath)
                self.labelList.append(int(x[1]))#just for occlusion label list:0~6

    def __getitem__(self,index):
        imgPath=self.imgList[index]
        # imgName_=list(imgName)
        # imgName_.insert(self.index,'_aligned')
        # imgName="".join(imgName_)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]])
        return img,label

    def __len__(self):
        return len(self.imgList)

class load_RAFDB_name(data.Dataset):
    def __init__(self,labelPath,dataRoot,transform=None):
        with open(labelPath,'r') as f:
            labels=f.readlines()
        self.tranform=transform
        self.dataRoot=dataRoot
        # self.size=size
        # self.index=index
        self.imgList=[]
        self.labelList=[]
        self.img_exist_list=os.listdir(dataRoot)
        for xx in labels:
            x =xx.split()
            x[0]+='.jpg'#just for occlusion label list
            imgPath=os.path.join(self.dataRoot,x[0])
            if x[0] in self.img_exist_list:
                self.imgList.append(imgPath)
                self.labelList.append(int(x[1]))#just for occlusion label list:0~6

    def __getitem__(self,index):
        imgPath=self.imgList[index]
        # imgName_=list(imgName)
        # imgName_.insert(self.index,'_aligned')
        # imgName="".join(imgName_)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]])
        return img,label,imgPath

    def __len__(self):
        return len(self.imgList)
# if __name__ == '__main__':
#     transform = transforms.Compose([transforms.Resize(size=(224, 224)),transforms.ToTensor()])
#     img_path=r'D:\Dataset\RAFDB\224'
#     label_file=r'D:\Dataset\RAFDB\train.txt'
#     train_dataset =  load_RAFDB(label_file,img_path,transform=transform)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=64, shuffle=True,
#         num_workers=2, pin_memory=True)

#     print(len(train_dataset),len(train_loader))
#     for images, targets in train_loader:
#         print(len(images))