from torch.utils import data
import torch,os
from PIL import Image
class load_RAFDB(data.Dataset):
    def __init__(self,labelPath,dataRoot,transform=None,index=0):
        with open(labelPath,'r') as f:
            labels=f.readlines()
        self.tranform=transform
        self.dataRoot=dataRoot
        self.imgList=[]
        self.labelList=[]
        self.index=index
        for xx in labels:
            x =xx.split()
            self.imgList.append(x[0])
            self.labelList.append(int(x[1])-1)

    def __getitem__(self,index):
        imgName=self.imgList[index]
        imgName_=list(imgName)
        imgName_.insert(self.index,'_aligned')
        imgName="".join(imgName_)
        imgPath=os.path.join(self.dataRoot,imgName)
        img = Image.open(imgPath)
        if self.tranform is not None:
            img=self.tranform(img)
        label=torch.IntTensor([self.labelList[index]])
        return img,label

    def __len__(self):
        return len(self.imgList)