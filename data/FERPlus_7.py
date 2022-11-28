import os, sys, pdb
import os.path as osp
import csv
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch

class FERPlus(data.Dataset):
    """ FERPlus dataset"""
    def __init__(self, data_path, phase = 'train', mode = 'majority', transform = None, lmk = False,RAFDB_format=False):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.lmk = lmk
        self.mode = mode
        self.RAFDB_format=RAFDB_format
        self.EMOTIONS = {0:"Neutral", 1:"Happiness", 2:"Surprise", 3:"Sadness", 4:"Anger", 5:"Disgust", 6:"Fear", 7:"contempt"}
        self.EMOTIONS2Index = {"Neutral":0, "Happiness":1, "Surprise":2, "Sadness":3, "Anger":4, "Disgust":5, "Fear":6, "contempt":7}
        #'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt'
        self.RAFDB_class=["Surprise","Fear","Disgust","Happiness","Sadness","Anger","Neutral"]

        # read annotations
        self.file_paths = []
        self.labels = []
        if phase == 'train':
            self.get_labels_('FER2013Train')
            self.get_labels_('FER2013Valid')
        else:
            self.get_labels_('FER2013Test')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path)
        label = torch.IntTensor([self.labels[idx]])
       
        if self.transform is not None:
            image = self.transform(image)
        
        if self.lmk:
            lmk_path = path.replace('Image/aligned','Landmarks')
            lmk_path = lmk_path.replace('.png', '_landmarks.txt')
            lmks = self.get_landmark_(lmk_path)
            return image, lmks, path, label, idx
        else:
            return image, label
        
    def get_landmark_(self, lmk_path):
        return 0

    def get_labels_(self, subfoler):
        with open(os.path.join(self.data_path, subfoler, 'label.csv')) as csvfile: 
            emotion_label = csv.reader(csvfile)
            for row in emotion_label: 
                emotion_raw = list(map(float, row[2:len(row)]))
                emotion = self.process_data_(emotion_raw, self.mode) 
                idx = np.argmax(emotion)
                if idx > 6: # not unknown or non-face 
                    continue
                if self.mode=='majority':
                    if self.RAFDB_format:
                        self.labels.append(self.RAFDB_class.index(self.EMOTIONS[idx]))
                    else:    
                        self.labels.append(idx)

                else:
                    self.labels.append(emotion)
                self.file_paths.append(os.path.join(self.data_path, subfoler,row[0]))

                
    def process_data_(self, emotion_raw, mode):
        '''
        Based on https://arxiv.org/abs/1608.01041, we process the data differently depend on the training mode:
        Majority: return the emotion that has the majority vote, or unknown if the count is too little.
        Probability or Crossentropty: convert the count into probability distribution.abs
        Multi-target: treat all emotion with 30% or more votes as equal.
        '''        
        size = len(emotion_raw)
        emotion_unknown     = [0.0] * size
        emotion_unknown[-2] = 1.0

        # remove emotions with a single vote (outlier removal) 
        for i in range(size):
            if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
                emotion_raw[i] = 0.0

        sum_list = sum(emotion_raw)
        emotion = [0.0] * size 

        if mode == 'majority': 
            # find the peak value of the emo_raw list 
            maxval = max(emotion_raw) 
            if maxval > 0.5*sum_list: 
                emotion[np.argmax(emotion_raw)] = maxval 
            else: 
                emotion = emotion_unknown   # force setting as unknown 
        elif (mode == 'probability') or (mode == 'crossentropy'):
            sum_part = 0
            count = 0
            valid_emotion = True
            while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
                maxval = max(emotion_raw) 
                for i in range(size): 
                    if emotion_raw[i] == maxval: 
                        emotion[i] = maxval
                        emotion_raw[i] = 0
                        sum_part += emotion[i]
                        count += 1
                        if i >= 8:  # unknown or non-face share same number of max votes 
                            valid_emotion = False
                            if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                                emotion[i] = 0
                                count -= 1
                            break
            if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
                emotion = emotion_unknown   # force setting as unknown 
        elif mode == 'multi_target':
            threshold = 0.3
            for i in range(size): 
                if emotion_raw[i] >= threshold*sum_list: 
                    emotion[i] = emotion_raw[i] 
            if sum(emotion) <= 0.5 * sum_list: # less than 50% of the votes are integrated, we discard this example 
                emotion = emotion_unknown   # set as unknown 
                                
        return [float(i)/sum(emotion) for i in emotion]


if __name__ == '__main__':
    fer = FERPlus('/home/ssd7T/FaceData/FERPlus/data-aligned/', 'train')
    # pdb.set_trace()
    fer_test = FERPlus('/home/ssd7T/FaceData/FERPlus/data-aligned/', 'test', mode='probability')

    print(fer.__len__())
    print(fer_test.__len__())