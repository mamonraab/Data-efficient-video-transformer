import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import torch
from torchvision import datasets, models, transforms  
from torch.utils.data.sampler import SubsetRandomSampler  
from torch.utils.data import Dataset, DataLoader

def capture(filename,timesep,rgb,h,w):
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    while i < timesep:
        tmp[:] = frm[:]
        rval, frame = vc.read()
        frm = resize(frame,( h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frm = np.moveaxis(frm, -1, 1)
        frames[i][:] = frm # - tmp
        i +=1
    del tmp
    del frm
    del rval
    return frames


class TaskDataset(Dataset):
    """Fire dataset."""

    def __init__(self, datas, timesep=10,rgb=3,h=90,w=90):
        """
        Args:
            datas: pandas dataframe contain path to videos files with label of them
            timesep: number of frames
            rgb: number of color chanles
            h: height
            w: width
                 
        """
        self.dataloctions = datas
        self.timesep,self.rgb,self.h,self.w = timesep,rgb,h,w


    def __len__(self):
        return len(self.dataloctions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video = capture(self.dataloctions.iloc[idx, 0],self.timesep,self.rgb,self.h,self.w)
        sample = {'video': torch.from_numpy(video), 'label': torch.from_numpy(np.asarray(self.dataloctions.iloc[idx, 1]))}


        return sample
