#import DataLoader
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image




class DataframeProcess():
    def __init__  (self):
        pass
    def Randomizer(self,DFrame):
        print("Randomizing row order...")
        Rand_DFrame=DFrame.sample(frac=1.0, random_state=32)
        Rand_DFrame.reset_index(drop=True, inplace=True)
        Rand_DFrame['dir']=Rand_DFrame['dir'].astype(int)
        return Rand_DFrame
    def getData(self,loc):
        return DataLoader.getData(loc)

    def OneHotEncoder(self,DFrame):
        print("One-hot Encoding...")
        target=np.zeros((len(DFrame),10))
        for i in range(len(DFrame)):
            target[i,DFrame.iloc[i,0]-1]=1
        return target
        
    class Unlabeled_Dataset(Dataset):
        def __init__(self, fnames,path, transform=None, target_transform=None):
            self.folder_path=path
            self.fnames=fnames
            self.transform = transform
            self.target_transform = target_transform
        def __len__(self):
            return len(self.fnames)

        def __getitem__(self, idx):
            half_path=self.fnames.iloc[idx].to_string(header=False, index=False)
            half_path=half_path.replace(" ","")
            img_path = self.folder_path+half_path
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            return image


    class ClassDataset(Dataset):
        def __init__(self, fnames,path, target_vect, transform=None, target_transform=None):
            self.target_vect=target_vect
            self.fnames=fnames
            self.folder_path=path
            self.transform = transform
            self.target_transform = target_transform
        def __len__(self):
            return len(self.fnames)

        def __getitem__(self, idx):
            half_path=self.fnames.iloc[idx,[0]].to_string(header=False, index=False)+'/'+self.fnames.iloc[idx,[1]].to_string(header=False, index=False)
            half_path=half_path.replace(" ","")
            img_path = self.folder_path+'/'+half_path
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            
            return image, self.target_vect[idx]
        def getImgPath(self,idx):
            half_path=self.fnames.iloc[idx,[0]].to_string(header=False, index=False)+'/'+self.fnames.iloc[idx,[1]].to_string(header=False, index=False)
            half_path=half_path.replace(" ","")
            img_path = self.folder_path+'/'+half_path
            return img_path

    def getDataset(self,DFrame,path):
        return self.Unlabeled_Dataset(DFrame,path)
    def getClassDataset(self,DFrame,path,target):
        return self.ClassDataset(DFrame,path,target)
    




##put to train.py
'''
Pre_train_loader=DataLoader(Pre_train_dataset,shuffle=False, batch_size=24)
Test_loader=DataLoader(test_dataset,shuffle=True, batch_size=24)
class_train_loader=DataLoader(class_train_dataset,shuffle=False, batch_size=24)
class_test_loader=DataLoader(class_test_dataset,shuffle=True, batch_size=24)
'''