import Datasets
import DataLoaders
import Model
from Model import AE2
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torchvision.io import read_image
import cv2
import torchvision.transforms as transforms
import __main__
setattr(__main__, "AE2", AE2)




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.003):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

DLoader=DataLoaders.Loader()
DframeProcess=Datasets.DataframeProcess()

train_df=DLoader.get_data(DLoader.get_train_data_loc())
test_df=DLoader.get_data(DLoader.get_test_data_loc())
unlabeled_df=DLoader.get_unlabeled_data()


rand_train_df=DframeProcess.Randomizer(train_df)
rand_test_df=DframeProcess.Randomizer(test_df)


train_target=DframeProcess.OneHotEncoder(rand_train_df)
test_target=DframeProcess.OneHotEncoder(rand_test_df)


Unlabeled_loc=DLoader.get_unlabeled_data_loc()

Pre_train_dataset = DframeProcess.getDataset(unlabeled_df,Unlabeled_loc)
class_train_dataset=DframeProcess.getClassDataset(rand_train_df,DLoader.get_train_data_loc(),train_target)
class_test_dataset=DframeProcess.ClassDataset(rand_test_df,DLoader.get_test_data_loc(),test_target)





Pre_train_loader=DataLoader(Pre_train_dataset,shuffle=False, batch_size=24,num_workers=8)
#Test_loader=DataLoader(test_dataset,shuffle=True, batch_size=24)


script_dir = os.path.dirname(os.path.abspath(__file__))
cwd=script_dir


model3 = Model.AE2()
model3=torch.load(cwd+'/Models/model_ae_small_40epoch.pth')
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model3.to(dtype=torch.float32, device=dev)
optimizer = torch.optim.Adam(model3.parameters(), lr=0.0001)


model2 = Model.Classifier()
model2.encoder.load_state_dict(model3.get_encoder_state_dict())
model2.to(dtype=torch.float32, device=dev)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0001)


are_weights_equal = all(torch.equal(ae_param, another_param) for ae_param, another_param in zip(model3.encoder.parameters(), model2.encoder.parameters()))

print("Models are on device: ",dev)
if are_weights_equal:
    print("Weights were successfully copied.")
else:
    print("Weights were not copied successfully.")

loss_function=torch.nn.MSELoss()
class_loss_function=torch.nn.KLDivLoss(reduction="batchmean")
epochs = 40
outputs = []
losses = []


def train_ae(_model,epoch,loader,loss_func):
    for epoch in range(epochs):
        id=0
        for image in loader:
            id=id+1
            imag=image[0]
            imag=imag.permute(1,2,0)
            image=image/255
            image=image.to(dtype=torch.float32,device=dev) 
            
            reconstructed = _model(image)

            img=reconstructed
            img=img.detach()
            img2=img[0]
            img2=img2.permute(1,2,0)
            img2=img2.cpu()
            
            '''
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 15))
            ax1.imshow(img2)
            ax2.imshow(imag)
            plt.show()
            '''
            
            loss = loss_func(reconstructed, image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            if(id%20==0):
                print(f'Train Epoch: {epoch}, {id}/{len(Pre_train_loader)}\tLoss: {loss.item()}')
 

def freeze_param(_model):
    for param in _model.encoder.parameters():
        param.requires_grad = False

    for i in range(10,12):
        for param in _model.encoder[i].parameters():
            param.requires_grad = True

    for name, param in _model.named_parameters():
        print(name,param.requires_grad)

def train_classifier(_model,epochs,loader,val_loader,loss_func,freeze):
    
    if(freeze):
        freeze_param(_model)
    Stopper=EarlyStopper()
    for epoch in range(epochs):
        id=0
        total_loss=0
        for image2,target2 in loader:
            id=id+1

            image2=image2/255      
            image2=image2.to(dtype=torch.float32,device=dev)
            target2=target2.to(dev)
            
            out=model2(image2)
            out_indice=torch.argmax(out,dim=1)
            target_indice=torch.argmax(target2,dim=1)
            target2=F.softmax(target2,dim=1)
            target2=target2.to(torch.float32)
            lossc = class_loss_function(out,target2)

            optimizer2.zero_grad()
            lossc.backward()
            optimizer2.step()
            
            if(id%20==0):
                print(f'Train Epoch: {epoch}, {id}/{len(loader)}\tLoss: {lossc.item()}')
            #del lossc,out,image2,target2,out_indice,target_indice
            
        
        print("Performing check..")
        
        for image,target in val_loader:
            with torch.no_grad():
                image=image/255      
                image=image.to(dtype=torch.float32,device=dev)
                target=target.to(dev)
                out=model2(image)
                out_indice=torch.argmax(out,dim=1)
                target_indice=torch.argmax(target,dim=1)
                target=F.softmax(target,dim=1)
                target=target.to(torch.float32)
                lossc = class_loss_function(out,target)
                total_loss += lossc.item()
        average_loss = total_loss / len(val_loader)
        print("Check complete, current val loss:",average_loss)
        if Stopper.early_stop(average_loss):
            print("val loss stopped decreasing, exiting..")
            break


def call_train_class(EPOCH,FREEZE,BATCH_SIZE,NUM_WORKERS):
    class_train_loader=DataLoader(class_train_dataset,shuffle=False, batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    class_test_loader=DataLoader(class_test_dataset,shuffle=True, batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
    print("runnning train with params: ",EPOCH,FREEZE,BATCH_SIZE,NUM_WORKERS)
    train_classifier(model2,EPOCH,class_train_loader,class_test_loader,class_loss_function,FREEZE)
    print("after train call")

def _interference(image):
    if(image.shape!=(3,96,96)):
        transform = transforms.Compose([
            transforms.Resize((96, 96)),  # Set your desired dimensions
            transforms.ToTensor()
        ])
        image=transform(image)
    with torch.no_grad():
        image=image/255  
        image=image.unsqueeze(0)    
        image=image.to(dtype=torch.float32,device=dev)
        out=model2(image)
        out_indice=torch.argmax(out,dim=1)
        return out_indice.item()



def random_interf(id):
    img, _ = class_test_dataset.__getitem__(id)
    a=_interference(img);
    path=class_test_dataset.getImgPath(id)
    return _interference(img),path
    

def specific_interf(img_path):
    img=read_image(img_path)
    print(img)
    return _interference(img)