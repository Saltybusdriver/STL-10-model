import os
import pandas as pd
import sys
cwd=os.getcwd()

script_dir = os.path.dirname(os.path.abspath(__file__))
print("Script directory in Python:", script_dir)

cwd=script_dir
class Loader:
    def __init__(self):
        self.cwd=script_dir
        self.folder_path = cwd+'/unlabelled/'
        self.files = os.listdir(self.folder_path)
        self.train_root_dir= self.cwd+'/train'
        self.test_root_dir= self.cwd+'/test'
    def get_unlabeled_data(self):
        image_files = [file for file in self.files if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        return pd.DataFrame({'ImageFileName': image_files})
    
    def _LoadData(self,Dir_path):
        Fname=[]
        Iname=[]
        print("Loading data from: ",Dir_path)
        for dirpath, dirnames, filenames in os.walk(Dir_path):
            for filename in filenames:
                if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    Fname.append(os.path.basename(dirpath))
                    Iname.append(filename)
        print("Load Complete!!")
        return Fname, Iname


        
    def get_data(self,root_dir):
        folder_names, image_names = self._LoadData(root_dir)
        return pd.DataFrame({'dir': folder_names,'ImageFileName': image_names})
    def get_train_data_loc(self):
        return self.train_root_dir
        
    def get_test_data_loc(self):
        return self.test_root_dir
    def get_unlabeled_data_loc(self):
        return self.folder_path
    def set_train_data_loc(self,loc):
        self.train_root_dir=loc
    def set_test_data_loc(self,loc):
        self.test_root_dir=loc


