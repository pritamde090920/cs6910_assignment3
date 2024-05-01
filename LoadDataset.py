import pandas as pd

class DatasetLoad:
    def loadDataset(self,root,lang):
        train_dataset_path=root+"/"+lang+"_train.csv"
        val_dataset_path=root+"/"+lang+"_valid.csv"
        train_dataframe=pd.read_csv(train_dataset_path,sep=",",header=None)
        val_dataframe=pd.read_csv(val_dataset_path,sep=",",header=None)
        self.train_dataset=train_dataframe.values
        self.val_dataset=val_dataframe.values
    
    def loadTestDataset(self,root,lang):
        test_dataset_path=root+"/"+lang+"_test.csv"
        self.test_dataframe=pd.read_csv(test_dataset_path,sep=",",header=None)
        self.test_dataset=self.test_dataframe.values