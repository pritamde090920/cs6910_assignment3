import pandas as pd

'''class to load dataset'''
class DatasetLoad:
    def loadDataset(self,root,lang):
        '''
            Parameters:
                root : path of the dataset
                lang : language which is chosen (taken from the path itself)
            Returns :
                None
            Function:
                Loads train and valiation dataset
        '''
        train_dataset_path=root+"/"+lang+"_train.csv"
        val_dataset_path=root+"/"+lang+"_valid.csv"
        train_dataframe=pd.read_csv(train_dataset_path,sep=",",header=None)
        val_dataframe=pd.read_csv(val_dataset_path,sep=",",header=None)
        self.train_dataset=train_dataframe.values
        self.val_dataset=val_dataframe.values
    
    def loadTestDataset(self,root,lang):
        '''
            Parameters:
                root : path of the dataset
                lang : language which is chosen (taken from the path itself)
            Returns :
                None
            Function:
                Loads test dataset
        '''
        test_dataset_path=root+"/"+lang+"_test.csv"
        self.test_dataframe=pd.read_csv(test_dataset_path,sep=",",header=None)
        self.test_dataset=self.test_dataframe.values