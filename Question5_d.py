from LoadDataset import DatasetLoad
from PrepareVocabulary import PrepareVocabulary
from WordEmbeddings import WordEmbeddings
from ModelTrainDriver import Model
import AttentionHeatmap
import Utilities
import torch.utils as utils
import itertools
import pandas as pd

'''The purpose of this code is to act like a driver code for generating the attention heatmaps'''

def main():
    '''loads dataset'''
    lang=""
    d=DatasetLoad()
    root="aksharantar_sampled/ben"
    lang=root[root.rfind("/")+1:]
    d.loadDataset(root,lang)
    d.loadTestDataset(root,lang)

    '''creates vocabulary from the dataset'''
    vocabulary=PrepareVocabulary()
    vocabulary.createVocabulary(d.train_dataset)

    '''create embeddings of words for train, validation and test dataset'''
    embeddingTrain=WordEmbeddings()
    embeddingTrain.createWordEmbeddings(d.train_dataset,vocabulary)

    embeddingVal=WordEmbeddings()
    embeddingVal.createWordEmbeddings(d.val_dataset,vocabulary)

    embeddingTest=WordEmbeddings()
    embeddingTest.createWordEmbeddings(d.test_dataset,vocabulary)

    '''create the dataloaders'''
    trainEmbeddedDataset=utils.data.TensorDataset(embeddingTrain.englishEmbedding,embeddingTrain.bengaliEmbedding)
    trainEmbeddedDataLoader=utils.data.DataLoader(trainEmbeddedDataset,batch_size=64,shuffle=True)

    valEmbeddedDataset=utils.data.TensorDataset(embeddingVal.englishEmbedding,embeddingVal.bengaliEmbedding)
    valEmbeddedDataLoader=utils.data.DataLoader(valEmbeddedDataset,batch_size=64)

    testEmbeddedDataset=utils.data.TensorDataset(embeddingTest.englishEmbedding,embeddingTest.bengaliEmbedding)
    testEmbeddedDataset=utils.data.DataLoader(testEmbeddedDataset,batch_size=64)

    '''setting the length of the source (english in this case) and the target (bengali in this case) embeddings'''
    englishLength=embeddingTest.englishEmbedding.size(1)
    bengaliLength=embeddingTest.bengaliEmbedding.size(1)

    '''read the files and create dataframe'''
    dataFrame=pd.read_csv("AttentionVsSeq2Seq.csv")
    englishWordSelected=dataFrame.sample(n=9,random_state=42).iloc[:,0]
    dataFrame2=pd.read_csv("modelPredictionsWithAttention.csv")
    requiredIndices=dataFrame2[dataFrame2['English'].isin(englishWordSelected)]
    requiredIndices=requiredIndices.index.tolist()

    '''create zero tensros for stroing the words'''
    englishWords=Utilities.getLongZeroTensor(len(requiredIndices),englishLength)
    bengaliWords=Utilities.getLongZeroTensor(len(requiredIndices),bengaliLength)

    '''store 9 source and their corresponding target words to create heatmap'''
    for heatmapIndex,position in enumerate(requiredIndices):
        batchPosition=Utilities.getBatchFloorValue(position,64)
        datasetPosition=position-batchPosition*64
        data=next(itertools.islice(testEmbeddedDataset,batchPosition,None))
        englishWord,bengaliWord=data[0][datasetPosition],data[1][datasetPosition]
        englishWords[heatmapIndex]=englishWord
        bengaliWords[heatmapIndex]=bengaliWord
        heatmapIndex+=1

    '''create an object of the model and run the architecture with the best configuration'''
    modelBestWithAttention=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=1,attention=1)
    modelBestWithAttention.createModelFramework(modelType="LSTM",embeddingSize=256,neruonsInFC=256,layersInEncoder=3,layersInDecoder=3,dropout=0,bidirectional="YES",learningRate=0.001,epochs=15,batchSize=64)

    '''plot the heatmaps'''
    AttentionHeatmap.plotAttn(modelBestWithAttention.framework,englishWords,bengaliWords,vocabulary)

if __name__ == "__main__":
    main()