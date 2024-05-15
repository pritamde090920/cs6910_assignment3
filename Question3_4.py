from LoadDataset import DatasetLoad
from PrepareVocabulary import PrepareVocabulary
from WordEmbeddings import WordEmbeddings
from ModelTrainDriver import Model
from ModelForTest import RunTestOnBestModel
import torch.utils as utils
import wandb

'''purpose of this code is to test the best vanilla model'''
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

    '''create an object of the encoder-decoder architecture with the best configuration for vanilla model'''
    model=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=1)
    model.createModelFramework(modelType="LSTM",embeddingSize=256,neruonsInFC=512,layersInEncoder=3,layersInDecoder=3,dropout=0,bidirectional="YES",learningRate=0.001,epochs=15,batchSize=64)

    '''call the function which calculates the accuracy and loss'''
    paramList=[model.framework,testEmbeddedDataset,d.test_dataframe,64,vocabulary.paddingIndex,vocabulary.endOfSequenceIndex,vocabulary.indexToCharDictForBengali]
    image=RunTestOnBestModel.testAndGivePredictions(paramList)

    '''plot the image to wandb'''
    wandb.login()
    wandb.init(project="Pritam CS6910 - Assignment 3",name="Question 3 Vanilla Predictions")
    wandb.log({"Vanilla Predictions":wandb.Image(image)})
    wandb.finish()


    #test accuracy 0.380615234375

if __name__ == "__main__":
    main()