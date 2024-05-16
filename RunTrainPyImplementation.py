from LoadDataset import DatasetLoad
from PrepareVocabulary import PrepareVocabulary
from WordEmbeddings import WordEmbeddings
from ModelTrainDriver import Model
import ModelForTestAttention
import ModelForTest
import AttentionHeatmap
import Utilities
import itertools
import torch.utils as utils
import random


class Train:
    def runTrain(root,epochs,batchSize,test,attention,heatmap,modelType,embeddingSize,layersInEncoder,layersInDecoder,neruonsInFC,bidirectional,dropoutProb,learningRate,fontName):
        '''loads dataset'''
        lang=""
        d=DatasetLoad()
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

        '''create an object of the encoder-decoder architecture with the best configuration for attention based model'''
        myModel=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=test,attention=attention,trainPy=1)
        framework=myModel.createModelFramework(modelType=modelType,embeddingSize=embeddingSize,neruonsInFC=neruonsInFC,layersInEncoder=layersInEncoder,layersInDecoder=layersInDecoder,dropout=dropoutProb,bidirectional=bidirectional,learningRate=learningRate,epochs=epochs,batchSize=batchSize)

        '''if prompted then do testing'''
        if test==1:
            if attention==1:                
                '''call the function which calculates the accuracy and loss'''
                paramList=[framework,testEmbeddedDataset,d.test_dataframe,64,vocabulary.paddingIndex,vocabulary.endOfSequenceIndex,vocabulary.indexToCharDictForBengali]
                ModelForTestAttention.RunTestOnBestModel.testAndGivePredictions(paramList,trainPy=1)

                '''if required then plot the heatmap'''
                if heatmap==1:
                    englishLength=embeddingTest.englishEmbedding.size(1)
                    bengaliLength=embeddingTest.bengaliEmbedding.size(1)

                    requiredIndices=random.sample(range(4097),9)

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
                    AttentionHeatmap.plotAttn(framework,englishWords,bengaliWords,vocabulary,trainPy=1,fontName=fontName)
            else:
                '''call the function which calculates the accuracy and loss'''
                paramList=[framework,testEmbeddedDataset,d.test_dataframe,64,vocabulary.paddingIndex,vocabulary.endOfSequenceIndex,vocabulary.indexToCharDictForBengali]
                ModelForTest.RunTestOnBestModel.testAndGivePredictions(paramList,trainPy=1)
