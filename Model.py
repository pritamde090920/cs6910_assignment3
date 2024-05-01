import torch
from Encoder import Encoder
from Decoder import Decoder
from CombinedModel import Seq2Seq
from Trainer import Trainer
from torch import optim
import torch.nn as nn

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model:
    def __init__(self,vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=0):
        self.paddingIndex=vocabulary.paddingIndex
        self.encoderInputSize=vocabulary.vocabularySizeForEnglish
        self.decoderInputSize=vocabulary.vocabularySizeForBengali
        self.outputWordSize=vocabulary.vocabularySizeForBengali
        self.trainEmbeddedDataLoader=trainEmbeddedDataLoader
        self.valEmbeddedDataLoader=valEmbeddedDataLoader
        self.test=test
    
    def createModelFramework(self,modelType,embeddingSize,neruonsInFC,layersInEncoder,layersInDecoder,dropout,bidirectional,learningRate,epochs,batchSize):
        self.encoderFramework=Encoder(modelType,self.encoderInputSize,embeddingSize,neruonsInFC,layersInEncoder,dropout,bidirectional)
        self.encoderFramework=self.encoderFramework.to(device=device)
        self.decoderFramework=Decoder(modelType,self.decoderInputSize,embeddingSize,neruonsInFC,self.outputWordSize,layersInDecoder,dropout)
        self.decoderFramework=self.decoderFramework.to(device=device)

        self.framework=Seq2Seq(self.encoderFramework,self.decoderFramework).to(device=device)
        if self.test==0:
          if torch.cuda.device_count()>1:
            self.framework=nn.DataParallel(self.framework)

          self.optimizer=optim.Adam(self.framework.parameters(),lr=learningRate)
          self.criterion=nn.CrossEntropyLoss()
          Trainer.trainModel(self.framework,self.criterion,self.optimizer,self.trainEmbeddedDataLoader,self.valEmbeddedDataLoader,epochs,batchSize,self.paddingIndex)
        else:
           self.framework.load_state_dict(torch.load('modelParam.pth',map_location=torch.device('cpu')))