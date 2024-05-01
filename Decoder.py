import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self,modelType,decoderInputSize,embeddingSize,neruonsInFC,outputWordSize,layersInDecoder,dropout):
        super(Decoder,self).__init__()
        self.modelType=modelType
        self.decoderInputSize=decoderInputSize
        self.embeddingSize=embeddingSize
        self.neruonsInFC=neruonsInFC
        self.layersInDecoder=layersInDecoder
        self.outputWordSize=outputWordSize
        self.dropout=nn.Dropout(dropout)

        self.embeddingLayer=nn.Embedding(self.decoderInputSize,self.embeddingSize)
        if self.layersInDecoder==1:
            dropout=0.0
        if self.modelType=="LSTM":
            self.model=nn.LSTM(self.embeddingSize,self.neruonsInFC,self.layersInDecoder,dropout=dropout)
        elif self.modelType=="GRU":
            self.model=nn.GRU(self.embeddingSize,self.neruonsInFC,self.layersInDecoder,dropout=dropout)
        else:
            self.model=nn.RNN(self.embeddingSize,self.neruonsInFC,self.layersInDecoder,dropout=dropout)
        self.fc=nn.Linear(self.neruonsInFC, self.outputWordSize)
    
    def forward(self,batchData,innerLayer,model):
        batchData=batchData.unsqueeze(0)
        embeddingLayer=self.dropout(self.embeddingLayer(batchData))
        if self.modelType=="LSTM":
            outputs,(innerLayer,model)=self.model(embeddingLayer,(innerLayer,model))
        else:
            outputs,innerLayer=self.model(embeddingLayer,innerLayer)
        predictions=self.fc(outputs)
        predictions=predictions.squeeze(0)

        return predictions,innerLayer,model