import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,modelType,encoderInputSize,embeddingSize,neruonsInFC,layersInEncoder,dropout,biDirectional):
        super(Encoder,self).__init__()
        self.modelType=modelType
        self.encoderInputSize=encoderInputSize
        self.embeddingSize=embeddingSize
        self.neruonsInFC=neruonsInFC
        self.layersInEncoder=layersInEncoder
        self.dropout=nn.Dropout(dropout)
        if biDirectional=="YES":
            self.biDirect=True
        else:
            self.biDirect=False
        
        self.embeddingLayer=nn.Embedding(self.encoderInputSize,self.embeddingSize)
        if self.layersInEncoder==1:
            dropout=0.0
        if self.modelType=="RNN":
            self.model=nn.RNN(self.embeddingSize,self.neruonsInFC,self.layersInEncoder,dropout=dropout,bidirectional=self.biDirect)
        elif self.modelType=="LSTM":
            self.model=nn.LSTM(self.embeddingSize,self.neruonsInFC,self.layersInEncoder,dropout=dropout,bidirectional=self.biDirect)
        else:
            self.model=nn.GRU(self.embeddingSize,self.neruonsInFC,self.layersInEncoder,dropout=dropout,bidirectional=self.biDirect)
    
    def forward(self,batchData):
        embeddedBatch=self.dropout(self.embeddingLayer(batchData))
        model=None
        if self.modelType=="LSTM":
            outputs,(innerLayer,model)=self.model(embeddedBatch)
            if self.biDirect:
                batchSize=model.size(1)
                model=(model.view(self.layersInEncoder,2,batchSize,-1))[-1]
                model=model.mean(axis=0)
            else:
                model=model[-1,:,:]
            model=model.unsqueeze(0)
        else:
            outputs,innerLayer=self.model(embeddedBatch)
        
        if self.biDirect:
            batchSize=innerLayer.size(1)
            innerLayer=(innerLayer.view(self.layersInEncoder,2,batchSize,-1))[-1]
            innerLayer=innerLayer.mean(axis=0)
            outputs=outputs[:,:,:self.neruonsInFC]+outputs[:,:,self.neruonsInFC:]
        else:
            innerLayer=innerLayer[-1,:,:]
        innerLayer=innerLayer.unsqueeze(0)

        return outputs,innerLayer,model