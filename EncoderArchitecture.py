import Utilities
import torch.nn as nn



'''class to represent the encoder architecture'''
class EncoderStack(nn.Module):

    '''constructor to intialize the class parameters'''
    def __init__(self,argList):
        '''inherit the constructor of the parent class'''
        super(EncoderStack,self).__init__()
        '''set all the class parameters based on the arguments passed'''
        modelType=argList[0]
        encoderInputSize=argList[1]
        embeddingSize=argList[2]
        neruonsInFC=argList[3]
        layersInEncoder=argList[4]
        dropout=argList[5]
        biDirectional=argList[6]
        attention=argList[7]

        self.neruonsInFC=neruonsInFC
        self.layersInEncoder=layersInEncoder
        if biDirectional=="YES":
            self.biDirect=True
        else:
            self.biDirect=False
        self.attention=attention
        
        '''select the cell type based on the value passed in argument'''
        model_dict={"LSTM":nn.LSTM,"GRU":nn.GRU,"RNN":nn.RNN}
        modelObj=model_dict.get(modelType)

        '''do not apply dropout if only one layer is present'''
        if self.layersInEncoder==1:
            self.dropout=Utilities.createDropoutLayer(0.0)
            self.model=modelObj(embeddingSize,self.neruonsInFC,self.layersInEncoder,dropout=0.0,bidirectional=self.biDirect)
        else:
            self.dropout=Utilities.createDropoutLayer(dropout)
            self.model=modelObj(embeddingSize,self.neruonsInFC,self.layersInEncoder,dropout=dropout,bidirectional=self.biDirect)

        '''create ambedding layer'''
        self.embeddingLayer=Utilities.createEmbeddingLayer(encoderInputSize,embeddingSize)
    


    def forward(self,batchData):
        '''
            Parameters:
                batchData : data sent in batches (as a 2D tensor)
            Returns :
                modelEval : output from the current state of the encoder
                innerLayer : hidden layers representation
                model : the object of the combined architecture with updated parameters
            Function:
                Performs forward propagation in the architecture
        '''

        '''sets embedding layer'''
        embeddedBatch=self.embeddingLayer(batchData)
        embeddedBatch=self.dropout(embeddedBatch)

        model=None

        '''create the gates for LSTM'''
        if isinstance(self.model,nn.LSTM):
            modelEval,(innerLayer,model)=self.model(embeddedBatch)
            '''implement bidirectional architecture'''
            if self.biDirect:
                batchSize=model.size(1)
                model=Utilities.resizeTensor(model,self.layersInEncoder,2,batchSize,-1)
                model=Utilities.reverseTensor(model)
                model=Utilities.getMean(model)
            else:
                model=model[-1,:,:]
            model=Utilities.increaseDimension(model)
        else:
            modelEval,innerLayer=self.model(embeddedBatch)
        
        '''check and implement bidirectional architecture'''
        if self.biDirect:
            batchSize=innerLayer.size(1)
            innerLayer=Utilities.resizeTensor(innerLayer,self.layersInEncoder,2,batchSize,-1)
            innerLayer=Utilities.reverseTensor(innerLayer)
            innerLayer=Utilities.getMean(innerLayer)
            '''apply attention'''
            if self.attention==1:
                modelEval=Utilities.addTensor(modelEval[:,:,:self.neruonsInFC],modelEval[:,:,self.neruonsInFC:])
        else:
            innerLayer=innerLayer[-1,:,:]
        
        innerLayer=Utilities.increaseDimension(innerLayer)

        return modelEval,innerLayer,model