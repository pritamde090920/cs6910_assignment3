import Utilities
import torch.nn as nn



'''class to represent the decoder architecture'''
class DecoderStack(nn.Module):

    '''constructor to intialize the class parameters'''
    def __init__(self,argList):
        '''inherit the constructor of the parent class'''
        super(DecoderStack,self).__init__()
        '''set all the class parameters based on the arguments passed'''
        modelType=argList[0]
        decoderInputSize=argList[1]
        embeddingSize=argList[2]
        neruonsInFC=argList[3]
        outputWordSize=argList[4]
        layersInDecoder=argList[5]
        dropout=argList[6]
        attention=argList[7]

        self.modelType=modelType
        self.layersInDecoder=layersInDecoder
        self.outputWordSize=outputWordSize
        self.attention=attention
        
        '''select the cell type based on the value passed in argument'''
        modelDict={"LSTM":nn.LSTM,"GRU":nn.GRU,"RNN":nn.RNN}
        modelObj=modelDict.get(modelType)

        '''apply attention'''
        if self.attention==0:
            '''do not apply dropout if only one layer is present'''
            if layersInDecoder==1:
                self.dropout=Utilities.createDropoutLayer(0.0)
                self.model=modelObj(embeddingSize,neruonsInFC,layersInDecoder,dropout=0.0)
            else:
                self.dropout=Utilities.createDropoutLayer(dropout)
                self.model=modelObj(embeddingSize,neruonsInFC,layersInDecoder,dropout=dropout)
            self.fullyConnectedLayer=nn.Linear(neruonsInFC,outputWordSize)
        else:
            '''do not apply dropout if only one layer is present'''
            if layersInDecoder==1:
                self.dropout=Utilities.createDropoutLayer(0.0)
                self.model=modelObj(embeddingSize+neruonsInFC,neruonsInFC,layersInDecoder,dropout=0.0)
            else:
                self.dropout=Utilities.createDropoutLayer(dropout)
                self.model=modelObj(embeddingSize+neruonsInFC,neruonsInFC,layersInDecoder,dropout=dropout)
            self.fullyConnectedLayer=nn.Linear(neruonsInFC*2,outputWordSize)
        
        '''create ambedding and linear layer'''
        self.embeddingLayer=Utilities.createEmbeddingLayer(decoderInputSize,embeddingSize)
        self.neuronsInAttentionFC=Utilities.createLinearLayer(neruonsInFC,neruonsInFC,False)
    


    def forward(self,batchData,encoderOutput,innerLayer,model):
        '''
            Parameters:
                batchData : data sent in batches (as a 2D tensor)
                encoderOutput : output from the encoder (on which the decoder will work)
                innerLayer : hidden layers representation
                model : the object of the combined architecture on which the decoder is working
            Returns :
                predictions : predicted outputs from the decoder
                innerLayer : hidden layers representation
                model : the object of the combined architecture with updated parameters
                finalAttentionWeights : updated attention weights
            Function:
                Performs forward propagation in the architecture
        '''

        '''sets batch size and embedding layer'''
        batchData=Utilities.increaseDimension(batchData)
        embeddedBatch=self.embeddingLayer(batchData)
        embeddingLayer=self.dropout(embeddedBatch)

        '''declare the attention matrix'''
        finalAttentionWeights=None

        '''appply attention and calculate the weights'''
        if self.attention==1:
            finalOutputFromEncoderBlock=self.neuronsInAttentionFC(encoderOutput)
            finalHiddenLayer=innerLayer[-1:]
            attentionValues=Utilities.mutiplyTensors(Utilities.reorderDimensions(finalOutputFromEncoderBlock,1,0,2),Utilities.reorderDimensions(finalHiddenLayer,1,2,0))
            attentionValues=Utilities.reorderDimensions(attentionValues,2,0,1)
            finalAttentionWeights=Utilities.setOutputFunction(attentionValues)
            attentionIntoDecoder=Utilities.mutiplyTensors(Utilities.reorderDimensions(finalAttentionWeights,1,0,2),Utilities.reorderDimensions(encoderOutput,1,0,2))
            attentionIntoDecoder=Utilities.reorderDimensions(attentionIntoDecoder,1,0,2)

        '''check and apply attention'''
        if self.attention==0:
            '''apply forget gate for LSTM'''
            if isinstance(self.model,nn.LSTM):
                modelEval,(innerLayer,model)=self.model(embeddingLayer,(innerLayer,model))
            else:
                modelEval,innerLayer=self.model(embeddingLayer,innerLayer)
            '''get decoder outputs by passing through the fully connected layer'''
            predictions=self.fullyConnectedLayer(modelEval)
        else:
            '''apply forget gate for LSTM'''
            if isinstance(self.model,nn.LSTM):
                concatenatedInput=Utilities.concatenateTensor(embeddingLayer,attentionIntoDecoder,2)
                modelEval,(innerLayer,model)=self.model(concatenatedInput,(innerLayer,model))
            else:
                concatenatedInput=Utilities.concatenateTensor(embeddingLayer,attentionIntoDecoder,2)
                modelEval,innerLayer=self.model(concatenatedInput,innerLayer)
            concatenatedInput=Utilities.concatenateTensor(modelEval,attentionIntoDecoder,2)
            '''get decoder outputs by passing through the fully connected layer'''
            predictions=self.fullyConnectedLayer(concatenatedInput)

        predictions=Utilities.decreaseDimension(predictions)

        if self.attention==1:
            finalAttentionWeights=Utilities.decreaseDimension(finalAttentionWeights)

        return predictions,innerLayer,model,finalAttentionWeights