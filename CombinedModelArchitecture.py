import Utilities
import torch.nn as nn
import random



def formMatrix(dim1,dim2,dim3):
    '''
        Parameters:
            dim1 : First dimension of a tensor
            dim2 : Second dimension of the tensor
            dim3 : Third dimension of the tensor
        Returns :
            A tensor
        Function:
            Creates a tensor with all zeros following the dimensions passed as parameters
    '''
    return Utilities.getZeroTensor(dim1,dim2,dim3)



def doTeacherForcing(decoderOutput,bengaliSequence,sequenceNumber,teacherRatio):
    '''
        Parameters:
            decoderOutput : Tensor representing the output of the previous state of the decoder
            bengaliSequence : Sequence of characters in the target language (bengali in this case)
            sequenceNumber : Index of the sequence to be considered
            teacherRatio : Threshold percentage on whether to apply teacher forching
        Returns :
            The function can return two things:
                if teacher forcing is not applied then return the output of the previous state of the decoder
                else return the actual target word
        Function:
            Performs teacher forcing in the decoder
    '''
    prediction=decoderOutput.argmax(dim=1)

    '''make a random guess and based on that decide whether or not to apply teacher forcing in the current timestamp'''
    currentGuess=random.random()
    if currentGuess<teacherRatio:
        return bengaliSequence[sequenceNumber]
    
    return prediction



'''class to represent the combined architecture of encoder and decoder'''
class EncoderDecoderStack(nn.Module):

    '''constructor to intialize the class parameters'''
    def __init__(self,argList):
        '''inherit the constructor of the parent class'''
        super(EncoderDecoderStack,self).__init__()
        '''encoder object'''
        self.encoderFramework=argList[0]
        '''decoder object'''
        self.decoderFramework=argList[1]
        '''attention(helps to decide whether to apply attention or not)'''
        self.attention=argList[2]
    

    def forward(self,englishSequence,bengaliSequence,teacherRatio=0.5):
        '''
            Parameters:
                englishSequence : Sequence of characters in the source language (english in this case)
                bengaliSequence : Sequence of characters in the target language (bengali in this case)
                teacherRatio : Threshold percentage on whether to apply teacher forching (set to 0.5 if not passed during function call)
            Returns :
                modelEval : output from the architecture
                attentions : updated attention weights
            Function:
                Performs forward propagation in the architecture
        '''

        '''sets batch size and maximum lengths of the words in the source and target dataset'''
        batchSize=Utilities.getShapeOfTensor(englishSequence,1)
        englishSequenceLength=Utilities.getShapeOfTensor(englishSequence,0)
        bengaliSequenceLength=Utilities.getShapeOfTensor(bengaliSequence,0)

        '''sets target vocabulary'''
        bengaliVocabulary=self.decoderFramework.outputWordSize

        '''forms the initial attention and output matrix'''
        attentions=formMatrix(bengaliSequenceLength,batchSize,englishSequenceLength)
        attentions=Utilities.setDevice(attentions)
        modelEval=formMatrix(bengaliSequenceLength,batchSize,bengaliVocabulary)
        modelEval=Utilities.setDevice(modelEval)

        '''passes the source word into the encoder'''
        encoderOutput,innerLayer,model=self.encoderFramework(englishSequence)

        '''resizes the tensor to match decoder architecture'''
        innerLayer=Utilities.expandTensor(innerLayer,self.decoderFramework.layersInDecoder,1,1)
        
        '''resize the tensor if the cell is LSTM'''
        if isinstance(self.decoderFramework.model,nn.LSTM):
            model=Utilities.expandTensor(model,self.decoderFramework.layersInDecoder,1,1)

        '''run the decoder based on whether attention is applied or not'''
        batchData=bengaliSequence[0]
        for sequenceNumber in range(1,bengaliSequenceLength):
            '''if no attention then no need to consider the attention weights being returned by decoder'''
            if self.attention==0:
                decoderOutput,innerLayer,model,_=self.decoderFramework(batchData,None,innerLayer,model)
            else:
                decoderOutput,innerLayer,model,attentionWeights=self.decoderFramework(batchData,encoderOutput,innerLayer,model)            
            modelEval[sequenceNumber]=decoderOutput

            '''if attention is applied then store the attention weights'''
            if self.attention==1:
                attentions[sequenceNumber]=attentionWeights

            '''call teacher forcing function to implement it''' 
            batchData=doTeacherForcing(decoderOutput,bengaliSequence,sequenceNumber,teacherRatio)

        return modelEval,attentions