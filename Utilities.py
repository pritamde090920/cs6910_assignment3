import torch
from torch import optim
import torch.nn as nn
import plotly.graph_objects as go
from matplotlib.ticker import NullFormatter,FixedFormatter
import numpy as np
from PIL import Image

'''The following are the utility functions used across various classes and functions across the whole assignment'''


def setDevice(objToSet):
    '''
        Parameters:
            objToSet : object on which to set the device
        Returns :
            objToSet : the same object after the device is set on it
        Function:
            Sets the device as cpu or gpu based on availability
    '''
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    objToSet=objToSet.to(device)
    return objToSet



def setOptimizer(framework,learningRate):
    '''
        Parameters:
            framework : the model on which to set the opotimizer
            learningRate : learning rate to be applied
        Returns :
            An object of the optimizer
        Function:
            Sets the optimizer
    '''
    return optim.Adam(framework.parameters(),lr=learningRate)



def setLossFunction():
    '''
        Parameters:
            None
        Returns :
            An object of the loss function
        Function:
            Sets the loss function
    '''
    return nn.CrossEntropyLoss()



def setOutputFunction(layer):
    '''
        Parameters:
            layer : layer on which to apply softmax
        Returns :
            An object of the softmax function
        Function:
            Sets the output function as softmax
    '''
    return nn.functional.softmax(layer,dim=2)



def clipGradient(framework):
    '''
        Parameters:
            framework : the model on which to do gradient clipping
        Returns :
            framework : the same model object after gradient clipping is done
        Function:
            Performs gradient clipping
    '''
    torch.nn.utils.clip_grad_norm_(framework.parameters(),max_norm=1)
    return framework



def runDecoderWithNoTeacherForcing(framework,input,output,neruonsInFC):
    '''
        Parameters:
            framework : the model on which to do gradient clipping
            input : input to the decoder
            output : output from the encoder
            neruonsInFC : number of neurons in the fully connected layer
        Returns :
            modelEval : output after running the encoder-decoder architecture
        Function:
            Performs teacher forcing
    '''
    modelEval,model=framework(input,output,neruonsInFC,teacherRatio=0.0)
    return modelEval,model



def increaseDimension(data):
    '''
        Parameters:
            data : tensor whose dimension to increase
        Returns :
            data : same tensor after dimension increase
        Function:
            Performs dimension increase in tensor
    '''
    data=data.unsqueeze(0)
    return data



def decreaseDimension(data):
    '''
        Parameters:
            data : tensor whose dimension to decrease
        Returns :
            data : same tensor after dimension decrease
        Function:
            Performs dimension decrease in tensor
    '''
    data=data.squeeze(0)
    return data



def expandTensor(tensor,dim1,dim2,dim3):
    '''
        Parameters:
            tensor : tensor whose dimensions are to be reproduced
            dim1,dim2,dim3 : dimensions along which to reproduce the tensor
        Returns :
            tensor : same tensor after reproducing dimension
        Function:
            Performs dimension reproducing in tensor
    '''
    tensor=tensor.repeat(dim1,dim2,dim3)
    return tensor



def reorderDimensions(data,dim1,dim2,dim3):
    '''
        Parameters:
            data : tensor whose dimensions are to be reordered
            dim1,dim2,dim3 : dimensions along which to reordered the tensor
        Returns :
            data : same tensor after reordering dimension
        Function:
            Performs dimension reordering in tensor
    '''
    data=data.permute(dim1,dim2,dim3)
    return data



def mutiplyTensors(tensor1,tensor2):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be multiplied
        Returns :
            a product of the two tensors
        Function:
            Performs tensor multiplicaton
    '''
    return tensor1 @ tensor2



def addTensor(tensor1,tensor2):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be added
        Returns :
            a sum of the two tensors
        Function:
            Performs tensor addtion
    '''
    return tensor1+tensor2



def createEmbeddingLayer(layerSize1,layerSize2):
    '''
        Parameters:
            layerSize1,layerSize2 : size of the layers to produce the embedding layer
        Returns :
            an object of the embedding layer
        Function:
            Creates embedding layer
    '''
    return nn.Embedding(layerSize1,layerSize2)



def createLinearLayer(neuronsInLayer1,neuronsInLayer2,bias):
    '''
        Parameters:
            neuronsInLayer1,neuronsInLayer2 : number of neurons to produce the linear layer
            bias : variable indicating whether to apply bias or not
        Returns :
            an object of the linear layer
        Function:
            Creates linear layer
    '''
    return nn.Linear(neuronsInLayer1,neuronsInLayer2,bias=bias)

def createDropoutLayer(percentage):
    '''
        Parameters:
            percentage : percentage of dropout to be applied
        Returns :
            an object of the dropout layer
        Function:
            Creates dropout layer
    '''
    return nn.Dropout(percentage)



def concatenateTensor(tensor1,tensor2,dimension):
    '''
        Parameters:
            tensor1,tensor2 : the tensors which are to be concatenated
            dimension : dimension along which to concatenate
        Returns :
            a concatenated tensor
        Function:
            Performs tensor concatenation
    '''
    return torch.cat([tensor1,tensor2],dim=dimension)



def getMean(data):
    '''
        Parameters:
            data : tensor to find the mean
        Returns :
            mean of the tensor
        Function:
            Calculates the mean of tensor values
    '''
    return data.mean(axis=0)



def getShapeOfTensor(tensor,dimension):
    '''
        Parameters:
            tenosr : tensor to find the shape
            dimension : which dimension to find the shape
        Returns :
            shape of the tensor along the dimension
        Function:
            Calculates the shape of tensor
    '''
    return tensor.shape[dimension]



def resizeTensor(tensor,dim1,dim2,dim3,orientation):
    '''
        Parameters:
            tenosr : tensor to resize
            dim1,dim2,dim3 : dimensions along which to resize the tensor
            orientation : orientation of the tensor
        Returns :
            tensor : same tensor after resizing
        Function:
            Resizes a tensor
    '''
    tensor=tensor.view(dim1,dim2,dim3,orientation)
    return tensor



def reverseTensor(tensor):
    '''
        Parameters:
            tenosr : tensor to reverse
        Returns :
            same tensor after reversing
        Function:
            Reverses a tensor
    '''
    return tensor[-1]



def getZeroTensor(dim1,dim2,dim3):
    '''
        Parameters:
            dim1,dim2,dim3 : dimensions to form the tensor
        Returns :
            a zero tensor
        Function:
            Creates a zero tensor
    '''
    return torch.zeros(dim1,dim2,dim3)



def getLongZeroTensor(dim1,dim2):
    '''
        Parameters:
            dim1,dim2 : dimensions to form the tensor
        Returns :
            a long zero tensor
        Function:
            Creates a long zero tensor
    '''
    return torch.zeros(dim1,dim2,dtype=torch.long)



def plotHtml(df,fileName):
    '''
        Parameters:
            df : the dataframe object on which to plot the image
            fileName : name of the file which is to be saved
        Returns :
            None
        Function:
            Plots and saves the table of predictions
    '''
    columnValues=[df.English,df.Original,df.Predicted,df.Differences]
    head=dict(values=list(df.columns),fill_color='yellow',align='center',font_size=15,height=25)
    value=dict(values=columnValues,fill_color='orange',align='center',font_size=13,height=25)
    columns=dict(l=0,r=0,b=0,t=0)
    table=go.Table(header=head,cells=value)
    plot=go.Figure(data=[table])
    plot.update_layout(autosize=False,width=650,height=500,margin=columns)
    plot.write_html(fileName)



def plotHtmlComparison(df,fileName):
    '''
        Parameters:
            df : the dataframe object on which to plot the image
            fileName : name of the file which is to be saved
        Returns :
            image : the image which is plotted
        Function:
            Saves the table of predictions
    '''
    columnValues=[df.English,df.Original,df.Seq2Seq,df.Attention,df.Differences_Seq2Seq,df.Differences_Attention]
    head=dict(values=list(df.columns),fill_color='yellow',align='center',font_size=15,height=25)
    value=dict(values=columnValues,fill_color='orange',align='center',font_size=13,height=25)
    columns=dict(l=0,r=0,b=0,t=0)
    table=go.Table(header=head,cells=value)
    plot=go.Figure(data=[table])
    plot.update_layout(autosize=False,width=1000,height=500,margin=columns)
    plot.write_html(fileName)
    image=Image.open("AttentionVsSeq2Seq.png")
    return image



def extractColumn(tensor):
    '''
        Parameters:
            tenosr : tensor to extract column
        Returns :
            same tensor after extracting column
        Function:
            Extracts column from tensor
    '''
    return tensor[1:]



def createXandYticks(bengaliLength,englishLength,vocabulary,attentionSequence,inputSequence,row):
    '''
        Parameters:
            bengaliLength : length of the target word
            englishLength : length of the source word
            vocabulary : vocabulary of the dataset
            attentionSequence : word generated by the attention model
            inputSequence : original source sequence
            row : row of the grid to plot
        Returns :
            xticklabels : labels of charatcer of the target word
            yticklabels : labels of charatcer of the source word
        Function:
            Creates the labels for the plot
    '''

    '''target words'''
    xticklabels=[]
    for column in range(bengaliLength):
        value=attentionSequence[row][column]
        value=value.item()
        label=vocabulary.indexToCharDictForBengali[value]
        xticklabels.append(label)
    
    '''source words'''
    yticklabels = []
    for column in range(englishLength):
        value=inputSequence[row][column]
        value=value.item()
        label=vocabulary.indexToCharDictForEnglish[value]
        yticklabels.append(label)
    
    return xticklabels,yticklabels



def getNullObject():
    '''
        Parameters:
           None
        Returns :
            an object of nullformatter
        Function:
            Creates the null formatter object
    '''
    obj=NullFormatter()
    return obj



def getFormatObject(value):
    '''
        Parameters:
            value : value to create the fixed formatter on
        Returns :
            an object of fixedformatter
        Function:
            Creates the fixed formatter object
    '''
    obj=FixedFormatter(value)
    return obj



def getBatchFloorValue(x,y):
    '''
        Parameters:
           x,y : Values whose floor to calculate
        Returns :
            an integer
        Function:
            Calculates and returns floor value
    '''
    floorValue=np.floor(x/y)
    return int(floorValue)