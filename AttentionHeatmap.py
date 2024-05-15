import torch
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator
import seaborn
import wandb
import Utilities



def createPlot():
    '''
        Parameters:
            None
        Returns :
            plot : the plot area for the graph
            axes : the axes lables of the graph
        Function:
            Creates a graph are to plot the heatmaps on
    '''
    plot,axes=plt.subplots(3,3,figsize=(15,15))
    plot.tight_layout(pad=5.0)
    plot.subplots_adjust(top=0.90)
    axes=axes.ravel()
    
    return plot,axes



def createAttentionPerCharacter(attention,character,bengaliLength,englishLength):
    '''
        Parameters:
            attention : attention matrix
            character : character of the attention matrix
            bengaliLength : length of target word
            englishLength : length of source word
        Returns :
            A tensor with the attention per character
        Function:
            Calculates attention per character
    '''
    att=attention[character]
    att=att[:bengaliLength]
    att=att[:,:englishLength]
    return att.T.cpu()



def createHeatMap(attentionPerCharacter,axes,row,bengaliLength,englishLength,xTicks,yTicks):
    '''
        Parameters:
            attentionPerCharacter : attention given to each character
            axes : axes of the graph to plot the heatmap
            row : which row of the grid of grpah to plot
            bengaliLength : length of the target word
            englishLength : length of the source word
            xTicks : x axis labels
            yTicks : y axis labels
        Returns :
            axes : axes of the graph after the heatmap is plotted on the particluar row
        Function:
            Creates heatmap for each position of the grid
    '''
    
    '''create the graph objects required'''
    nullObj=Utilities.getNullObject()
    xObj=Utilities.getFormatObject(xTicks)
    yObj=Utilities.getFormatObject(yTicks)

    '''create the heatmap structure'''
    seaborn.heatmap(attentionPerCharacter, ax=axes[row], cmap='magma', cbar=False, vmin=0.0, vmax=1.0)

    '''edit the axes as per requirement'''
    axes[row].xaxis.set_major_formatter(nullObj)
    minorTickLocator=list()
    for pos in range(bengaliLength):
        minorTickLocator.append(pos+0.5)
    minorObj=FixedLocator(minorTickLocator)
    axes[row].xaxis.set_minor_locator(minorObj)
    axes[row].xaxis.set_minor_formatter(xObj)
    axes[row].yaxis.set_major_formatter(nullObj)
    minorTickLocator=list()
    for pos in range(englishLength):
        minorTickLocator.append(pos+0.5)
    minorObj=FixedLocator(minorTickLocator)
    axes[row].yaxis.set_minor_locator(minorObj)
    axes[row].yaxis.set_minor_formatter(yObj)
    axes[row].set_yticklabels(yTicks,rotation=0,fontdict={'fontsize':12})  
    axes[row].set_xticklabels(xTicks,fontproperties=FontProperties(fname='BengaliFont.TTF'),fontdict={'fontsize':12})
    axes[row].xaxis.tick_top()
    axes[row].set_xlabel('Predicted Bengali Word',size=14,labelpad=-300)
    axes[row].set_ylabel('Original English Word',size=14)
    return axes



def plotAttn(model,inputSequence,outputSequence,vocabulary,trainPy=0):
    '''
        Parameters:
            model : model on which to create the heatmaps
            inputSequence : word in source language
            outputSequence : word in target language
            vocabulary : vocabulary of the dataset
            trainPy : variable indicating whether this is trian.py call or not
        Returns :
            None
        Function:
            Creates 9x9 heatmap grid
    '''
    model.eval()    
    with torch.no_grad():
        '''get the original source and target words'''
        inputSequence=inputSequence.T
        inputSequence=Utilities.setDevice(inputSequence)
        outputSequence=outputSequence.T
        outputSequence=Utilities.setDevice(outputSequence)

        '''run the encoder-decoder architecture and get the model predictions'''
        modelEval,attention=model(inputSequence,outputSequence,teacherRatio=0.0)
        
        modelEval=Utilities.extractColumn(modelEval)
        attention=Utilities.extractColumn(attention)
        attentionSequence=modelEval.argmax(dim=2)
        
        attention=Utilities.reorderDimensions(attention,1,0,2)
        inputSequence=inputSequence.T
        attentionSequence=attentionSequence.T
        
        fig,axes=createPlot()
        
        '''iterate on each character of the word'''
        for row in range(inputSequence.size(0)):
            englishLength=inputSequence.size(1)
            bengaliLength=outputSequence.size(1)-1
            
            '''source word'''
            column=0
            flag=True
            while(flag):
                if inputSequence[row][column].item()==vocabulary.endOfSequenceIndex:
                    englishLength=column+1
                    flag=False
                column+=1
            
            '''target word'''
            column=0
            flag=True
            while(flag):
                if attentionSequence[row][column].item()==vocabulary.endOfSequenceIndex:
                    bengaliLength=column+1
                    flag=False
                column+=1

            '''calculate attention per character'''
            attentionPerCharacter=createAttentionPerCharacter(attention,row,bengaliLength,englishLength)

            '''generate the x and y labels'''
            xTicks,yTicks=Utilities.createXandYticks(bengaliLength,englishLength,vocabulary,attentionSequence,inputSequence,row)

            '''create the heatmap'''
            axes=createHeatMap(attentionPerCharacter,axes,row,bengaliLength,englishLength,xTicks,yTicks)

        '''save the plot and log into wandb'''
        plt.savefig('AttentionHeatMap1.png')
        if trainPy==0:
            wandb.login()
            wandb.init(project="Pritam CS6910 - Assignment 3",name="Question 5 Attention Heatmap")
        wandb.log({'Attention Heatmap1':wandb.Image(plt)})
        if trainPy==0:
            wandb.finish()
        plt.close()