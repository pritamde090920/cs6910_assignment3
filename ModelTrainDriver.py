import torch
import Utilities
from EncoderArchitecture import EncoderStack
from DecoderArchitecture import DecoderStack
from CombinedModelArchitecture import EncoderDecoderStack
from RunTrainer import Trainer

'''setting device to cpu to load the saved model during testing'''
device=torch.device('cpu')


'''class to drive the steps of training the model'''
class Model:
    
    '''constructor to intialize the class parameters'''
    def __init__(self,vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader,test=0,attention=0,trainPy=0):
        '''
            Parameters:
                vocabulary : vocabulary of the dataset
                trainEmbeddedDataLoader : training data
                valEmbeddedDataLoader : validation data
                test : variable indicating whether to do test or not
                attention : variable indicating whether to apply attention or not
                root : path of the dataset
                trainPy : variable indicating whether to this is train.py call or not
            Returns :
                None
            Function:
                Sets class parameters
        '''
        self.paddingIndex=vocabulary.paddingIndex
        self.encoderInputSize=vocabulary.vocabularySizeForEnglish
        self.decoderInputSize=vocabulary.vocabularySizeForBengali
        self.outputWordSize=vocabulary.vocabularySizeForBengali
        self.trainEmbeddedDataLoader=trainEmbeddedDataLoader
        self.valEmbeddedDataLoader=valEmbeddedDataLoader
        self.test=test
        self.attention=attention
        self.trainPy=trainPy



    def createModelFramework(self,modelType,embeddingSize,neruonsInFC,layersInEncoder,layersInDecoder,dropout,bidirectional,learningRate,epochs,batchSize):
        '''
            Parameters:
                modelType : type of cell (RNN, LSTM, GRU)
                embeddingSize : size of the embeddings
                neruonsInFC : number of neurons in the fully connected layer
                layersInEncoder : number of layers in the encoder
                layersInDecoder : number of layers in the decoder
                dropout : probability of dropout
                bidirectional : variable indicating whether to apply bidirectional flow or not
                learningRate : learning rate of the model
                epochs : number of epochs to run
                batchSize : batch size used
            Returns :
                None
            Function:
                Runs the encoder-decoder architecture on the data passed
        '''

        '''create encoder object'''
        paramList=[modelType,self.encoderInputSize,embeddingSize,neruonsInFC,layersInEncoder,dropout,bidirectional,self.attention]
        self.encoderFramework=EncoderStack(paramList)
        self.encoderFramework=Utilities.setDevice(self.encoderFramework)

        '''create decoder object'''
        paramList=[modelType,self.decoderInputSize,embeddingSize,neruonsInFC,self.outputWordSize,layersInDecoder,dropout,self.attention]
        self.decoderFramework=DecoderStack(paramList)
        self.decoderFramework=Utilities.setDevice(self.decoderFramework)

        '''create the combined architecture'''
        paramList=[self.encoderFramework,self.decoderFramework,self.attention]
        self.framework=EncoderDecoderStack(paramList)
        self.framework=Utilities.setDevice(self.framework)
        
        '''
            check if this is a train.py call.
            If yes then train the model and return the trained model
        '''
        if self.trainPy==1:
            paramList=[self.framework,learningRate,self.trainEmbeddedDataLoader,self.valEmbeddedDataLoader,epochs,batchSize,self.paddingIndex]
            framework=Trainer.runModelTrainer(paramList,self.trainPy)
            return framework

        else:
            '''if testing is done then no need of training (load the best model that is saved)'''
            if self.test==0:
                paramList=[self.framework,learningRate,self.trainEmbeddedDataLoader,self.valEmbeddedDataLoader,epochs,batchSize,self.paddingIndex]
                Trainer.runModelTrainer(paramList)
            else:
                '''load the best model based on whether attention is applied or not'''
                if self.attention==0:
                    self.framework.load_state_dict(torch.load('bestModelVanilla.pth',map_location=device))
                else:
                    self.framework.load_state_dict(torch.load('bestModelAttention.pth',map_location=device))