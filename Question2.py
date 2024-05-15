from LoadDataset import DatasetLoad
from PrepareVocabulary import PrepareVocabulary
from WordEmbeddings import WordEmbeddings
from ModelTrainDriver import Model
import torch.utils as utils
import wandb
import torch

'''login to wandb to generate plot'''
wandb.login()


def main():
    '''initialize to project and create a config'''
    wandb.init(project="Pritam CS6910 - Assignment 3")
    config=wandb.config

    '''loads dataset'''
    lang=""
    d=DatasetLoad()
    root="aksharantar_sampled/ben"
    lang=root[root.rfind("/")+1:]
    d.loadDataset(root,lang)

    '''creates vocabulary from the dataset'''
    vocabulary=PrepareVocabulary()
    vocabulary.createVocabulary(d.train_dataset)

    '''create embeddings of words for train and validation dataset'''
    embeddingTrain=WordEmbeddings()
    embeddingTrain.createWordEmbeddings(d.train_dataset,vocabulary)

    embeddingVal=WordEmbeddings()
    embeddingVal.createWordEmbeddings(d.val_dataset,vocabulary)

    '''create the dataloaders'''
    trainEmbeddedDataset=utils.data.TensorDataset(embeddingTrain.englishEmbedding,embeddingTrain.bengaliEmbedding)
    trainEmbeddedDataLoader=utils.data.DataLoader(trainEmbeddedDataset,batch_size=config.batch_size,shuffle=True)

    valEmbeddedDataset=utils.data.TensorDataset(embeddingVal.englishEmbedding,embeddingVal.bengaliEmbedding)
    valEmbeddedDataLoader=utils.data.DataLoader(valEmbeddedDataset,batch_size=config.batch_size)

    '''give a name for the run'''
    run="EP_{}_CELL_{}_EMB_{}_ENC_{}_DEC_{}_FC_{}_DRP_{}_BS_{}_BIDIREC_{}".format(config.epochs,config.cell_type,config.embedding_size,config.encoder_layers,config.decoder_layers,config.neurons_in_fc,config.dropout,config.batch_size,config.bidirectional)
    wandb.run.name=run
    print("run name = {}".format(run))

    '''
        create an object of the encoder-decoder model which has all the required functions.
        pass the parameters to the constructor as a sweep value. this will change the values with each run of the sweep.
    '''
    model=Model(vocabulary,trainEmbeddedDataLoader,valEmbeddedDataLoader)
    model.createModelFramework(modelType=config.cell_type,embeddingSize=config.embedding_size,neruonsInFC=config.neurons_in_fc,layersInEncoder=config.encoder_layers,layersInDecoder=config.decoder_layers,dropout=config.dropout,bidirectional=config.bidirectional,learningRate=0.001,epochs=config.epochs,batchSize=config.batch_size)



'''sweep configuration'''
configuration_values={
    'method': 'bayes',
    'name': 'ACCURACY AND LOSS',
    'metric': {
        'goal': 'maximize',
        'name': 'validation_accuracy'
    },
    'parameters': {
        'embedding_size' : {'values' : [16,32,64,128,256,512]},
        'encoder_layers' : {'values' : [1,2,3]},
        'decoder_layers' : {'values' : [1,2,3]},
        'neurons_in_fc' : {'values' : [16,32,64,128,256,512]},
        'cell_type' : {'values' : ["RNN","LSTM","GRU"]},
        'bidirectional' : {'values' : ["YES","NO"]},
        'batch_size' : {'values' : [32,64,128]},
        'epochs' : {'values' : [5,10,15]},
        'dropout' : {'values' : [0,0.2,0.4]},
    }
}

'''create a sweep id in the current project'''
sweep_id=wandb.sweep(sweep=configuration_values,project='Pritam CS6910 - Assignment 3')
'''generate a sweep agent to run the sweep'''
wandb.agent(sweep_id,function=main,count=150)
wandb.finish()