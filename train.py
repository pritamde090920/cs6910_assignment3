import wandb
from RunTrainPyImplementation import Train
import warnings
warnings.filterwarnings("ignore")
import argparse

'''login to wandb to generate plot'''
wandb.login()

def arguments():
    '''
      Parameters:
        None
      Returns :
        A parser object
      Function:
        Does command line argument parsing and returns the arguments passed
    '''
    commandLineArgument=argparse.ArgumentParser(description='Model Parameters')
    commandLineArgument.add_argument('-wp','--wandb_project',help="Project name used to track experiments in Weights & Biases dashboard")
    commandLineArgument.add_argument('-we','--wandb_entity',help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    commandLineArgument.add_argument('-r','--root',help="Absolute path of the dataset")
    commandLineArgument.add_argument('-e','--epochs',type=int,help="Number of epochs to train neural network")
    commandLineArgument.add_argument('-b','--batch',type=int,help="Batch size to divide the dataset")
    commandLineArgument.add_argument('-n','--neurons',type=int,help="Number of neurons in the fully connected layer")
    commandLineArgument.add_argument('-d','--dropout',type=float,help="Percentage of dropout in the network")
    commandLineArgument.add_argument('-em','--embedding',type=int,help="Size of the embedding layer")
    commandLineArgument.add_argument('-enc','--encoder',type=int,help="Number of layers in the encoder")
    commandLineArgument.add_argument('-dec','--decoder',type=int,help="Number of layers in the decoder")
    commandLineArgument.add_argument('-c','--cell',help="Type of cell")
    commandLineArgument.add_argument('-bid','--bidir',help="Bidirectional flow")
    commandLineArgument.add_argument('-t','--test',type=int,help="choices: [0,1]")
    commandLineArgument.add_argument('-att','--attention',type=int,help="choices: [0,1]")
    commandLineArgument.add_argument('-h','--heatmap',type=int,help="choices: [0,1]")

    return commandLineArgument.parse_args()

'''main driver function'''
def main():
    '''default values of each of the hyperparameter. it is set according to the config of my best model'''
    project_name='Pritam CS6910 - Assignment 3'
    entity_name='cs23m051'
    modelType="LSTM"
    embeddingSize=256
    neruonsInFC=256
    layersInEncoder=3
    layersInDecoder=3
    bidirectional="YES"
    learningRate=0.001
    epochs=15
    batchSize=64
    dropoutProb=0
    test=0
    root='aksharantar_sampled/ben'
    attention=1
    heatmap=1

    '''call to argument function to get the arguments'''
    argumentsPassed=arguments()

    '''checking if a particular argument is passed thorugh commadn line or not and updating the values accordingly'''
    if argumentsPassed.wandb_project is not None:
        project_name=argumentsPassed.wandb_project
    if argumentsPassed.wandb_entity is not None:
        entity_name=argumentsPassed.wandb_entity
    if argumentsPassed.cell is not None:
        modelType=argumentsPassed.cell
    if argumentsPassed.embedding is not None:
        embeddingSize=argumentsPassed.embedding
    if argumentsPassed.neurons is not None:
        neruonsInFC=argumentsPassed.neurons
    if argumentsPassed.encoder is not None:
        layersInEncoder=argumentsPassed.encoder
    if argumentsPassed.decoder is not None:
        layersInDecoder=argumentsPassed.decoder
    if argumentsPassed.bidir is not None:
        bidirectional=argumentsPassed.bidir
    if argumentsPassed.epochs is not None:
        epochs=argumentsPassed.epochs
    if argumentsPassed.batch is not None:
        batchSize=argumentsPassed.batch
    if argumentsPassed.dropout is not None:
        dropoutProb=argumentsPassed.dropout
    if argumentsPassed.test is not None:
        test=argumentsPassed.test
    if argumentsPassed.root is not None:
        root=argumentsPassed.root
    if argumentsPassed.attention is not None:
        attention=argumentsPassed.attention
    if argumentsPassed.heatmap is not None:
        heatmap=argumentsPassed.heatmap

    '''initializing to the project'''
    wandb.init(project=project_name,entity=entity_name)

    '''calling the functions with the parameters'''
    if(attention==0):
        run="EP_{}_CELL_{}_EMB_{}_ENC_{}_DEC_{}_FC_{}_DRP_{}_BS_{}_BIDIREC_{}".format(epochs,modelType,embeddingSize,layersInEncoder,layersInDecoder,neruonsInFC,dropoutProb,batchSize,bidirectional)
    else:
        run="ATT_{}_EP_{}_CELL_{}_EMB_{}_ENC_{}_DEC_{}_FC_{}_DRP_{}_BS_{}_BIDIREC_{}".format("YES",epochs,modelType,embeddingSize,layersInEncoder,layersInDecoder,neruonsInFC,dropoutProb,batchSize,bidirectional)
    print("run name = {}".format(run))
    wandb.run.name=run

    Train.runTrain(root,epochs,batchSize,test,attention,heatmap,modelType,embeddingSize,layersInEncoder,layersInDecoder,neruonsInFC,bidirectional,dropoutProb,learningRate)
    wandb.finish()

if __name__ == '__main__':
    main()