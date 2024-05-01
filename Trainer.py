import torch.nn as nn
import torch
from FindAccuracyAndLoss import FindAccuracyAndLoss
import wandb
from copy import deepcopy

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def trainModel(framework,criterion,optimizer,trainEmbeddedDataLoader,valEmbeddedDataLoader,epochs,batchSize,paddingIndex,saveBestModel=0):
        trainAccuracyPerEpoch=list()
        trainLossPerEpoch=list()
        valAccuracyPerEpoch=list()
        valLossPerEpoch=list()

        modelSavePath="/modelParam.pth"

        for epoch in range(epochs):
            framework.train()
            for id,data in enumerate(trainEmbeddedDataLoader):
                inputSequence=data[0]
                outputSequence=data[1]
                inputSequence=inputSequence.T.to(device)
                outputSequence=outputSequence.T.to(device)

                output=framework(inputSequence,outputSequence)
                
                output=output[1:].reshape(-1,output.shape[2])
                target=outputSequence[1:].reshape(-1)

                optimizer.zero_grad()
                loss=criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(framework.parameters(),max_norm=1)
                optimizer.step()
            
            trainingLoss,trainingAccuracy=FindAccuracyAndLoss.findAccuracyAndLoss(framework,trainEmbeddedDataLoader,criterion,batchSize,paddingIndex)
            trainLossPerEpoch.append(trainingLoss)
            trainAccuracyPerEpoch.append(trainingAccuracy)

            valLoss,valAccuracy=FindAccuracyAndLoss.findAccuracyAndLoss(framework,valEmbeddedDataLoader,criterion,batchSize,paddingIndex)
            valLossPerEpoch.append(valLoss)
            valAccuracyPerEpoch.append(valAccuracy)

            print("\n===================================================================================================================")
            print("Epoch : {}".format(epoch+1))
            print("Training Accuracy : {}".format(trainAccuracyPerEpoch[-1]))
            print("Validation Accuracy : {}".format(valAccuracyPerEpoch[-1]))
            print("Training Loss : {}".format(trainLossPerEpoch[-1]))
            print("Validation Loss : {}".format(valLossPerEpoch[-1]))
            wandb.log({"training_accuracy":trainAccuracyPerEpoch[-1],"validation_accuracy":valAccuracyPerEpoch[-1],"training_loss":trainLossPerEpoch[-1],"validation_loss":valLossPerEpoch[-1],"Epoch":(epoch+1)})

        if(saveBestModel==1):
            torch.save(deepcopy(framework.state_dict()),modelSavePath)