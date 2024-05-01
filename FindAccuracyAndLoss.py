import torch

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FindAccuracyAndLoss:
    def findAccuracyAndLoss(framework,dataLoader,criterion,batchSize,paddingIndex):
        framework.eval()
    
        datasetSize=len(dataLoader)*batchSize
        totalLoss=0.0
        correctPredictions=0
        
        with torch.no_grad():
            for id,data in enumerate(dataLoader):
                inputSequence=data[0]
                outputSequence=data[1]
                inputSequence=inputSequence.T.to(device)
                outputSequence=outputSequence.T.to(device)

                output=framework(inputSequence,outputSequence,teacher_force_ratio=0.0)
                
                predictedSequence=output.argmax(dim=2)
                correctPredictions+=torch.logical_or(predictedSequence==outputSequence,outputSequence==paddingIndex).all(dim=0).sum().item()
                output=output[1:].reshape(-1,output.shape[2])
                target=outputSequence[1:].reshape(-1)
                loss=criterion(output,target)
                totalLoss+=loss.item()
            
            accuracy=correctPredictions/datasetSize
            totalLoss/=len(dataLoader)
            return totalLoss,accuracy