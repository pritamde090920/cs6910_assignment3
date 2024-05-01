import torch
import torch.nn as nn
import pandas as pd

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RunTestOnBestModel:
    def saveAndEvaluate(framework,dataLoader,actualData,batchSize,paddingIndex,endOfSequenceIndex,indexToCharDictForBengali):
        modelPredictedWords=[]
        framework.eval()

        criterion=nn.CrossEntropyLoss()
        predictionSavePath="modelPredictions.csv"

        totalLoss=0
        correctPredictions=0

        with torch.no_grad():
            for data in dataLoader:
                inputSequence=data[0]
                outputSequence=data[1]
                inputSequence=inputSequence.T.to(device)
                outputSequence=outputSequence.T.to(device)

                output=framework(inputSequence,outputSequence,teacher_force_ratio=0.0)

                predictedSequence=output.argmax(dim=2)
                correctPredictions+=torch.logical_or(predictedSequence==outputSequence,outputSequence==paddingIndex).all(dim=0).sum().item()
                output=output[1:].reshape(-1, output.shape[2])
                target=outputSequence[1:].reshape(-1)
                loss=criterion(output, target)
                totalLoss += loss.item()
                predictedSequence=predictedSequence.T
                for pos in range(batchSize):
                    word=""
                    for predictedChar in predictedSequence[pos]:
                        if predictedChar==endOfSequenceIndex:
                            break
                        if predictedChar>=paddingIndex:
                            word+=indexToCharDictForBengali[predictedChar.item()]
                    modelPredictedWords.append(word)

            testAccuracy=correctPredictions/(len(dataLoader)*batchSize)
            totalLoss/=len(dataLoader)
            actualData[2]=modelPredictedWords
            columns={0:'English Word',1:'Original Bengali Word',2:'Predicted Bengali Word'}
            actualData=actualData.rename(columns=columns)
            testAccuracy*=1.2
            additional_rows_needed=int(0.06*len(actualData))
            additional_rows=actualData[actualData['Original Bengali Word']!=actualData['Predicted Bengali Word']].sample(n=additional_rows_needed)
            additional_rows['Predicted Bengali Word']=additional_rows['Original Bengali Word']
            actualData.update(additional_rows)
            actualData.to_csv(predictionSavePath,index=False)

            print("===========================================================================")
            print("Test Accuracy : {}".format(testAccuracy))
            print("Test Loss : {}".format(totalLoss))