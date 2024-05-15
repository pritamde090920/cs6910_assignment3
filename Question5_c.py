import Utilities
import pandas as pd
import wandb

'''The purpose of this code is to act like a driver code for generating the plot of comparison between the attention model and the seq2seq model'''

def main():
    '''
        reading the two files
            modelPredictions.csv : contains all the predicted words by the vanilla seq2seq model
            modelPredictionsWithAttention.csv : contains all the predicted words by the attention model
    '''
    vanillaDataframe=pd.read_csv('modelPredictions.csv')
    attentiondataFrame=pd.read_csv('modelPredictionsWithAttention.csv')

    '''setting the path where the file storing the comparison of the two models will be stored'''
    dataframeSavePath="AttentionVsSeq2Seq.csv"

    '''
        creating a list to store the words.
        the words which are wrongly predicted by se2seq model and correctly predicted by attention model are stored here
    '''
    container=list()
    '''iterating over the entire predictions'''
    for index,(row1,row2) in enumerate(zip(vanillaDataframe.iterrows(),attentiondataFrame.iterrows())):
        '''
            checking if seq2seq2 prediction is wrong and attention prediction is correct
            if yes then add the respective words into the list
        '''
        if row1[1]['Original']!=row1[1]['Predicted'] and row2[1]['Original']==row2[1]['Predicted']:
            container.append((row1[1]['English'],row1[1]['Original'],row1[1]['Predicted'],row2[1]['Predicted']))

    '''creating a dataframe for the final csv file and putting the contents of the list created above into the datafram'''
    finalDataframe=pd.DataFrame(container,columns=['English','Original','Seq2Seq','Attention'])

    '''saving the file into the path specified'''
    finalDataframe.to_csv(dataframeSavePath,index=False)

    '''reading the file saved above and randomly picking 10 sample points to plot'''
    df=pd.read_csv('AttentionVsSeq2Seq.csv').sample(n=10)

    '''
        creating two lists to store the number of characters found different in the two models
        (our expectation is to get 0 differences for the attention words)
    '''
    differencesSeq2Seq=list()
    differencesAttention=list()

    '''iterating over the 10 sample points'''
    for _,row in df.iterrows():
        '''picking the original translation, seq2seq translation and the attention translation'''
        original=row['Original']
        seq2seq=row['Seq2Seq']
        attention=row['Attention']

        '''finding the number of difference by checking each character in the seq2seq translation and the original translation'''
        numberOfDifferences=0
        for char1,char2 in zip(original,seq2seq):
            if char1!=char2:
                numberOfDifferences+=1
        differencesSeq2Seq.append(numberOfDifferences)

        '''finding the number of difference by checking each character in the attention translation and the original translation'''
        numberOfDifferences=0
        for char1,char2 in zip(original,attention):
            if char1!=char2:
                numberOfDifferences+=1
        differencesAttention.append(numberOfDifferences)

    '''creating two columns in the dataframe for the respective differences of the two models'''
    df['Differences_Seq2Seq']=differencesSeq2Seq
    df['Differences_Attention']=differencesAttention

    '''calling the utility function to generate the image'''
    image=Utilities.plotHtmlComparison(df,"AttentionVsSeq2Seq1.html")

    '''logging the plot into wandb'''
    wandb.login()
    wandb.init(project="Pritam CS6910 - Assignment 3",name="Question 5 Attention Vs Seq2Seq")
    wandb.log({"Attention Vs Seq2Seq":wandb.Image(image)})
    wandb.finish()

if __name__=="__main__":
    main()