import Utilities
import torch
import torch.nn as nn



'''class to create the word embeddings'''
class WordEmbeddings:

    def createWordEmbeddings(self,dataset,vocabulary):
        '''
            Parameters:
                dataset : dataset on which to create the embeddings
                vocabulary : vocabulary of the dataset
            Returns :
                None
            Function:
                Creates embeddings of the words
        '''
        englishDataset=dataset[:,0]
        bengaliDataset=dataset[:,1]

        tensorListEnglish=list()
        tensorListBengali=list()

        '''embeddings for source language'''
        language="english"
        for one_word in englishDataset:
            tensor=self.translateWordToTensor(one_word,vocabulary,language)
            tensor=Utilities.setDevice(tensor)
            tensorListEnglish.append(tensor)
        self.englishEmbedding=nn.utils.rnn.pad_sequence(tensorListEnglish,padding_value=vocabulary.paddingIndex,batch_first=True)
        self.englishEmbedding=Utilities.setDevice(self.englishEmbedding)

        '''embeddings for target language'''
        language="bengali"
        for one_word in bengaliDataset:
            tensor=self.translateWordToTensor(one_word,vocabulary,language)
            tensor=Utilities.setDevice(tensor)
            tensorListBengali.append(tensor)
        self.bengaliEmbedding=nn.utils.rnn.pad_sequence(tensorListBengali,padding_value=vocabulary.paddingIndex,batch_first=True)
        self.bengaliEmbedding=Utilities.setDevice(self.bengaliEmbedding)
    


    def translateWordToTensor(self,word,vocabulary,language):
        '''
            Parameters:
                word : word on which to create the embeddings
                vocabulary : vocabulary of the dataset
                language : language of the dataset
            Returns :
                trans : embedding of the word
            Function:
                Generates the embeddings
        '''

        tensorList=list()
        if language=="english":
            tensorList.append(vocabulary.charToIndexDictForEnglish[vocabulary.startOfSequenceToken])
        else:
            tensorList.append(vocabulary.charToIndexDictForBengali[vocabulary.startOfSequenceToken])

        for one_char in word:
            if(language=="english"):
                tensorList.append(vocabulary.charToIndexDictForEnglish[one_char])
            else:
                tensorList.append(vocabulary.charToIndexDictForBengali[one_char])
        
        if language=="english":
            tensorList.append(vocabulary.charToIndexDictForEnglish[vocabulary.endOfSequenceToken])
        else:
            tensorList.append(vocabulary.charToIndexDictForBengali[vocabulary.endOfSequenceToken])
        
        trans=torch.tensor(tensorList,dtype=torch.int64)
        return trans