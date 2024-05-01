import torch
import torch.nn as nn

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class WordEmbeddings:
    def createWordEmbeddings(self,dataset,vocabulary):
        englishDataset=dataset[:,0]
        bengaliDataset=dataset[:,1]

        tensorListEnglish=list()
        tensorListBengali=list()

        language="english"
        for one_word in englishDataset:
            tensor=self.translateWordToTensor(one_word,vocabulary,language).to(device=device)
            tensorListEnglish.append(tensor)
        self.englishEmbedding=nn.utils.rnn.pad_sequence(tensorListEnglish,padding_value=vocabulary.paddingIndex,batch_first=True).to(device=device)

        language="bengali"
        for one_word in bengaliDataset:
            tensor=self.translateWordToTensor(one_word,vocabulary,language).to(device=device)
            tensorListBengali.append(tensor)
        self.bengaliEmbedding=nn.utils.rnn.pad_sequence(tensorListBengali,padding_value=vocabulary.paddingIndex,batch_first=True).to(device=device)
    
    def translateWordToTensor(self,word,vocabulary,language):
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
        
        return torch.tensor(tensorList,dtype=torch.int64)