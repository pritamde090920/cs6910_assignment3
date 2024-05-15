
'''class to prepare the vocabulary of the dataset'''
class PrepareVocabulary:

    '''constructor to intialize the class parameters'''
    def __init__(self):

        '''define the start token, end token and padding token'''
        self.startOfSequenceToken="~"
        self.endOfSequenceToken="%"
        self.paddingToken="`"
        self.startOfSequenceIndex=0
        self.endOfSequenceIndex=1
        self.paddingIndex=2

        '''current vocabulary size is 3 (start token, end token, padding token)'''
        self.vocabularySizeForEnglish=3
        self.vocabularySizeForBengali=3

        self.charToIndexDictForEnglish=dict()
        self.indexToCharDictForEnglish=dict()
        self.charCounterForEnglish=dict()

        self.charToIndexDictForBengali=dict()
        self.indexToCharDictForBengali=dict()
        self.charCounterForBengali=dict()

        self.initializeVocabularyDictionaries()
    


    def initializeVocabularyDictionaries(self):
        '''
            Parameters:
                None
            Returns :
                None
            Function:
                Initializes the vocabulary dictionaries
        '''

        '''dictionary for source language'''
        self.charToIndexDictForEnglish[self.startOfSequenceToken]=self.startOfSequenceIndex
        self.charToIndexDictForEnglish[self.endOfSequenceToken]=self.endOfSequenceIndex
        self.charToIndexDictForEnglish[self.paddingToken]=self.paddingIndex

        self.indexToCharDictForEnglish[self.startOfSequenceIndex]=self.startOfSequenceToken
        self.indexToCharDictForEnglish[self.endOfSequenceIndex]=self.endOfSequenceToken
        self.indexToCharDictForEnglish[self.paddingIndex]=self.paddingToken

        '''dictionary for target language'''
        self.charToIndexDictForBengali[self.startOfSequenceToken]=self.startOfSequenceIndex
        self.charToIndexDictForBengali[self.endOfSequenceToken]=self.endOfSequenceIndex
        self.charToIndexDictForBengali[self.paddingToken]=self.paddingIndex

        self.indexToCharDictForBengali[self.startOfSequenceIndex]=self.startOfSequenceToken
        self.indexToCharDictForBengali[self.endOfSequenceIndex]=self.endOfSequenceToken
        self.indexToCharDictForBengali[self.paddingIndex]=self.paddingToken
    


    def createVocabulary(self,dataset):
        '''
            Parameters:
                dataset : dataset on which to create the vocabulary
            Returns :
                None
            Function:
                creates vocabulary of each word in the dataset
        '''

        '''iterate over the entire dataset'''
        for each_pair in dataset:
            english_word=each_pair[0]
            bengali_word=each_pair[1]

            '''create vocabulary for the source language'''
            for one_char in english_word:
                '''if the character is not already recorded then add it to the dictionary'''
                if one_char not in self.charToIndexDictForEnglish:
                    self.charToIndexDictForEnglish[one_char]=self.vocabularySizeForEnglish
                    self.charCounterForEnglish[one_char]=1
                    self.indexToCharDictForEnglish[self.vocabularySizeForEnglish]=one_char
                    self.vocabularySizeForEnglish+=1
                else:
                    self.charCounterForEnglish[one_char]+=1
            
            '''create vocabulary for the target language'''
            for one_char in bengali_word:
                '''if the character is not already recorded then add it to the dictionary'''
                if one_char not in self.charToIndexDictForBengali:
                    self.charToIndexDictForBengali[one_char]=self.vocabularySizeForBengali
                    self.charCounterForBengali[one_char]=1
                    self.indexToCharDictForBengali[self.vocabularySizeForBengali]=one_char
                    self.vocabularySizeForBengali+=1
                else:
                    self.charCounterForBengali[one_char]+=1