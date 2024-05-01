class PrepareVocabulary:
    def __init__(self):
        self.startOfSequenceToken="~"
        self.endOfSequenceToken="%"
        self.paddingToken="`"
        self.startOfSequenceIndex=0
        self.endOfSequenceIndex=1
        self.paddingIndex=2


        self.vocabularySizeForEnglish=3 #startToken,endToken,paddingToken
        self.vocabularySizeForBengali=3 #startToken,endToken,paddingToken

        self.charToIndexDictForEnglish=dict()
        self.indexToCharDictForEnglish=dict()
        self.charCounterForEnglish=dict()

        self.charToIndexDictForBengali=dict()
        self.indexToCharDictForBengali=dict()
        self.charCounterForBengali=dict()

        self.initializeVocabularyDictionaries()
    
    def initializeVocabularyDictionaries(self):
        self.charToIndexDictForEnglish[self.startOfSequenceToken]=self.startOfSequenceIndex
        self.charToIndexDictForEnglish[self.endOfSequenceToken]=self.endOfSequenceIndex
        self.charToIndexDictForEnglish[self.paddingToken]=self.paddingIndex

        self.indexToCharDictForEnglish[self.startOfSequenceIndex]=self.startOfSequenceToken
        self.indexToCharDictForEnglish[self.endOfSequenceIndex]=self.endOfSequenceToken
        self.indexToCharDictForEnglish[self.paddingIndex]=self.paddingToken

        self.charToIndexDictForBengali[self.startOfSequenceToken]=self.startOfSequenceIndex
        self.charToIndexDictForBengali[self.endOfSequenceToken]=self.endOfSequenceIndex
        self.charToIndexDictForBengali[self.paddingToken]=self.paddingIndex

        self.indexToCharDictForBengali[self.startOfSequenceIndex]=self.startOfSequenceToken
        self.indexToCharDictForBengali[self.endOfSequenceIndex]=self.endOfSequenceToken
        self.indexToCharDictForBengali[self.paddingIndex]=self.paddingToken
    
    def createVocabulary(self,dataset):
        for each_pair in dataset:
            english_word=each_pair[0]
            bengali_word=each_pair[1]
            for one_char in english_word:
                if one_char not in self.charToIndexDictForEnglish:
                    self.charToIndexDictForEnglish[one_char]=self.vocabularySizeForEnglish
                    self.charCounterForEnglish[one_char]=1
                    self.indexToCharDictForEnglish[self.vocabularySizeForEnglish]=one_char
                    self.vocabularySizeForEnglish+=1
                else:
                    self.charCounterForEnglish[one_char]+=1
                
            for one_char in bengali_word:
                if one_char not in self.charToIndexDictForBengali:
                    self.charToIndexDictForBengali[one_char]=self.vocabularySizeForBengali
                    self.charCounterForBengali[one_char]=1
                    self.indexToCharDictForBengali[self.vocabularySizeForBengali]=one_char
                    self.vocabularySizeForBengali+=1
                else:
                    self.charCounterForBengali[one_char]+=1