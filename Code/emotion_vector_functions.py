
import pandas as pd
# from operator import itemgetter

# nltk is what does all the magic language parsing
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk


import numpy as np
import re

# for stop words
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# On your first time running this program, you'll need to run the following line to download the stuff from nltk to your local machine
# nltk.download(['stopwords', 'wordnet'])

stops:list[str] = list(set(ENGLISH_STOP_WORDS))


# define the file which contains all the words with their associated emotion + intensities
fileEmotion = "./Code/emotion_itensity.txt"
table = pd.read_csv(fileEmotion,  names=["word", "emotion", "intensity"], sep='\t')

# create a dictionary with the word/emotion/score
emotion_dic = dict()
lmtzr = WordNetLemmatizer()
for index, row in table.iterrows():
    word, emotion, intensity = row['word'], row['emotion'], row['intensity']
    
    # add the given word to our dictionary
    temp_key = word + "#" + emotion
    emotion_dic[temp_key] = intensity

    # add in the normal noun form
    temp_key_n = lmtzr.lemmatize(word) + '#' + emotion;
    emotion_dic[temp_key_n] = intensity
    
    # add in the normal verb form
    temp_key_v = lmtzr.lemmatize(word, 'v') + '#' + emotion
    emotion_dic[temp_key_v] = intensity
        


def getEmotionItensity(word:str, emotion:str):
    key = word + "#" + emotion
    try:
        return emotion_dic[key]
    except:
        return 0.0
        
#Check if the word is in the Lexicon
def isWordInEmotionFile(word:str):
    # Slightly faster implementation
    for key in emotion_dic.keys():
        if key.startswith(word + "#"):
            return True
    return False

#Stopping checker 
def isStopWord(word:str):
    return word in stops

#Assign the emotion itensity to the dictionary
def calculateEmotion(emotions:dict[str,float], word:str):
    emotions["Anger"] += getEmotionItensity(word, "anger")
    emotions["Anticipation"] += getEmotionItensity(word, "anticipation")
    emotions["Disgust"] += getEmotionItensity(word, "disgust")
    emotions["Fear"] += getEmotionItensity(word, "fear")
    emotions["Joy"] += getEmotionItensity(word, "joy")
    emotions["Sadness"] += getEmotionItensity(word, "sadness")
    emotions["Surprise"] += getEmotionItensity(word, "surprise")
    emotions["Trust"] += getEmotionItensity(word, "trust")
    
def cleanString(text:str):        
    clean = re.sub("[^a-zA-Z]+", " ", text) # replace all non-letters with a space
    clean = re.sub("[^a-zA-Z ]+", '', clean).lower() #  convert to lowercase
    
    return clean
    
#get the emotion vector of a given text
def getEmotionVector(text:str, removeObj = False, useSynset = True) -> dict[str, float]:
    
    #  create the initial emotions
    emotions = {"Anger": 0.0,
                "Anticipation": 0.0,
                "Disgust": 0.0,
                "Fear": 0.0,
                "Joy": 0.0,
                "Sadness": 0.0,
                "Surprise": 0.0,
                "Trust": 0.0,
                "Objective": 0.0}
    
    # parse the description
    str = cleanString(text)

    # split string into words
    tokens = str.split()

    
    #iterate over words array
    for word in tokens:
        if not isStopWord(word):
            #first check if the word appears as it does in the text
            if isWordInEmotionFile(word): 
                calculateEmotion(emotions, word)
                
            # check the word in noun form (bats -> bat)
            elif isWordInEmotionFile(lmtzr.lemmatize(word)):
                calculateEmotion(emotions, lmtzr.lemmatize(word))
                
            # check the word in verb form (ran/running -> run)
            elif isWordInEmotionFile(lmtzr.lemmatize(word, 'v')):
                calculateEmotion(emotions, lmtzr.lemmatize(word, 'v'))  
                
            # check synonyms of this word
            elif useSynset and wordnet.synsets(word) is not None:
                # only check the first two "senses" of a word, so we don't stray too far from its intended meaning
                for syn in wordnet.synsets(word)[0:1]:
                    for l in syn.lemmas():
                        if isWordInEmotionFile(l.name()):
                            calculateEmotion(emotions, l.name())
                            continue
                            
                # none of the synonyms matched something in the file
                emotions["Objective"] += 1
                
            else:
                # not found in the emotion file, assign a score to Objective instead
                emotions["Objective"] += 1

    # remove the Objective category if requested
    if removeObj:
        del emotions['Objective']
        
    # normalize the emotion vector
    total = sum(emotions.values())
    for key in sorted(emotions.keys()):
        try:
            emotions[key] /= total
        except:
            emotions[key] = 0

    return emotions
