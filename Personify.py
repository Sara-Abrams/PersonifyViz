import nltk
import string
import pandas as pd
import colorsys
import csv
import numpy as np
import colorsys
#nltk.download('vader_lexicon')
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#from nltk import vader_lexicon
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
stem = PorterStemmer()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
StopWords = stopwords.words('english')
SIA = SentimentIntensityAnalyzer()

def clean_tokens (text_input):
    tokens = [word for word in text_input]
    #print("**************")
    #print(tokens)
    for t in tokens:
        if t in StopWords:
            tokens.remove(t)
    #print(tokens)
    for i,t in enumerate(tokens):
        tokens[i] = lem.lemmatize(t, "v")
    return (tokens)

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

ClientJournals = pd.read_csv("dcleaned.csv", sep = ",", header=None)
ClientEntries = ClientJournals[0].tolist()
ClientEntries = ''.join(ClientEntries)
sentences = tokenize.sent_tokenize(ClientEntries)

MasterWerd = {}

for sent in sentences:
    #print (sent)
    
    pos = SIA.polarity_scores(sent)['pos']
    neg = SIA.polarity_scores(sent)['neg']
    neu = SIA.polarity_scores(sent)['neu']
    compound = SIA.polarity_scores(sent)['compound']
    #print(pos, neg, neu, compound)
    
    ##Strip Punctuation
    WordTokens = [word.strip(string.punctuation) for word in sent.split()]
    
    #print(WordTokens)
    ##Lowercase
    WordTokens = [word.lower() for word in WordTokens]
    
    #print(WordTokens)
    ##Remove Stopwords
    cleanWordTokens = clean_tokens(WordTokens)
    
    #print("#########")
    print (cleanWordTokens)
    for token in cleanWordTokens:
        if token in MasterWerd.keys():
            #print(1)
            (num, pos_list, neg_list) = MasterWerd[token]
            (num, pos_list.append(pos), neg_list.append(neg))
            MasterWerd[token] = (num+1, pos_list, neg_list)
        else:
            #print(2)
            MasterWerd[token] = (1, [pos], [neg])

test = pd.DataFrame.from_dict(MasterWerd, orient = 'index')
test['pos'] = np.nan
test['neg'] = np.nan

##Get average positive and negative emotion
trows = len(test.index)
for i in range(0,trows):
    test.iloc[i,3]=sum(test.iloc[i,1])/test.iloc[i,0]
    test.iloc[i,4]=sum(test.iloc[i,2])/test.iloc[i,0]

#Relative Overall Sentiment
test['val'] = test['pos'] - test['neg']

##Calculate Hue (from green to blue)
test['hue'] = test['val']*180 + 180

##Calculate Saturation, based on total amount of positive and negative (not avg)
for i in range(0,trows):
    test.loc[test.index[i],'sat']=sum(test.iloc[i,1])+sum(test.iloc[i,2])

#total 
test['satperc'] = test['sat']/test[0]

for i in range(0,trows):
    rgb = hsv2rgb(test.loc[test.index[i],'hue'], test.loc[test.index[i],'satperc'], .80)
    test.loc[test.index[i],'red'] = rgb[0]
    test.loc[test.index[i],'green'] = rgb[1]
    test.loc[test.index[i],'blue'] = rgb[2]


