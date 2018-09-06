trainSetf = open('F:\\ABCD\\PESIT\\Year4\\Sem7\\NLP\\Final\\train.txt','r')
datasetf = open('F:\\ABCD\\PESIT\\Year4\\Sem7\\NLP\\Final\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\datasetSentences.txt','r')
phraseSetf = open('F:\\ABCD\\PESIT\\Year4\\Sem7\\NLP\\Final\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\dictionary.txt','r')
sentPhrase = open('F:\\ABCD\\PESIT\\Year4\\Sem7\\NLP\\Final\\stanfordSentimentTreebank\\stanfordSentimentTreebank\\sentiment_labels.txt','r')

phraseType = {1:[],2:[],3:[]}
phraseId = dict()
trainDataString=""
for i in trainSetf.readlines():
    trainDataString+=i.strip()+" "
#print trainDataString[:1000]

for i in phraseSetf.readlines():
    k = i.strip().split('|')
    if len(k[0])>2 and k[0] in trainDataString:
        phraseType[1].append(k[0])
        phraseId[k[1]] = k[0]
        if k[1]=="!":
            print k

phraseSent = dict()
sentPhrase.readline()
for i in sentPhrase.readlines():
    k = i.strip().split('|')
    phraseSent[k[0]] = k[1]

f = open('phrase_train.txt','w')

for i in phraseId:
    phraseId[i]
    phraseSent[i]
    f.write(phraseId[i]+"|"+phraseSent[i]+"\n")

trainSetf.close()
datasetf.close()
phraseSetf.close()
sentPhrase.close()
f.close()

