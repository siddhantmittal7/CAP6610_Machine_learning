import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

#Training label
train_label = open('20news-bydate/matlab/train.label')

#Class Distribution Probability
cdp = {}

for i in range(1,21):
    cdp[i] = 0
    
lines = train_label.readlines()
total = len(lines)

for line in lines:
    val = int(line.split()[0])
    cdp[val] += 1

for key in pi:
    cdp[key] = float(cdp[key])/total
    
#Training data
train_data = open('20news-bydate/matlab/train.data')
df = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])

#Training label
label = []
train_label = open('20news-bydate/matlab/train.label')
lines = train_label.readlines()
for line in lines:
    label.append(int(line.split()[0]))

docIdx = df['docIdx'].values
i = 0
new_label = []
for index in range(len(docIdx)-1):
    new_label.append(label[i])
    if docIdx[index] != docIdx[index+1]:
        i += 1
new_label.append(label[i])

df['classIdx'] = new_label

#Alpha value for smoothing
a = 0.001

pb_ij = df.groupby(['classIdx','wordIdx'])
pb_j = df.groupby(['classIdx'])
Pr =  (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + 16689)    
Pr = Pr.unstack()

for c in range(1,21):
    Pr.loc[c,:] = Pr.loc[c,:].fillna(a/(pb_j['count'].sum()[c] + 16689))

Pr_dict = Pr.to_dict()

#Common stop words from online resources
stop_words = [
"a", "about", "above", "across", "after", "afterwards", 
"again", "all", "almost", "alone", "along", "already", "also",    
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", 
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]

vocab = open('20news-bydate/matlab/vocabulary.txt') 
vocab_df = pd.read_csv(vocab, names = ['word']) 
vocab_df = vocab_df.reset_index() 
vocab_df['index'] = vocab_df['index'].apply(lambda x: x+1) 

#Index of all words
tot_list = set(vocab_df['index'])

#Index of good words
vocab_df = vocab_df[~vocab_df['word'].isin(stop_words)]
good_list = vocab_df['index'].tolist()
good_list = set(good_list)

#Index of stop words
bad_list = tot_list - good_list

#Set all stop words to 0
for bad in bad_list:
    for j in range(1,21):
        Pr_dict[j][bad] = a/(pb_j['count'].sum()[j] + 16689)

#Part A
#Since we can ignore the word's count in a document that makes probability 
#of document d belonging to class j as multiplication of probability of each 
#word belonging to that class.

def ClassifierA(df):

    df_dict = df.to_dict()
    new_dict = {}
    prediction = []
    
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try: 
            new_dict[docIdx][wordIdx] = count 
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    for docIdx in range(1, len(new_dict)+1):
        score_dict = {}
        for classIdx in range(1,21):
            score_dict[classIdx] = 1
            for wordIdx in new_dict[docIdx]:
                try:
                    probability = Pr_dict[wordIdx][classIdx]                     
                    score_dict[classIdx]+=np.log(
                                           probability) 
                except:
                    score_dict[classIdx] += 0          
            score_dict[classIdx] +=  np.log(cdp[classIdx])                          

        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)
        
    return prediction

def ClassifierB(df):
    
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []
    
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try: 
            new_dict[docIdx][wordIdx] = count 
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    for docIdx in range(1, len(new_dict)+1):
        score_dict = {}
        for classIdx in range(1,21):
            score_dict[classIdx] = 1
            for wordIdx in new_dict[docIdx]:
                try:
                        probability = Pr_dict[wordIdx][classIdx]        
                        power = new_dict[docIdx][wordIdx]               
                        score_dict[classIdx]+=power*np.log(
                                           probability) 
                except:
                    score_dict[classIdx] += 0              
            score_dict[classIdx] +=  np.log(cdp[classIdx])                          

        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)
        
    return prediction

#Calculate IDF 
tot = len(df['docIdx'].unique()) 
pb_ij = df.groupby(['wordIdx']) 
IDF = np.log(tot/pb_ij['docIdx'].count()) 
IDF_dict = IDF.to_dict()

def ClassifierC(df):
    
    df_dict = df.to_dict()
    new_dict = {}
    prediction = []
    
    for idx in range(len(df_dict['docIdx'])):
        docIdx = df_dict['docIdx'][idx]
        wordIdx = df_dict['wordIdx'][idx]
        count = df_dict['count'][idx]
        try: 
            new_dict[docIdx][wordIdx] = count 
        except:
            new_dict[df_dict['docIdx'][idx]] = {}
            new_dict[docIdx][wordIdx] = count

    for docIdx in range(1, len(new_dict)+1):
        score_dict = {}
        for classIdx in range(1,21):
            score_dict[classIdx] = 1
            for wordIdx in new_dict[docIdx]:
                try:
                        probability = Pr_dict[wordIdx][classIdx]        
                        power = new_dict[docIdx][wordIdx]               
                        score_dict[classIdx]+= power*np.log(
                                   probability*IDF_dict[wordIdx]) 
                except:
                    score_dict[classIdx] += 0              
            score_dict[classIdx] +=  np.log(cdp[classIdx])                          

        max_score = max(score_dict, key=score_dict.get)
        prediction.append(max_score)
        
    return prediction

#Get test data
test_data = open('20news-bydate/matlab/test.data')
df = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])

#Get list of labels
test_label = pd.read_csv('20news-bydate/matlab/test.label', names=['t'])
test_label= test_label['t'].tolist()

predictA = ClassifierA(df)

total = len(test_label)
val = 0
for i,j in zip(predictA, test_label):
    if i == j:
        val +=1
    else:
        pass
ratio = float(val)/total
print("Prediction accuracy of classifier A:\t",(ratio * 100), "%")

predictB = ClassifierB(df)

total = len(test_label)
val = 0
for i,j in zip(predictB, test_label):
    if i == j:
        val +=1
    else:
        pass
ratio = float(val)/total
print("Prediction accuracy of classifier B:\t",(ratio * 100), "%")

predictC = ClassifierC(df)

total = len(test_label)
val = 0
for i,j in zip(predictC, test_label):
    if i == j:
        val +=1
    else:
        pass
ratio = float(val)/total
print("Prediction accuracy of classifier C:\t",(ratio * 100), "%")