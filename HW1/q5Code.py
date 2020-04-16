{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import operator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Training label\n",
    "train_label = open('20news-bydate/matlab/train.label')\n",
    "\n",
    "#Class Distribution Probability\n",
    "cdp = {}\n",
    "\n",
    "for i in range(1,21):\n",
    "    cdp[i] = 0\n",
    "    \n",
    "lines = train_label.readlines()\n",
    "total = len(lines)\n",
    "\n",
    "for line in lines:\n",
    "    val = int(line.split()[0])\n",
    "    cdp[val] += 1\n",
    "\n",
    "for key in pi:\n",
    "    cdp[key] = float(cdp[key])/total\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#Training data\n",
    "train_data = open('20news-bydate/matlab/train.data')\n",
    "df = pd.read_csv(train_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])\n",
    "\n",
    "#Training label\n",
    "label = []\n",
    "train_label = open('20news-bydate/matlab/train.label')\n",
    "lines = train_label.readlines()\n",
    "for line in lines:\n",
    "    label.append(int(line.split()[0]))\n",
    "\n",
    "docIdx = df['docIdx'].values\n",
    "i = 0\n",
    "new_label = []\n",
    "for index in range(len(docIdx)-1):\n",
    "    new_label.append(label[i])\n",
    "    if docIdx[index] != docIdx[index+1]:\n",
    "        i += 1\n",
    "new_label.append(label[i])\n",
    "\n",
    "df['classIdx'] = new_label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Alpha value for smoothing\n",
    "a = 0.001\n",
    "\n",
    "pb_ij = df.groupby(['classIdx','wordIdx'])\n",
    "pb_j = df.groupby(['classIdx'])\n",
    "Pr =  (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + 16689)    \n",
    "Pr = Pr.unstack()\n",
    "\n",
    "for c in range(1,21):\n",
    "    Pr.loc[c,:] = Pr.loc[c,:].fillna(a/(pb_j['count'].sum()[c] + 16689))\n",
    "\n",
    "Pr_dict = Pr.to_dict()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Common stop words from online resources\n",
    "stop_words = [\n",
    "\"a\", \"about\", \"above\", \"across\", \"after\", \"afterwards\", \n",
    "\"again\", \"all\", \"almost\", \"alone\", \"along\", \"already\", \"also\",    \n",
    "\"although\", \"always\", \"am\", \"among\", \"amongst\", \"amoungst\", \"amount\", \"an\", \"and\", \"another\", \"any\", \"anyhow\", \"anyone\", \"anything\", \"anyway\", \"anywhere\", \"are\", \"as\", \"at\", \"be\", \"became\", \"because\", \"become\",\"becomes\", \"becoming\", \"been\", \"before\", \"behind\", \"being\", \"beside\", \"besides\", \"between\", \"beyond\", \"both\", \"but\", \"by\",\"can\", \"cannot\", \"cant\", \"could\", \"couldnt\", \"de\", \"describe\", \"do\", \"done\", \"each\", \"eg\", \"either\", \"else\", \"enough\", \"etc\", \"even\", \"ever\", \"every\", \"everyone\", \"everything\", \"everywhere\", \"except\", \"few\", \"find\",\"for\",\"found\", \"four\", \"from\", \"further\", \"get\", \"give\", \"go\", \"had\", \"has\", \"hasnt\", \"have\", \"he\", \"hence\", \"her\", \"here\", \"hereafter\", \"hereby\", \"herein\", \"hereupon\", \"hers\", \"herself\", \"him\", \"himself\", \"his\", \"how\", \"however\", \"i\", \"ie\", \"if\", \"in\", \"indeed\", \"is\", \"it\", \"its\", \"itself\", \"keep\", \"least\", \"less\", \"ltd\", \"made\", \"many\", \"may\", \"me\", \"meanwhile\", \"might\", \"mine\", \"more\", \"moreover\", \"most\", \"mostly\", \"much\", \"must\", \"my\", \"myself\", \"name\", \"namely\", \"neither\", \"never\", \"nevertheless\", \"next\",\"no\", \"nobody\", \"none\", \"noone\", \"nor\", \"not\", \"nothing\", \"now\", \"nowhere\", \"of\", \"off\", \"often\", \"on\", \"once\", \"one\", \"only\", \"onto\", \"or\", \"other\", \"others\", \"otherwise\", \"our\", \"ours\", \"ourselves\", \"out\", \"over\", \"own\", \"part\",\"perhaps\", \"please\", \"put\", \"rather\", \"re\", \"same\", \"see\", \"seem\", \"seemed\", \"seeming\", \"seems\", \"she\", \"should\",\"since\", \"sincere\",\"so\", \"some\", \"somehow\", \"someone\", \"something\", \"sometime\", \"sometimes\", \"somewhere\", \"still\", \"such\", \"take\",\"than\", \"that\", \"the\", \"their\", \"them\", \"themselves\", \"then\", \"thence\", \"there\", \"thereafter\", \"thereby\", \"therefore\", \"therein\", \"thereupon\", \"these\", \"they\",\n",
    "\"this\", \"those\", \"though\", \"through\", \"throughout\",\n",
    "\"thru\", \"thus\", \"to\", \"together\", \"too\", \"toward\", \"towards\",\n",
    "\"under\", \"until\", \"up\", \"upon\", \"us\",\n",
    "\"very\", \"was\", \"we\", \"well\", \"were\", \"what\", \"whatever\", \"when\",\n",
    "\"whence\", \"whenever\", \"where\", \"whereafter\", \"whereas\", \"whereby\",\n",
    "\"wherein\", \"whereupon\", \"wherever\", \"whether\", \"which\", \"while\", \n",
    "\"who\", \"whoever\", \"whom\", \"whose\", \"why\", \"will\", \"with\",\n",
    "\"within\", \"without\", \"would\", \"yet\", \"you\", \"your\", \"yours\", \"yourself\", \"yourselves\"\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "vocab = open('20news-bydate/matlab/vocabulary.txt') \n",
    "vocab_df = pd.read_csv(vocab, names = ['word']) \n",
    "vocab_df = vocab_df.reset_index() \n",
    "vocab_df['index'] = vocab_df['index'].apply(lambda x: x+1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Index of all words\n",
    "tot_list = set(vocab_df['index'])\n",
    "\n",
    "#Index of good words\n",
    "vocab_df = vocab_df[~vocab_df['word'].isin(stop_words)]\n",
    "good_list = vocab_df['index'].tolist()\n",
    "good_list = set(good_list)\n",
    "\n",
    "#Index of stop words\n",
    "bad_list = tot_list - good_list\n",
    "\n",
    "#Set all stop words to 0\n",
    "for bad in bad_list:\n",
    "    for j in range(1,21):\n",
    "        Pr_dict[j][bad] = a/(pb_j['count'].sum()[j] + 16689)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Part A\n",
    "#Since we can ignore the word's count in a document that makes probability \n",
    "#of document d belonging to class j as multiplication of probability of each \n",
    "#word belonging to that class.\n",
    "\n",
    "def ClassifierA(df):\n",
    "\n",
    "    df_dict = df.to_dict()\n",
    "    new_dict = {}\n",
    "    prediction = []\n",
    "    \n",
    "    for idx in range(len(df_dict['docIdx'])):\n",
    "        docIdx = df_dict['docIdx'][idx]\n",
    "        wordIdx = df_dict['wordIdx'][idx]\n",
    "        count = df_dict['count'][idx]\n",
    "        try: \n",
    "            new_dict[docIdx][wordIdx] = count \n",
    "        except:\n",
    "            new_dict[df_dict['docIdx'][idx]] = {}\n",
    "            new_dict[docIdx][wordIdx] = count\n",
    "\n",
    "    for docIdx in range(1, len(new_dict)+1):\n",
    "        score_dict = {}\n",
    "        for classIdx in range(1,21):\n",
    "            score_dict[classIdx] = 1\n",
    "            for wordIdx in new_dict[docIdx]:\n",
    "                try:\n",
    "                    probability = Pr_dict[wordIdx][classIdx]                     \n",
    "                    score_dict[classIdx]+=np.log(\n",
    "                                           probability) \n",
    "                except:\n",
    "                    score_dict[classIdx] += 0          \n",
    "            score_dict[classIdx] +=  np.log(cdp[classIdx])                          \n",
    "\n",
    "        max_score = max(score_dict, key=score_dict.get)\n",
    "        prediction.append(max_score)\n",
    "        \n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "def ClassifierB(df):\n",
    "    \n",
    "    df_dict = df.to_dict()\n",
    "    new_dict = {}\n",
    "    prediction = []\n",
    "    \n",
    "    for idx in range(len(df_dict['docIdx'])):\n",
    "        docIdx = df_dict['docIdx'][idx]\n",
    "        wordIdx = df_dict['wordIdx'][idx]\n",
    "        count = df_dict['count'][idx]\n",
    "        try: \n",
    "            new_dict[docIdx][wordIdx] = count \n",
    "        except:\n",
    "            new_dict[df_dict['docIdx'][idx]] = {}\n",
    "            new_dict[docIdx][wordIdx] = count\n",
    "\n",
    "    for docIdx in range(1, len(new_dict)+1):\n",
    "        score_dict = {}\n",
    "        for classIdx in range(1,21):\n",
    "            score_dict[classIdx] = 1\n",
    "            for wordIdx in new_dict[docIdx]:\n",
    "                try:\n",
    "                        probability = Pr_dict[wordIdx][classIdx]        \n",
    "                        power = new_dict[docIdx][wordIdx]               \n",
    "                        score_dict[classIdx]+=power*np.log(\n",
    "                                           probability) \n",
    "                except:\n",
    "                    score_dict[classIdx] += 0              \n",
    "            score_dict[classIdx] +=  np.log(cdp[classIdx])                          \n",
    "\n",
    "        max_score = max(score_dict, key=score_dict.get)\n",
    "        prediction.append(max_score)\n",
    "        \n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Calculate IDF \n",
    "tot = len(df['docIdx'].unique()) \n",
    "pb_ij = df.groupby(['wordIdx']) \n",
    "IDF = np.log(tot/pb_ij['docIdx'].count()) \n",
    "IDF_dict = IDF.to_dict()\n",
    "\n",
    "def ClassifierC(df):\n",
    "    \n",
    "    df_dict = df.to_dict()\n",
    "    new_dict = {}\n",
    "    prediction = []\n",
    "    \n",
    "    for idx in range(len(df_dict['docIdx'])):\n",
    "        docIdx = df_dict['docIdx'][idx]\n",
    "        wordIdx = df_dict['wordIdx'][idx]\n",
    "        count = df_dict['count'][idx]\n",
    "        try: \n",
    "            new_dict[docIdx][wordIdx] = count \n",
    "        except:\n",
    "            new_dict[df_dict['docIdx'][idx]] = {}\n",
    "            new_dict[docIdx][wordIdx] = count\n",
    "\n",
    "    for docIdx in range(1, len(new_dict)+1):\n",
    "        score_dict = {}\n",
    "        for classIdx in range(1,21):\n",
    "            score_dict[classIdx] = 1\n",
    "            for wordIdx in new_dict[docIdx]:\n",
    "                try:\n",
    "                        probability = Pr_dict[wordIdx][classIdx]        \n",
    "                        power = new_dict[docIdx][wordIdx]               \n",
    "                        score_dict[classIdx]+= power*np.log(\n",
    "                                   probability*IDF_dict[wordIdx]) \n",
    "                except:\n",
    "                    score_dict[classIdx] += 0              \n",
    "            score_dict[classIdx] +=  np.log(cdp[classIdx])                          \n",
    "\n",
    "        max_score = max(score_dict, key=score_dict.get)\n",
    "        prediction.append(max_score)\n",
    "        \n",
    "    return prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Prediction accuracy of classifier A:\\t', 79.00066622251832, '%')\n",
      "('Prediction accuracy of classifier B:\\t', 79.14723517654897, '%')\n",
      "('Prediction accuracy of classifier C:\\t', 79.14723517654897, '%')\n"
     ]
    }
   ],
   "source": [
    "#Get test data\n",
    "test_data = open('20news-bydate/matlab/test.data')\n",
    "df = pd.read_csv(test_data, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])\n",
    "\n",
    "#Get list of labels\n",
    "test_label = pd.read_csv('20news-bydate/matlab/test.label', names=['t'])\n",
    "test_label= test_label['t'].tolist()\n",
    "\n",
    "predictA = ClassifierA(df)\n",
    "\n",
    "total = len(test_label)\n",
    "val = 0\n",
    "for i,j in zip(predictA, test_label):\n",
    "    if i == j:\n",
    "        val +=1\n",
    "    else:\n",
    "        pass\n",
    "ratio = float(val)/total\n",
    "print(\"Prediction accuracy of classifier A:\\t\",(ratio * 100), \"%\")\n",
    "\n",
    "predictB = ClassifierB(df)\n",
    "\n",
    "total = len(test_label)\n",
    "val = 0\n",
    "for i,j in zip(predictB, test_label):\n",
    "    if i == j:\n",
    "        val +=1\n",
    "    else:\n",
    "        pass\n",
    "ratio = float(val)/total\n",
    "print(\"Prediction accuracy of classifier B:\\t\",(ratio * 100), \"%\")\n",
    "\n",
    "predictC = ClassifierC(df)\n",
    "\n",
    "total = len(test_label)\n",
    "val = 0\n",
    "for i,j in zip(predictC, test_label):\n",
    "    if i == j:\n",
    "        val +=1\n",
    "    else:\n",
    "        pass\n",
    "ratio = float(val)/total\n",
    "print(\"Prediction accuracy of classifier C:\\t\",(ratio * 100), \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
