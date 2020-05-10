# -*- coding: utf-8 -*-
"""homework5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cngkvovnpNCoYdWSrJt6QJ5M8m_nAc-l
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.cm as cm
from six.moves import urllib

from sklearn.feature_extraction.text import TfidfVectorizer
newsgroups_train = fetch_20newsgroups(subset='train')
data_target = newsgroups_train.target
vectorizer = TfidfVectorizer(stop_words='english')
TFIDF = vectorizer.fit_transform(newsgroups_train.data)

data = TFIDF.toarray().T

url = "http://qwone.com/~jason/20Newsgroups/vocabulary.txt"
file = urllib.request.urlopen(url)

dict = {}
i=0
for line in file:
  dict[i] = line.decode("utf-8")
  i = i + 1

print("Data loaded for 20 news groups usign TF-IDF and removing english stop words. Also created the vocabulary dict")

class MultinomialMixture:
    def __init__(self, c, k):

        self.C = c
        self.K = k
        self.smoothing = 0.001

    def train(self, dataset, threshold=0, max_epochs=10):

        likelihood_list = []
        current_epoch = 1
        old_likelihood = - np.inf
        delta = np.inf

        # Initialisation of the model's parameters.
        # probility of each class
        pr = np.random.uniform(size=self.C)
        pr = pr / np.sum(pr)
        self.pi = pr

        self.p = np.empty((self.K, self.C))
        for i in range(0, self.C):
            em = np.random.uniform(size=self.K)
            em = em / np.sum(em)
            self.p[:, i] = em

        while current_epoch <= max_epochs and delta > threshold:
            # E-step
            posterior_estimate = np.divide(np.multiply(self.p[dataset, :], np.reshape(self.pi, (1, self.C))), 
                                           np.dot(self.p[dataset, :], np.reshape(self.pi, (self.C, 1))))
            # Compute the likelihood
            likelihood = np.sum(np.log(self.p[dataset, :]*np.reshape(self.pi, (1, self.C))))
            likelihood_list.append(likelihood)
            # M-step
            self.pi = np.divide(self.smoothing + np.sum(posterior_estimate, axis=0), 
                                self.smoothing * self.C + np.sum(posterior_estimate))
            self.p = np.divide(np.add.at(self.smoothing + np.zeros((self.K, self.C)), dataset, posterior_estimate), 
                               np.reshape(self.smoothing * self.K + np.sum(posterior_estimate[:, :], axis=0), (1, self.C)))
            delta = likelihood - old_likelihood
            old_likelihood = likelihood
            current_epoch += 1
        return likelihood_list

    def predict(self, prediction_set):
        prods = self.p[prediction_set, :]*np.reshape(self.pi, (1, self.C))
        return np.argmax(prods, axis=1)

    def mean(self, size):
        mean = []
        for _ in range(0, size):
            state = np.random.choice(np.arange(0, self.C), p=self.pi)
            emitted_label = np.random.choice(np.arange(0, self.K), p=self.p[:, state])
            mean.append(emitted_label)
        return mean

C = 11314
K = 20
dim_dataset = 129796

mixture = MultinomialMixture(C, K)
mixture.train(data, threshold=0.01)

mean = model.means_

theta_mean = np.matmul(data,mean.transpose())

for i in range(20):
    result = []
    temp = theta_mean[:,i]
    temp_index = temp.argsort()[-10:][::-1]
    print("Top 10 key words for cluster:",i)
    for j in temp_index:
      result.append(dict[j])
    print(result)