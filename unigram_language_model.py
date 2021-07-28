# Functions are inspired by IR week 9 lab 

import numpy as np
import pandas as pd

class UnigramLanguageModel:
    def __init__(self, titles, smoothing=True):
        '''
            titles: list of lists of words (e.g. [["hello", "world"], ["how", "are", "you"]])
        '''
        
        self.titles = []
        self.smoothing = smoothing

        self.word_frequency = {}
        self.total_count = 0

        for each_prod in titles:
            new_title = []
            for each_word in each_prod:
                self.total_count += 1
                new_title.append(each_word.lower())

                if each_word not in self.word_frequency:
                    self.word_frequency[each_word] = 1
                else:
                    self.word_frequency[each_word] += 1
            
            self.titles.append(new_title)

    def calculate_unigram_probability(self, word):
        try:
            return self.word_frequency[word] / self.total_count
        except:
            return 0

    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        '''
            calculate score/probability of a sentence or query using the unigram language model.
            sentence: input sentence or query
            normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
        '''
        score = 1
        for each_word in sentence:
            score *= self.calculate_unigram_probability(each_word)
        
        return score
    

def calculate_interpolated_sentence_probability(sentence, doc, collection, alpha=0.75, normalize_probability=True):
    '''
        calculate interpolated sentence/query probability using both sentence and collection unigram models.
        sentence: input sentence/query
        doc: unigram language model a doc. HINT: this can be an instance of the UnigramLanguageModel class
        collection: unigram language model a collection. HINT: this can be an instance of the UnigramLanguageModel class
        alpha: the hyperparameter to combine the two probability scores coming from the document and collection language models.
        normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
    '''
    score = 1
    query = sentence.lower().split()
 
    if query in doc.titles:
        return 1

    for each_word in query:
        doc_prob = alpha * doc.calculate_unigram_probability(each_word)
        collection_prob = (1-alpha) * collection.calculate_unigram_probability(each_word)
        score *= (doc_prob + collection_prob)
    
    return score

def read_csv_titles(file_name):
    df = pd.read_csv(file_name)
    all_titles = df["title"]
    
    list_of_titles = all_titles.tolist()
    
    processed_titles = []

    # we are only seperating by strings, not accounting for stuff like '/' and '-'
    # so dashes will come up as "words" in this model

    # TODO: pre-process the words by making them lower case
    for each_title in list_of_titles:
        processed_titles.append(each_title.lower().split())
    
    return processed_titles

if __name__ == '__main__':
    train_file = 'train.csv'
    train_titles = read_csv_titles(train_file)

    smoothing=True

    train_model = UnigramLanguageModel(train_titles, smoothing) # train model is our entire corpus

    query = "Paper Bag Victoria Secret"
    #print("Smoothing:", smoothing)
    all_scores = []

    # for each title in train.csv, we make a model for them
    # interpolated score is then calculated for each title
    # retrieve top 5 scores

    # for doc in train_titles:
    for doc in range(len(train_titles)):
        current_model = UnigramLanguageModel([train_titles[doc]], smoothing)

        current_score = calculate_interpolated_sentence_probability(query, current_model, train_model)
        all_scores.append(current_score)

    top_five_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])[-5:]
    top_five_scores = sorted(all_scores)[-5:]
    
    for i in range(0, 5):
        print(f"{i+1}, Index: {top_five_indices[4-i]}, Score: {top_five_scores[4-i]}")