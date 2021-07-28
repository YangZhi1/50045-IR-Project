# combination of both bigram and unigram, but with a higher weight for bigram

import numpy as np
import pandas as pd

class BigramLanguageModel:
    def __init__(self, titles, smoothing=True):
        '''
            titles: list of list of titles. Each title should start with <s> and end with </s>
                    e.g. ['<s>', 'Nano', 'water', 'spray', '</s>']
        '''

        self.titles = titles
        self.smoothing = smoothing

        self.unigram_occurrences = {}
        self.occurrences = {}
        self.total_unigram_word_count = 0
        self.total_bigram_word_count = 0

        for each_title in self.titles:
            for word_index in range(0, len(each_title)):
                ####### FOR BIGRAM CALCULATION #######
                self.total_bigram_word_count += 1
                # two_words could be '<s> Nano'
                # or just '</s>' if it is the last word
                try:
                    two_words = (each_title[word_index] + ' ' + each_title[word_index+1]).lower()
                except:
                    two_words = each_title[word_index].lower()
                
                if two_words in self.occurrences:
                    self.occurrences[two_words] += 1
                else:
                    self.occurrences[two_words] = 1

                ####### FOR UNIGRAM CALCULATION #######
                if(word_index == 0 or word_index == len(each_title)-1):
                    pass
                else:
                    self.total_unigram_word_count +=1
                    if(each_title[word_index] not in self.unigram_occurrences):
                        self.unigram_occurrences[each_title[word_index]] = 1
                    else:
                        self.unigram_occurrences[each_title[word_index]] += 1

    def calculate_unigram_probability(self, word):
        # includes add-one smoothing so we do not return 0 for words that are not present even in the collection
        try:
            return self.unigram_occurrences[word] / self.total_unigram_word_count
        except:
            if(self.smoothing):
                return 1 / (self.total_unigram_word_count + len(self.unigram_occurrences))
            return 0

    def calculate_bigram_probability(self, pair_of_words):
        '''
            pair_of_words (STRING): can be a single word or a pair of words separated by a space 
        '''
        try:
            return self.occurrences[pair_of_words]/self.total_bigram_word_count
        except:
            if(self.smoothing):
                return 1 / (self.total_bigram_word_count + len(self.occurrences))
            return 0
    
    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        '''
            sentence: can be a string separated by spaces or a list of words
        '''
        total_probability = 1

        if(type(sentence) == str):
            processed_sentence = sentence.split()
        else:
            processed_sentence = []
            for word_index in range(0, len(sentence)):
                try:
                    new_word = sentence[word_index] + sentence[word_index+1]
                except:
                    new_word = sentence[word_index]
                processed_sentence.append(new_word)
        
        for each_word in processed_sentence:
            word_prob = self.calculate_bigram_probability(each_word)

            if(normalize_probability):
                total_probability *= np.log2(word_prob)
            else:
                total_probability *= word_prob
            
        return total_probability

def calculate_interpolated_sentence_probability(sentence, doc, collection, alpha=0.75, normalize_probability=True):
    '''
        calculate interpolated sentence/query probability using both sentence and collection unigram models.
        sentence: input sentence/query
        doc: unigram language model a doc. HINT: this can be an instance of the UnigramLanguageModel class
        collection: unigram language model a collection. HINT: this can be an instance of the UnigramLanguageModel class
        alpha: the hyperparameter to combine the two probability scores coming from the document and collection language models.
        normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
    '''
    processed_sentence_list = []

    if(type(sentence) == str):
        processed_sentence = "<s> " + sentence + " </s>"
        temp_list = processed_sentence.lower().split()
        for i in range(len(temp_list)):
            try:
                two_words = (temp_list[i] + ' ' +  temp_list[i+1]).lower()
            except:
                two_words = (temp_list[i]).lower()
            processed_sentence_list.append(two_words)
    else:
        temp_list = [each_string.lower() for each_string in sentence]
        temp_list.insert(0, "<s>")
        temp_list.append("</s>")
        for i in range(len(temp_list)):
            try:
                two_words = (temp_list[i] + ' ' + temp_list[i+1]).lower()
            except:
                two_words = (temp_list[i]).lower()
            processed_sentence_list.append(two_words)

    if(temp_list in doc.titles):
        return 1, 1


    ###### CALCULATION OF UNIGRAM SCORE ######
    unigram_score = 1
    query = sentence.lower().split()

    for each_word in query:
        doc_prob = alpha * doc.calculate_unigram_probability(each_word)
        collection_prob = (1-alpha) * collection.calculate_unigram_probability(each_word)
        unigram_score *= (doc_prob + collection_prob)
    
    ####### CALCULATION OF BIGRAM SCORE #######
    bigram_score = 1
    for each_pair in processed_sentence_list:
        doc_prob = alpha * doc.calculate_bigram_probability(each_pair)
        collection_prob = (1-alpha) * collection.calculate_bigram_probability(each_pair)
        bigram_score *= (doc_prob + collection_prob)

    beta = 0.9
    score = (beta * bigram_score) + (1-beta) * unigram_score

    return unigram_score, bigram_score


def read_csv_titles(file_name):
    df = pd.read_csv(file_name)
    all_titles = df["processed title"]
    
    list_of_titles = all_titles.tolist()
    
    processed_titles = []

    # we are only seperating by strings, not accounting for stuff like '/' and '-'
    # so dashes will come up as "words" in this model
    for each_title in list_of_titles:
        each_title = each_title.lower()
        processed_titles.append(each_title.split())

    return processed_titles
    
# tit = [["<s>", "Hello", "WoRlD"], ["<s>", "nano", "spray"], ["<s>", "nano", "end", "</s>"], ["<s>", "Hello", "</s>"]]
# a = BigramLanguageModel(tit)

if __name__ == "__main__":
    processed_file = "processed_unigram.csv"

    smoothing = True

    query = "mens body lotion"
    
    normalized_scores = []
    all_unigram_scores = []
    all_bigram_scores = []

    train_titles = read_csv_titles(processed_file)
    collection_model = BigramLanguageModel(train_titles)
    
    beta = 0.9    

    for doc_index in range(len(train_titles)):
        current_model = BigramLanguageModel([train_titles[doc_index]], smoothing)

        unigram_score, bigram_score = calculate_interpolated_sentence_probability(query, current_model, collection_model)
        all_unigram_scores.append(unigram_score)
        all_bigram_scores.append(bigram_score)

    # normalize the unigram and bigram scores
    s = sum(all_unigram_scores)
    normalized_unigram = [float(i)/s for i in all_unigram_scores]

    s = sum(all_bigram_scores)
    normalized_bigram = [float(i)/s for i in all_bigram_scores]

    for i in range(len(normalized_unigram)):
        new_normalized_score = (beta * normalized_bigram[i]) + (1 - beta) * normalized_unigram[i]
        normalized_scores.append(new_normalized_score)
    
    top_five_indices = sorted(range(len(normalized_scores)), key=lambda i: normalized_scores[i])[-5:]
    top_five_scores = sorted(normalized_scores)[-5:]

    for i in range(0, 5):
        print(f"{i+1}, Index: {top_five_indices[4-i]}, Score: {top_five_scores[4-i]}")

    