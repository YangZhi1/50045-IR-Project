import numpy as np
import pandas as pd

# for the start and end of title
START_TITLE = '<t>'
END_TITLE = '</t>'

class BigramLanguageModel:
    def __init__(self, titles, smoothing=True):
        '''
            titles: list of list of titles. Each title should start with <s> and end with </s>
                    e.g. ['<s>', 'Nano', 'water', 'spray', '</s>']
        '''

        self.titles = titles
        self.smoothing = smoothing

        self.occurrences = {}
        self.total_word_count = 0

        for each_title in self.titles:
            for word_index in range(0, len(each_title), 2):
                self.total_word_count += 1
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
    
    def calculate_bigram_probability(self, pair_of_words):
        '''
            pair_of_words (STRING): can be a single word or a pair of words separated by a space 
        '''
        try:
            toreturn = self.occurrences[pair_of_words]/self.total_word_count
        except:
            if(self.smoothing):
                toreturn = 1/(self.total_word_count + len(self.occurrences))
            else:
                toreturn = 0
        return toreturn
    
    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        '''
            sentence: can be a string separated by spaces or a list of words
        '''
        total_probability = 1

        if(type(sentence) == str):
            processed_sentence = sentence.split()
        else:
            processed_sentence = []
            for word_index in range(0, len(sentence), 2):
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
    score = 1

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
        return 1
    
    score = 1
    for each_pair in processed_sentence_list:
        doc_prob = alpha * doc.calculate_bigram_probability(each_pair)
        collection_prob = (1-alpha) * collection.calculate_bigram_probability(each_pair)
        score *= (doc_prob + collection_prob)

    return score


def read_csv_titles(file_name):
    df = pd.read_csv(file_name)
    all_titles = df["processed title"]
    
    list_of_titles = all_titles.tolist()
    
    processed_titles = []

    # we are only seperating by strings, not accounting for stuff like '/' and '-'
    # so dashes will come up as "words" in this model

    # TODO: pre-process the words by making them lower case
    for each_title in list_of_titles:
        each_title = each_title.lower()
        processed_titles.append(each_title.split())

    return processed_titles
    
# tit = [["<s>", "Hello", "WoRlD"], ["<s>", "nano", "spray"], ["<s>", "nano", "end", "</s>"], ["<s>", "Hello", "</s>"]]
# a = BigramLanguageModel(tit)

if __name__ == "__main__":
    train_file = "train.csv"
    processed_file = "processed_unigram.csv"

    smoothing = False

    query = "Paper BAG victoria secret"
    all_scores = []

    collection_model = BigramLanguageModel(processed_file)
    train_titles = read_csv_titles(processed_file)

    for doc in train_titles:
        current_model = BigramLanguageModel([doc], smoothing)
        current_score = calculate_interpolated_sentence_probability(query, current_model, collection_model)
        all_scores.append(current_score)
    
    top_five_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i])[-5:]
    top_five_scores = sorted(all_scores)[-5:]

    for i in range(0, 5):
        print(f"{i+1}, Index: {top_five_indices[4-i]}, Score: {top_five_scores[4-i]}")

    