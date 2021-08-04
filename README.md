# 50045 Information Retrieval Project

This repository contains the code for our project. The project uses kaggle competition dataset from: https://www.kaggle.com/c/shopee-product-matching

### Dataset
We will be using train.csv and the provided images in train_images folder.

train.csv contains the following information for each product:
* posting_id
* image
* image_phash
* title
* label_group

posting_id: ID of each product <br />
image: file name of image of that product <br />
image_phash: perceptual hash of image. Refer to [Perceptual_hashing](https://en.wikipedia.org/wiki/Perceptual_hashing) for more information <br />
title: title of product <br />
label_group: label group, same for similar products

## Approaches
### TF-IDF and Cosine Similarity
Using TF-IDF and cosine similarity to measure how similar a title (document) is to the query. 

### BM25
BM25 is a ranking model used to find the relevance of each document with respect to a given query. Each title in the dataset is treated as an independent query and the corpus consisted of all 34250 title descriptions found in the entire dataset.

### Unigram Language Model
Each product title is taken as an individual document and the collection of all product titles were used as the entire collection.

The occurrences of the words were calculated for each document and for the entire collection, which were used to calculate the Unigram probability and interpolated sentence probability. The documents were then scored using the interpolated sentence probability and highest scoring documents were the most relevant ones.

### Bigram and Mixgram Language Model
Similar to Unigram Language Model, the Bigram Language Model instead calculates the occurrences of pair of words and uses those to calculate the probabilities. Start and stop words were added to account for single-word documents.

The Mixgram Language Model makes use of both Unigram and Bigram probabilities to calculate the final score of a document. 

### Image Search
Convolutional Neural Network was used for image search through the product images. AlexNet was the model for the neural network used. Similar looking images to the query image were returned as relevant files