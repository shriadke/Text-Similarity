# Fetch Rewards Coding Exercise - Text Similarity

## Task

This challenge is focused on the similarity between two texts. The objective is to write a program that takes as inputs two texts and uses a metric to determine how similar they are. Documents that are exactly the same should get a score of 1, and documents that don’t have any words in common should get a score of 0.

## File Structure
This repo contains python implementation of text similarity app. 

1. [textSimilarity.ipynb](https://github.com/shriadke/Text-Similarity/blob/main/textSimilarity.ipynb)
	The notebook  contains my initial approach towards this project and includes line by line implementation of my approach along with explanation.
	
2. [textSimilarity.py](https://github.com/shriadke/Text-Similarity/blob/main/textSimilarity.py)
	This is the Python module that encapsulates all the differen methods performed in this approach along with appropriate comments. This module is further used by the API service in [app.py](https://github.com/shriadke/Text-Similarity/blob/main/app.py)

3. [app.py](https://github.com/shriadke/Text-Similarity/blob/main/app.py)
	This is the API POST service implementation of textSimilarity app that uses `get_text_similarity(texts)` method to compute the similarity matrix between 3 given texts.

4. [templates/index.html](https://github.com/shriadke/Text-Similarity/blob/main/templates/index.html)
	This is the web page which will be loaded to use the above service through a web browser.
	
5. [requirements.txt](https://github.com/shriadke/Text-Similarity/blob/main/requirements.txt)
	This file contains the necessary packages required for this project.

6. [Dockerfile](https://github.com/shriadke/Text-Similarity/blob/main/Dockerfile)
	This file contains build instructions for API deployment using Docker.
	
7. [Procfile](https://github.com/shriadke/Text-Similarity/blob/main/Procfile)
	This file is used to deploy the app using Heroku platform.

## Approach

Here I have considered 3 different basic text similarity approaches that are easy to implement without the use of external libraries such as Scikit-Learn, NLTK, Gensim, Spacy, etc. These approaches consider individual texts as a list of tokenized words and performs mathematical similarity operations. The details of all of these can be found in [Approach](https://github.com/shriadke/Text-Similarity/blob/main/docs/APPROACH.md). Follwing are the steps involved in this project:

1. Data Cleaning
	
	Tokenization and filtering words to create text vocabulary
	
2. Data Processing

	Vectorization to compute similarity based on word vectors.

3. Text Similarity Metrics

	Three metrics used to compute text similarity: Jaccard Index, Cosine Similarity, Euclidian Distance.
