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

Here I have considered 3 different basic text similarity approaches that are easy to implement without the use of external libraries such as Scikit-Learn, NLTK, Gensim, Spacy, etc. These approaches considers individual texts as a list of tokenized words and performs mathematical similarity operations. Following are the steps implemented in my solution:

1. Data Cleaning
	- Word/Sentence Tokenization [`def tokenize_sents(text, stop_char=".")`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L47) and [`def tokenize_words_from_sent(sents, stop_words=[], punct=[])`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L50)
		* The sentences or words can be extracted form the given text strings using delimiters such as "."(period for sentences), " "(blank space for words).
		* To get tokenized words, I have tokenized the words from individual sentences.
		* The sentences are extracted and the final period(".") is padded with space for the case in which punctuations is not removed.
		
	- Expansion of common terms [`def decontracted(text)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L54)
		* It is very common in english language to use contractions of 2 words such as "we'll" --> "we will".
		* While computing text similarity, these words can make a difference if they appear in both texts, expanded in one and contracted in other.
		* Thus, we need to expand these contractions in order to consider the similarity in both the texts.
		* I have considered common contractions and replaced these using regular expression.
		
	- Removing Stop words and Punctuations
		* In most of the NLP tasks, stop words and punctutations are removed in order to get the relations between core words and remove extraneous information.
		* This implementations supports such functionality as well as these terms can be allowed by excluding them from removal.
		* This is implemented in [`def tokenize_words_from_sent(sents, stop_words=[], punct=[])`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L50) where one can specify custom stop words and punctuations if want to rermove.
		* By default, these will not be removed. For simplicity, the module has two lists containing a few common stop words(NOT ALL, others can be found on NLTK stopwords data) and punctuations.
		* It was observed that for smaller texts, considering both of these gives better similarity results as they matter in context as well.
		* For example, texts "I am Shrinidhi.", "Am I Shrinidhi?" will give a perfect similarity score if we remove "I", "am", ".", "?". But instead if we consider these, we will get a different result.
		* Thus, in my view, punctuations and stop words matters and should be considered.
		
	- Building a Custom Vocabulary [`def prepare_vocab(all_words)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L70)
		* Considering all the separate texts contributes to our entire vocabulary/corpus, the single vocabulary is created.
		* The original problem statement asks to consider only two texts at a time, but this is my assumption that we can input any number of texts for comparison and all of them will be considered as a part of single corpus.
		* This is done considering the real world use case of computing text similarity in which there will always be a huge corpus containing multiple documents.
		* For example, a product can be sold at different places with different descriptions but in the end product is the same. Consider, Walmart sell "Coke" with a description of "A refreshing, sweetened, carbonated soft drink.". Kroger sell it with "Coke, is a carbonated soft drink manufactured by The Coca-Cola Company." as description. Target has "It is a carbonated, sweet soft drink and is the world's best-selling drink.". In all the cases, considering all three descriptions together helps us to have a larger vocabulary that contributes in similarity comparison.
		
2. Data Processing

	It is important for computation that the word data is in some form of numbers. The most common way of achieving it is Vectorization. In my approach I have used [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) as a feature vector to represent the individual texts with the custom vocabulary/corpus. However, the choice of a vectorizer depends on application, TF is good for general similarity but TF-IDF is better for identifying relevance between multiple texts as well.
	
	- Term Frequency [`def get_word_tf(sorted_vocab, vocab_to_int, words)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L79)
		* The term frequency is the normalized frequency of individual word in an individual text/document in our corpus.
		* Term Frequency = (# of occurrences of a word in a document) / (total words in the document).
		* In many cases of text similarity, only TF is sufficient to get a satisfactory result. But it may lead to inaccurate results if the number of documents are significantly high.
		
	- Inverse Document Frequency [`def get_word_idf(all_words, n_texts, sorted_vocab, vocab_to_int)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L88)
		* Inverse Document Frequency portrays the information about the commonness of a word in the corpus/ all documents.
		* IDF = (Total number of Documents)/ (Total number of documents containing word w)
	
	- TF-IDF [`def get_tfidf_vectors(word_tf, word_idf)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L100)
		* This is a simple product of TFs and IDFs for individual words.
		* We can obtain an n-dimensional tf-idf vector representing individual text, where n is number of distinct words in the custom vocabulary.

3. Text Similarity Metrics

	Once we have a vector representation, we can compute similarity based on various vector similarity methods. Following are the three methods/metrics implemented in this project:
	
	- Jaccard Index [`def get_jaccard_sim(words1, words2)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L108)
		* This method does not need word vector representations. The tokens are enough to get the index.
		* It is the Intersection over Union (IoU) for two lists of words (one for each text).
		* IoU = (Intersection of two lists) / (Union of two lists)
		* IoU represents the similarity between texts.
		
	- Cosine Similarity [`def get_cosine_similarity(x,y)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L118)
		* This method needs word vector representations.
		* After converting words to vectors, we can compute the similarity between the vectors by getting the cosine of the angle between them.
		* The dot product of two vectors gives us the cosine of the angle between them.
		* The dot product is calculated as (sum of products of individual vector components) / (product of magnitudes of two vectors)
		
	- Euclidian Distance [`def get_euclidian_similarity(x,y)`](https://github.com/shriadke/Text-Similarity/blob/76d0d496f4ed0fa8f12ce75a7a4f776f6ca40091/textSimilarity.py#L124)
		* It is nothng but the direct distance between the two points in space.
		* As we have word vectors, we can compute the distance between each vector component using distance formula.
		* euclidian distance = sq.root(sum over all components i (xi - yi)^2)
		* This gives the direct distance but we need a metric in 0 to 1.
		* This is achieved by introducing a threshold value.
		* The similarity = (threshold - distance) / threshold
