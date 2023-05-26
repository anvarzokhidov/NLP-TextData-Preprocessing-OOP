import numpy as np
import pandas as pd
import nltk

class data_manipulation():
    """
    Performs manipulations with datasets
    """

    def __init__(self, dataset = None):
        """
        Assigns the dataset into an object variable
        """
        self.dataset = dataset


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def preprocess(self, df_train):
        """
        Performs preprocessing operations on raw textual data initially reading and assigning into list 
        column of the pandas dataframe
            Steps:
                1. Lowercase
                2. Tokenization
                3. Stopwords removal
                4. Stemming

        :param
            df_train (pd dataframe): pandas dataframe with the training data.
        :return:
            Preprocessed data (nested list)
        """

        
        # ~~~~~~~~~~~~~~~~~~ Lowercase
        self.df_train_text = df_train['text'].tolist().copy()  # getting and copying the text column data = x data points
        self.df_train_text = [str(x).lower() for x in self.df_train_text]  # Lowercasing the dataset


        
        # ~~~~~~~~~~~~~~~~~~ Tokenization
        nltk.download('punkt') 
        from nltk import word_tokenize  # import tokenizer method from the nltk package

        self.train_tokenized = []  # list to keep the tokenized sentences. Needed to create a 2-dimensional list
        for sentence in self.df_train_text:
            self.train_tokenized.append(word_tokenize(sentence))  # store tokenized sentence in a list


        # ~~~~~~~~~~~~~~~~~ Stopwords removal
        from nltk.corpus import stopwords  # stopword method of nltk package
        self.stop_words_nltk = stopwords.words('english')  # getting the french stopwords form

        self.worded = []  # list to keep the words of each sentence
        self.tokenized_stopworded = []  # list to keep previously tokenized and currently got rid of stopwords sentences
        self.stop_words = ["whatever", "word",] # defining our own stop words
        for sentence in self.train_tokenized:  # iterating through sentences
            for word in sentence:  # itereating through the words of the sentences
                if word not in self.stop_words and word not in self.stop_words_nltk: 
                    self.worded.append(word)  # if the word is absent in the list of stop_words than append it to a list

            self.tokenized_stopworded.append(self.worded)  # append the sentence without stop words into a new sentence list
            self.worded = []  # clean the list to get rid of the previous information (sentence)


        # ~~~~~~~~~~~~~~~~~~ Stemming
        from nltk.stem.porter import PorterStemmer # import stemmer from the nltk library
        self.stemmer = PorterStemmer() # assigning object to a stemmer (PorterStemmer)
        self.processed_dataset = [] # instantiating precessed_dataset object
        for sentence in self.tokenized_stopworded:
            self.processed_dataset.append([self.stemmer.stem(word) for word in sentence]) # stemming and appending processed_dataset object

        return self.processed_dataset # returning preprocessed dataset after different preprocessing steps listed above
