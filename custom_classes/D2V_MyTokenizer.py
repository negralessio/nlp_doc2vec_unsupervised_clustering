import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import random
import re
import string
nltk.download('wordnet')
nltk.download("stopwords")
nltk.download("punkt")

class MyTokenizer:
    
    def __init__(self, master:pd.DataFrame, TARGET_COLUMN:str, custom_stopwords:list):
        self.master = master
        self.TARGET_COLUMN = TARGET_COLUMN
        self.custom_stopwords = custom_stopwords
        self.stopwords_final:list
        
        
    def get_tfidf_stopwords(self, max_docfreq = 0.2, print_interval = False, lower_max_df = 0.05, upper_max_df = 0.5):
        """
        Function to find and print tfidf based stopwords.
        
        max_docfreq:float Ignore terms that have a document frequency strictly higher than the given threshold
        print_interval:bool Parameter whether to print and find tfidf based stopwords based on a given max_df interval
        lower_max_df:float Lower bound of the search interval
        upper_max_df:float Upper bound of the search interval
        """
        # Get content from TARGET_COLUMN
        docs = self.master[self.TARGET_COLUMN].values
        docs = list(docs)

        # If print_interval is set to False, only take and print one tfidf iteration
        if print_interval == False:
            # Init TFIDF Vectorizer
            vectorizer = TfidfVectorizer(lowercase=True, max_df=max_docfreq, stop_words = set(stopwords.words('english')))
            tfidf_matrix = vectorizer.fit_transform(docs)
            # Get stopwords, i.e. terms with DF that are higher than MAX_DF
            self.tfidf_stopwords = vectorizer.stop_words_
            self.tfidf_stopwords = list(self.tfidf_stopwords)
            # Print Info
            print(120 * "_")
            print(f"Identified Stop Words through TFIDF (max_df = {max_docfreq}):")
            print(self.tfidf_stopwords)
            print(120 * "_")
        
        # If print_interval is set to True, print all tfidf iteration in max_df_range
        if print_interval == True:
            # get nparray of max_df [0.05, 0.10, 0.15, ...]
            steps = 0.05
            max_df_range = np.arange(lower_max_df, upper_max_df+steps, steps)
            # round to two decimals
            max_df_range = np.around(max_df_range, 2)
            # sort descending
            max_df_range[::-1].sort()
            for i in max_df_range:
                # Init TFIDF Vectorizer
                vectorizer = TfidfVectorizer(lowercase=True, max_df=i, stop_words = set(stopwords.words('english')))
                tfidf_matrix = vectorizer.fit_transform(docs)
                # Get stopwords, i.e. terms with DF that are higher than MAX_DF
                self.tfidf_stopwords = vectorizer.stop_words_
                self.tfidf_stopwords = list(self.tfidf_stopwords)
                # Print Info
                print(f"Identified Stop Words through TFIDF (max_df = {i}):")
                print(self.tfidf_stopwords)
                print(120 * "_")
                
            # However, take the max_df as global variable for custom tfidf stopwords (even if print_interval is set to True)
            vectorizer = TfidfVectorizer(lowercase=True, max_df=max_docfreq, stop_words = set(stopwords.words('english')))
            tfidf_matrix = vectorizer.fit_transform(docs)
            # Get stopwords, i.e. terms with DF that are higher than MAX_DF
            self.tfidf_stopwords = vectorizer.stop_words_
            self.tfidf_stopwords = list(self.tfidf_stopwords)
            print(f"Attention: print_interval was set to True. Taking the tfidf stopwords yielded by setting max_df = {max_docfreq}")
        
        # Safety variable
        self.did_tfidf_run = True
        
        
    def create_stopword_list(self, stopwords_to_remove = []):
        """
        Creates and prints out information about the final stopword list.
        Final stopword list is created by using the nltk predefined stopwords, plus our own custom stopwords,
        plus our own tfidf stopwords, minus the stopwords to remove
        
        stopwords_to_remove:list Stopwords you want to remove from the final stopword list
        """
        assert self.did_tfidf_run, "Attention: TFIDF based stopwords has not been searched for"
            
        # Add custom stopwords to the imported stopwords list
        self.stopwords_final = set(stopwords.words('english') + self.custom_stopwords + self.tfidf_stopwords)
            
        # Remove stopwords
        self.stopwords_final = [x for x in self.stopwords_final if x not in stopwords_to_remove]

        # Preview
        print(120 * "_")
        print("TFIDF stopwords:")
        print(f"{self.tfidf_stopwords}\n")
        print("Custom stopwords:")
        print(f"{self.custom_stopwords}\n")
        print("Stopwords to remove:")
        print(f"{stopwords_to_remove}\n")
        print("Review of all stopwords:")
        print(f"{self.stopwords_final}\n")
        print(f"Length of all stopwords: {len(self.stopwords_final)}")
        print(120 * "_")
        
        
    def clean_text(self, text, tokenizer, stopwords):
        """
        Important function to preprocess target column and generate tokens.
        
        text: Text to tokenize (usually the sentence column of master df)
        Returns: Tokenized text.
        """
        text = str(text).lower()  # Lowercase words
        text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
        text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
        text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
        text = re.sub(
            f"[{re.escape(string.punctuation)}]", "", text
        )  # Remove punctuation

        # Create lemmatizer
        lemmatizer = WordNetLemmatizer()

        tokens = tokenizer(text)  # Get tokens from text
        tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Apply Lemmatizing
        tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
        tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
        return tokens
    
    
    def create_tokens(self, min_length_of_a_token = 1):
        """
        Function to create tokens by calling the clean_text() function. Appends the results to the master dataframe
        """
        # Apply clean text function for each cell in 'Content' column to create tokens
        self.master["tokens"] = self.master[self.TARGET_COLUMN].map(lambda x: self.clean_text(x, word_tokenize, self.stopwords_final))
        
        # Delete rows where token list is smaller than min_length_of_a_token
        self.master = self.master.loc[self.master["tokens"].map(lambda x: len(x) >= min_length_of_a_token)]

        # Reset index
        self.master.reset_index(inplace=True)
        
        # Drop unnecessary index
        self.master = self.master.drop(["index"], axis = 1)
        
        # Safety Variable
        self.did_create_tokens_run = True
    
        
    def get_master_df(self) -> pd.DataFrame:
        """
        Simple function that outputs the master dataframe after being processed by the class
        """
        assert self.did_create_tokens_run, "Attention: Token has not been created yet. Call 'create_tokens' before!"
        return self.master
    
    def get_final_stopwords_list(self) -> list:
        """
        Simple function to return the final stopwords list
        """
        return self.stopwords_final
        