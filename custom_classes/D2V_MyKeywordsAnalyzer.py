from math import floor
import operator
from nltk import bigrams
from nltk.util import trigrams
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()


class MyKeywordsAnalyzer:
    
    def __init__(self, X_transformed_df:pd.DataFrame, n_clusters:int, n_keywords:int, verbose:bool, n_grams:int, palette:tuple):
        self.X_transformed_df = X_transformed_df
        self.n_clusters = n_clusters
        self.n_keywords = n_keywords
        self.verbose = verbose
        self.n_grams = n_grams
        self.palette = palette
        
        # Cluster occurence for keywords plotting
        self.cluster_occurence = dict(self.X_transformed_df["kmeans_cluster"].value_counts())
        
        # Call function in constructor to get global keywords list that is required in further functions
        self.keywords_list = self.get_keywords_list()
        
    
    def get_keywords(self, c:int, n = 25, verbose = False, n_grams = 2):
        """
        c:int Indicates from which cluster c we want to have our keywords
        n:int Indicates how many top n keyword shall be extracted (descending order)
        verbose:bool Indicates whether or not the values shall be printed during the process
        n_grams:int Indicates the size of the n-grams windows

        returns descending dictionary with keys being the words/n-grams and values the number of occurences
        """

        # Throw print error if cluster c does not exist
        if c > self.n_clusters:
            print (f"Attention: Cluster {c} does not exist. Must be in [0, {n_clusters})")
            return 0

        # Extract only token and cluster label from given cluster c
        X_filtered = self.X_transformed_df[self.X_transformed_df["kmeans_cluster"] == c]
        X_filtered = X_filtered[["kmeans_cluster", "tokens"]]

        # Get n_gram list by iterating through each cell in tokens column
        onegram_list, bigram_list, trigram_list = [], [], []
        for cell in X_filtered["tokens"]:
            onegram_list.append(cell)
            bigram_list.append(list(bigrams(cell)))
            trigram_list.append(list(trigrams(cell)))

        # Flatten n-gram_lists
        if n_grams==1:
            flattened = [val for sublist in onegram_list for val in sublist]
        elif n_grams==2:
            flattened = [val for sublist in bigram_list for val in sublist]
        elif n_grams==3:
            flattened = [val for sublist in trigram_list for val in sublist]
        else:
            print("Error: n_grams parameter must be in [1, 3]")
            return 0

        # Count Appearenced of words
        word_dict = dict(Counter(flattened))
        # Sort descending
        word_dict_sorted = sorted(word_dict.items(),key=operator.itemgetter(1),reverse=True)

        # If verbose is set to True (default is False) print all words
        if verbose == True:
            for i in range(0, n):
                print(word_dict_sorted[i])

        res = word_dict_sorted[: n]
        return dict(res)

    
    def get_keywords_list(self):
        """
        Computes keywords list for all clusters and stores it into the 'keywords_list' (instance variable)
        """
        keywords_list = []
        for i in range(0, self.n_clusters):
            keywords_cluster_i = self.get_keywords(c = i, n = self.n_keywords, n_grams=self.n_grams)
            keywords_list.append(keywords_cluster_i)
            
        return keywords_list
            
            
    def plot_keywords_all_clusters(self):
        """
        Functions to plot barplots of keywords for all clusters
        """

        # Define run variable for plot title
        i = 0
        for keywords in self.keywords_list:
            plt.figure(figsize=(30,9))
            plt.title(f"Barplot of top n = {len(keywords)} keywords / {self.n_grams}-Grams of cluster {i} (n_cluster = {self.cluster_occurence.get(i)})")
            sns.barplot(y=list(map(str, list(keywords.keys()))), x=list(keywords.values()), color=self.palette[i], orient='h')
            plt.show()
            i = i + 1
        
        
    def plot_keywords(self, c: int):
        """
        c:int Indicates which keywords of cluster c to plot
        Functions to plot keywords for a specific cluster c
        """

        keywords = self.keywords_list[c]
        plt.figure(figsize=(30,9))
        plt.title(f"Barplot of top n = {len(keywords)} keywords / {self.n_grams}-Grams of cluster {c} (n_cluster = {self.cluster_occurence.get(c)})")
        sns.barplot(y=list(map(str, list(keywords.keys()))), x=list(keywords.values()), color=self.palette[c], orient='h')
        plt.show()
        

    def plot_keywords_grid(self, n_cols = 2):
        """
        Function to print the keywords in a grid
        n_cols:int (default: 2) Number of columns per row
        """

        i, k = 0, 0
        for keywords in self.keywords_list:
            if i % n_cols == 0:
                fig, axs = plt.subplots(ncols=n_cols, figsize=(30, 7))
            axs[k].title.set_text(f"Barplot of top n = {len(keywords)} keywords / {self.n_grams}-Grams of cluster {i} (n_cluster = {self.cluster_occurence.get(i)})")
            sns.barplot(y=list(map(str, list(keywords.keys()))), x=list(keywords.values()), color=self.palette[i], orient='h', ax=axs[k])
            fig.tight_layout()
            i = i + 1
            k = k + 1
            if k == n_cols:
                k = 0

