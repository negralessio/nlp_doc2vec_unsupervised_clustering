import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class MyTFIDFAnalyzer:
    
    def __init__(self, X_transformed_df:pd.DataFrame, n_clusters:int, target_column:str, max_docfreq:float, stopwords:list, tfidf_n_output:int, tfidf_n_grams:tuple):
        self.X_transformed_df = X_transformed_df
        self.n_clusters = n_clusters
        self.target_column = target_column
        self.max_docfreq = max_docfreq
        self.stopwords_final = stopwords
        self.tfidf_n_output = tfidf_n_output
        self.tfidf_n_grams = tfidf_n_grams
        self.tfidf_keyword_list = self.fill_global_keyword_list_variable()

        
    def get_top_n_keyword_of_cluster_c(self, c:int, n: int, n_grams:tuple, verbose:bool):
        """
        Function to print the top n keywords with the highsest tfidf score
        for a input cluster c.

        c:int Cluster to consider
        n:int Number of top n keywords with highest tfidf score
        """
        # Filter matrix
        X = self.X_transformed_df[self.X_transformed_df["kmeans_cluster"] == c]

        if verbose == True:
            print(80 * "_")
            print(f"Cluster c = {c} filtered matrix X has shape {X.shape}")
            print(f"Print top n = {n} keywords based on tfidf-score:\n")

        # Extract docs from X
        docs = X[self.target_column].values
        docs = list(docs)

        vectorizer = TfidfVectorizer(lowercase=True, max_df=self.max_docfreq, stop_words=self.stopwords_final, ngram_range=self.tfidf_n_grams)
        tfidf_matrix = vectorizer.fit_transform(docs)

        importance = np.argsort(np.asarray(tfidf_matrix.sum(axis=0)).ravel())[::-1]
        tfidf_feature_names = np.array(vectorizer.get_feature_names())
        return tfidf_feature_names[importance[:n]]
    
    
    def print_all_tfidf_keywords(self):
        """
        Function to print all tfidf keywords and store it into global list 'tfidf_keyword_list'
        """
        self.tfidf_keyword_list = []
        
        # Get tfidf-keywords for each cluster c and store it into list
        for i in range(0, self.n_clusters):
            res = self.get_top_n_keyword_of_cluster_c(c=i, n=self.tfidf_n_output, n_grams=self.tfidf_n_grams, verbose=True)
            self.tfidf_keyword_list.append(res)
            print(res)
            
            
    def print_tfidf_keywords_of_cluster_c(self, c:int):
        """
        Function to print only tfidf keywords of cluster c specified as the parameter
        """
        res = self.get_top_n_keyword_of_cluster_c(c=c, n=self.tfidf_n_output, n_grams=self.tfidf_n_grams, verbose=True)
        print(res)
        
        
    def fill_global_keyword_list_variable(self) -> list:
        """
        Assist method to fill the instance variable 'tfidf_keyword_list' after an object is instantiated.
        """
        res_list = []
        for i in range(0, self.n_clusters):
            res = self.get_top_n_keyword_of_cluster_c(c=i, n=self.tfidf_n_output, n_grams=self.tfidf_n_grams, verbose=False)
            res_list.append(res)
        return res_list
            
    
    def get_tfidf_keywords_list(self) -> list:
        return self.tfidf_keyword_list