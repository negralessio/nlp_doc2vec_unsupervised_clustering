import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
sns.set()

class LoadAndPreprocessor:
    
    def __init__(self, PATH_TO_DATA:str, TARGET_COLUMN:str, sep=',', type='csv'):
        """
        PATH_TO_DATA:str Define the path to the data where your data lies
        TARGET_COLUMN:str Define the column that you want to be analyized.
        sep:str (default: ',') Seperator that seperates the data
        type:str (default: 'csv') Whether to read the data as a csv or excel file (xlsx)
        """
        self.PATH_TO_DATA = PATH_TO_DATA
        self.TARGET_COLUMN = TARGET_COLUMN
        if type=='csv':
            self.master = pd.read_csv(PATH_TO_DATA, sep = sep)
        if type=='xlsx':
            self.master = pd.read_excel(PATH_TO_DATA)
        self.original_shape = self.master.shape
        print(f"Data in {PATH_TO_DATA} has been sucessfully read.\nOriginal shape of DataFrame: {self.original_shape}\nTarget column: {self.TARGET_COLUMN}")

        
    def clean_data(self):
        """
        Function to do the first steps in the preprocessing: Dropping duplicates
        """
        n_before = self.master.shape[0]
        # Convert target column to string
        self.master[self.TARGET_COLUMN] = self.master[self.TARGET_COLUMN].astype(str)
        
        # Drop unnecessary columns
        if "Unnamed: 0" in self.master.columns:
            self.master.drop(["Unnamed: 0"], axis=1, inplace=True)
        
        # Drop Duplicates on column 'Content'
        self.master.drop_duplicates(subset=[self.TARGET_COLUMN], inplace=True)
        n_after = self.master.shape[0]
        print(f"Table Dimensions after dropping duplicates on '{self.TARGET_COLUMN}' column: {self.master.shape}")
        print(f"Removed rows by dropping duplicates: {n_before - n_after} ({round(((n_before - n_after)/n_before)*100, 2)}%)")
   

    def do_sentence_split(self, column:str, set_to_target_column:bool, name_of_new_sentence_column:str):
        """
        Function to do sentence split on whole customer feedback, i.e. on the column where the text to analyze is represented
        as a whole paragraph. 
        
        column:str Name of the column where the data is represented as a whole paragraph
        set_to_target_column:bool If set to True, the newly created sentence column will be set as the TARGET_COLUMN
        name_of_new_sentence_column:str Name of the new column that will be created during the sentence splitting
        """
        assert "sentence" not in self.master.columns, "Attention: There is already a column named 'sentence' in master."
        assert name_of_new_sentence_column not in self.master.columns, f"Attention: The column {name_of_new_sentence_column} already exists in master"

        # Set column to string
        self.master[column] = self.master[column].astype(str)
        
        # Apply sentence tokenizer to each row in the column parameter
        self.master[name_of_new_sentence_column] = self.master[column].apply(lambda x: sent_tokenize(x))
        # Get each list entry and put it into new row
        self.master = self.master.explode(name_of_new_sentence_column, ignore_index = True)
        
        # Set to target column
        if set_to_target_column == True:
            self.TARGET_COLUMN = name_of_new_sentence_column
    
    
    def print_head(self, n=5):
        """
        n:int Number of rows to show in the preview
        Simple function to print the thead of the master dataframe
        """
        return self.master.head(n)
    
    
    def cut_head_and_tail(self, a_percentile=2, b_percentile=95):
        """
        Function to remove outliers by removing all sentences where its length is SMALLER than the a_percentile 
        and removing all sentences where its length is GREATER than the b_percentile.
        Thus all sentences where len(sentence) is not in [a, b] will be cut
        
        Note that this function does not return a master dataframe. Instead it alters the instance attribute (self.master).
        To get the dataframe, call the get_master_df() function, after calling this method.
        
        a_percentile:int Define lower bound of len(sentence)
        b_percentile:int Define upper bound of len(sentence)
        """
        
        # Compute length list for each content entry
        length_list = []
        for doc in self.master[self.TARGET_COLUMN]:
            length_list.append(len(doc))

        # Append to master df
        self.master["content_length"] = length_list
        
        # Define lower and upper bound for docs length
        LOWER_BOUND = np.percentile(length_list, a_percentile)
        UPPER_BOUND = np.percentile(length_list, b_percentile)
        
        # Rounding
        LOWER_BOUND = round(LOWER_BOUND, 2)
        UPPER_BOUND = round(UPPER_BOUND, 2)

        # Plot Boxplot
        plt.figure(figsize=(12,5))
        plt.title("Boxplot of content_length (= number of letters)")
        sns.boxplot(data=self.master, x="content_length")
        #plt.xticks(np.arange(0, max(length_list), 100))
        # Add verticale lines indicating the lower and upper bound
        plt.axvline(x= LOWER_BOUND,linewidth=2, color='r',ymin=0.4, ymax=0.6)
        plt.axvline(x= UPPER_BOUND,linewidth=2, color='r',ymin=0.4, ymax=0.6)
        plt.show()

        # Get information about quartiles
        describe_content = self.master["content_length"].describe()
        print(describe_content)

        # Print information
        print(90 * "_")
        print(f"Lower Bound is {LOWER_BOUND} ({a_percentile}-percentile), Upper Bound is {UPPER_BOUND} ({b_percentile}-percentile) -- see red vlines")
        print(f"Master Shape before content length removal: {self.master.shape}")
        n_before = self.master.shape[0]

        # Remove all rows that are not in [LOWER_BOUND, UPPER_BOUND] Interval
        self.master = self.master[self.master["content_length"] >= LOWER_BOUND]
        self.master = self.master[self.master["content_length"] <= UPPER_BOUND]

        # Print information
        print(f"Master Shape after content length removal: {self.master.shape}")

        # Print information about delta
        n_after = self.master.shape[0]
        print(f"Removed rows: {n_before - n_after} ({round(((n_before - n_after)/n_before)*100, 2)}%)")
        print(90 * "_")
        
        
    def filter_master(self, cols_to_drop:list):
        """
        Dynamically change this function depending on your needs within the 'try'-block
        """ 
        try:
            self.master = self.master.drop(columns=cols_to_drop)
        except KeyError:
            print("Attention (KeyError): You called the filter_master() method and used filters on columns that " +  
                  "do not exist on this data set.\nThus no filter was applied.\n")


    def take_sample_matrix(self, n:int) -> pd.DataFrame:
        """
        Function to set global master df variable to a sample of that
        n:int Number of samples to take
        """
        self.master = self.master.sample(n=n)
        print(f"Sample of size {n} has been taken instead.")

        

    def get_master_df(self) -> pd.DataFrame:
        """
        Simple function that outputs the master dataframe. Should be called after applying all cleaning functions.
        """
        return self.master
    
    
    def get_target_column(self) -> str:
        return self.TARGET_COLUMN