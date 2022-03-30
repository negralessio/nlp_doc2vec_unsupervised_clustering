from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pandas as pd
import re
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class MyDoc2Vec:
    
    def __init__(self, master:pd.DataFrame, name_of_token_column = "tokens", target_column = "sentence"):
        self.master = master
        self.name_of_token_column = name_of_token_column
        self.target_column = target_column
        self.doc2vec_model:Doc2Vec
        
        
    def use_pretrained_model(self, path_to_pretrained_model:str, allow_finetune = True, train_epochs = 50, epochs_of_infering_vectors = 300) -> pd.DataFrame:
        """
        Function that loads the pretrained model. It is optional (default True) to allow finetuning of the master model. 
        The embedded documents will then be infered by the infer_vector() method from gensim and appended to the master dataframe.
        
        path_to_pretrained_model:str Define the path to the pretrained model to use
        allow_finetune:bool Whether to allow finetuning or not; Finetuning adjusts the weights of the current master model
        train_epochs:int Number of epochs to train using the model.train() function. Only relevant if allow_finetune is set to True.
        epochs_of_infering_vectors:int Number of epochs to use for infering vectors. The higher it is, the more stable the vectors become.
        
        return Master df with DocEmbedding column appended
        """
        # Count time to execute
        start_time = time.time()
        
        # Load pretrained doc2vec model
        self.doc2vec_model = Doc2Vec.load(path_to_pretrained_model)
        
        # Create TaggedDocument Object for training
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.master[self.name_of_token_column])]
        
        # If allow_finetune is set to True: Finetune the model by calling the train method
        if allow_finetune == True:
            # Train existing doc2vec model to adjust weights for considered data
            # Usually epochs is in [10, 20]; If data set is small, increasing it to [20, 50] may yield better results
            self.doc2vec_model.train(corpus_iterable = documents, epochs=train_epochs, total_examples=self.master.shape[0])
            
        # Get the infered vectors from pretrained Doc2Vec Models
        infered_vectors = []
        tokens_list_iv = list(self.master[self.name_of_token_column])
        for i in range(0, self.master.shape[0]):
            infered_vectors.append(self.doc2vec_model.infer_vector(tokens_list_iv[i], epochs=epochs_of_infering_vectors, alpha=0.025))
        
        # Append DocEmbeddings to master df
        self.master["vectorized_docs"] = infered_vectors 
        
        # Time information and return
        print(f"DocEmbeddings created in {round(time.time() - start_time, 2)}s")
        return self.master
    
    
    def create_new_model(self, vector_size = 200, epochs=20, seed=1997) -> pd.DataFrame:
        """
        Read more about the Model here: https://radimrehurek.com/gensim/models/doc2vec.html 
        
        vector_size:int Define the shape of the vectors (x,). Usually in [100, 700]. The more data, the higher that value should be
        epochs:int Number of training epochs for the model
        
        return Master df with DocEmbedding column appended
        """
        # Count time to execute
        start_time = time.time()
        
        # Create TaggedDocument Object for new model creation
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.master[self.name_of_token_column])]
        
        # Create new model
        self.doc2vec_model = Doc2Vec(documents, vector_size = vector_size, workers=16, seed=seed, epochs=epochs)
        
        # Extract document embeddings to list and append to master df
        vectorized_docs = []
        for i in range(0, self.master.shape[0]):
            vectorized_docs.append(self.doc2vec_model.dv[i])

        # Append DocEmbeddings to master df
        self.master["vectorized_docs"] = vectorized_docs
        
        # Time information and return
        print(f"DocEmbeddings created in {round(time.time() - start_time, 2)}s")
        return self.master

    
    def save_model(self, file_name:str):
        """
        Simple function to save the current model
        """
        self.doc2vec_model.save(file_name)
        
        
    def get_feature_matrix(self) -> pd.DataFrame:
        """
        return Feature Matrix X, i.e. Matrix containing all DocEmbeddings
        """
        X = pd.DataFrame(list(self.master["vectorized_docs"]))

        print(f"Shape of Feature Matrix X: {X.shape}")
        return X
    
    
    def get_mydoc2vec_model(self) -> Doc2Vec:
        return self.doc2vec_model
    
    
    def clean_text(self, text, tokenizer, stopwords):
        """
        Pre-process text and generate tokens
        
        text: Text to tokenize (usually the sentence column of master df)
        tokenizer: Tokenizer, usually the nltk tokenizer
        stopwords: All final stopwords that will be removed as a token
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
    
    
    def print_similar(self, input_doc:str, n:int, custom_stopwords:list):
        """
        Function to print the most similar documents in comparison to input_doc. Similarity is based on the cosine similarity.
        
        input_doc:str The input document to compare with
        n:int Number of top n similar sentences to output
        custom_stopwords:list List of all final stopwords
        """
        # Create tokens from input doc by calling clean_text function defined earlier
        tokens = self.clean_text(text = input_doc,
                                 tokenizer = word_tokenize,
                                 stopwords = custom_stopwords)
        print(f"Tokens from input text: {tokens}")

        # Create vector representation from tokens
        text_vector_represented = self.doc2vec_model.infer_vector(tokens, epochs=300, alpha=0.025)
        print(f"Shape of text in vector representation: {text_vector_represented.shape}")

        # Get top 3000 most similar docs; Choose big number, since there might
        # exist similar docs in the pretrained corpus, but that are not in the trained corpus
        res = self.doc2vec_model.dv.most_similar([text_vector_represented], topn=3000)
        # print(res)

        # Run variable to only get n most similar docs
        i = 0
        print(80 * "_")
        for doc_tag, doc_similarity in res:
            try:
                # Locate doc in master and print it together with information
                print(self.master.loc[doc_tag, self.target_column])
                print(self.master.loc[doc_tag, self.name_of_token_column])
                print("(Cosine Similarity:", round(doc_similarity * 100, 2), f"% | doc_tag = {doc_tag}) \n")
                i = i + 1
                if (i == n):
                    break
            # Catch Error, if doc_tag index is out of range (due to big corpus of the pretraining)
            except KeyError:
                continue 