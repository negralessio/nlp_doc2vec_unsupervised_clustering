#from MulticoreTSNE import MulticoreTSNE as MC_TSNE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import time
import umap.umap_ as umap 

class MyDimensionalityReducer:
    
    def __init__(self, feature_matrix:pd.DataFrame):
        self.X = feature_matrix

        
    def apply_tsne(self, n_components = 2, learning_rate = 200, perplexity = 20) -> pd.DataFrame:
        """
        n_components:int Remaining number of dimensions after dimensionality reduction (default: 2)
        learning_rate:int Learning rate of TSNE
        perplexity:int 
        
        Learn more about TSNE here: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        
        return DataFrame with shape (n_master, n_components)
        """
        
        # Count time to execute
        start_time = time.time()
        
        # Apply MC TSNE
        X_transformed = TSNE(n_components=n_components, learning_rate=learning_rate, n_jobs=-1, perplexity=perplexity).fit_transform(self.X)
        
        # Create dataframe from transformed and dimensionality reduced data
        X_transformed_df = pd.DataFrame(X_transformed, columns=["TC{}".format(i+1) for i in range(n_components)])
        print(f"Shape of X_transformed_df: {X_transformed_df.shape}")
        
        # Time information and return
        print(f"TSNE was applied in {round(time.time() - start_time, 2)}s")
        return X_transformed_df
    
    
    def apply_pca(self, n_components = 5) -> pd.DataFrame:
        """
        n_components:int Number of Principal Components
        
        return DataFrame with shape (n_master, n_components)
        """
        # Count time to execute
        start_time = time.time()
    
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(self.X)
        print(f"Explained Variance Ratio (Singular Values): {pca.explained_variance_ratio_}")
        print(f"Sum of Singular Values: {sum(pca.explained_variance_ratio_)}")
        
        # Create dataframe from transformed and dimensionality reduced data
        X_transformed_df = pd.DataFrame(X_transformed, columns=["PC{}".format(i+1) for i in range(n_components)])
        print(f"Shape of X_transformed_df: {X_transformed_df.shape}")        
        
        # Time information and return
        print(f"PCA was applied in {round(time.time() - start_time, 2)}s")
        return X_transformed_df
    
    
    def apply_umap(self, n_neighbors = 10) -> pd.DataFrame:
        """
        n_neighbors:int Number of neighbors umap considers
        
        Documentation: https://umap-learn.readthedocs.io/en/latest/basic_usage.html
        
        return DataFrame with shape (n_master, n_components)
        """
        # Count time to execute
        start_time = time.time()
        
        # Apply UMAP
        X_transformed = umap.UMAP(n_neighbors=n_neighbors).fit_transform(self.X)
        
        # Create dataframe from transformed and dimensionality reduced data
        X_transformed_df = pd.DataFrame(X_transformed, columns=["UC{}".format(i+1) for i in range(2)])
        print(f"Shape of X_transformed_df: {X_transformed_df.shape}")
        
        # Time information and return
        print(f"UMAP was applied in {round(time.time() - start_time, 2)}s")
        return X_transformed_df
        