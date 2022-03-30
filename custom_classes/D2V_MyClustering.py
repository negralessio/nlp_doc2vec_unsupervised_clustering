from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
sns.set()

class MyClustering:
    
    def __init__(self, X_transformed_df:pd.DataFrame):
        self.X_transformed_df = X_transformed_df
        # Initially set centroids to zero; will get updated in apply_kmeans() function
        self.centroids = 0
        self.mean_silhouette_coefficient = 0
        
        
    def apply_kmeans(self, n_clusters:int) -> pd.DataFrame:
        """
        n_clusters:int Number of centroids and clusters to group by
        
        return DataFrame with cluster labels appended
        """
        
        # Apply KMeans
        clustering = KMeans(n_clusters = n_clusters).fit(self.X_transformed_df)
        # Save centroids
        self.centroids = clustering.cluster_centers_
        
        # Compute silhouette score and "save" it by overwriting the instance variable
        self.mean_silhouette_coefficient = round(silhouette_score(X=self.X_transformed_df, labels=clustering.labels_), 4)
        
        # Append labels to dataframe
        if "kmeans_cluster" in self.X_transformed_df.columns:
            print("KMeans Cluster Column was already inserted into the df. You called 'apply_kmeans' twice on the same object.")
            print("Thus old kmeans cluster labels were dropped and new labels were added instead")
            self.X_transformed_df = self.X_transformed_df.drop(["kmeans_cluster"], axis=1)
            self.X_transformed_df.insert(loc=0, column="kmeans_cluster", value=clustering.labels_)
        else:
            self.X_transformed_df.insert(loc=0, column="kmeans_cluster", value=clustering.labels_)
        
        return self.X_transformed_df
    
    
    def plot_elbow_method(self, max_k:int):
        """
        max_k:int Maximum number of k clusters to consider

        Plot of all inertia (Sum of squared distances of samples to their closest cluster center)
        Inertia = WCSS = Within_Cluster_Sum_of_Square = ∑(x_i - C_k)^2, where C_k is the Centroid of cluster k and x_i an object within cluster k
        """

        # Apply kMeans on data with different parameter k
        ss_distances = []
        for k in range(1, max_k+1):
            if "kmeans_cluster" in self.X_transformed_df.columns:
                km = KMeans(n_clusters = k).fit(self.X_transformed_df.drop(["kmeans_cluster"], axis=1))
            else:
                km = KMeans(n_clusters = k).fit(self.X_transformed_df)
            ss_distances.append(km.inertia_)
        
        # Define text position for annotation and text
        x_pos = round((max_k+1) * 0.77, 2)
        y_pos = round(max(ss_distances)*0.92, 2)
        text = 'Hint: Determine k where the line forms a sharp bend, i.e. the elbow'

        # Plot settings
        plt.figure(figsize=(25, 6))
        plt.title("Plot Inertia of KMeans (Elbow Method to determine number of clusters)")
        plt.xlabel("k")
        plt.ylabel("Inertia (WCSS)")
        plt.xticks(np.arange(1, len(ss_distances) + 1, 1))
        sns.lineplot(x=np.arange(1, len(ss_distances) + 1), y=ss_distances)
        plt.text(x_pos, y_pos, text, size=15, color='black', style='italic', 
                 bbox={'facecolor': 'yellow', 'alpha': 0.2, 'pad': 10}, 
                 horizontalalignment='center', verticalalignment='center')
        plt.show()
        
        
    def plot_distribution_of_clusters(self):
        """
        Draws the distribution of the cluster labels (histogram)
        """
        
        # Count Cluster Occurences
        cluster_occurence = dict(self.X_transformed_df["kmeans_cluster"].value_counts())
        print("Cluster and its number of occurence: {}".format(cluster_occurence))

        # Draw Distribution of Cluster labels 
        plt.figure(figsize=(25,6))
        plt.title("Distribution of the Clusters")
        plt.xlabel("Cluster Label")
        plt.ylabel("Count")
        plt.bar(range(len(cluster_occurence)), list(cluster_occurence.values()), align='center')
        plt.xticks(range(len(cluster_occurence)), list(cluster_occurence.keys()))
        plt.show()
        
        
    def get_silhouette_score(self) -> float:
        return self.mean_silhouette_coefficient
