from bokeh.io import push_notebook, show, output_notebook, curdoc, save
from bokeh.layouts import row 
from bokeh.plotting import figure, output_file, save
import bokeh.plotting as bpl
import bokeh.models as bmo
from bokeh.palettes import Viridis256, viridis, magma, Cividis256, Turbo256, Spectral6, Category20c, Bokeh, Category20, inferno
from bokeh.transform import factor_cmap
from bokeh.transform import linear_cmap
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MyVisualizer:
    
    def __init__(self, X_transformed_df:pd.DataFrame, n_clusters:int, kind:str, target_column:str, centroids, silhouette_score):
        self.X_transformed_df = X_transformed_df
        self.n_clusters = n_clusters
        self.kind = kind
        self.target_column = target_column
        self.axis_name:list
        self.centroids = centroids
        self.silhouette_score = round(silhouette_score, 2)
        output_notebook()
        
    def get_axis_name(self):
        """
        Function to define the axis name, e.g. if algorithm PCA was used, the axis and the column are 'PC1', 'PC2', ...
        """
        if self.kind == "TSNE":
            self.axis_name = ['TC1', 'TC2']
        elif self.kind == "UMAP":
            self.axis_name = ['UC1', 'UC2']
        elif self.kind == "PCA":
            self.axis_name = ["PC{}".format(i+1) for i in range(self.n_clusters)]
        
    def plot(self):
        """
        Function to plot a bokeh scatterplot
        """
        # Call get_axis_name first
        self.get_axis_name()
        
        # Add color column to df so that bokeh is finally satisfied and the coloring works
        palette = viridis(self.n_clusters)
        self.X_transformed_df["color"] = [palette[i] for i in self.X_transformed_df.kmeans_cluster]

        # Create bokeh DataSource from Pandas DF
        source = bpl.ColumnDataSource(self.X_transformed_df)

        # Color settings
        color_map = bmo.CategoricalColorMapper(factors=self.X_transformed_df['kmeans_cluster'].astype(str).unique(), palette=palette)

        # Plot settings
        p = figure(plot_width=1200, plot_height=700, title = f"{self.kind} Clustering | n_clusters = {self.n_clusters} | N = {self.X_transformed_df.shape[0]} | Mean Silhouette Coefficient = {self.silhouette_score}",toolbar_location=None, 
                   tools="hover", tooltips="[Cluster: @kmeans_cluster] @" + self.target_column)
        # Scatter settings
        p.scatter(self.axis_name[0], self.axis_name[1], source=source, fill_alpha=0.8, size=5, legend_group="kmeans_cluster", color='color')
        p.scatter(self.centroids[:, 0], self.centroids[:, 1], marker="cross", size=25, color="#FF0000")  # draw centroids
        p.xaxis.axis_label = self.axis_name[0]
        p.yaxis.axis_label = self.axis_name[1]
        p.legend.location = "top_right"
        show(p)
        

    def plot_seaborn(self):
        self.get_axis_name()

        plt.figure(figsize=(20, 12))
        plt.title(f"{self.kind} Clustering | n_clusters = {self.n_clusters} | N = {self.X_transformed_df.shape[0]} | Mean Silhouette Coefficient = {round(self.silhouette_score, 2)}")
        sns.scatterplot(data=self.X_transformed_df, x="TC1", y="TC2", hue="kmeans_cluster", legend="full", palette=sns.color_palette("viridis", as_cmap=True))
        sns.scatterplot(x = self.centroids[:, 0], y = self.centroids[:, 1], markers="X", s=30, palette="#fc0303")
        plt.plot()
        

    def draw_sub_clusters(self, c:int, n_clusters:int, blob_size:int, drawDendro:bool, tfidf_keyword_list:list):
        """
        c:int Initital cluster to consider
        n_clusters:int Number of subclusters to compute
        drawDendro:boolean Whether or not to draw dendrogram
        """
        # Take only data from input cluster c
        X = self.X_transformed_df[self.X_transformed_df["kmeans_cluster"] == c].copy()

        # Clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering.fit(X[self.axis_name])

        # Append Subcluster label to X
        X["subcluster"] = clustering.labels_

        # Add color column to df
        palette = Category20[n_clusters]
        
        # If n_clusters is too high for Category20, change palette to viridis
        if n_clusters > 20:
            palette = viridis(n_clusters)
        
        X["subcolor"] = [palette[i] for i in X.subcluster]

        if drawDendro == True:
            # Draw Dendrogram
            plt.figure(figsize=(30, 7))
            plt.title("Dendograms of Subcluster")
            dend = shc.dendrogram(shc.linkage(X[['TC1','TC2']], method='ward'))
            plt.show()
            print(125 * "_")

        # Create bokeh DataSource from Pandas DF
        source = bpl.ColumnDataSource(X)
        # Colors
        color_map = bmo.CategoricalColorMapper(factors=X['subcluster'].astype(str).unique(), palette=palette)

        # Plot settings
        p = figure(plot_width=1200, plot_height=700, 
                   title = f"Agglomerative subclustering of cluster c = {c} \t|\t Top 7 TFIDF-Keywords: {tfidf_keyword_list[c][: 5]}",toolbar_location=None, 
                   tools="hover", tooltips="[Subcluster: @subcluster] @" + self.target_column)
        # Scatter settings
        p.scatter(self.axis_name[0], self.axis_name[1], source=source, fill_alpha=0.8, size=blob_size, legend_group="subcluster", color='subcolor')
        p.xaxis.axis_label = self.axis_name[0]
        p.yaxis.axis_label = self.axis_name[1]
        p.legend.location = "top_right"
        show(p)