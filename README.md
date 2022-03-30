⚠️ Note that this notebook was originally developed for BMW data. Instead, I used google play store reviews data from [kaggle](https://www.kaggle.com/datasets/shivkumarganesh/tinder-google-play-store-review) to get a similiar result. Doc2Vec model is trained on around 800k cleaned and tokenized sentences. Then we took a sample of 10.000 reviews, did a sentence split and finally infer DocEmbeddings for these sentences.

**INTRODUCTION**

This Model has been created in order to **cluster unsupervised data** during my internship at BMW. It uses an Doc2Vec Model as the core of this notebook, to obtain numerical representation of the text data, so called **DocEmbeddings**.

This Notebook can be splitted into the following sections:
- **Load Data**: Load and clean the data (dropping duplicates, eliminating outliers, filtering, ...); Also allows sentence splitting if the data is a whole paragraph
- **Tokenize Data**: Tokenize the Data using the 'MyTokenizer' Class. Tokens serve as the input for our Doc2Vec Model and can be interpreted as a cleaned version of a word
- **Create DocEmbeddings**: Create numerical representations of our text data, which we then use for dimensionality reduction and clustering
- **Reduce Dimensionality**: DocEmbeddings have a default shape of 200 dimensions, which we cannot visualize and correctly work with. Thats why we use TSNE in order to reduce the dimensions to two dimensions while preservering the overall structure of the data
- **Clustering and Visualization**: We then use the two dimensional TSNE data to cluster and visualize it via bokeh. Note that github does not show bokeh visualization (I choose seaborn instead). If you're interested in the bokeh plot, which allows hovering over the scatter plot, run the notebook locally and call _myvisualizer_TSNE.plot()_ in the section 'TSNE Clustering'.
- **Additional Function Sections**: We can visualize n-grams to see which keywords get captured by each cluster (Explainability AI). We can also find similar documents given an arbitrary input doc. We can also dive deeper into a certain cluster using the 'draw_subcluster' function. Lastly we can use the _OpenAI Text Summarization_ to generate a summarization of each cluster.

The 'Backend' has been written in the _custom_classes_ directory as classes each (OOP paradigm). This ensure that the code can be changed easily without losing the overview. In general, this notebook serves as a console, where we only look at the results and define the hyperparameters for each section. Hyperparameters are written in capitalized letters and can be defined arbitrary.

**OBJECTIVES** 🎯

- Allows to **automatically read and understand the data** without reading all feedbacks manually
- Allows to **find similar feedbacks** of a given input text by using the cosine similarity
- Allows to **summarize all feedbacks** of a given cluster by using OpenAI Text Summarization
- Allows to **easily label the data**

**POTENTIAL USE CASES**
- Explore new, unknown data to understand what it is about (i.e. app store feedback)
- Challenge and validate existing classes in a supervised classification approach
- Identify new topics in “global other” bucket of a supervised classification approach
- Drill-down into a specific subset of data (i.e. “seat/ defects” à cluster + summarize)

**POINTS TO IMPROVE / POSSIBLE EXTENSIONS**
- Improve our Doc2Vec Model by training it with more data (> 1 Million). Current master model is trained on ~850k cleaned and tokenized **sentences**
- Fine tuning is a long and difficult process and each use cases needs different hyperparameters
- Challenge Doc2Vec approach by a transformer approach (e.g. BERT)
_______
_Author: Alessio Negrini ([LinkedIn](https://www.linkedin.com/in/alessio-negrini-9a7847230/)), 23rd March 2022_
