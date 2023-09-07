# K-means
In this repository, we will discuss the work of K-means algorithm using Python's Sklearn (scikit-learn). K-means is an unsupervised machine learning technique used to cluster or partition similar the data objects (data records) of your dataset into groups or clusters. Between the wide range of clustering techniques, the k-means is one of the oldest and most used in applications and domains. The conventional k-means algorithm follows a set of sequential steps to partition the input data into k disjoint clusters with the goal of by minimizing the variance in each cluster.

- Step1: Define K (known as centroids), which are initially random points as cluster centers and the value of k reflects the number of clusters.
- Step2: For each data point in the dataset:
- Step2.1: Calculate the distance between such data point and each of the K centroids
- Step2.2: Assign the data point to the cluster of the shortest calculated distance
- Step2.3: Calculate the new cluster's center (the centroid) by computing the average of the data points within such cluster


The data points are assigned to a cluster in a way to minimize the sum of the squared distance between the data points and centroid. The quality of the cluster assignments is determined by computing the sum of the squared error (SSE) which is the sum of the squared Euclidean distances of each point to its closest centroid. 

## Step1: Prepare the required libraries

Python version 3.8 is the most stable version, used in the different tutorials. Scikit-learn (or commonly referred to as sklearn) is probably one of the most powerful and widely used Machine Learning libraries in Python.

```
python3 --version
pip3 --version
pip3 install --upgrade pip  
pip3 install pandas
pip3 install numpy
pip3 install scikit-learn
```

## Step2: Prepare/build your Dataset


```
import pandas as pd
from sklearn import cluster
```
Your dataset is composed of a set of records in a table format, and the titles of each column. Next, we build a dataframe out of the our dataset and display columns. A DataFrame is a 2D data structure (you can visualize it as a table or a spreadsheet) and is most commonly used pandas object. A DataFrame optionally accepts index (row labels) and columns (column lables) arguments

```
titles = ['user','Jaws','Star Wars','Exorcist','Omen']
records = [['john',5,5,2,1],['mary',4,5,3,2],['lisa',2,2,4,5],['bob',4,4,4,3],['lee',1,2,3,4],['harry',2,1,5,5]]

dataSet = pd.DataFrame(records,columns=titles)
print(dataSet)
```

We will not need the users' names in our clustering mission, so let's drop them first. The drop() is used to remove specific row or entire column and two important arguments to consider with drop(). The labels (can be a single label or a list) to specify Index or column labels to drop.
- the axis to specifiy whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
- By specifying the column axis (axis='columns'), the drop() method removes the specified column.
- By specifying the row axis (axis='index'), the drop() method removes the specified row.

```
ratings = dataSet.drop('user',axis='columns')
print(ratings)
```

## Step3: Build and train the model


Let's now clustering these individual ratings. The first function we will use is the sklearn.cluster.KMeans, and this function accepts a set of arguments to guide the clustering operation, we will focus on the following arguments:

- n_clusters (int value with default=8) is the number of clusters to form as well as the number of centroids to generate.
- n_init ('auto', 'random', or an int value with default=10) is the number of times the k-means algorithm is run with different centroid seeds. 
-- The final results is the best output of n_init consecutive runs in terms of inertia. 
-- Several runs are recommended for sparse high-dimensional problems.
- random_state (int value with default=None) determines random number generation for centroid initialization. Use an int to make the randomness deterministic.
- max_iter (int value with default=300) specifies the maximum number of iterations of the k-means algorithm for a single run.


Then we allow our KMeans algorithm to learn, using the fit() function. The fit() is used to train your model with respect to some input data

```
k_means = cluster.KMeans(n_clusters=2, n_init='auto', max_iter=50, random_state=0)
k_means.fit(ratings)
```


A centroid, in k-means, is a data point that represents the center of the cluster (the mean). It is important to note here that a centroid might not be a member of the dataset. In order to access the calculated centroids, we use the .cluster_centers_

```
outputCentroids = k_means.cluster_centers_
centroidsInfo = pd.DataFrame(outputCentroids,columns=ratings.columns)
print(centroidsInfo)
```


Since our goal is to cluster the ratings into 2 clusters (as mentioned in the n_cluster argument of KMeans), we can see which class each rating is assigned to. In order to access the new classes, we can access the .labels_ variable. We can now add the new labels to our orignial dataset, and make the users' names column to be the index.

```
outputLabels = k_means.labels_
dataSet['Cluster Labels'] = outputLabels
dataSet.set_index('user')
print(dataSet)
```



## Step4: Agree on the number of clusters

Now we utilize the elbow method to visualize the intertia for different values of K. The elbow method is a technique used to determine the optimal number of clusters in KMeans clustering.  It is based on the idea that the optimal K value is the point at which the decrease in the sum of squared distances between data points and their assigned clusters starts to level off. Inertia measures how well a dataset was clustered by K-Means. It is calculated by measuring the distance between each data point and its centroid, squaring this distance, and summing these squares across one cluster.


```

inertias = []
for i in range(1,4):
    k_means = cluster.KMeans(n_clusters=i, n_init='auto', max_iter=50, random_state=0)
    k_means.fit(ratings) 
    inertias.append(k_means.inertia_)


import matplotlib.pyplot as plt  
plt.plot(range(1,4), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
```


## Step5: Agree on the number of clusters


Transformers are for pre-processing before modeling. Models are used to make predictions (e.g., Decision Tree model, Random Forest model).  You will usually pre-process your data (with transformers) before putting it in a model. Now the usage of methods fit(), transform(), fit_transform() and predict() depend on the type of object.


- For Transformers:
-- fit() is used to calculate the initial filling of parameters on the training data and saves them as an internal objects state
-- transform() uses the above calculated values and return modified training data
-- fit_transform() joins above two steps by just calling first fit() and then transform() on the same data.

- For Models:
-- fit() calculates the parameters/weights on training data and saves them as an internal objects state.
-- predict() uses the above calculated weights on test data to make the predictions

The purpose of .predict() or .transform() is to apply a trained model to data.

```
k_means = cluster.KMeans(n_clusters=2, n_init='auto', max_iter=50, random_state=0)
k_means.fit(ratings)  # k_means.fit_transform(ratings) 

newRecords = [[4,5,3,2],[2,1,5,5]]
predictedClusters = k_means.predict(newRecords)

print(predictedClusters)

```
