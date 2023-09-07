import pandas as pd
from sklearn import cluster

# let's create a fake dataset as follows:

# the columns
titles = ['user','Jaws','Star Wars','Exorcist','Omen']
# list of users along with their ratings
records = [['john',5,5,2,1],['mary',4,5,3,2],['lisa',2,2,4,5],['bob',4,4,4,3],['lee',1,2,3,4],['harry',2,1,5,5]]

# build a dataframe out of the our dataset and display columns
dataSet = pd.DataFrame(records,columns=titles)

# we will not need the users' names in our clustering mission, so let's drop them first
ratings = dataSet.drop('user',axis='columns')

# let's now clustering these individual ratings
k_means = cluster.KMeans(n_clusters=2, n_init='auto', max_iter=50, random_state=0)
k_means.fit(ratings) 


# since our goal is to cluster the ratings into 2 clusters (as mentioned in the n_cluster argument of KMeans), we can see which class each rating is assigned to
# to access the new classes, we can access the .labels_ variable
outputLabels = k_means.labels_

# we can now add the new labels to our orignial dataset, and make the users' names column to be the index
dataSet['Cluster Labels'] = outputLabels
dataSet.set_index('user')
print(dataSet)



# Predict the clusters of new data points

newRecords = [[4,5,3,2],[2,1,5,5]]
predictedClusters = k_means.predict(newRecords)

print(predictedClusters)
