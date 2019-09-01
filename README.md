# K Nearest Neighbor

Implementation of kNN in Python (3.6).


## Description

k-nearest neighbors (or "neighbours" for us Canadians) is a non-parametric method used in classification. The input consists of the k closest training examples in the feature space. The output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
In kNN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors. kNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification.

source: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

## Dataset

We will use the iris dataset to demo the kNN classifier (Fig. 1)

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/iris.PNG)

Figure 1: Iris Flower Species [source](https://www.flickr.com/photos/gmayfield10/3352170798/in/photostream/)


This data sets consists of 3 different types of irises (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray. The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

source: https://en.wikipedia.org/wiki/Iris_flower_data_set


## Overview

For illustration purposes we will only be using the two features (Sepal Width, Petal Width). We will also split the dataset into training (120 samples) and testing (30 samples) A scatterplot illustrating the distribution of iris flower species based on these features (Fig. 2).

![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/scatter1.png)

Figure 2: Scatterplot of samples in iris dataset.

Now that we split the dataset into training and testing, we can run our kNN model (Fig. 3)
```
train accuracy: 0.97
test accuracy: 0.92
```
![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/scatter2.png) <br/>
Figure 3: Scatterplot of iris dataset labeled by species (sepal length vs sepal width). Background colour represents best guess of the knn classifier for the class label of the hypothetical point in this feature space.

While this performance is good, we can further improve the accuracy by tuning the value of `k` on the test set (Figs. 4, 5).

![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/tune_k.png) <br/>
Figure 4: Accuracy for each value of k evaluated on the training and testing data.
```
optimal value for k: 12
train accuracy: 0.97
test accuracy: 0.94
```
![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/scatter2b.png) <br/>
Figure 5: Scatterplot of iris dataset with predicted (knn - k=12) and actual class labels (o - train set, 
x - test set).

Of course, we are not limited to using these two features, or any two features (Fig. 6). 
![Image](https://github.com/amourav/kNN_from_scratch/blob/readme/readme_imgs/knn_plots.png) <br/>
Figure 6: Pairwise comparison of features in the iris dataset and predicted labels (knn - k=12).

## Dependencies

To run kNearestNeighbor you only need the `numpy` package.
To run the demo notebook you will need a few additional packages:
`matplotlib`
`sklearn`


## Usage

`knn = kNearestNeighbor(k=k)` Initialize knn classifier with the number of neighbours.

`knn.fit(X_trn, y_trn)` Fit the classifier to the training data. (Note: all this does is evaluate the training accuracy and save the training set.

`y_pred = knn.predict(X_test)` This will run inference on new input data by measuring distance of points in X_trn to each point in X_test.

example:

```
from kNearestNeighbors import kNearestNeighbor, accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Load Iris Dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target

# For illustration purposes we will only be using the two features in the dataset
feature_idxs = [1, 3] # SET FEATURES BY INDEX <------------------
feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

# We will also split the dataset into training and testing so we can evaluate the kNN classifier
X_trn_, X_test_, y_trn, y_test = train_test_split(X, 
                                                 y, 
                                                 test_size=0.333, 
                                                 random_state=0,
                                                 stratify=y)
X_trn, X_test = X_trn_[:, feature_idxs], X_test_[:, feature_idxs]

print("X_trn.shape = {}, X_test.shape = {}".format(X_trn.shape, X_test.shape))
print("Features: {}, {}".format(feature_names[feature_idxs[0]], feature_names[feature_idxs[1]]))

# fit classifier
k = 12
knn = kNearestNeighbor(k=k)
knn.fit(X_trn, y_trn, v=False)
y_trn_pred = knn.predict(X_trn)
trn_acc = accuracy(y_trn_pred, y_trn)
y_test_pred = knn.predict(X_test)
test_acc = accuracy(y_test_pred, y_test)
print('train accuracy: {}'.format(trn_acc))
print('test accuracy: {}'.format(test_acc))

>> train accuracy: 0.97
>> test accuracy: 0.94
```


## Author

Andrei Mouraviev
