# K Nearest Neighbour

Implementation of kNN in Python (3.6).


## Description

k-nearest neighbors (kNN) is a non-parametric method used in  classification. The input consists of the k closest training examples in the feature space. The output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
In kNN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors. kNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification.

source: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

## Dataset


We will use the iris dataset to demo the kNN classifier

The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".

![Image](https://github.com/amourav/kNearestNeighbour/readme_imgs/iris.png)

Iris Flower Species ([source] (https://www.flickr.com/photos/gmayfield10/3352170798/in/photostream/))


This data sets consists of 3 different types of irises (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray. The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.

source: https://en.wikipedia.org/wiki/Iris_flower_data_set


## Overview
For illustration purposes we will only be using the two features.

![Image](https://github.com/amourav/kNearestNeighbour/readme_imgs/scatter1.png)

Scatterplot of samples in iris dataset.


We can tune the value of `k` on the test set.

![Image](https://github.com/amourav/kNearestNeighbour/readme_imgs/tune_k.png)

Using the optimal value of `k` we can run inference on the dataset and clasify each point in the feature space.

![Image](https://github.com/amourav/kNearestNeighbour/readme_imgs/scatter2.png)



## Dependencies

To run kNearestNeighbor you only need the `numpy` package.
To run the demo notebook you will need a few additional packages:
`matplotlib`
`sklearn`


## Usage

`knn = kNearestNeighbor(k=k)` Initialize knn classifier
`knn.fit(X_trn, y_trn)` Fit the classifier to the training data. (Note: all this does is evaluate the training accuracy and save the training set.
`knn.predict(X_test)` This will run inference on new input data by measuring distance of points in X_trn.

example:

```
from kNearestNeighbors import kNearestNeighbor, accuracy
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# load dataset
iris = datasets.load_iris()
X = iris.data  
y = iris.target

# trn/test split - only use first two features
X_trn, X_test, y_trn, y_test = train_test_split(X[:, :2], 
                                                y, 
                                                test_size=0.2, 
                                                random_state=0)

# fit classifier
k = 8
knn = kNearestNeighbor(k=k)
knn.fit(X_trn, y_trn)
y_trn_pred = knn.predict(X_trn)
trn_acc = accuracy(y_trn_pred, y_trn)
y_test_pred = knn.predict(X_test)
test_acc = accuracy(y_test_pred, y_test)
print('train accuracy: {}'.format(trn_acc))
print('test accuracy: {}'.format(test_acc))

>> train accuracy: 0.8666666666666667
>> test accuracy: 0.6
```


## Author

Andrei Mouraviev

