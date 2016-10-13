## Synopsis

A Label Propagation semi-supervised clustering algorithm based on the paper "**Semi-supervised Learning in Gigantic Image Collections**" by Fergus, Weiss and Torralba with modifications to fit Spark.  

## Code Example

```python
from LabelPropagationDistributed import LabelPropagationDistributed as LPD  
import numpy as np  
import matplotlib as plt  
dataX = np.array([[1,1], [2,3], [3,1], [4,10], [5,12], [6,13]])  
dataY = np.array([0,0,0,1,1,1])  
newdataY = np.array([0,-1,-1,-1,-1,1])  
testX = np.array([[1,-1], [3,-0.5],[7,5]])  
testY = np.array([0,0,1])  
dX = sc.parallelize(dataX)  
dy = sc.parallelize(newdataY)  
lpd.fit(dX,dy)  
plabels_ = lpd.predict(sc.parallelize(testX))  
plt.scatter(dataX[:, 0], dataX[:, 1], marker='o', c=dataY, cmap = ('GnBu'))  
plt.show()  

Training Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/testGT.png)  

plt.scatter(dataX[:,0], dataX[:,1], marker = 'o', c=np.array(lpd.labels_.collect()), cmap = (('GnBu')))  
plt.show()  

Training Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/trainPredicted.png)  

plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('GnBu'))  
plt.show()  

Training Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/testGT.png)  

plt.scatter(testX[:,0], testX[:,1], c=np.array(plabels_.collect()), cmap = (('GnBu')))  
plt.show()

Test Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/testPredicted.png)  

```
## Motivation

The idea behind implementing this algorithm was to attempt classification of millions of Olfactory Compounds with only a few labeled compounds making it a semi-supervised problem.
Being a Spark compatible algorithm, millions of Olfactory data can be trained and tested with this model.  

## Installation

Pre-Requisites needed to be installed:  
Apache Spark 2.0  
Python 2.7.0  
numPy 1.8.2  
sciPy 0.14.1  
sklearn 0.18  


## API Reference

The proposed algorithm for semi-supervised clustering is based on the paper "Semi-supervised Learning in Gigantic Image Collections" by Fergus, Weiss and Torralba.
The link to the paper can be found here: https://cs.nyu.edu/~fergus/papers/fwt_ssl.pdf  

## Tests

Describe and show how to run the tests with code examples.  

## Contributors

https://www.github.com/mgadgil09  
