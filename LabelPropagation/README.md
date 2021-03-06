## Synopsis

A Label Propagation semi-supervised clustering algorithm based on the paper "**Semi-supervised Learning in Gigantic Image Collections**" by Fergus, Weiss and Torralba.  

## Installation

Pre-Requisites needed to be installed:  
 - Python 2.7.0  
 - NumPy 1.8.2  
 - SciPy 0.14.1  
 - scikit-learn 0.18

Once you download the zipped folder, it will contain 2 different implementations. Choose the one which you want
to use and then run the setup.py file.  
To use the algorithm, download the folder and run the setup.py  
NOTE: This is not a published package hence it is required to be downloaded.  
```
wget https://github.com/quinngroup/fergus-ssl/archive/master.zip -O fergus-ssl.zip
unzip -q fergus-ssl.zip
cd fergus-ssl-master/LabelPropagation
sudo python setup.py clean build install

```

## Code Example

```python
from LabelPropagation import LabelPropagation as LP  
import numpy as np  
import matplotlib.pyplot as plt  
dataX = np.array([1,1,2,3,3,1,4,10,5,12,6,13]).reshape(6,2)
dataY = np.array([0,0,0,1,1,1])  
newdataY = np.array([0,-1,-1,-1,-1,1])  
testX = np.array([1,2,5,10]).reshape(2,2)  
testY = np.array([0,1])  
lp = LP(numBins=2)
lp.fit(dataX,newdataY)  
plabels_ = lp.predict(testX)  
plt.scatter(dataX[:, 0], dataX[:, 1], marker='o', c=dataY, cmap = ('GnBu'))  
plt.show()  
plt.scatter(dataX[:,0], dataX[:,1], marker = 'o', c=np.array(lp.labels_), cmap = (('GnBu')))  
plt.show()  
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('GnBu'))  
plt.show() 
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=plabels_, cmap = ('GnBu'))  
plt.show() 
plt.scatter(testX[:,0], testX[:,1], c=np.array(plabels_), cmap = (('GnBu')))  
plt.show()

```
Training Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagation/Images/trainGT.png)  

Training Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagation/Images/trainPredicted.png)  

Test Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagation/Images/testGT.png)  

Test Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagation/Images/testPredicted.png)  
## Motivation

The idea behind implementing this algorithm was to attempt classification of millions of Olfactory Compounds with only a few labeled compounds making it a semi-supervised problem.
To come up with a scalable Apache Spark compatible algorithm which can handle millions of data, a stand-alone version of the same was required which would later be used to create
the distributed one.

## API Reference

The proposed algorithm for semi-supervised clustering is based on the paper "Semi-supervised Learning in Gigantic Image Collections" by Fergus, Weiss and Torralba.
The link to the paper can be found here: https://cs.nyu.edu/~fergus/papers/fwt_ssl.pdf  

## Tests

Describe and show how to run the tests with code examples.  

## Contributors

https://www.github.com/mgadgil09  
https://github.com/magsol  
