## Synopsis

A Label Propagation semi-supervised clustering algorithm based on the paper "**Semi-supervised Learning in Gigantic Image Collections**" by Fergus, Weiss and Torralba with modifications to fit Spark.  


## Installation

Pre-Requisites needed to be installed:  
Apache Spark 2.0  
Python 2.7.0  
numPy 1.8.2  
sciPy 0.14.1  
sklearn 0.18    
Once you download the zipped folder, it will contain 2 different implementations. Choose the one which you want
to use and then run the setup.py file.  
To use the algorithm, download the folder and run the setup.py  
NOTE: This is not a published package hence it is required to be downloaded.
```
wget https://github.com/quinngroup/fergus-ssl/archive/master.zip -O fergus-ssl.zip
unzip -q fergus-ssl.zip
cd fergus-ssl-master/LabelPropagationDistributed
python setup.py clean build install

```

## Code Example

```python
from LabelPropagationDistributed import LabelPropagationDistributed as LPD  
import numpy as np  
import matplotlib.pyplot as plt  
dataX = np.array([[1,1], [2,3], [3,1], [4,10], [5,12], [6,13]])  
dataY = np.array([0,0,0,1,1,1])  
newdataY = np.array([0,-1,-1,-1,-1,1])  
testX = np.array([[1,-1], [3,-0.5],[7,5]])  
testY = np.array([0,0,1])  
dX = sc.parallelize(dataX)  
dy = sc.parallelize(newdataY)  
lpd = LPD(sc=sc, sqlContext= sqlContext)
lpd.fit(dX,dy)  
plabels_ = lpd.predict(sc.parallelize(testX))  
plt.scatter(dataX[:, 0], dataX[:, 1], marker='o', c=dataY, cmap = ('GnBu'))  
plt.show()  
plt.scatter(dataX[:,0], dataX[:,1], marker = 'o', c=np.array(lpd.labels_.collect()), cmap = (('GnBu')))  
plt.show()  
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, cmap = ('GnBu'))  
plt.show()  
plt.scatter(testX[:,0], testX[:,1], c=np.array(plabels_.collect()), cmap = (('GnBu')))  
plt.show()

```
Training Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/trainGT.png)  

Training Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/trainPredicted.png)  

Test Dataset GroundTruth:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/testGT.png)  

Test Dataset Predicted Labels:  
![alt tag](https://github.com/quinngroup/fergus-ssl/blob/master/LabelPropagationDistributed/Images/testPredicted.png)  
## Motivation

The idea behind implementing this algorithm was to attempt classification of millions of Olfactory Compounds with only a few labeled compounds making it a semi-supervised problem.
Being a Spark compatible algorithm, millions of Olfactory data can be trained and tested with this model.  

## API Reference

The proposed algorithm for semi-supervised clustering is based on the paper "Semi-supervised Learning in Gigantic Image Collections" by Fergus, Weiss and Torralba.
The link to the paper can be found here: https://cs.nyu.edu/~fergus/papers/fwt_ssl.pdf  

## Tests

Describe and show how to run the tests with code examples.  

## Contributors

https://www.github.com/mgadgil09  
https://github.com/magsol  
