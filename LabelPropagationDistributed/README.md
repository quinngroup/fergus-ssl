## Synopsis

A Label Propagation semi-supervised clustering algorithm based on the paper "**Semi-supervised Learning in Gigantic Image Collections**" by Fergus, Weiss and Torralba with modifications to fit Spark.

## Code Example

>>> from LabelPropagationDistributed import LabelPropagationDistributed as LPD
>>> dataX = array([ 5.76961775,  -6.55673209, 11.30752027,  -1.56316985,
        8.76722337,  -1.54995049, 10.23511359,  -6.20912033,
        3.49161828,  -3.02917744]).reshape(5,2)
>>> dataY = array([ 1,  -1,  0, -1, -1])
>>> test = array([2.1159109 ,   6.03520684,   1.04347698,  -4.44740207,
       -8.33902404,   4.20918959,   1.38447488,  -1.50363493]).reshape(4,2)
>>> lpd = LPD(sc=sc, sqlContext = sqlContext, numBins = 5)
>>> lpd.fit(sc.parallelize(dataX),sc.parallelize(dataY))
>>> plabels_ = lpd.predict(sc.parallelize(test))

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
