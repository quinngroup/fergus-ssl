//testing
//get the moons dataset
//plot the data
//Test classes
//1. Pick 2 data points keep them labelled
//2. randomly assign 2 data points and keep them labelled
//3. vary data sizes
//4. check ground truth predicted correctly
//5. vary hyper parameters
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
X1,Y1=datasets.make_moons(100,True,None,None)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, cmap = ('Paired'))
plt.show()
newX1 = np.around(X1,decimals=2)
newY1=np.copy(Y1)
'''Test case 1 '''
point1 = np.where((newX1==(1.,0.)).all(axis=1))
point2 = np.where((newX1==(0.13,0.01)).all(axis=1))
for y in range(0,len(newY1)):
    newY1[y]=-1

newY1[point1[0][0]] = Y1[point1[0][0]]
newY1[point2[0][0]] = Y1[point2[0][0]]
execfile('/home/madhura/Computational_Olfaction/fergus-ssl/fergus_propagation.py')
fp = FergusPropagation()
fp.fit(X1,newY1)
oldLabels = np.copy(Y1)
newLabels = np.copy(fp.labels_)
//plotting eigenvectors
plt.plot(range(1,len(fp.evects)+1),fp.evects[:,0], c='yellow', label='Eigenvector1')
plt.plot(range(1,len(fp.evects)+1),fp.evects[:,1], c='gray', label='Eigenvector2')
//plotting eigenvalues
plt.plot(1,fp.evals[0], c='yellow', label='Eigenvalue1',marker='o')
plt.plot(2,fp.evals[1], c='gray', label='Eigenvalue2',marker='o')

//Data plot
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=fp.labels_, cmap = ('Paired'))

mu = X1.mean(axis=0)
sigma = X1.std(axis=0).mean()
def annotate(ax, name, start, end):
    arrow = ax.annotate(name,
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(facecolor='red', width=2.0))
    return arrow

fig, ax = plt.subplots()
ax.scatter(X1)
ax.scatter(X1[:,0],X1[:,1])
ax.set_aspect('equal')
for axis in evects:
    annotate(ax, '', mu, mu + sigma * axis)

plt.show()
plt.hist(fp.func)
plt.show()
fp.gdash
fp.gdash.weights_
fp.gdash.means_
fp.gdash.covars_
fp.gdash.converged_
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=newLabels)
plt.show
(oldLabels!=newLabels).argmax(axis=0)
diff = oldLabels.size-np.sum((oldLabels==newLabels))
print 'incorrect labels: ' + repr(diff)
error =  (float)(diff/(oldLabels.size))
print 'error is ' + repr(error)
print error
//Test case 2
newY1 = np.copy(Y1)
//label 4=0 and label 99=1
for y in range(0,len(newY1)):
    newY1[y]=-1

newY1[4] = 0
newY1[99] = 1
execfile('/home/madhura/Computational_Olfaction/fergus-ssl/fergus_propagation.py')
fp = FergusPropagation()
fp.fit(X1,newY1)
oldLabels = np.copy(Y1)
newLabels = np.copy(fp.labels_)
diff = np.setdiff1d(oldLabels,newLabels)
error = 1-(diff.shape[0]/oldLabels.shape[0])
//Testcase 3
X1,Y1=datasets.make_moons(100000,True,None,None)
