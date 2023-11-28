# import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import csv
import math
import scikitplot as skplt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pylab import scatter,show,plot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from pandas.tools.plotting import parallel_coordinates
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from itertools import cycle
from sklearn.metrics import precision_score,recall_score,accuracy_score

# load dataset
sam1 = pd.read_csv('/media/smartspace/New Volume1/Clustering/kmeansnknn3218/Analysis.csv')

le = preprocessing.LabelEncoder()

for col in sam1.columns.values:
	if sam1[col].dtypes=='object':
		le.fit(sam1[col])
		sam1[col]=le.transform(sam1[col])


data = sam1
nclusters=3
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(data)
class_labels = kmeans.labels_
print('class labels ',class_labels)
data2 = data
idx = 11
data2.insert(loc=idx, column = 'Class', value = class_labels) # to insert new column in dataframe

neigh = KNeighborsClassifier(n_neighbors=int(math.sqrt(nclusters)))

X_train, X_test, y_train, y_test = train_test_split(data, class_labels, test_size=0.30, random_state=42)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


print('Total No. of Data',len(data))
print('70% of Training Data',len(X_train))
print('30% of Testing Data',len(X_test))

print('Predicted Output', neigh.predict(X_test))
print('Actual Output' , y_test)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = [0,1]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
#plt.figure()
#plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')

plt.show()

target_names = ['class 0', 'class 1']
graph_cl = [0,1]
print('KMeans & KNN Classifier Performance Metrics')
print(classification_report(y_test, y_pred, target_names=target_names))
precision = precision_score(y_test, y_pred, average=None)
#print(precision)
recall = recall_score(y_test, y_pred, average=None)
#rint(recall)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy:',accuracy)

plt.scatter(graph_cl,precision,c=graph_cl)
plt.xlabel('Classes')
plt.ylabel('Precision')
plt.show()

plt.scatter(graph_cl,recall,c=graph_cl)
plt.xlabel('Classes')
plt.ylabel('Recall')
plt.show()





