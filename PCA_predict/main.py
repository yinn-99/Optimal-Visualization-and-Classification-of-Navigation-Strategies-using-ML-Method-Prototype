# importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn import metrics


# importing data set
df = pd.read_csv('Spatial_Navigation_Research_DATA_2012-2013(NO_NAMES)_updated_JZ02.21.23.csv')
print(df)

X = df.iloc[:, 1:45].values
y = df.iloc[:, 0].values
X_train = df.iloc[:62, 1:45].values
X_test = df.iloc[63:, 1:45].values
y_train = df.iloc[:62, 0].values
y_test = df.iloc[63:, 0].values

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# apply PCA function
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

# fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predicting the test set result
y_pred = classifier.predict(X_train)
y_pred_test = classifier.predict(X_test)

# find the accuracy
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_train,y_pred)))

# find the confusion matrix
cm = confusion_matrix(y_train, y_pred, labels=classifier.classes_)
print("Confusion matrix:")
print(cm)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)

cm_display.plot()
plt.show()

# plotting pattern
marker1 = ["x", "*", "^"]
label1 = ['1-Procedural route sketchmappers', '2-Allocentric-survey sketchmappers', '3-Egocentric-survey sketchmappers']

# predicting the training set result
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                  X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), marker=marker1[i], label=label1[j-1])

# write data into csv
with open('train_Set_Output.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject ID", "LDA Dimension 1", "LDA Dimension 2"])
    k = 1 #index number
    for i, j in enumerate(np.unique(y_set)):
        for c, v, in zip(X_set[y_set == j, 0], X_set[y_set == j, 1]):
            writer.writerow([k, round(c, 2), round(v, 2)])
            k+=1
file.close()

plt.title('Training set')
plt.xlabel('PCA 1')  # for Xlabel
plt.ylabel('PCA 2')  # for Ylabel
plt.legend()  # to show legend

# show scatter plot
plt.show()

# Visualising the Test set results through scatter plot


X_set, y_set = X_test, y_pred_test

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                               stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1,
                               stop=X_set[:, 1].max() + 1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                  X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('yellow', 'white', 'aquamarine')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green', 'blue'))(i), marker=marker1[i], label=label1[j-1])

# write data into csv
with open('train_Set_Output.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject ID", "LDA Dimension 1", "LDA Dimension 2"])
    k = 1 #index number
    for i, j in enumerate(np.unique(y_set)):
        for c, v, in zip(X_set[y_set == j, 0], X_set[y_set == j, 1]):
            writer.writerow([k, round(c, 2), round(v, 2)])
            k+=1
file.close()

# title for scatter plot
plt.title('Test set')
plt.xlabel('PCA 1')  # for Xlabel
plt.ylabel('PCA 2')  # for Ylabel
plt.legend()

# show scatter plot
plt.show()
