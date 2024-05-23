# KPCA model for dataset (62participants, 44NSQ)
# importing required libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score
from sklearn import metrics



# importing data set
df = pd.read_csv('Spatial_Navigation_Research_DATA_2012-2013(NO_NAMES)_62sketchmappers_complete.csv')
print(df)

X = df.iloc[:, 1:45].values
y = df.iloc[:, 0].values

# loop the model to find the average accuracy
num_iterations = 20
total = 0
for i in range(num_iterations):
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, y, test_size=0.5, random_state=2*i)

    # feature scaling
    sca = StandardScaler()

    Xa_train = sca.fit_transform(Xa_train)
    Xa_test = sca.transform(Xa_test)

    kpca = KernelPCA(kernel='rbf', n_components=2)
    Xa_train = kpca.fit_transform(Xa_train)
    Xa_test = kpca.transform(Xa_test)

    # fitting logistic regression to the training set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(Xa_train, ya_train)
    ya_pred = classifier.predict(Xa_test)
    score = accuracy_score(ya_test, ya_pred)
    total += score

# Print the accuracy scores for each iteration
result = total / num_iterations
print(f"\n{num_iterations} iteration : Accuracy = {result}")


# splitting the dataset into the Training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# apply kernel KPCA model
kpca = KernelPCA(kernel='rbf', n_components=2)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# fitting logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predicting the test set result
y_pred = classifier.predict(X_test)

# calculate the accuracy for one iteration
accuracy_score_1 = accuracy_score(y_test,y_pred)
print("1 iteration: Accuracy = ", accuracy_score_1)

# find the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:", "\n", cm)
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

plt.title('Logistic Regression (Training set)')
plt.xlabel('kPCA 1')  # for Xlabel
plt.ylabel('kPCA 2')  # for Ylabel
plt.legend()  # to show legend

# show scatter plot
plt.show()

# write data into csv
with open('train_Set_Output.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject ID", "KPCA Dimension 1", "KPCA Dimension 2"])
    k = 1 #index number
    for i, j in enumerate(np.unique(y_set)):
        for c, v, in zip(X_set[y_set == j, 0], X_set[y_set == j, 1]):
            writer.writerow([k, round(c, 2), round(v, 2)])
            k+=1
file.close()

# Visualising the Test set results through scatter plot
X_set, y_set = X_test, y_test

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
                color=ListedColormap(('red', 'green', 'blue'))(i), marker=marker1[i], label=label1[j-1])

# title for scatter plot
plt.title('Logistic Regression (Test set)')
plt.xlabel('kPCA 1')
plt.ylabel('kPCA 2')
plt.legend()

# show scatter plot
plt.show()

# write data into csv
with open('test_Set_Output.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject ID", "KPCA Dimension 1", "KPCA Dimension 2"])
    k = 1 #index number
    for i, j in enumerate(np.unique(y_set)):
        for c, v, in zip(X_set[y_set == j, 0], X_set[y_set == j, 1]):
            writer.writerow([k, round(c, 2), round(v, 2)])
            k+=1
file.close()
