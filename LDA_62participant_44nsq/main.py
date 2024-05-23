# LDA model for dataset (62participants, 44NSQ)
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
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import metrics

# importing data set
df = pd.read_csv('Spatial_Navigation_Research_DATA_2012-2013(NO_NAMES)_62sketchmappers_complete.csv')
print(df)

X = df.iloc[:, 1:45].values
y = df.iloc[:, 0].values

# loop the model to calculate average accuracy
num_iterations = 20
total = 0
for i in range(num_iterations):
    # Split the dataset into training and testing sets
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(X, y, test_size=0.2, random_state=2*i)

    # feature scaling
    sca = StandardScaler()

    Xa_train = sca.fit_transform(Xa_train)
    Xa_test = sca.transform(Xa_test)

    clf = LinearDiscriminantAnalysis(n_components=2)
    clf.fit(X, y)
    Xa_train = clf.transform(Xa_train)
    Xa_test = clf.transform(Xa_test)

    # fitting logistic regression to the training set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(Xa_train, ya_train)

    # predicting the test set result
    ya_pred = classifier.predict(Xa_test)
    # calculate the accuracy
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

# apply lda model
clf = LinearDiscriminantAnalysis(n_components=2)
clf.fit(X, y)
X_train = clf.transform(X_train)
X_test = clf.transform(X_test)

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
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
cm_display.plot()
plt.show()


# plotting pattern
marker1 = ["x", "*", "^"]
label1 = ['1-Procedural route sketchmappers', '2-Allocentric-survey sketchmappers', '3-Egocentric-survey sketchmappers']
color1 = ["red", "green", "blue"]
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

print("train data")
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(color1)(i), marker=marker1[i], label=label1[j-1])

plt.title('Training set')
plt.xlabel('LDA 1')  # for Xlabel
plt.ylabel('LDA 2')  # for Ylabel
plt.legend()  # to show legend
# show scatter plot
plt.show()

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
                c=ListedColormap(color1)(i), marker=marker1[i], label=label1[j-1])
plt.title('Test set')
plt.xlabel('LDA 1')  # for Xlabel
plt.ylabel('LDA 2')  # for Ylabel
plt.legend()

# show scatter plot
plt.show()

# write data into csv
with open('test_Set_Output.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["Subject ID", "LDA Dimension 1", "LDA Dimension 2"])
    k = 1 #index number
    for i, j in enumerate(np.unique(y_set)):
        for c, v, in zip(X_set[y_set == j, 0], X_set[y_set == j, 1]):
            writer.writerow([k, round(c, 2), round(v, 2)])
            k+=1
file.close()



