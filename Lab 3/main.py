from numpy import split
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib
import pylab as pl
import numpy as np
from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn import naive_bayes
import random as random


irisData = datasets.load_iris()
# print (irisData.data[0])
# print (irisData.target[0])


def plot_2D(data, target, target_names):
    colors = cycle('rgbcmykw')  # cycle de couleurs
    target_ids = range(len(target_names))
    # print(data[target==1,1])
    # print(data)
    pl.figure()

    for i, c, label in zip(target_ids, colors, target_names):
        pl.scatter(data[target == i, 0],
                   data[target == i, 1], c=c, label=label)
    pl.legend()
    # pl.show()


def k_neighbors_classifier_plot():

    n_neighbors = 15

    # import some data to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    h = 0.02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    cmap_bold = ["darkorange", "c", "darkblue"]

    for weights in ["uniform", "distance"]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        pl.figure(figsize=(8, 6))
        pl.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=iris.target_names[y],
            palette=cmap_bold,
            alpha=1.0,
            edgecolor="black",
        )
        pl.xlim(xx.min(), xx.max())
        pl.ylim(yy.min(), yy.max())
        pl.title(
            "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
        )
        pl.xlabel(iris.feature_names[0])
        pl.ylabel(iris.feature_names[1])

    pl.show()


# plot_2D(irisData.data, irisData.target,["Setosa", "Versicolor", "Verginica"])
# k_neighbors_classifier_plot()


def naive_bayes_classifier(fit_data, fit_target, predict_data):
    nb_classifier = naive_bayes.MultinomialNB(
        fit_prior=True)  # un algo d'apprentissage
    nb_classifier.fit(fit_data, fit_target)
    # p31 = nb.predict([irisData.data[31]])
    # print(p31)
    # plast = nb.predict([irisData.data[-1]])
    # print(plast)
    predictions = nb_classifier.predict(predict_data)
    return nb_classifier, predictions


def error(prediction, target, classfier, data):
    ea = 0
    for i in range(len(target)):
        if (prediction[i] != target[i]):
            ea = ea+1
    # print('method 1 :' + str(ea/len(target)))
    # print('method 2 :'+str(sum(x != 0 for x in prediction-target)/len(target)))
    # print('method 3 :'+str(1-classfier.score(data, target)))
    return 1-classfier.score(data, target)


# nb_classifier, predictions = naive_bayes_classifier(
#     irisData.data[0:150], irisData.target[0:150], irisData.data[0:150])
# print(predictions,irisData.target[0:50])
# error(predictions,irisData.target[0:150],nb_classifier,irisData.data[0:150])


def split1(data, target, training_percent):
    sample = list(zip(data, target))
    random.shuffle(sample)
    data, target = zip(*sample)
    learningLength = int(training_percent*len(data))
    dataS1 = data[:learningLength]
    targetS1 = target[:learningLength]
    dataS2 = data[learningLength:]
    targetS2 = target[learningLength:]
    return dataS1, targetS1, dataS2, targetS2


def average_error_split1():
    sum = 0
    cycles = 1000
    for i in range(cycles):
        print(i)
        dataS1, targetS1, dataS2, targetS2 = split(
            irisData.data, irisData.target, 2/3)
        classifier, predictions = naive_bayes_classifier(
            dataS1, targetS1, dataS2)

        sum += error(predictions, targetS2, classifier, dataS2)

    print(sum/cycles)

def average_error_train_test_split():
    sum = 0
    cycles = 1000
    for i in range(cycles):
        train_data, test_data, train_target, test_target = train_test_split(
            irisData.data, irisData.target, test_size=0.2)

        classifier, predictions = naive_bayes_classifier(
            train_data, train_target, test_data)

        sum += error(predictions, test_target, classifier, test_data)
    print(sum/cycles)
average_error_train_test_split()
        