import pandas as pd

from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    
    path = './Bagging/data/heart.csv'
    dataset = pd.read_csv(path)

    print(dataset.head(5))
    print('')
    print(dataset['target'].describe())

    x = dataset.drop(['target'], axis=1, inplace=False)
    y = dataset['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)

    knn_class = KNeighborsClassifier().fit(x_train, y_train)
    knn_pred = knn_class.predict(x_test)

    print('')
    print('Accuracy KNeighbors:', accuracy_score(knn_pred, y_test))
    print('')

    #bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(x_train, y_train)
    #bag_pred = bag_class.predict(x_test)

    #print('')
    #print('Accuracy Bagging with KNeighbors:', accuracy_score(bag_pred, y_test))
    #print('')

    classifier = {
        'KNeighbors': KNeighborsClassifier(),
        'LinearSCV': LinearSVC(),
        'SVC': SVC(),
        'SGDC': SGDClassifier(),
        'DecisionTree': DecisionTreeClassifier()
    }

    for name, estimator in classifier.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=5).fit(x_train, y_train)
        bag_pred = bag_class.predict(x_test)

        print('Accuracy Bagging with {}:'.format(name), accuracy_score(bag_pred, y_test))
        print('')