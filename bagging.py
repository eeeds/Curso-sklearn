import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__== "__main__":
    dt_heart = pd.read_csv('./data/heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv')
    print(dt_heart.head(5))
    print(dt_heart['target'].describe())

    #inplace sacaría de dt_heart a target, ahora solo copiaremos.
    X = dt_heart.drop(['target'],axis=1)
    y = dt_heart['target']


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.35)

    knn_class =  KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)

    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator = KNeighborsClassifier(), n_estimators = 50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))