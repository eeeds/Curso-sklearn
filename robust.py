import pandas as pd


from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
#Se compara el resultado con un modelo basado en máquinas de soporte vectorial, el regresor es SVR(Support vector regressor)
from sklearn.svm import SVR


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__== "__main__":
    dataset = pd.read_csv("./data/felicidad_corrupt.csv")
    print(dataset.head(5))
    #axis=0 para fila
    X = dataset.drop(['country', 'score'], axis= 1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state= 42)

    estimadores = {
        'SVR': SVR(gamma = 'auto', C=1.0, epsilon = 0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)

        print("="*64)
        print(name)
        print('MSE:', mean_squared_error(y_test, predictions))