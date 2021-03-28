import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt 
#Kernel

from sklearn.decomposition import KernelPCA

#Se usa como clasificador lineal
from sklearn.linear_model import LogisticRegression

#Utilidades para pre procesar datos antes de enviarlos
#Para que los datos estén entre 0 y 1
from sklearn.preprocessing import StandardScaler
#Para partir datos 
from sklearn.model_selection import train_test_split

#Si este es el script principal que siga el sgte proceso
if __name__ == "__main__":
    dt_heart = pd.read_csv('.\data\heart_bde64b4c-2d72-4cd3-a964-62ee94855f5b.csv')
    print(dt_heart.head(5))
    #Se sacan los targets que son lo que se quiere clasificar y como es columnas por eso axis=1
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    dt_features = StandardScaler().fit_transform(dt_features)
    

    #Xtrain, etc.
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size = 0.3, random_state=42)

    #Aplicación de Kernel

    kpca = KernelPCA(n_components= 4, kernel = 'poly')

    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)
    
    logistic = LogisticRegression(solver= 'lbfgs')

    logistic.fit(dt_train, y_train)
    print("SCORE KPCA: ", logistic.score(dt_test, y_test))
    