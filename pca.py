import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt 
#Se importa los módulos para usar PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    print(X_train.shape)
    print(y_train.shape)

    #Configuración algoritmo PCA
    #n_componentes= min(n_muestras, n_features) si no especificamos
    pca = PCA(n_components= 3 )
    pca.fit(X_train)
    #Para hacer batch y entrenar por pequeños conjuntos(bueno para malas PC's)
    ipca = IncrementalPCA(n_components=3, batch_size = 10)
    ipca.fit(X_train)

    #Números entre 0 y cantidad de componentes escogidas, la variable de y indica que tan importante son las variables.
    #Se ve básicamente las variables que aportan más info
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    logistic = LogisticRegression( solver = 'lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)

    logistic.fit(dt_train, y_train)
    print("SCORE PCA:", logistic.score(dt_test, y_test))
    #IPCA
    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)

    logistic.fit(dt_train, y_train)
    print("SCORE IPCA:", logistic.score(dt_test, y_test))
    






