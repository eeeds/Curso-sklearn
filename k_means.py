import pandas as pd


#Útil para pc's con poca ram y malos procesadores
#Como no es supervisado, no se divide los datos
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":

    dataset = pd.read_csv('./data/candy_a74a49fd-6364-4c16-9381-406cdb66f338.csv')
    print(dataset.head(10))

    X = dataset.drop(['competitorname'], axis=1)
    
    kmeans = MiniBatchKMeans(n_clusters = 4, batch_size = 8).fit(X) #se formaran los algoritmos y los resultados de a 8

    print("Total de centros:", len(kmeans.cluster_centers_))
    print("="*64)

    print(kmeans.predict(X))

    dataset['group'] = kmeans.predict(X)
    print(dataset)

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.pairplot(dataset, hue='group')
    plt.show()

    #sns.pairplot(dataset[['sugarpercent','pricepercent','winpercent','group']], hue = 'group')


