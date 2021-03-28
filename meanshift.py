#Para datos moderados

import pandas as pd

from sklearn.cluster import MeanShift

if __name__== "__main__":

    dataset = pd.read_csv('./data/candy_a74a49fd-6364-4c16-9381-406cdb66f338.csv')

    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1)
    meanshift = MeanShift().fit(X)
    print(max(meanshift.labels_))
    #MeanShift envía 3 clusters(él toma esa decisión)

    print("="*64)
    print(meanshift.cluster_centers_)

    dataset['meanshift']=meanshift.labels_
    print("="*64)
    print(dataset)