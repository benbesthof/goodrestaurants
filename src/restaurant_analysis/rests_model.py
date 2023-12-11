"""Predictive and clustering models for good restaurants data"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing as pre
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans

class Kmean:
    """Class that creates Kmeans clustering model to group data in a way previously unseen
        Returns:
        kmeanmanual: creates a klustering model and returns results of the model 
         
    """
    def kmeanmanual(self):
        """
          Kmeans clustering with sklearn
          fitting of scikit learn kmeans clustering model with preprocessed data 

          Args: 
            self(series): original dataframe to be analyzed
          Returns:
          results of cluster model fitting with preprocessed data """
        k = 8
        model = KMeans(n_clusters = k, random_state = 0)
        model.fit(self.modeldf)
        clusters = model.predict(self.modeldf)
        manualdf = self.df.copy()
        manualdf.clusters = clusters
        return manualdf