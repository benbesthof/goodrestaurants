""" Exploring restaurants data and the data on their respective counties.
    Returns: 
        reader: creates a pandas dataframe object with scraped tabular data.
        peprocessor: preprocesser for clustering algorithm
        hclustering: fits data to clustering algorithm and returns results in a dendrogram

"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import preprocessing as pre
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import scipy as sp
from scipy.cluster.hierarchy import dendrogram, ward 

class Scrape:
    """ Create DataFrame to land scraped data into Pandas DataFrame

        Returns:
            Reader: creates a pandas dataframe object with legally scraped tabular data."""
    def __init__(self, websiteurl):
        """
        Args:
            websiteurl(str): url of chosen website
        Returns:
            landing of tabular data in pandas dataframe """
        self.df = pd.read_html(websiteurl)

    def reader(self):
        """ Create pandas df object  
        Args:
            dataframe(series): pandas dataframe of tabular data scraped from U.S Census Bureau website
        
        Returns: 
            pandas dataframe object of scraped tabular data from U.S Census Bureau website"""
        
        table = self.dataf[1]

        return table

class YelpData:
    """ Create DataFrame from Yelp data in csv"""
    def __init__(self, dfpath):
        """ Create pandas df object
        Args:
            dataframe(str): pandas dataframe of yelp data 
        
        Returns: 
            pandas dataframe object of yelp data"""
        self.df = pd.read_csv(dfpath)

    

    
    def peprocessor(self):
        """ Preprocessing of data for clustering
        
        Args:
            self(series): original dataframe to be analyzed
        
        Returns:
            Preprocessed dataframe for clustering model"""

        self.df['medincome'] = self.df.apply(lambda row: 47036 if self.df[self.df['where'] == 'The Bronx'] else(
            82431 if self.df[self.df['where'] == 'Queens'] else(
            96185 if self.df[self.df['where'] == 'Staten Island'] else(
                74692 if self.df[self.df['where'] == 'Brooklyn'] else(
                    99880 if self.df[self.df['where'] == 'Manhattan'] else 0)))))
        
        self.df['population'] = self.df.apply(lambda row: 1379946 if self.df[self.df['where'] == 'The Bronx'] else(
            2278029 if self.df[self.df['where'] == 'Queens'] else(
            491133 if self.df[self.df['where'] == 'Staten Island'] else(
                2590516 if self.df[self.df['where'] == 'Brooklyn'] else(
                    1596273 if self.df[self.df['where'] == 'Manhattan'] else 0)))))
        
        self.df['foreign born persons pct'] = self.df.apply(lambda row: 33.9 if self.df[self.df['where'] == 'The Bronx'] else(
            47.1 if self.df[self.df['where'] == 'Queens'] else(
            24.8 if self.df[self.df['where'] == 'Staten Island'] else(
                35.3 if self.df[self.df['where'] == 'Brooklyn'] else(
                    28.1 if self.df[self.df['where'] == 'Manhattan'] else 0)))))
        
        self.df['whitenolatino'] = self.df.apply(lambda row: 8.7  if self.df[self.df['where'] == 'The Bronx'] else(
            23.9 if self.df[self.df['where'] == 'Queens'] else(
            8.7 if self.df[self.df['where'] == 'Staten Island'] else(
                36.7 if self.df[self.df['where'] == 'Brooklyn'] else(
                    45.5 if self.df[self.df['where'] == 'Manhattan'] else 0)))))
        
        
        
        
        categorical_variables = [i for i in self.df.select_dtypes(include = object)]
        numerical_variables = [i for i in self.df.select_dtypes(exclude = object) if i != 'zip_code']
        dummy_variables = pd.get_dummies(self.df[categorical_variables], drop_first = True, dtype = 'int64')
        scaled_numerical_variables = [i for i in numerical_variables]
        scaled_numerical_variables = [i for i in numerical_variables]
        array = self.df[numerical_variables].values
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        dfscaled = pd.DataFrame(datascaler.fit_transform(array), columns = scaled_numerical_variables)
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        self.modeldf = pd.concat([dummy_variables, dfscaled], axis = 1)
        self.df.to_csv('combined_rests_data.csv')
        return self.modelf.head()
    

    def hclustering(self):
        """
          Hierararchical clustering with scipy
          fitting of scipy hierarchachal clustering model with preprocessed data
          Args: 
            self(series): original dataframe to be analyzed
          Returns:
          results of hierarchachal cluster model fitting with preprocessed data """
        matrices = ward(self.modeldf.values)
        dendrogram(matrices, orientation = 'right', labels = list(self.df['name']))
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
        showtime = plt.show()
        return showtime