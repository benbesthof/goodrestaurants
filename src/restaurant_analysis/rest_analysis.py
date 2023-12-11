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
import warnings


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
        
        table = self.df
        return table


class Ranalytics:
    """ Analytics class for displaying restaurant and US census data
    
        Returns: 
            preprocessor: data preprocessor for clustering algorithm
            hclustering: fits data to clustering algorithm and returns results in a dendrogram"""
    def __init__(self, dfpath):
        """ Read terrorsim datasetwith pandas and creating dataframe object
        Args:
            self(series): original dataframe to be analyzed
            dfpath(str): name of csv file in which dataset is contained

        Returns:
            First rows of restaurant data object"""
        self.df = pd.read_csv('dfpath.csv')
        
        return self.df
    def peprocessor(self):
        """ Preprocessing of data
        
        Args:
            self(series): original dataframe to be analyzed
        
        Returns:
            Preprocessed dataframe"""

        #self.df.good_restaurant = [lambda x: 1 if ]
        categorical_variables = [i for i in self.df.select_dtypes(include = object)]
        numerical_variables = [i for i in self.df.select_dtypes(exclude = object)]
        dummy_variables = pd.get_dummies(self.f[categorical_variables], drop_first = True, dtype = 'int64')
        scaled_numerical_variables = [i for i in numerical_variables]
        scaled_numerical_variables = [i for i in numerical_variables]
        array = self.df[numerical_variables].values
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        dfscaled = pd.DataFrame(datascaler.fit_transform(array), columns = scaled_numerical_variables)
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        self.modeldf = pd.concat([dummy_variables, dfscaled], axis = 1)
        return self.modelf.head()
    

    def hclustering(self):
        """
          Hierararchical clustering with scipy
          fitting of scikit learn kmeans clustering model with preprocessed data via pipeline

          Args: 
            self(series): original dataframe to be analyzed
          Returns:
          results of hierarchachal cluster model fitting with preprocessed data """
        matrices = ward(self.modeldf.values)
        dendrogram(matrices, orientation = 'right', labels = list(self.df['name']))
        plt.tick_params(axis = 'x', which = 'both', bottom = 'off', top = 'off', labelbottom = 'off')
        showtime = plt.show()
        return showtime



class ExploratoryDataAnalysis:
    """
    A class for conducting various analyses on a DataFrame.
    """
    def __init__(self, data):
        """
        Initializes the DataAnalysis object with the provided data.
        
        Parameters:
            data (pd.DataFrame): The data to be analyzed.
        """
        self.data = data

    def unique_cuisines(self):
        """
        Retrieves the unique values in the 'cuisine' column of the data.
        
        Returns:
        np.ndarray: An array containing unique cuisines.
        """
        return self.data['cuisine'].unique()

    def cuisine_value_counts(self):
        """
        Counts the occurrences of each unique value in the 'cuisine' column of the data.
        
        Returns:
        pd.Series: A Series containing the counts of unique cuisines.
        """
        return self.data['cuisine'].value_counts()

    def unique_boroughs(self):
        """
        Retrieves the unique values in the 'borough' column of the data.
        
        Returns:
        np.ndarray: An array containing unique boroughs.
        """
        return self.data['borough'].unique()

    def borough_value_counts(self):
        """
        Counts the occurrences of each unique value in the 'borough' column of the data.
        
        Returns:
        pd.Series: A Series containing the counts of unique boroughs.
        """
        return self.data['borough'].value_counts()

    def bar_plot_cuisine(self):
        """
        Creates a horizontal bar plot for the frequency distribution of the 'cuisine' column.
        """
        self.data['cuisine'].value_counts().sort_values().plot.barh(title="Freq Dist of Cuisine", rot=0)

    def bar_plot_borough(self):
        """
        Creates a horizontal bar plot for the frequency distribution of the 'borough' column.
        """
        self.data['borough'].value_counts().sort_values().plot.barh(title="Freq Dist of Borough", rot=0)

    
