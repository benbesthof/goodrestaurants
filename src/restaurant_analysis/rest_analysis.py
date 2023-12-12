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
        
        table = self.df[1]
        return table
    
    def peprocessor(self):
        """ Preprocessing of data
        
        Args:
            self(series): original dataframe to be analyzed
        
        Returns:
            Preprocessed dataframe"""

        #self.good_restaurant = [lambda row: 1 if row self.rating > 2.5 else 0]
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



    def average_rating_by_category(self, category):
        """
        Computes the average rating for each category (cuisine or borough).
        
        Parameters:
            category (str): The category to group by ('cuisine' or 'borough').
            
        Returns:
        pd.Series: A Series containing average ratings for each category.
        """
        return self.data.groupby(category)['rating'].mean()

    def average_price_by_category(self, category):
        """
        Computes the average price level for each category (cuisine or borough).
        
        Parameters:
            category (str): The category to group by ('cuisine' or 'borough').
            
        Returns:
        pd.Series: A Series containing average price levels for each category.
        """
        # Converting price to numerical value if it's not already
        if isinstance(self.data['price'].iloc[0], str):
            self.data['price_level'] = self.data['price'].apply(lambda x: len(x))
        else:
            self.data['price_level'] = self.data['price']

        return self.data.groupby(category)['price_level'].mean()
    
    def plot_rating_distribution(self):
         """
         Create a histogram for the distribution of ratings.

         This method plots a histogram of the 'rating' column in the data attribute of the class,
         showing the frequency distribution of ratings across the dataset.

         Returns:
            None
         """
         self.data['rating'].plot.hist(title='Histogram of Ratings', bins=10, ec='black')
         plt.xlabel('Rating')
         plt.ylabel('Frequency')
         plt.show()

    def plot_cuisine_count(self):
         """
         Create a count plot for the different cuisines present in the dataset.

         This method uses seaborn's countplot to display the counts of different cuisines
         sorted by their frequency.

         Returns:
         None
         """
         sns.countplot(y='cuisine', data=self.data, order=self.data['cuisine'].value_counts().index)
         plt.title('Count Plot of Cuisines')
         plt.show()

    def plot_borough_count(self):
        """
        Create a count plot for the different boroughs present in the dataset.

        This method uses seaborn's countplot to display the counts of different boroughs
        sorted by their frequency.
    
        Returns:
            None
        """
        sns.countplot(y='borough', data=self.data, order=self.data['borough'].value_counts().index)
        plt.title('Count Plot of Boroughs')
        plt.show()

    def correlation_analysis(self):
        """
        Perform a correlation analysis on the numerical columns of the dataset.

        This method calculates the correlation matrix for numerical columns and
        displays it using seaborn's heatmap for easier interpretation.

        Returns:
            None
        """
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()