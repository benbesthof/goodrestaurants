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
import lxml
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
        self.df = pd.read_html(websiteurl)[1]

    def reader(self):
        """ Create pandas df object  
        Args:
            dataframe(series): pandas dataframe of tabular data scraped from U.S Census Bureau website
        
        Returns: 
            pandas dataframe object of scraped tabular data from U.S Census Bureau website"""
    
        table = self.df.query('Population in ["Population Estimates, July 1, 2022, (V2022)", "White alone, not Hispanic or Latino, percent", "Foreign born persons, percent, 2018-2022", "Median household income (in 2022 dollars), 2018-2022", "Median Gross Rent, 2018-2022"]').reset_index()
        table.rename(columns = {'Population' : 'Statistic', 'Unnamed: 1': 'Staten Island', 'Unnamed: 2' : 'The Bronx', 'Unnamed: 3' : 'Queens', 'Unnamed: 4' : 'Brooklyn', 'Unnamed: 5' : 'Manhattan'}, inplace = True)
        return table
class Dapp:
    """ data preprocessor for yelp api data from csv
        Returns:
            preprocessor: preprocessing of data for yelp reviews and clustering
            hclustering: initial view of how data might group together

    """
    def __init__(self, dfpath):
        """
        Loading data into preprocessor

        Args: 
            self(dataframe): original dataframe to be analyzed 

        """
        self.df = pd.read_csv('restaurants_data.csv')
    def preprocessor(self):
        """ Preprocessing of data for clustering
        
        Args:
            self(series): original dataframe to be analyzed
        
        Returns:
            Preprocessed dataframe for clustering model"""
        
        manhattan_zip_codes = ['10001', '10002', '10003', '10004', '10005',
                               '10006', '10007', '10009', '10010', '10011',
                               '10012', '10013', '10014', '10016', '10017',
                               '10018', '10019', '10020', '10021', '10022',
                               '10023', '10024', '10025', '10026', '10027',
                               '10028', '10029', '10030', '10031', '10032',
                               '10033', '10034', '10035', '10036', '10037',
                               '10038', '10039', '10040', '10128', '10280']

        brooklyn_zip_codes = ['11201', '11203', '11204', '11205', '11206',
                              '11207', '11208', '11209', '11210', '11211',
                              '11212', '11213', '11214', '11215', '11216',
                              '11217', '11218', '11219', '11220', '11221',
                              '11222', '11223', '11224', '11225', '11226',
                              '11228', '11229', '11230', '11231', '11232',
                              '11233', '11234', '11235', '11236', '11237',
                              '11238', '11239', '11249']

        queens_zip_codes = ['11001', '11004', '11101', '11102', '11103',
                            '11104', '11105', '11106', '11354', '11355',
                            '11356', '11357', '11358', '11359', '11360',
                            '11361', '11362', '11363', '11364', '11365',
                            '11366', '11367', '11368', '11369', '11370',
                            '11371', '11372', '11373', '11374', '11375',
                            '11377', '11378', '11379', '11385', '11411',
                            '11412', '11413', '11414', '11415', '11416',
                            '11417', '11418', '11419', '11420', '11421',
                            '11422', '11423', '11426', '11427', '11428',
                            '11429', '11430', '11432', '11433', '11434',
                            '11435', '11436', '11691', '11692', '11693',
                            '11694', '11697']

        staten_island_zip_codes = ['10301', '10302', '10303', '10304', '10305',
                                   '10306', '10307', '10308', '10309', '10310',
                                   '10311', '10312', '10314']

        bronx_zip_codes = ['10451', '10452', '10453', '10454', '10455',
                           '10456', '10457', '10458', '10459', '10460',
                           '10461', '10462', '10463', '10464', '10465',
                           '10466', '10467', '10468', '10469', '10470',
                           '10471', '10472', '10473', '10474', '10475']
        rests = self.df

        rests['where'] = rests.apply(lambda row: 'Manhattan' if str(row['zip_code']).strip() in manhattan_zip_codes else(
            'Brooklyn' if str(row['zip_code']).strip() in brooklyn_zip_codes
            else(
                'Queens' if str(row['zip_code']).strip() in queens_zip_codes
                  else(
                      'The Bronx' if str(row['zip_code']).strip() in bronx_zip_codes
                        else(
                            'Staten Island' if str(row['zip_code']).strip() in staten_island_zip_codes
                              else 'other')))), axis=1)
        
        rests['price'] = rests['price'].apply(lambda x: '1' if x == '$' else ('2' if x == '$$' else ('3' if x == '$$$' else('4' if x == '$$$$' else( '0')))))
        rests.drop('borough', axis = 1, inplace = True)
        rests = rests[rests['where'] != 'other'].reset_index(drop=True)
        rests = rests[rests['price'] != 'Not Available'].reset_index(drop=True)
        rests.rename(columns = {'where': 'borough'}, inplace = True)

        rests['medincome'] = rests.apply(lambda row: 47036 if row['borough'] == 'The Bronx' else(
            82431 if row['borough'] == 'Queens' else(
                96185 if row['borough'] == 'Staten Island' else(
                    74692 if row['borough'] == 'Brooklyn' else(
                        99880 if row['borough'] == 'Manhattan' else 0)))), axis = 1)
        
        rests['population'] = rests.apply(lambda row: 1379946 if row['borough'] == 'The Bronx' else(
            2278029 if row['borough'] == 'Queens'   else(
                491133 if row['borough'] == 'Staten Island' else(
                    2590516 if row['borough'] == 'Brooklyn' else(
                        1596273 if row['borough'] == 'Manhattan' else 0)))), axis = 1)
        
        rests['foreign born persons pct'] = rests.apply(lambda row: 33.9 if row['borough'] == 'The Bronx' else(
            47.1 if row['borough'] == 'Queens' else(
                24.8 if row['borough'] == 'Staten Island' else(
                    35.3 if row['borough'] == 'Brooklyn' else(
                        8.1 if row['borough'] == 'Manhattan' else 0)))), axis = 1)
        
        rests['whitenolatino'] = rests.apply(lambda row: 8.7  if row['borough'] == 'The Bronx' else(
            23.9 if row['borough'] == 'Queens' else(
                8.7 if row['borough'] == 'Staten Island' else(
                    36.7 if row['borough'] == 'Brooklyn' else(
                        45.5 if row['borough'] == 'Manhattan' else 0)))), axis = 1)
        rests.to_csv('combined_rests_data.csv', index = False)
        return rests
        
    def precluster(self):
        """ Preprocessing data for clustering algorithm
            Args: 
                self(dataframe): dataset to be analyzed 
            Returns: preprocessed dataset for clustering model
        """
        rests = pd.read_csv('combined_rests_data.csv')
        categorical_variables = [i for i in rests.select_dtypes(include = object) if i != 'name' and i != 'categories']
        numerical_variables = [i for i in rests.select_dtypes(exclude = object) if i != 'zip_code']
        dummy_variables = pd.get_dummies(rests[categorical_variables], drop_first = True, dtype = 'int64')
        scaled_numerical_variables = [i for i in numerical_variables]
        scaled_numerical_variables = [i for i in numerical_variables]
        array = rests[numerical_variables].values
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        dfscaled = pd.DataFrame(datascaler.fit_transform(array), columns = scaled_numerical_variables)
        datascaler = pre.MinMaxScaler(feature_range = (0,1))
        self.modeldf = pd.concat([dummy_variables, dfscaled], axis = 1)
        
        self.modeldf.to_csv('hclustering_data.csv', index = False)
        dataprev = self.modeldf.head()
        return dataprev
    

    def hclustering(self):
        """
          Hierararchical clustering with scipy
          fitting of scipy hierarchachal clustering model with preprocessed data
          Args: 
            self(series): original dataframe to be analyzed
          Returns:
          results of hierarchachal cluster model fitting with preprocessed data """
        rests = pd.read_csv('hclustering_data.csv')
        forlabels = pd.read_csv('combined_rests_data.csv')
        matrices = ward(rests.values)
        dendrogram(matrices, orientation = 'right', labels = list(forlabels['name']))
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
        Creates a histogram for the distribution of ratings.
        """
        self.data['rating'].plot.hist(title='Histogram of Ratings', bins=10, ec='black')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.show()

    def plot_cuisine_count(self):
        """
        Creates a count plot for cuisines.
        """
        sns.countplot(y='cuisine', data=self.data, order=self.data['cuisine'].value_counts().index)
        plt.title('Count Plot of Cuisines')
        plt.show()

    def plot_borough_count(self):
        """
        Creates a count plot for boroughs.
        """
        sns.countplot(y='borough', data=self.data, order=self.data['borough'].value_counts().index)
        plt.title('Count Plot of Boroughs')
        plt.show()

    def correlation_analysis(self):
        """
        Performs correlation analysis on numerical columns of the dataset.
        """
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
