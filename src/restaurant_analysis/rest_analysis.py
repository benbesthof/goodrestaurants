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