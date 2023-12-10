import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def summarize_restaurants(csv_file_path):
    # Read the CSV file
    data = pd.read_csv(csv_file_path)

    # Basic statistics
    num_restaurants = len(data)
    avg_rating = data['rating'].mean()

    # Distribution by cuisine
    cuisine_distribution = data['cuisine'].value_counts()

    # Distribution by borough
    borough_distribution = data['borough'].value_counts()

    # Returning the summary as a dictionary
    summary = {
        "Number of Restaurants": num_restaurants,
        "Average Rating": avg_rating,
        "Cuisine Distribution": cuisine_distribution,
        "Borough Distribution": borough_distribution
    }

    return summary

class DataSummary:
    """
    A class to perform data summary and handle missing data in a DataFrame.
    """
    def __init__(self, data):
        """
        Initializes the DataSummary object with the provided DataFrame.
        
        Parameters:
            data (pd.DataFrame): The data to be summarized.
        """
        self.data = data


    def display_head(self):
        """
        Displays the first five rows of the data.
        
        Returns:
        None
        """
        display(self.data.head())

    def display_tail(self):
        """
        Displays the last five rows of the data.
        
        Returns:
        None
        """
        display(self.data.tail())


    def get_shape(self):
        """
        Returns the shape of the data (number of rows and columns).
        
        Returns:
        tuple: A tuple containing the number of rows and columns.
        """
        return self.data.shape


    def missing_value_percent(self):
        """
        Calculates and returns the percentage of missing values in the data.
        
        Returns:
        pd.Series: A Series containing the percentage of missing values for each column.
        """
        return self.data.isna().sum() / len(self.data) * 100


    def data_info(self):
        """
        Displays a brief summary of the data including data types and non-null value counts.
        
        Returns:
        None
        """
        self.data.info()
        

    def categorical_descriptive_statistics(self):
        """
        Computes descriptive statistics for categorical variables in the data.
        
        The result's index will include count, unique, top, and freq. Analyzes both 
        numeric and object series, as well as DataFrame columns.
        
        Returns:
        pd.DataFrame: Descriptive statistics of categorical variables.
        """
        return self.data.describe(include=['object'])

    def numerical_descriptive_statistics(self):
        """
        Computes descriptive statistics for numerical variables in the data.
        
        The result's index will include: count, mean, std, min, 25%, 50%, 75%, and max.
        
        Returns:
        pd.DataFrame: Descriptive statistics of numerical variables.
        """
        return self.data.describe()

    def data_types(self):
        """
        Returns the data types of each column in the data.
        
        Returns:
        pd.Series: A Series containing data types of each column.
        """
        return self.data.dtypes
    

    def missing_value_summary(self):
        """
        Summarizes missing values in the DataFrame.
        
        Returns:
            A DataFrame with count and percentage of missing values for each column.
        """
        missing_count = self.data.isnull().sum()
        missing_percent = (missing_count / len(self.data)) * 100
        return pd.DataFrame({'count': missing_count, 'percent': missing_percent})
    

    def fill_missing_values(self, column, method='mean'):
        """
        Fills missing values in a specified column using a defined method.
        
        Parameters:
            column (str): The column to fill missing values in.
            method (str): The method to use for filling missing values ('mean', 'median', 'mode').
        """
        if method == 'mean':
            fill_value = self.data[column].mean()
        elif method == 'median':
            fill_value = self.data[column].median()
        elif method == 'mode':
            fill_value = self.data[column].mode()[0]
        else:
            raise ValueError("Method must be 'mean', 'median', or 'mode'")

        self.data[column].fillna(fill_value, inplace=True)


    def drop_column(self,dataframe, column):
        """
        Drops a specified column from a DataFrame.

        :param dataframe: A pandas DataFrame from which the column will be dropped.
        :param column: The name of the column to be dropped.
        :return: The DataFrame with the specified column dropped.
        """
        if column in dataframe.columns:
            return dataframe.drop(columns=[column])
        else:
            print(f"The column {column} does not exist in the DataFrame.")
            return dataframe