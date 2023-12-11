import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

class Inference:
    """
    The Inference class is used for conducting various inference-based analyses 
    on a given dataset. It provides functionalities for hypothesis testing and generating statistical summaries.
    """
    
    def __init__(self, data):
        """
        Initializes the Inference class with a dataset.

        Args:
            data (pd.DataFrame): The dataset to be used for inference.
        """
        self.data = data

    def hypothesis_test_rating_cuisine(self):
        """
        Conducts an ANOVA test to determine if there are statistically 
        significant differences in ratings across different cuisines.

        This method assumes that the dataset has 'cuisine' and 'rating' columns.

        Returns:
          dict: A dictionary containing the F-statistic and the p-value of the test.
        """
        # Check if required columns are in the dataset
        if 'cuisine' not in self.data or 'rating' not in self.data:
            raise ValueError("Dataset must contain 'cuisine' and 'rating' columns.")

        # Prepare data for ANOVA
        groups = self.data.groupby('cuisine')['rating']

        # Conduct ANOVA
        f_value, p_value = stats.f_oneway(*[group for name, group in groups])

        # Return the F-statistic and p-value
        return {'F-Statistic': f_value, 'p-value': p_value}

    def statistical_summary(self, column):
        """
        Provides a statistical summary for a specified column in the dataset.

        Args:
            column (str): The name of the column for which the summary is required.

        Returns:
            pd.Series: A series containing descriptive statistics of the column.
        """
        return self.data[column].describe()

    def scatter_plot_rating_price(self):
        """
        Creates a scatter plot between the 'rating' and 'price' columns of the data.
        """
        # Mapping string values of 'price' to numerical values
        price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'Not Available': 0}
        numeric_price = self.data['price'].map(price_mapping)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data['rating'], y=numeric_price)
        plt.title("Scatter plot of Rating vs Price")
        plt.ylabel('Price Level')
        plt.show()


    def hypothesis_test_rating_cuisine(self):
        """
        Conducts an ANOVA test to determine if there are statistically 
        significant differences in ratings across different cuisines.

        This method assumes that the dataset has 'cuisine' and 'rating' columns.

        Returns:
          dict: A dictionary containing the F-statistic and the p-value of the test.
        """
        # Check if required columns are in the dataset
        if 'cuisine' not in self.data or 'rating' not in self.data:
            raise ValueError("Dataset must contain 'cuisine' and 'rating' columns.")

        # Prepare data for ANOVA
        groups = self.data.groupby('cuisine')['rating']

        # Conduct ANOVA
        f_value, p_value = stats.f_oneway(*[group for name, group in groups])

        # Return the F-statistic and p-value
        return {'F-Statistic': f_value, 'p-value': p_value}

    def statistical_summary(self, column):
        """
        Provides a statistical summary for a specified column in the dataset.

        Args:
            column (str): The name of the column for which the summary is required.

        Returns:
            pd.Series: A series containing descriptive statistics of the column.
        """
        return self.data[column].describe()

    def scatter_plot_rating_price(self):
        """
        Creates a scatter plot between the 'rating' and 'price' columns of the data.
        """
        # Mapping string values of 'price' to numerical values
        price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'Not Available': None}
        self.data['numeric_price'] = self.data['price'].map(price_mapping)

        # Handling 'Not Available' prices
        # Option 1: Drop rows where price is 'Not Available'
        plot_data = self.data.dropna(subset=['numeric_price'])

        # Option 2: Treat 'Not Available' as a specific numeric value (e.g., 0)
        # plot_data = self.data.fillna({'numeric_price': 0})

        # Creating the scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=plot_data['rating'], y=plot_data['numeric_price'])
        plt.title("Scatter plot of Rating vs Price")
        plt.ylabel('Price Level')
        plt.show()


    def chi_square_test(self, column1, column2):
        """
        Conducts a Chi-Square test of independence between two categorical columns.

        Args:
            column1 (str): The name of the first categorical column.
            column2 (str): The name of the second categorical column.

        Returns:
            dict: A dictionary containing the Chi-Square statistic and the p-value.
        """
        contingency_table = pd.crosstab(self.data[column1], self.data[column2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        return {'Chi-Square Statistic': chi2, 'p-value': p}

    def boxplot_rating_by_category(self, category):
        """
        Creates a box plot for ratings grouped by a specified category (e.g., 'cuisine' or 'borough').

        Args:
            category (str): The category to group ratings by.
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=self.data[category], y=self.data['rating'])
        plt.title(f'Boxplot of Ratings by {category.title()}')
        plt.xticks(rotation=45)
        plt.show()

    def kruskal_wallis_test(self, category):
        """
        Conducts a Kruskal-Wallis H-test to determine if there are statistically 
        significant differences in a numeric column (e.g., 'rating') across different categories.

        Args:
            category (str): The category to group the numeric column by.

        Returns:
            dict: A dictionary containing the H-statistic and the p-value.
        """
        groups = [group for name, group in self.data.groupby(category)['rating']]
        h_value, p_value = stats.kruskal(*groups)
        return {'H-Statistic': h_value, 'p-value': p_value}

    def correlation_analysis(self):
        """
        Performs correlation analysis between 'rating' and 'price'.

        Returns:
            float: The correlation coefficient.
        """
        # Converting price to a numeric scale
        price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4, 'Not Available': 0}
        self.data['price_level'] = self.data['price'].map(price_mapping)
        return self.data[['rating', 'price_level']].corr().iloc[0, 1]




   
