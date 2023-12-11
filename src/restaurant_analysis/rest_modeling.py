import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import warnings

class RestaurantModeling:
    def __init__(self, data, target, test_size=0.2, random_state=42):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.pipeline = self.create_pipeline()

    def prepare_data(self):

        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def create_pipeline(self):
        # Adjusting the data types for preprocessing
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object', 'bool']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        return pipeline

    def train(self):
        """
        Fits the model to the training data using the prepared pipeline.
        """
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline.named_steps['regressor']

    def evaluate(self):
        """
        Evaluates the model's performance on the test set using mean squared error and R squared metrics.
        """
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')

    def cross_validate(self, cv=5):
        """
        Applies cross-validation to the training data to assess model performance.
        
        Parameters:
        - cv (int): Number of cross-validation folds.
        """
        cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv, scoring='neg_mean_squared_error')
        print(f'CV Mean Squared Error: {-cv_scores.mean()}')
        return cv_scores

    def grid_search(self, param_grid):
        """
        Executes a grid search over a parameter grid to find the best model parameters.
        
        Parameters:
        - param_grid (dict): Dictionary with parameter names and lists of settings to try.
        """
        search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_.named_steps['regressor']
        print(f'Best parameters: {search.best_params_}')
        return search.best_params_

    def plot_residuals(self):
        """
        Creates a histogram of the residuals to visualize the distribution of prediction errors.
        """
        predictions = self.pipeline.predict(self.X_test)
        residuals = self.y_test - predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel('Residuals')
        plt.title('Histogram of Residuals')
        plt.show()

    def plot_actual_vs_predicted(self):
        """
        Displays a scatter plot comparing actual and predicted ratings.
        """
        predictions = self.pipeline.predict(self.X_test)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.y_test, y=predictions)
        plt.xlabel('Actual Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Actual vs Predicted Ratings')

        # Plotting a diagonal line for perfect predictions
        max_rating = max(self.y_test.max(), predictions.max())
        plt.plot([0, max_rating], [0, max_rating], '--k', linewidth=2)
        plt.show()

    def analyze_geographical_impact(self):
        """
        Analyzes the impact of location on Yelp ratings.
        """
        location_ratings = self.data.groupby(['borough', 'cuisine'])['rating'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='borough', y='rating', hue='cuisine', data=location_ratings)
        plt.title('Average Yelp Ratings by Borough and Cuisine')
        plt.show()
        return location_ratings

    def analyze_economic_status_correlation(self):
        """
        Analyzes the correlation between economic status and restaurant types.
        """
        economic_corr = self.data.groupby('borough').agg({'medincome': 'mean', 'rating': 'mean'})
        sns.scatterplot(x='medincome', y='rating', data=economic_corr)
        plt.title('Correlation between Median Income and Average Restaurant Rating')
        plt.show()
        return economic_corr.corr().iloc[0, 1]

    def analyze_demographic_influences(self):
        """
        Analyzes the influence of demographic factors on restaurant success.
        """
        demographic_corr = self.data[['foreign born persons pct', 'whitenolatino', 'rating']].corr()
        sns.heatmap(demographic_corr, annot=True)
        plt.title('Correlation Heatmap of Demographic Factors and Restaurant Rating')
        plt.show()
        return demographic_corr


    def analyze_cuisine_pricing_and_popularity(self):
        """
        Analyzes how the pricing of different cuisines is related to their ratings and popularity.
        """
        cuisine_analysis = self.data.groupby('cuisine').agg({'price': 'mean', 'rating': 'mean'}).reset_index()
        sns.scatterplot(x='price', y='rating', hue='cuisine', data=cuisine_analysis)
        plt.title('Cuisine Pricing vs. Yelp Ratings')
        plt.show()
        return cuisine_analysis

    
