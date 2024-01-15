# Five Borough Restaurant Analysis

## Overview

This study investigates the food scene in the five boroughs of New York. Based on demographic data from the Census and Yelp API, the analysis explores how geography, economy, and various cultural variables intersect with restaurant success and customer preferences.

## Team 
- [@sauceboss7](https://github.com/sauceboss7) 
- [@VGiannac](https://github.com/VGiannac)
- [@Mapalo2023](https://github.com/Mapalo2023)

## Research Questions

1. **Geographical Impact on Yelp Ratings:**
   - Investigates how the specific location within the Tri-State area influences restaurant Yelp ratings, especially concerning particular cuisines.

2. **Economic Status and Restaurant Types:**
   - Explores whether there is a discernible correlation between the economic status of a neighborhood and the types of restaurants it hosts.

3. **Demographic Influences on Restaurant Success:**
   - Examines the extent to which demographic factors, such as ethnic composition, contribute to the success of various restaurant types.

4. **Optimal Locations for New Restaurants:**
   - Identifies current trends and optimal locations for opening new restaurants based on Yelp rating patterns.

5. **Cuisine Pricing, Ratings, and Popularity:**
   - Investigates how the pricing of different cuisines relates to their Yelp ratings and overall popularity.

## Data Collection

- **Yelp API:**
  - Gathered restaurant ratings, cuisine types, pricing, and customer reviews.
  - [Yelp API Documentation](https://docs.developer.yelp.com/docs/fusion-intro)

- **Census Data (CSV):**
  - Obtained demographic and economic data for the region.
  - [Census Data for New York Tri-State Area](https://www.census.gov/quickfacts/fact/table/bergencountynewjersey,richmondcountynewyork,bronxcountynewyork,queenscountynewyork,kingscountynewyork,newyorkcountynewyork/PST045222)

## Exploratory Data Analysis (EDA)

- Utilized Python and data visualization libraries (Pandas, Matplotlib, Seaborn).
- Conducted EDA to uncover trends and connections in restaurant ratings, cuisine categories, and prices.

## Data Summary

- **Number of Restaurants:** 1000
- **Average Rating:** 4.154
- **Cuisine Distribution:** Italian, American, Japanese, Mexican, Kosher
- **Borough Distribution:** Brooklyn, Queens, Staten Island, The Bronx

## Data Analysis

- Explored average ratings, unique cuisines, and correlation analyses for deeper insights.
- Investigated the relationship between rating and price, conducting hypothesis tests and creating scatter plots.

## Conclusion

To sum up, this study offers a thorough examination of the restaurant scene in the Tri-State region. The results provide insightful information for investors, restaurant operators, and anyone else curious about the intricate interactions between demographic, economic, and geographic variables that shape the region's varied food landscape.

## How to Use the Python Package

### Installation

To use the Python package for this project, follow these steps:

1. **Clone the Repository:**

2. **Install Dependencies:**

## Running the Analysis

1. **Run the Data Collection Script:**
- Open a Jupyter notebook or your preferred Python environment.
- Navigate to the `/src/restaurant_analysis/` directory.
- Run the `yelpapi.py` script to collect restaurant data using the Yelp API.

2. **Data Cleaning:**
- If needed, refer to the data cleaning Excel file (`/path/to/data_cleaning.xlsx`) for any preprocessing steps.

3. **Exploratory Data Analysis (EDA):**
- Open the EDA notebook.
- Execute cells to run the exploratory analysis and visualize trends.

4. **Utilize the Python Package:**
- Import the necessary modules from the `src.restaurant_analysis` package.
- Create instances of classes like `ExploratoryDataAnalysis` or `DataSummary` with your dataset.
- Utilize the provided methods for various analyses, such as correlation, average ratings, and hypothesis testing.

Example:
```python
from src.restaurant_analysis.rest_analysis import ExploratoryDataAnalysis

# Load your dataset into a DataFrame
df_restaurants = pd.read_csv('restaurants_data.csv')## Running the Analysis

1. **Run the Data Collection Script:**
- Open a Jupyter notebook or your preferred Python environment.
- Navigate to the `/src/restaurant_analysis/` directory.
- Run the `yelpapi.py` script to collect restaurant data using the Yelp API.

2. **Data Cleaning:**
- If needed, refer to the data cleaning Excel file (`/path/to/data_cleaning.xlsx`) for any preprocessing steps.

3. **Exploratory Data Analysis (EDA):**
- Open the EDA notebook (`/notebooks/eda.ipynb`).
- Execute cells to run the exploratory analysis and visualize trends.

4. **Utilize the Python Package:**
- Import the necessary modules from the `src.restaurant_analysis` package.
- Create instances of classes like `ExploratoryDataAnalysis` or `DataSummary` with your dataset.
- Utilize the provided methods for various analyses, such as correlation, average ratings, and hypothesis testing.

Example:
```python
from src.restaurant_analysis.rest_analysis import ExploratoryDataAnalysis

# Load your dataset into a DataFrame
df_restaurants = pd.read_csv('restaurants_data.csv') 
