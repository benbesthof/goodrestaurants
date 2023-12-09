import requests
import json

# Yelp Fusion API endpoint
API_ENDPOINT = 'https://api.yelp.com/v3/businesses/search'
# Your API Key (replace with your actual API key)
API_KEY = 'KQnEhTUQYWDeocagAGP5tjMLjMRw2VhQlekKCOoSfZ8ssZrwtyaVMuKqWqyCZFGboLwNMCEzKxh0sPdZb38vmeVijeTXgcGkSZvpUytCSKrXjMrhlNhMYDgDv4tzZXYx'

def get_restaurant_data(location, cuisines, term='restaurants', limit=50):
    """
    Fetches restaurant data from the Yelp API.

    This function queries the Yelp API to gather data about restaurants
    based on the specified location and list of cuisines. It makes separate
    API calls for each cuisine type and aggregates the results.

    Parameters:
    - location (str): The location to search for restaurants.
    - cuisines (list of str): The list of cuisines to filter the search.
    - term (str, optional): The type of business to search for. Defaults to 'restaurants'.
    - limit (int, optional): The number of results to return per cuisine. Defaults to 50. Maximum is 50.

    Returns:
    - list of dict: A list of dictionaries, each containing data about a restaurant, including
                    its name, rating, price level, and categories.
    """
    headers = {
        'Authorization': 'Bearer ' + API_KEY,
    }
    all_restaurants = []
    for cuisine in cuisines:
        params = {
            'location': location,
            'term': f'{cuisine} {term}',
            'limit': limit,
        }

        response = requests.get(API_ENDPOINT, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            restaurants = data['businesses']
            for restaurant in restaurants:
                all_restaurants.append({
                    'name': restaurant['name'],
                    'rating': restaurant['rating'],
                    'price': restaurant.get('price', 'Not Available'),
                    'categories': [category['title'] for category in restaurant['categories']],
                    'cuisine': cuisine
                })
        else:
            print(f'Error fetching data for {cuisine}: {response.status_code}')

    return all_restaurants
