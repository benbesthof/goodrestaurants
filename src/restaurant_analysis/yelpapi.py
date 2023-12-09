import requests
import json

# Yelp Fusion API endpoint
API_ENDPOINT = 'https://api.yelp.com/v3/businesses/search'
# Your API Key (replace with your actual API key)
API_KEY = 'KQnEhTUQYWDeocagAGP5tjMLjMRw2VhQlekKCOoSfZ8ssZrwtyaVMuKqWqyCZFGboLwNMCEzKxh0sPdZb38vmeVijeTXgcGkSZvpUytCSKrXjMrhlNhMYDgDv4tzZXYx'

def get_restaurant_data(location, term='restaurants', limit=50):
    """
    Fetches restaurant data from Yelp API based on location and term.
    :param location: The location to search for restaurants.
    :param term: The type of business to search for. Default is 'restaurants'.
    :param limit: Number of results to return. Maximum is 50.
    :return: A list of dictionaries containing restaurant data.
    """
    headers = {
        'Authorization': 'Bearer ' + API_KEY,
    }
    params = {
        'location': location,
        'term': term,
        'limit': limit,
    }

    response = requests.get(API_ENDPOINT, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        restaurants = data['businesses']
        return [
            {
                'name': restaurant['name'],
                'rating': restaurant['rating'],
                'categories': [category['title'] for category in restaurant['categories']]
            }
            for restaurant in restaurants
        ]
    else:
        print(f'Error fetching data: {response.status_code}')
        return []


