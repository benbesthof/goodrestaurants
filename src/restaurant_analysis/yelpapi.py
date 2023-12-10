import requests
import json
import csv

# Yelp Fusion API endpoint
API_ENDPOINT = 'https://api.yelp.com/v3/businesses/search'
# Your API Key (replace with your actual API key)
API_KEY = 'cUDi-X8cHtlgplTSopy2RDatH3rGO37tSFJ9lTmHeF2xe2oHGF0F10GQx9A0zoLtChbqmndNwhxMt6kh3zaqUNoN72a5peuofw5R6Z3vr-aI13sptMpxbGg0NCB1ZXYx'

def get_restaurant_data(locations, cuisines, term='restaurants', limit=50, save_csv=False, csv_file_path='restaurants.csv'):
    """
    Fetches restaurant data from the Yelp API and optionally saves it to a CSV file.

    Parameters:
    - locations (list of str): The locations within New York to search for restaurants.
    - cuisines (list of str): The list of cuisines to filter the search.
    - term (str, optional): The type of business to search for. Defaults to 'restaurants'.
    - limit (int, optional): The number of results to return per location and cuisine. Defaults to 50. Maximum is 50.
    - save_csv (bool, optional): Whether to save the data to a CSV file. Defaults to False.
    - csv_file_path (str, optional): The file path for the CSV file if save_csv is True. Defaults to 'restaurants.csv'.

    Returns:
    - list of dict: A list of dictionaries, each containing data about a restaurant.
    """
    headers = {
        'Authorization': 'Bearer ' + API_KEY,
    }
    all_restaurants = []
    for location in locations:
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
                    restaurant_data = {
                        'name': restaurant['name'],
                        'rating': restaurant['rating'],
                        'price': restaurant.get('price', 'Not Available'),
                        'zip_code': restaurant['location'].get('zip_code', 'Not Available'),
                        'categories': ', '.join([category['title'] for category in restaurant['categories']]),
                        'cuisine': cuisine,
                        'borough': location
                    }
                    all_restaurants.append(restaurant_data)
            else:
                print(f'Error fetching data for {cuisine} in {location}: {response.status_code}')

    # Save to CSV if requested
    if save_csv:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['name', 'rating', 'price', 'zip_code', 'categories', 'cuisine', 'borough'])
            writer.writeheader()
            for data in all_restaurants:
                writer.writerow(data)

    return all_restaurants



