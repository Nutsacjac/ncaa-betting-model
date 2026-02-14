import requests

response = requests.get('https://api.odds-api.io/v3/sports')
sports = response.json()
print(sports)
