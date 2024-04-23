import requests
from bs4 import BeautifulSoup
import os
import urllib.request

url = "https://www.trollandtoad.com/pokemon/psa-graded-pokemon-cards/10377"

r = requests.get(url)

soup = BeautifulSoup(r.text, 'html.parser')

listings = soup.find_all('div', class_='item')

for listing in listings:
    image_url = listing.find('img')['src']
    image_filename = os.path.join("./data/trollandtoad/", image_url.split("/")[-1])
    urllib.request.urlretrieve(image_url, image_filename)
