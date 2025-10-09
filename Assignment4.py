import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Q1: Web Scraping - Books to Scrape

# Define the URL
url = "https://books.toscrape.com/"

# Create a list to store book details
book_details = []

# Function to scrape book details from a single page
def scrape_books(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all book containers
    books = soup.find_all('article', class_='product_pod')

    # Loop through each book and extract details
    for book in books:
        title = book.find('h3').find('a')['title']
        price = book.find('p', class_='price_color').text
        availability = book.find('p', class_='instock availability').text.strip()
        star_rating = book.find('p', class_='star-rating')['class'][1]

        # Append the book details to the list
        book_details.append({
            'Title': title,
            'Price': price,
            'Availability': availability,
            'Star Rating': star_rating
        })

# Scrape the first page
scrape_books(url)

# Handle pagination
next_page = True
while next_page:
    # Find the next page URL
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    next_button = soup.find('li', class_='next')
    if next_button:
        next_url = url + next_button.find('a')['href']
        scrape_books(next_url)
        url = next_url
    else:
        next_page = False

# Convert the list to a Pandas DataFrame and save to CSV
df_books = pd.DataFrame(book_details)
df_books.to_csv('books.csv', index=False)


# Q2: Web Scraping - IMDB Top 250 Movies

# Set up Selenium WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Navigate to the IMDB Top 250 Movies list
driver.get("https://www.imdb.com/chart/top/")

# Allow time for the page to load
time.sleep(3)

# Create a list to store movie details
movie_details = []

# Scrape the top 250 movies
movies = driver.find_elements(By.CSS_SELECTOR, ".lister-list tr")

for movie in movies:
    rank = movie.find_element(By.CSS_SELECTOR, ".titleColumn span").text
    title = movie.find_element(By.CSS_SELECTOR, ".titleColumn a").text
    year = movie.find_element(By.CSS_SELECTOR, ".titleColumn span.secondaryInfo").text.strip('()')
    rating = movie.find_element(By.CSS_SELECTOR, ".imdbRating strong").text

    # Append the movie details to the list
    movie_details.append({
        'Rank': rank,
        'Movie Title': title,
        'Year of Release': year,
        'IMDB Rating': rating
    })

# Convert the list to a Pandas DataFrame and save to CSV
df_movies = pd.DataFrame(movie_details)
df_movies.to_csv('imdb_top250.csv', index=False)

# Close the browser window
driver.quit()


# Q3: Web Scraping - Weather Information


# Define the base URL for weather information
base_url = "https://www.timeanddate.com/weather/"

# List of world cities to scrape
cities = ["New York", "London", "Tokyo", "Paris", "Sydney", "Berlin", "Moscow", "Dubai"]

# Create a list to store weather details
weather_details = []

# Function to scrape weather data for each city
def scrape_weather(city):
    url = base_url + "usa/" + city.lower().replace(" ", "-") + "/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract weather information
    city_name = city
    temperature = soup.find('div', class_='h2').text.strip()
    weather_condition = soup.find('div', class_='h2').find_next('p').text.strip()

    # Append the weather details to the list
    weather_details.append({
        'City Name': city_name,
        'Temperature': temperature,
        'Weather Condition': weather_condition
    })

# Scrape weather data for each city
for city in cities:
    scrape_weather(city)

# Convert the list to a Pandas DataFrame and save to CSV
df_weather = pd.DataFrame(weather_details)
df_weather.to_csv('weather.csv', index=False)

# --------------------------------------------------
# Name: Adityavir Singh Randhawa
# Roll No: 102483009
# Sub Group: 3C53
# --------------------------------------------------
