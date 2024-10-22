import pandas as pd

# Load the dataset
df = pd.read_csv("../../data/weather_long.csv")

# Fix the datetime column by removing UTC and parsing it correctly
df['dt_iso'] = pd.to_datetime(df['dt_iso'].str.replace(r" \+0000 UTC", "", regex=True), format='%Y-%m-%d %H:%M:%S')

# Extract the date from the 'dt_iso' column
df['date'] = df['dt_iso'].dt.date

# Group by both date and city_name to include all unique city_name entries
daily_data = df.groupby(['date', 'city_name']).agg({
    'lat': 'mean',
    'lon': 'mean',
    'temp': 'mean',
    'visibility': 'mean',
    'dew_point': 'mean',
    'feels_like': 'mean',
    'temp_min': 'mean',
    'temp_max': 'mean',
    'pressure': 'mean',
    'sea_level': 'mean',
    'grnd_level': 'mean',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'wind_deg': 'mean',
    'wind_gust': 'mean',
    'rain_1h': 'mean',
    'rain_3h': 'mean',
    'snow_1h': 'mean',
    'snow_3h': 'mean',
    'clouds_all': 'mean',
    'weather_id': 'first',
    'weather_main': 'first',
    'weather_description': 'first',
    'weather_icon': 'first'
}).reset_index()

# Save the modified dataset to a new CSV file
file_path = "../../data/weather.csv"
daily_data.to_csv(file_path, index=False)
