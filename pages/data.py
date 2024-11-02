import pandas as pd
from dash import html, dcc
import plotly.express as px
import plotly.graph_objs as go

# Load the datasets
weather_data = pd.read_csv('./data/weather_data.csv')
crop_data = pd.read_csv('./data/crop_data.csv')

# Column explanations
weather_column_explanations = {
    'date': 'Date and time of the weather observation',
    'city_name': 'Name of the city where the data was collected',
    'lat': 'Latitude coordinate of the location',
    'lon': 'Longitude coordinate of the location',
    'temp': 'Temperature in degrees Fahrenheit',
    'visibility': 'Visibility in meters',
    'dew_point': 'Dew point temperature in degrees Fahrenheit',
    'feels_like': 'Perceived temperature in degrees Fahrenheit',
    'temp_min': 'Minimum temperature at the time of observation',
    'temp_max': 'Maximum temperature at the time of observation',
    'pressure': 'Atmospheric pressure in hPa',
    'sea_level': 'Sea-level atmospheric pressure in hPa',
    'grnd_level': 'Ground-level atmospheric pressure in hPa',
    'humidity': 'Humidity percentage',
    'wind_speed': 'Wind speed in meters per second',
    'wind_deg': 'Wind direction in degrees (meteorological)',
    'wind_gust': 'Wind gust speed in meters per second',
    'clouds_all': 'Cloudiness percentage',
    'rain_1h': 'Rain volume for the last 1 hour in mm',
    'rain_3h': 'Rain volume for the last 3 hours in mm',
    'snow_1h': 'Snow volume for the last 1 hour in mm',
    'snow_3h': 'Snow volume for the last 3 hours in mm',
    'weather_id': 'Weather condition ID',
    'weather_main': 'Group of weather parameters (e.g., Rain, Snow)',
    'weather_description': 'Weather condition within the group',
    'weather_icon': 'Weather icon ID'
}

crop_column_explanations = {
    'Year': 'Year of the crop data record',
    'Commodity Code': 'Unique code assigned to each commodity',
    'Crop Name': 'Name of the crop',
    'County Code': 'Unique code assigned to each county',
    'County': 'Name of the county',
    'Harvested Acres': 'Area harvested in acres',
    'Yield': 'Crop yield per unit area (e.g., tons per acre)',
    'Production': 'Total production volume',
    'Price P/U': 'Price per unit of the crop',
    'Unit': 'Unit of measurement for the crop (e.g., tons, bushels)',
    'Value': 'Total value of the crop production in dollars'
}

# Prepare data for tables
def prepare_column_info(data, explanations):
    col_info = pd.DataFrame({
        'Column Name': data.columns,
        'Data Type': [str(dtype) for dtype in data.dtypes],
        'Description': [explanations.get(col, 'No description available') for col in data.columns]
    })
    return col_info

# Basic Properties
weather_properties = {
    "Number of Rows": weather_data.shape[0],
    "Number of Columns": weather_data.shape[1],
    "Column Names and Types": weather_data.dtypes.to_dict()
}
crop_properties = {
    "Number of Rows": crop_data.shape[0],
    "Number of Columns": crop_data.shape[1],
    "Column Names and Types": crop_data.dtypes.to_dict()
}

# Summary statistics
weather_summary = weather_data.describe().transpose()
crop_summary = crop_data.describe().transpose()

# Missing values
weather_missing = weather_data.isnull().sum().reset_index()
weather_missing.columns = ["Column", "Missing Values"]
crop_missing = crop_data.isnull().sum().reset_index()
crop_missing.columns = ["Column", "Missing Values"]

# Visualization functions
def plot_missing_values(data_missing, title):
    fig = px.bar(data_missing, x="Column", y="Missing Values", title=title)
    fig.update_layout(xaxis_title="Columns", yaxis_title="Missing Values Count")
    return fig

weather_col_info = prepare_column_info(weather_data, weather_column_explanations)
crop_col_info = prepare_column_info(crop_data, crop_column_explanations)

# Layout function for the Data page
def layout():
    return html.Div([
        html.H1("Data Page"),
        
        html.H2("Weather Data Properties"),
        html.P(f"Number of Rows: {weather_data.shape[0]}"),
        html.P(f"Number of Columns: {weather_data.shape[1]}"),
        
        html.H3("Columns Information"),
        dcc.Graph(figure=go.Figure(data=[go.Table(
            header=dict(values=["Column Name", "Data Type", "Description"],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[weather_col_info['Column Name'],
                               weather_col_info['Data Type'],
                               weather_col_info['Description']],
                       fill_color='lavender',
                       align='left'))
        ])),
        
        html.H2("Crop Data Properties"),
        html.P(f"Number of Rows: {crop_data.shape[0]}"),
        html.P(f"Number of Columns: {crop_data.shape[1]}"),
        
        html.H3("Columns Information"),
        dcc.Graph(figure=go.Figure(data=[go.Table(
            header=dict(values=["Column Name", "Data Type", "Description"],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[crop_col_info['Column Name'],
                               crop_col_info['Data Type'],
                               crop_col_info['Description']],
                       fill_color='lavender',
                       align='left'))
        ])),
        
        # Existing code for Summary Statistics and Missing Values...
        html.H2("Summary Statistics for Weather Data"),
        dcc.Graph(figure=go.Figure(data=[go.Table(
            header=dict(values=["Statistic"] + list(weather_summary.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[weather_summary.index] + [weather_summary[col] for col in weather_summary.columns],
                       fill_color='lavender',
                       align='left'))
        ])),
        
        html.H2("Summary Statistics for Crop Data"),
        dcc.Graph(figure=go.Figure(data=[go.Table(
            header=dict(values=["Statistic"] + list(crop_summary.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[crop_summary.index] + [crop_summary[col] for col in crop_summary.columns],
                       fill_color='lavender',
                       align='left'))
        ])),
        
        html.H2("Missing Values in Weather Data"),
        dcc.Graph(figure=plot_missing_values(weather_missing, "Missing Values in Weather Data")),
        
        html.H2("Missing Values in Crop Data"),
        dcc.Graph(figure=plot_missing_values(crop_missing, "Missing Values in Crop Data")),
    ])
