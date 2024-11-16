import pandas as pd
from dash import html, dcc
import plotly.express as px
import plotly.graph_objs as go

# Load the datasets
weather_data = pd.read_csv('./data/weather_monthly.csv')
crop_data = pd.read_csv('./data/crops_yearly.csv')

# Column descriptions
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
    'weather_icon': 'Weather icon ID',
    
    # Extreme weather columns
    'high_temp_days': 'Days with temperatures above a high threshold',
    'low_temp_days': 'Days with temperatures below a low threshold',
    'heavy_rain_days': 'Days with rainfall above a threshold',
    'snow_days': 'Days with measurable snowfall',
    'high_wind_days': 'Days with high wind speeds exceeding a threshold',
    'low_visibility_days': 'Days with low visibility conditions',
    'cloudy_days': 'Days with high cloud cover'
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
    'Value': 'Total value of the crop production in dollars',
    
    # Per-acre columns
    'Yield Per Acre': 'Average yield per harvested acre for the crop',
    'Production Per Acre': 'Total production volume divided by harvested acres',
    'Value Per Acre': 'Total value of production divided by harvested acres'
}

# Helper functions
def prepare_column_info(data, explanations):
    """Prepare column information with descriptions for a dataset."""
    col_info = pd.DataFrame({
        'Column Name': data.columns,
        'Data Type': [str(dtype) for dtype in data.dtypes],
        'Description': [explanations.get(col, 'No description available') for col in data.columns]
    })
    return col_info

def plot_missing_values(data_missing, title):
    """Plot missing values as a bar chart."""
    fig = px.bar(data_missing, x="Column", y="Missing Values", title=title)
    fig.update_layout(xaxis_title="Columns", yaxis_title="Missing Values Count")
    return fig

# Data properties, summaries, and missing values
weather_col_info = prepare_column_info(weather_data, weather_column_explanations)
crop_col_info = prepare_column_info(crop_data, crop_column_explanations)
weather_summary = weather_data.describe().transpose()
crop_summary = crop_data.describe().transpose()
weather_missing = weather_data.isnull().sum().reset_index()
weather_missing.columns = ["Column", "Missing Values"]
crop_missing = crop_data.isnull().sum().reset_index()
crop_missing.columns = ["Column", "Missing Values"]

# Layout function for the Data page
def layout():
    return html.Div(className="data-overview-page-container", children=[
        
        # Page Header
        html.H1("Data Overview", className="data-overview-page-header"),
        
        # Weather Data Properties Section
        html.Div(className="weather-data-properties-section", children=[
            html.H2("Weather Data Properties", className="weather-data-properties-header"),
            html.P(f"Number of Rows: {weather_data.shape[0]}", className="weather-data-num-rows"),
            html.P(f"Number of Columns: {weather_data.shape[1]}", className="weather-data-num-columns"),
            
            html.H3("Columns Information", className="weather-data-columns-info-header"),
            dcc.Graph(className="weather-data-columns-info-graph", figure=go.Figure(data=[go.Table(
                header=dict(values=["Column Name", "Data Type", "Description"],
                            fill_color='paleturquoise', align='left'),
                cells=dict(values=[weather_col_info['Column Name'], weather_col_info['Data Type'], weather_col_info['Description']],
                           fill_color='lavender', align='left'))
            ])),
        ]),
    
        # Crop Data Properties Section
        html.Div(className="crop-data-properties-section", children=[
            html.H2("Crop Data Properties", className="crop-data-properties-header"),
            html.P(f"Number of Rows: {crop_data.shape[0]}", className="crop-data-num-rows"),
            html.P(f"Number of Columns: {crop_data.shape[1]}", className="crop-data-num-columns"),
            
            html.H3("Columns Information", className="crop-data-columns-info-header"),
            dcc.Graph(className="crop-data-columns-info-graph", figure=go.Figure(data=[go.Table(
                header=dict(values=["Column Name", "Data Type", "Description"],
                            fill_color='paleturquoise', align='left'),
                cells=dict(values=[crop_col_info['Column Name'], crop_col_info['Data Type'], crop_col_info['Description']],
                           fill_color='lavender', align='left'))
            ])),
        ]),
    
        # Summary Statistics for Weather Data
        html.Div(className="weather-data-summary-section", children=[
            html.H2("Summary Statistics for Weather Data", className="weather-data-summary-header"),
            dcc.Graph(className="weather-data-summary-graph", figure=go.Figure(data=[go.Table(
                header=dict(values=["Statistic"] + list(weather_summary.columns),
                            fill_color='paleturquoise', align='left'),
                cells=dict(values=[weather_summary.index] + [weather_summary[col] for col in weather_summary.columns],
                           fill_color='lavender', align='left'))
            ])),
        ]),
    
        # Summary Statistics for Crop Data
        html.Div(className="crop-data-summary-section", children=[
            html.H2("Summary Statistics for Crop Data", className="crop-data-summary-header"),
            dcc.Graph(className="crop-data-summary-graph", figure=go.Figure(data=[go.Table(
                header=dict(values=["Statistic"] + list(crop_summary.columns),
                            fill_color='paleturquoise', align='left'),
                cells=dict(values=[crop_summary.index] + [crop_summary[col] for col in crop_summary.columns],
                           fill_color='lavender', align='left'))
            ])),
        ]),
    
        # Missing Values Section for Weather Data
        html.Div(className="weather-data-missing-values-section", children=[
            html.H2("Missing Values in Weather Data", className="weather-data-missing-values-header"),
            dcc.Graph(className="weather-data-missing-values-graph", figure=plot_missing_values(weather_missing, "Missing Values in Weather Data")),
        ]),
    
        # Missing Values Section for Crop Data
        html.Div(className="crop-data-missing-values-section", children=[
            html.H2("Missing Values in Crop Data", className="crop-data-missing-values-header"),
            dcc.Graph(className="crop-data-missing-values-graph", figure=plot_missing_values(crop_missing, "Missing Values in Crop Data")),
        ]),
    ])