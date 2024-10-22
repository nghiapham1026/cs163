import plotly.express as px
from dash import dcc, html

# Define the cities and their coordinates
cities = ['San Francisco', 'Riverside', 'Fresno', 'South Lake Tahoe', 'Palm Springs', 'Eureka', 'Los Angeles']
latitudes = [37.7749, 33.9806, 36.7378, 38.9399, 33.8303, 40.8021, 34.0522]
longitudes = [-122.4194, -117.3755, -119.7871, -119.9772, -116.5453, -124.1637, -118.2437]

# Create the map
def render_home_page():
    # Create the mapbox scatter plot with fixed marker size
    fig = px.scatter_mapbox(
        lat=latitudes,
        lon=longitudes,
        hover_name=cities,
        zoom=5,
        height=500,
        size=[10] * len(cities),  # Fixed marker size for all cities
        size_max=15,  # Maximum size of the markers
        center={"lat": 36.7783, "lon": -119.4179},  # Centered over California
        mapbox_style="open-street-map"  # Choose the map style (can be 'satellite', 'open-street-map', etc.)
    )
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  # Adjust layout to remove extra margins
    
    return html.Div([
        html.H1("Welcome to the California Weather Data and Crop Yield Analysis"),
        html.Img(src='/assets/illustration.png', style={'width': '60%', 'margin': 'auto', 'display': 'block'}),
        html.P("This site provides a comprehensive analysis of historical and real-time weather data in California "
               "along with crop yield trends. The goal is to offer insights for agricultural planning, climate studies, "
               "and prediction of future weather and crop yield patterns."),
        html.Hr(),
        html.H2("California Cities"),
        html.P("Below is a map showing the locations of key cities for weather data analysis:"),
        
        # Display the map
        dcc.Graph(figure=fig),
        
        html.Hr(),
        html.H2("Site Navigation"),
        html.P("Use the tabs above to explore the following sections:"),
        html.Ul([
            html.Li("Weather Data: Explore detailed weather data for different regions in California."),
            html.Li("Crop Data: Examine trends and insights related to crop yield."),
            html.Li("Prediction: See predictions for future weather patterns and crop yields based on machine learning models."),
            html.Li("Analysis: Dive deeper into data analysis, including anomaly detection and correlation between climate factors and crop yields.")
        ]),
        html.P("This project applies data science techniques such as time series analysis, anomaly detection, and machine learning "
               "to provide valuable insights into California's agricultural and climate trends.")
    ])
