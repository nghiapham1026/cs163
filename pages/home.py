from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd

# City coordinates
city_data = [
    {"name": "San Francisco", "lat": 37.7749, "lon": -122.4194},
    {"name": "Fresno", "lat": 36.7378, "lon": -119.7871},
    {"name": "Eureka", "lat": 40.8021, "lon": -124.1637},
    {"name": "Palm Springs", "lat": 33.8303, "lon": -116.5453},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Riverside", "lat": 33.9533, "lon": -117.3962},
    {"name": "South Lake Tahoe", "lat": 38.9336, "lon": -119.9843}
]

# County coordinates (approximate centroids)
county_data = [
    {"name": "Sonoma", "lat": 38.5780, "lon": -122.9888},
    {"name": "Napa", "lat": 38.5025, "lon": -122.2654},
    {"name": "Santa Clara", "lat": 37.3337, "lon": -121.8907},
    {"name": "Alameda", "lat": 37.6017, "lon": -121.7195},
    {"name": "Tulare", "lat": 36.1342, "lon": -118.7625},
    {"name": "Kings", "lat": 36.0758, "lon": -119.8151},
    {"name": "Fresno", "lat": 36.9859, "lon": -119.2321},
    {"name": "Riverside", "lat": 33.9533, "lon": -117.3962},
    {"name": "Mendocino", "lat": 39.5500, "lon": -123.4384},
    {"name": "Nevada", "lat": 39.3030, "lon": -120.7401},
    {"name": "El Dorado", "lat": 38.7426, "lon": -120.4358}
]

def layout():
    return html.Div([
        html.H1("California Crop and Weather Analysis"),
        
        # Project Objective
        html.P(
            "This project aims to analyze the correlation between extreme weather conditions "
            "and crop yield, production, and harvested acres in various California counties. "
            "By examining historical weather patterns and crop data, we seek to understand how "
            "extreme weather events affect agricultural productivity and inform decision-making "
            "in climate adaptation strategies for California’s agricultural sector."
        ),

        # Stock Image
        html.Img(
            src="https://d17ocfn2f5o4rl.cloudfront.net/wp-content/uploads/2020/02/weather-monitoring-technologies-to-save-crops-from-mother-nature_optimized_optimized-1920x600.jpg", 
            alt="Illustration of weather impact on crops", 
            style={"width": "100%", "height": "auto", "margin-top": "20px"}
        ),

        # Map of California
        html.Div([
            dcc.Graph(
                id="california-map",
                figure=california_map()
            )
        ], style={"margin-top": "20px", "height": "500px"}),  # Adjust height as necessary
    ])

def california_map():
    # Initialize the map
    fig = go.Figure()

    # Add city markers
    for city in city_data:
        fig.add_trace(go.Scattermapbox(
            lat=[city["lat"]],
            lon=[city["lon"]],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=10, color="blue"),
            text=city["name"],
            textposition="top right",
            name="Cities",
            hoverinfo="text"
        ))

    # Add county markers
    for county in county_data:
        fig.add_trace(go.Scattermapbox(
            lat=[county["lat"]],
            lon=[county["lon"]],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=8, color="green"),
            text=county["name"],
            textposition="top right",
            name="Counties",
            hoverinfo="text"
        ))

    # Update layout with map style and centering
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",  # Light-themed map style
            center={"lat": 37.5, "lon": -119.5},  # Center on California
            zoom=5.5  # Adjust zoom level as needed
        ),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=True
    )

    return fig