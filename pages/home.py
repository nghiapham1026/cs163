from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import json

# Load California county boundaries GeoJSON
with open('./data/California_County_Boundaries.geojson') as f:
    california_counties = json.load(f)

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

# Selected counties
selected_counties = ["Sonoma", "Napa", "Santa Clara", "Alameda", "Tulare",
                     "Kings", "Fresno", "Riverside", "Mendocino", "Nevada", "El Dorado"]

# Filter GeoJSON data to only include selected counties
filtered_counties = {
    "type": "FeatureCollection",
    "features": [
        feature for feature in california_counties["features"]
        if feature["properties"]["CountyName"] in selected_counties
    ]
}

def layout():
    return html.Div(
        className="main-container",
        children=[
            # Main Title
            html.H1(
                "California Crop and Weather Analysis",
                className="main-title"
            ),

            # Project Objective Section
            html.Div(
                className="project-objective-section",
                children=[
                    html.P(
                        "This project aims to analyze the correlation between extreme weather conditions "
                        "and crop yield, production, and harvested acres in various California counties. "
                        "By examining historical weather patterns and crop data, we seek to understand how "
                        "extreme weather events affect agricultural productivity and inform decision-making "
                        "in climate adaptation strategies for California’s agricultural sector.",
                        className="project-objective-text"
                    ),
                    html.Img(
                        src="https://d17ocfn2f5o4rl.cloudfront.net/wp-content/uploads/2020/02/weather-monitoring-technologies-to-save-crops-from-mother-nature_optimized_optimized-1920x600.jpg",
                        alt="Illustration of weather impact on crops",
                        className="project-objective-image"
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Geographical Analysis Section
            html.Div(
                className="geographical-analysis-section",
                children=[
                    html.H2(
                        "Geographical Locations Used in Analysis",
                        className="section-title"
                    ),
                    html.Div(
                        className="map-container",
                        children=[
                            dcc.Graph(
                                id="california-map",
                                figure=california_map(),  # Assuming this function generates the map figure
                                className="california-map-graph"
                            )
                        ]
                    )
                ]
            )
        ]
    )

def california_map():
    fig = go.Figure()

    # Add county boundaries as filled areas from the filtered GeoJSON
    fig.update_layout(mapbox=dict(
        style="carto-positron",
        center={"lat": 37.5, "lon": -119.5},
        zoom=5
    ))

    # Add GeoJSON polygons for each selected county with a semi-transparent fill
    fig.update_layout(mapbox_layers=[
        {
            "source": filtered_counties,
            "type": "fill",
            "below": "traces",
            "color": "rgba(0, 128, 0, 0.3)"  # Semi-transparent green fill
        }
    ])

    # Add city markers with labels
    for city in city_data:
        fig.add_trace(go.Scattermapbox(
            lat=[city["lat"]],
            lon=[city["lon"]],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=10, color="blue"),
            text=city["name"],
            textposition="top right",
            hoverinfo="text",
            name="Cities"
        ))

    # Add county labels at the centroids
    for county_item in county_data:
        fig.add_trace(go.Scattermapbox(
            lat=[county_item["lat"]],
            lon=[county_item["lon"]],
            mode="text",
            text=county_item["name"],
            textposition="middle center",
            textfont=dict(size=12, color="green"),
            hoverinfo="none",
            name="Counties"
        ))

    # Update layout settings
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False
    )

    return fig