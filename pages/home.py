from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
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

# Load the data
df = pd.read_csv('./data/merged_yearly.csv')

# Define extreme weather variables
extreme_weather_vars = ['high_temp_days', 'low_temp_days', 'heavy_rain_days', 'snow_days',
                        'high_wind_days', 'low_visibility_days', 'cloudy_days']

def layout():
    return html.Div(className="main-container", children=[
        html.H1("California Crop and Weather Analysis", className="main-title"),

        # Project Objective
        html.P(
            "This project aims to analyze the correlation between extreme weather conditions "
            "and crop yield, production, and harvested acres in various California counties. "
            "By examining historical weather patterns and crop data, we seek to understand how "
            "extreme weather events affect agricultural productivity and inform decision-making "
            "in climate adaptation strategies for California’s agricultural sector.",
            className="project-objective"
        ),

        # Stock Image
        html.Img(
            src="https://d17ocfn2f5o4rl.cloudfront.net/wp-content/uploads/2020/02/weather-monitoring-technologies-to-save-crops-from-mother-nature_optimized_optimized-1920x600.jpg",
            alt="Illustration of weather impact on crops",
            style={"width": "100%", "height": "auto", "margin-top": "20px"},
            className="stock-image"
        ),

        html.H1("Geographical Locations Used in Analysis", className="section-title"),

        # Map of California
        html.Div(className="map-container", children=[
            dcc.Graph(
                id="california-map",
                figure=california_map(),
                className="california-map-graph"
            )
        ], style={"margin-top": "20px", "height": "500px"}),

        # New Plot Section
        html.H2("Impact of Extreme Weather on Crop Yield", className="section-title"),
        html.P(
            "Select a crop and an extreme weather variable to see how extreme weather conditions impact crop yields across different counties."
        ),
        html.Div([
            html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown'),
            dcc.Dropdown(
                id='crop-dropdown',
                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                value=sorted(df["Crop Name"].unique())[0],
                className='dropdown'
            ),
            html.Label("Select Extreme Weather Variable:", className='dropdown-label', htmlFor='extreme-variable-dropdown'),
            dcc.Dropdown(
                id='extreme-variable-dropdown',
                options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in extreme_weather_vars],
                value=extreme_weather_vars[0],
                className='dropdown'
            ),
        ], className='dropdown-container', style={'width': '50%', 'margin': 'auto'}),
        dcc.Graph(id='yield-comparison-graph', className='graph'),
    ])

def california_map():
    fig = go.Figure()

    # Add county boundaries as filled areas from the filtered GeoJSON
    fig.update_layout(mapbox=dict(
        style="carto-positron",
        center={"lat": 37.5, "lon": -119.5},
        zoom=5.5
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
    for county in county_data:
        fig.add_trace(go.Scattermapbox(
            lat=[county["lat"]],
            lon=[county["lon"]],
            mode="text",
            text=county["name"],
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

# Callback for the plot
@callback(
    Output('yield-comparison-graph', 'figure'),
    [Input('crop-dropdown', 'value'),
     Input('extreme-variable-dropdown', 'value')]
)
def update_graph(selected_crop, selected_var):
    # Get unique counties
    counties = df["County"].unique()

    # DataFrame to store results
    yield_comparison_list = []

    for county in counties:
        # Filter data for the current county and selected crop
        county_data = df[(df["County"] == county) & (df["Crop Name"] == selected_crop)]
        if county_data.empty:
            continue  # Skip if no data for this county and crop

        # Handle possible NaNs in the selected variable
        var_values = county_data[selected_var].dropna()
        if var_values.empty:
            continue  # Skip if variable data is missing

        # Calculate the 65th percentile threshold for the selected variable
        threshold = var_values.quantile(0.65)

        # Identify extreme years for the current variable in the current county
        extreme_years = county_data[county_data[selected_var] > threshold]["Year"].unique()

        # Calculate average crop yield for extreme years
        extreme_yield = county_data[county_data["Year"].isin(extreme_years)]["Yield Per Acre"].mean()
        if pd.isna(extreme_yield):
            extreme_yield = 0

        # Calculate average crop yield for non-extreme years
        non_extreme_yield = county_data[~county_data["Year"].isin(extreme_years)]["Yield Per Acre"].mean()
        if pd.isna(non_extreme_yield):
            non_extreme_yield = 0

        # Append to list
        yield_comparison_list.append({
            "County": county,
            "Extreme Years Yield": extreme_yield,
            "Non-Extreme Years Yield": non_extreme_yield
        })

    if not yield_comparison_list:
        return px.bar(title="No data available for the selected crop and variable.")

    # Create DataFrame
    yield_comparison_df = pd.DataFrame(yield_comparison_list)

    # Melt the DataFrame for plotting
    yield_comparison_melted = yield_comparison_df.melt(
        id_vars="County",
        value_vars=["Extreme Years Yield", "Non-Extreme Years Yield"],
        var_name="Condition",
        value_name="Yield Per Acre"
    )

    # Create the bar plot
    fig = px.bar(
        yield_comparison_melted,
        x="County",
        y="Yield Per Acre",
        color="Condition",
        barmode="group",
        title=f"Impact of {selected_var.replace('_', ' ').title()} on {selected_crop} Yield Across Counties",
        labels={"Yield Per Acre": "Yield Per Acre", "Condition": "Condition"},
        category_orders={"County": sorted(counties)}
    )

    fig.update_layout(
        xaxis_title="County",
        yaxis_title="Average Yield Per Acre",
        legend_title="Condition"
    )

    return fig
