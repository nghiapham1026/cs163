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

# List of weather features and target variables
weather_features = ['high_temp_days', 'low_temp_days', 'heavy_rain_days', 'snow_days',
                    'high_wind_days', 'low_visibility_days', 'cloudy_days']

# Define extreme weather variables and assign colors
extreme_weather_vars_colors = {
    "high_temp_days": "red",
    "low_temp_days": "blue",
    "heavy_rain_days": "green",
    "high_wind_days": "orange",
    "cloudy_days": "purple"
}

# Define county-specific thresholds
county_thresholds = {}
for county_name in df["County"].unique():
    county_df = df[df["County"] == county_name]
    thresholds = {}
    for var in extreme_weather_vars_colors.keys():
        thresholds[var] = {
            "high": county_df[var].quantile(0.9),
        }
    county_thresholds[county_name] = thresholds

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

        html.H2("Weather Impact on Crop Yield and Production", className="section-title"),
        html.P(
            "Select a crop and an extreme weather variable to explore the relationship between weather conditions and crop yield and production."
        ),
        html.Div([
            html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-weather-impact'),
            dcc.Dropdown(
                id='crop-dropdown-weather-impact',
                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                value=sorted(df["Crop Name"].unique())[0],
                className='dropdown'
            ),
            html.Label("Select Extreme Weather Variable:", className='dropdown-label', htmlFor='weather-feature-dropdown-weather-impact'),
            dcc.Dropdown(
                id='weather-feature-dropdown-weather-impact',
                options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in weather_features],
                value=weather_features[0],
                className='dropdown'
            ),
        ], className='dropdown-container', style={'width': '50%', 'margin': 'auto'}),
        html.Div([
            dcc.Graph(id='yield-per-acre-graph-weather-impact'),
            dcc.Graph(id='production-per-acre-graph-weather-impact')
        ], className='graph-container'),

        # New Plot Section
        html.H2("Weather Anomalies and Crop Yield", className="section-title"),
        html.P(
            "Select a county and a crop to see how weather anomalies impact crop yield and production over time."
        ),
        html.Div([
            html.Label("Select County:", className='dropdown-label', htmlFor='county-dropdown-anomalies'),
            dcc.Dropdown(
                id='county-dropdown-anomalies',
                options=[{'label': county, 'value': county} for county in sorted(df["County"].unique())],
                value=sorted(df["County"].unique())[0],
                className='dropdown'
            ),
            html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-anomalies'),
            dcc.Dropdown(
                id='crop-dropdown-anomalies',
                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                value=sorted(df["Crop Name"].unique())[0],
                className='dropdown'
            ),
        ], className='dropdown-container', style={'width': '50%', 'margin': 'auto'}),
        html.Div([
            dcc.Graph(id='yield-per-acre-graph-anomalies'),
            dcc.Graph(id='production-per-acre-graph-anomalies')
        ], className='graph-container'),
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

@callback(
    [Output('yield-per-acre-graph-weather-impact', 'figure'),
     Output('production-per-acre-graph-weather-impact', 'figure')],
    [Input('crop-dropdown-weather-impact', 'value'),
     Input('weather-feature-dropdown-weather-impact', 'value')]
)
def update_graphs(selected_crop, selected_weather_feature):
    # Filter data for selected crop
    crop_data = df[df['Crop Name'] == selected_crop]
    
    # Check if data is available
    if crop_data.empty:
        return {}, {}
    
    # Remove rows with NaN in selected weather feature
    crop_data = crop_data.dropna(subset=[selected_weather_feature, 'Yield Per Acre', 'Production Per Acre', 'County'])
    
    if crop_data.empty:
        return {}, {}
    
    # For 'Yield Per Acre' plot
    fig_yield = px.scatter(
        crop_data,
        x=selected_weather_feature,
        y='Yield Per Acre',
        color='County',
        trendline='ols',
        trendline_scope='trace',
        title=f"{selected_crop}: {selected_weather_feature.replace('_', ' ').title()} vs Yield Per Acre",
        labels={selected_weather_feature: selected_weather_feature.replace('_', ' ').title(), 'Yield Per Acre': 'Yield Per Acre'}
    )
    
    # For 'Production Per Acre' plot
    fig_production = px.scatter(
        crop_data,
        x=selected_weather_feature,
        y='Production Per Acre',
        color='County',
        trendline='ols',
        trendline_scope='trace',
        title=f"{selected_crop}: {selected_weather_feature.replace('_', ' ').title()} vs Production Per Acre",
        labels={selected_weather_feature: selected_weather_feature.replace('_', ' ').title(), 'Production Per Acre': 'Production Per Acre'}
    )
    
    # Update layouts if needed
    fig_yield.update_layout(
        xaxis_title=selected_weather_feature.replace('_', ' ').title(),
        yaxis_title='Yield Per Acre',
        legend_title='County',
        height=500
    )
    
    fig_production.update_layout(
        xaxis_title=selected_weather_feature.replace('_', ' ').title(),
        yaxis_title='Production Per Acre',
        legend_title='County',
        height=500
    )
    
    return fig_yield, fig_production

@callback(
    [Output('yield-per-acre-graph-anomalies', 'figure'),
     Output('production-per-acre-graph-anomalies', 'figure')],
    [Input('county-dropdown-anomalies', 'value'),
     Input('crop-dropdown-anomalies', 'value')]
)
def update_graphs_anomalies(selected_county, selected_crop):
    # Filter data for the selected county and crop
    county_crop_data = df[(df["County"] == selected_county) & (df["Crop Name"] == selected_crop)]

    if county_crop_data.empty:
        return {}, {}

    # Get thresholds for the current county
    thresholds = county_thresholds[selected_county]

    # Create the Yield Per Acre plot
    fig_yield = go.Figure()

    # Add crop yield trend line
    fig_yield.add_trace(go.Scatter(
        x=county_crop_data["Year"],
        y=county_crop_data["Yield Per Acre"],
        mode="lines+markers",
        name="Yield Per Acre",
        line=dict(color="black"),
        marker=dict(size=6)
    ))

    # Keep track of variables added to legend
    variables_plotted = set()

    # Add markers for each extreme weather variable
    for var, color in extreme_weather_vars_colors.items():
        for _, row in county_crop_data.iterrows():
            if row[var] > thresholds[var]["high"]:
                if var not in variables_plotted:
                    showlegend = True
                    variables_plotted.add(var)
                else:
                    showlegend = False
                fig_yield.add_trace(go.Scatter(
                    x=[row["Year"]],
                    y=[row["Yield Per Acre"]],
                    mode="markers",
                    name=var.replace('_', ' ').title(),
                    marker=dict(size=12, color=color, opacity=0.8),
                    hovertext=f"{var.replace('_', ' ').title()}: {row[var]}",
                    showlegend=showlegend
                ))

    # Customize layout
    fig_yield.update_layout(
        title=f"Weather Anomalies and Yield in {selected_county} - {selected_crop}",
        xaxis_title="Year",
        yaxis_title="Yield Per Acre",
        legend_title="Legend",
        height=500
    )

    # Create the Production Per Acre plot
    fig_production = go.Figure()

    # Add crop production trend line
    fig_production.add_trace(go.Scatter(
        x=county_crop_data["Year"],
        y=county_crop_data["Production Per Acre"],
        mode="lines+markers",
        name="Production Per Acre",
        line=dict(color="black"),
        marker=dict(size=6)
    ))

    # Keep track of variables added to legend
    variables_plotted = set()

    # Add markers for each extreme weather variable
    for var, color in extreme_weather_vars_colors.items():
        for _, row in county_crop_data.iterrows():
            if row[var] > thresholds[var]["high"]:
                if var not in variables_plotted:
                    showlegend = True
                    variables_plotted.add(var)
                else:
                    showlegend = False
                fig_production.add_trace(go.Scatter(
                    x=[row["Year"]],
                    y=[row["Production Per Acre"]],
                    mode="markers",
                    name=var.replace('_', ' ').title(),
                    marker=dict(size=12, color=color, opacity=0.8),
                    hovertext=f"{var.replace('_', ' ').title()}: {row[var]}",
                    showlegend=showlegend
                ))

    # Customize layout
    fig_production.update_layout(
        title=f"Weather Anomalies and Production in {selected_county} - {selected_crop}",
        xaxis_title="Year",
        yaxis_title="Production Per Acre",
        legend_title="Legend",
        height=500
    )

    return fig_yield, fig_production