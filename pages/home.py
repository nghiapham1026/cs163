from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import json

# Load California county boundaries GeoJSON
with open('./data/California_County_Boundaries.geojson') as f:
    california_counties = json.load(f)

techniques_df = pd.read_csv('./data/techniques.csv')

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
                    html.H2(
                        "Project Objective",
                        className="section-title"
                    ),
                    html.P(
                        "This project aims to analyze the correlation between extreme weather conditions "
                        "and crop yield, production, and harvested acres in various California counties. "
                        "By examining historical weather patterns and crop data, we seek to understand how "
                        "extreme weather events affect agricultural productivity and inform decision-making "
                        "in climate adaptation strategies for California’s agricultural sector.",
                        className="project-objective-text"
                    ),
                    # Methodologies Section
                    html.Div(
                        className="project-methodologies-section",
                        children=[
                            html.H2(
                                "Methodologies Used",
                                className="section-title"
                            ),
                            html.P(
                                "We employed a combination of data integration, statistical analysis, predictive modeling, "
                                "and visualization techniques to assess the impact of extreme weather on crop outcomes. "
                                "Our methodologies include:",
                                className="methodologies-intro-text"
                            ),
                            html.Ul(
                                className="methodologies-list",
                                children=[
                                    html.Li(
                                        "🗂️ Data Integration and Feature Engineering: Merged historical weather data with crop yield data "
                                        "to create a comprehensive dataset. Engineered new features to quantify extreme weather events "
                                        "using location-specific thresholds based on historical weather statistics."
                                    ),
                                    html.Li(
                                        "📊 Statistical Analysis: Conducted correlation analyses and hypothesis testing to identify significant "
                                        "weather variables affecting crop yields, harvested acres, and production per acre for each county-crop combination."
                                    ),
                                    html.Li(
                                        "🤖 Predictive Modeling: Developed machine learning models, including Random Forest and Ridge Regression, "
                                        "incorporating significant weather features and lagged variables to predict future crop yields. Evaluated model "
                                        "performance using metrics like R-squared and RMSE."
                                    ),
                                    html.Li(
                                        "📈 Visualization: Created heatmaps, coefficient plots, and time series charts to visualize relationships "
                                        "between weather variables and crop outcomes, aiding in interpretation and decision-making."
                                    ),
                                ]
                            ),
                        ]
                    ),
                    # Website Layout Section
                    html.Div(
                        className="website-layout-section",
                        children=[
                            html.H2(
                                "Website Layout",
                                className="section-title"
                            ),
                            html.P(
                                "Our website is structured into several pages to facilitate exploration of the data and findings:",
                                className="layout-intro-text"
                            ),
                            html.Ul(
                                className="layout-list",
                                children=[
                                    html.Li(
                                        [
                                            html.B("🏠 Home Page: "),
                                            "An overview of the project objectives and methodologies."
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("📄 Data Overview Page: "),
                                            "Summary and exploration of the datasets used, including data sources and preprocessing steps."
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("🌾 Crops Visualization Page: "),
                                            "Interactive visualizations of the crops dataset, allowing users to explore crop yields, production, "
                                            "and harvested acres across different counties and years."
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("☁️ Weather Visualization Page: "),
                                            "Visualizations of the weather dataset, showcasing trends and patterns in extreme weather events across California counties."
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("📊 Visualization Page: "),
                                            "Combined visualizations illustrating the relationships between crop outcomes and extreme weather variables."
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.B("🔮 Prediction Page: "),
                                            "Presentation of the predictive models' performance, including training results and a model demonstration "
                                            "for forecasting crop yields based on weather inputs."
                                        ]
                                    ),
                                ]
                            ),
                        ]
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
                        className="geographical-description",
                        children=[
                            html.H3(
                                "Incorporating Geographical Data into the Analysis",
                                className="description-title"
                            ),
                            html.P(
                                "Geographical data plays a crucial role in our analysis by enabling us to assess "
                                "the impact of extreme weather conditions on crop yields across different regions "
                                "in California. We have selected key counties known for their significant production "
                                "of almonds, wine grapes, and processing tomatoes. By focusing on these areas, "
                                "we capture a diverse range of climatic conditions and agricultural practices, "
                                "providing a comprehensive understanding of how location-specific factors influence "
                                "crop performance.",
                                className="description-text"
                            )
                        ]
                    ),
                    html.Div(
                        className="map-container",
                        children=[
                            dcc.Graph(
                                id="california-map",
                                figure=california_map(),
                                className="california-map-graph"
                            )
                        ]
                    ),
                    html.Div(
                        className="map-description",
                        children=[
                            html.H3(
                                "Map Visualization Details",
                                className="description-title"
                            ),
                            html.P(
                                "This interactive map highlights the California counties (in yellow) and cities (in blue) "
                                "included in our study. Hover over each county to view detailed information about its primary crops, "
                                "irrigation and farming methods, and pesticide usage levels. This visualization showcases the "
                                "geographical distribution of agricultural practices and crop types, offering insights into regional "
                                "differences in farming techniques, environmental factors, and how they may relate to crop yields "
                                "and weather impacts.",
                                className="description-text"
                            )
                        ]
                    )
                ]
            )
        ]
    )

def california_map():
    # Initialize the map figure
    fig = go.Figure()

    # Add GeoJSON polygons for each selected county with a semi-transparent fill
    fig.update_layout(mapbox=dict(
        style="carto-positron",
        center={"lat": 37.5, "lon": -119.5},
        zoom=5
    ))

    fig.update_layout(mapbox_layers=[
        {
            "source": filtered_counties,
            "type": "fill",
            "below": "traces",
            "color": "rgba(0, 128, 0, 0.3)"  # Semi-transparent green fill
        }
    ])

    # Add city markers with labels
    for i, city in enumerate(city_data):
        fig.add_trace(go.Scattermapbox(
            lat=[city["lat"]],
            lon=[city["lon"]],
            mode="markers+text",
            marker=go.scattermapbox.Marker(size=10, color="blue", symbol="circle"),
            text=city["name"],
            textposition="top right",
            hoverinfo="text",
            name="City Markers" if i == 0 else None,  # Show legend only once
            showlegend=i == 0  # Show legend for the first trace only
        ))

    # Add county markers for farming techniques
    for i, county_item in enumerate(county_data):
        county_name = county_item["name"]

        # Filter techniques for this county
        county_techniques = techniques_df[techniques_df["County"] == county_name]

        if not county_techniques.empty:
            # Create hover text for techniques
            hover_text = (
                f"<b>{county_name}</b><br>" +
                "<br>".join([
                    f"<b>Crop:</b> {row['Primary Crops']}<br>"
                    f"<b>Farming:</b> {row['Farming Methods']}<br>"
                    f"<b>Irrigation:</b> {row['Irrigation Techniques']}<br>"
                    f"<b>Pesticide:</b> {row['Pesticide Usage']}"
                    for _, row in county_techniques.iterrows()
                ])
            )

            # Add marker for the county
            fig.add_trace(go.Scattermapbox(
                lat=[county_item["lat"]],
                lon=[county_item["lon"]],
                mode="markers",
                marker=go.scattermapbox.Marker(
                    size=15,
                    color="rgba(255, 100, 0, 0.7)",
                    symbol="circle"
                ),
                text=hover_text,
                hoverinfo="text",
                name="County Markers" if i == 0 else None,  # Show legend only once
                showlegend=i == 0  # Show legend for the first trace only
            ))

    # Update layout settings
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="rgba(200, 200, 200, 0.5)",
            borderwidth=1
        )
    )

    return fig
