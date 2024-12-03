import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px

# Load the weather data
weather_data = pd.read_csv('./data/weather_monthly.csv')
yearly_weather_merged = pd.read_csv('./data/yearly_weather_merged.csv')

# Get unique city names for the dropdown
cities = weather_data['city_name'].unique()

extreme_weather_features = [
    'high_temp_days', 'low_temp_days', 'heavy_rain_days',
    'snow_days', 'high_wind_days', 'low_visibility_days', 'cloudy_days'
]

# Layout function for the Visualization page
def layout():
    return html.Div(
        className="weather-visualization-container",
        children=[
            # Page Title
            html.H1(
                "Weather Data Visualization",
                className="page-title"
            ),
            
            # City Selection Section
            html.Div(
                className="city-selection-section",
                children=[
                    html.Label(
                        "Select a City:",
                        className="city-dropdown-label"
                    ),
                    dcc.Dropdown(
                        id="city-dropdown",
                        options=[{"label": city, "value": city} for city in cities],
                        value=cities[0],
                        clearable=False,
                        className="city-dropdown"
                    ),
                    html.P(
                        "Use this dropdown to select a city and view its weather data visualizations. "
                        "Each city includes data on temperature, extreme weather events, rainfall, dew point, and cloud cover over time.",
                        className="city-selection-description"
                    )
                ]
            ),
            
            # Temperature Plot Section
            html.Div(
                className="temperature-plot-container",
                children=[
                    html.H3(
                        "Temperature Over Time",
                        className="temperature-plot-title"
                    ),
                    html.P(
                        "This graph displays the temperature trends over time for the selected city. "
                        "You can use this visualization to understand seasonal variations and identify long-term patterns.",
                        className="temperature-plot-description"
                    ),
                    dcc.Graph(
                        id="temp-plot",
                        className="temperature-plot-graph"
                    )
                ]
            ),
            
            # Extreme Weather Plot Section
            html.Div(
                className="extreme-weather-plot-container",
                children=[
                    html.H3(
                        "Extreme Weather Events",
                        className="extreme-weather-plot-title"
                    ),
                    html.P(
                        "This plot highlights extreme weather events, such as high wind days and heavy rain occurrences. "
                        "It provides insights into the frequency and intensity of extreme conditions.",
                        className="extreme-weather-plot-description"
                    ),
                    dcc.Graph(
                        id="extreme-weather-plot",
                        className="extreme-weather-plot-graph"
                    )
                ]
            ),

            # Rainfall Plot Section
            html.Div(
                className="rainfall-plot-container",
                children=[
                    html.H3(
                        "Rain (1 Hour) Over Time",
                        className="rainfall-plot-title"
                    ),
                    html.P(
                        "Visualize hourly rainfall data over time to identify precipitation patterns and assess periods of drought or heavy rain.",
                        className="rainfall-plot-description"
                    ),
                    dcc.Graph(
                        id="rain-plot",
                        className="rainfall-plot-graph"
                    )
                ]
            ),
            
            # Dew Point Plot Section
            html.Div(
                className="dew-point-plot-container",
                children=[
                    html.H3(
                        "Dew Point Over Time",
                        className="dew-point-plot-title"
                    ),
                    html.P(
                        "Explore dew point data to understand humidity levels and their implications on comfort and weather patterns in the selected city.",
                        className="dew-point-plot-description"
                    ),
                    dcc.Graph(
                        id="dew-point-plot",
                        className="dew-point-plot-graph"
                    )
                ]
            ),
            
            # Cloud Cover Plot Section
            html.Div(
                className="cloud-cover-plot-container",
                children=[
                    html.H3(
                        "Cloud Cover Over Time",
                        className="cloud-cover-plot-title"
                    ),
                    html.P(
                        "This plot illustrates cloud cover data over time, providing insights into overcast days and periods of clear skies.",
                        className="cloud-cover-plot-description"
                    ),
                    dcc.Graph(
                        id="cloud-cover-plot",
                        className="cloud-cover-plot-graph"
                    )
                ]
            ),
        ]
    )
    
# Callback to update all plots based on selected city
@callback(
    Output("temp-plot", "figure"),
    Output("rain-plot", "figure"),
    Output("dew-point-plot", "figure"),
    Output("cloud-cover-plot", "figure"),
    Input("city-dropdown", "value")
)
def update_plots(selected_city):
    # Filter data for the selected city
    city_data = weather_data[weather_data["city_name"] == selected_city]
    
    # Temperature plot
    temp_fig = px.line(
        city_data, x="month", y="temp",
        title=f"Temperature Over Time in {selected_city}"
    )
    temp_fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°F)")

    # Rainfall plot
    rain_fig = px.line(
        city_data, x="month", y="rain_1h",
        title=f"Rain (1 Hour) Over Time in {selected_city}"
    )
    rain_fig.update_layout(xaxis_title="Date", yaxis_title="Rain Volume (mm)")

    # Dew Point plot
    dew_point_fig = px.line(
        city_data, x="month", y="dew_point",
        title=f"Dew Point Over Time in {selected_city}"
    )
    dew_point_fig.update_layout(xaxis_title="Date", yaxis_title="Dew Point (°F)")

    # Cloud Cover plot
    cloud_cover_fig = px.line(
        city_data, x="month", y="clouds_all",
        title=f"Cloud Cover Over Time in {selected_city}"
    )
    cloud_cover_fig.update_layout(xaxis_title="Date", yaxis_title="Cloud Cover (%)")

    return temp_fig, rain_fig, dew_point_fig, cloud_cover_fig

# Callback to update the extreme weather features plot based on selected city
@callback(
    Output("extreme-weather-plot", "figure"),
    Input("city-dropdown", "value")
)
def update_extreme_weather_plot(selected_city):
    # Filter data for the selected city
    city_data = yearly_weather_merged[
        yearly_weather_merged["city_name"] == selected_city
    ]
    
    # Melt the data to have a long format suitable for line plotting
    melted_data = city_data.melt(
        id_vars=["Year"], 
        value_vars=extreme_weather_features, 
        var_name="Feature", 
        value_name="Days"
    )

    # Create time series line plot
    fig = px.line(
        melted_data, x="Year", y="Days", color="Feature",
        title=f"Extreme Weather Days Over Time in {selected_city}"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Days of Extreme Weather")

    return fig
