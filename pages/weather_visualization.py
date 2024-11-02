import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px

# Load the weather data
weather_data = pd.read_csv('./data/weather_data.csv')

# Get unique city names for the dropdown
cities = weather_data['city_name'].unique()

# Layout function for the Visualization page
def layout():
    return html.Div([
        html.H1("Weather Data Visualization"),
        
        # Dropdown for city selection
        html.Label("Select a City:"),
        dcc.Dropdown(
            id="city-dropdown",
            options=[{"label": city, "value": city} for city in cities],
            value=cities[0],  # Default value set to the first city
            clearable=False
        ),
        
        # Temperature plot
        html.Div([
            html.H3("Temperature Over Time"),
            dcc.Graph(id="temp-plot")
        ]),
        
        # Rainfall plot
        html.Div([
            html.H3("Rain (1 Hour) Over Time"),
            dcc.Graph(id="rain-plot")
        ]),
        
        # Dew point plot
        html.Div([
            html.H3("Dew Point Over Time"),
            dcc.Graph(id="dew-point-plot")
        ]),
        
        # Cloud cover plot
        html.Div([
            html.H3("Cloud Cover Over Time"),
            dcc.Graph(id="cloud-cover-plot")
        ]),
    ])
    
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

    # Ensure date is in datetime format
    city_data["date"] = pd.to_datetime(city_data["date"])
    
    # Temperature plot
    temp_fig = px.line(city_data, x="date", y="temp", title=f"Temperature Over Time in {selected_city}")
    temp_fig.update_layout(xaxis_title="Date", yaxis_title="Temperature (°F)")

    # Rainfall plot
    rain_fig = px.line(city_data, x="date", y="rain_1h", title=f"Rain (1 Hour) Over Time in {selected_city}")
    rain_fig.update_layout(xaxis_title="Date", yaxis_title="Rain Volume (mm)")

    # Dew Point plot
    dew_point_fig = px.line(city_data, x="date", y="dew_point", title=f"Dew Point Over Time in {selected_city}")
    dew_point_fig.update_layout(xaxis_title="Date", yaxis_title="Dew Point (°F)")

    # Cloud Cover plot
    cloud_cover_fig = px.line(city_data, x="date", y="clouds_all", title=f"Cloud Cover Over Time in {selected_city}")
    cloud_cover_fig.update_layout(xaxis_title="Date", yaxis_title="Cloud Cover (%)")

    return temp_fig, rain_fig, dew_point_fig, cloud_cover_fig