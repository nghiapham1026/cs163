import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px

# Load the weather data
weather_data = pd.read_csv('./data/weather_data.csv')

# Layout function for the Analysis page
def layout():
    return html.Div([
        html.H1("Extreme Weather Threshold Analysis"),
        
        # Dropdown for county selection
        html.Label("Select a County:"),
        dcc.Dropdown(
            id="county-dropdown",
            options=[{"label": county, "value": county} for county in weather_data['city_name'].unique()],
            value=weather_data['city_name'].unique()[0],  # Default to the first county
            clearable=False
        ),
        
        # Dropdown for weather variable selection
        html.Label("Select a Weather Variable:"),
        dcc.Dropdown(
            id="weather-variable-dropdown",
            options=[
                {"label": "Temperature", "value": "temp"},
                {"label": "Rain (1 Hour)", "value": "rain_1h"},
                {"label": "Dew Point", "value": "dew_point"},
                {"label": "Cloud Cover", "value": "clouds_all"}
            ],
            value="temp",  # Default to temperature
            clearable=False
        ),
        
        # Extreme Weather Plot
        html.Div([
            html.H3("Extreme Weather Thresholds Over Time"),
            dcc.Graph(id="extreme-weather-graph")
        ]),
    ])
    
# Callback to calculate and plot extreme weather thresholds
@callback(
    Output("extreme-weather-graph", "figure"),
    Input("county-dropdown", "value"),
    Input("weather-variable-dropdown", "value")
)
def plot_extreme_weather(selected_county, selected_variable):
    # Filter data for the selected county
    county_data = weather_data[weather_data["city_name"] == selected_county]

    # Calculate 10th and 90th percentiles
    lower_threshold = county_data[selected_variable].quantile(0.1)
    upper_threshold = county_data[selected_variable].quantile(0.9)
    
    # Flag values within the extreme thresholds
    county_data["Extreme"] = county_data[selected_variable].apply(
        lambda x: "Low (10th percentile)" if x <= lower_threshold 
                  else ("High (90th percentile)" if x >= upper_threshold else "Normal")
    )

    # Convert date column to datetime for time-series plotting
    county_data["date"] = pd.to_datetime(county_data["date"])

    # Plot
    fig = px.scatter(
        county_data, x="date", y=selected_variable, color="Extreme",
        title=f"{selected_variable.capitalize()} Extremes Over Time in {selected_county}",
        labels={"Extreme": "Extreme Level"},
        color_discrete_map={
            "Low (10th percentile)": "blue",
            "High (90th percentile)": "red",
            "Normal": "gray"
        }
    )

    # Add threshold lines for reference
    fig.add_hline(y=lower_threshold, line_dash="dash", line_color="blue", 
                  annotation_text="10th Percentile", annotation_position="bottom left")
    fig.add_hline(y=upper_threshold, line_dash="dash", line_color="red", 
                  annotation_text="90th Percentile", annotation_position="top left")

    fig.update_layout(xaxis_title="Date", yaxis_title=selected_variable.capitalize())
    
    return fig
