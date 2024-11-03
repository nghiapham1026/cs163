import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Load the weather data
weather_data = pd.read_csv('./data/weather_data.csv')
merged_yearly = pd.read_csv('./data/merged_yearly.csv')

correlation_columns = [
    'Yield Per Acre', 'Production Per Acre', 'Value Per Acre', 'high_temp_days',
    'low_temp_days', 'heavy_rain_days', 'high_wind_days', 'cloudy_days', 'low_visibility_days', 'snow_days'
]

def layout():
    return html.Div([
        html.H1("Extreme Weather Threshold Analysis"),
        
        # Extreme Weather Section
        html.Div([
            html.Label("Select a County (Extreme Weather):"),
            dcc.Dropdown(
                id="extreme-weather-county-dropdown",
                options=[{"label": county, "value": county} for county in weather_data['city_name'].unique()],
                value=weather_data['city_name'].unique()[0],
                clearable=False
            ),
            html.Label("Select a Weather Variable:"),
            dcc.Dropdown(
                id="extreme-weather-variable-dropdown",
                options=[
                    {"label": "Temperature", "value": "temp"},
                    {"label": "Rain (1 Hour)", "value": "rain_1h"},
                    {"label": "Dew Point", "value": "dew_point"},
                    {"label": "Cloud Cover", "value": "clouds_all"}
                ],
                value="temp",
                clearable=False
            ),
            dcc.Graph(id="extreme-weather-graph"),
        ], style={"margin-bottom": "50px"}),
        
        html.H1("County-Crop Correlation Analysis"),
        
        # Correlation Matrix Section
        html.Div([
            html.Label("Select a County (Correlation Matrix):"),
            dcc.Dropdown(
                id="correlation-county-dropdown",
                options=[{"label": county, "value": county} for county in merged_yearly['County'].unique()],
                value=merged_yearly['County'].unique()[0],
                clearable=False
            ),
            html.Label("Select a Crop:"),
            dcc.Dropdown(
                id="correlation-crop-dropdown",
                options=[],  # Populated by callback based on county selection
                value=None,
                clearable=False
            ),
            dcc.Graph(id="correlation-matrix-plot"),
        ]),
    ])
    
@callback(
    Output("extreme-weather-graph", "figure"),
    Input("extreme-weather-county-dropdown", "value"),
    Input("extreme-weather-variable-dropdown", "value")
)
def plot_extreme_weather(selected_county, selected_variable):
    county_data = weather_data[weather_data["city_name"] == selected_county]
    lower_threshold = county_data[selected_variable].quantile(0.1)
    upper_threshold = county_data[selected_variable].quantile(0.9)
    
    county_data["Extreme"] = county_data[selected_variable].apply(
        lambda x: "Low (10th percentile)" if x <= lower_threshold 
                  else ("High (90th percentile)" if x >= upper_threshold else "Normal")
    )

    county_data["date"] = pd.to_datetime(county_data["date"])
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
    fig.add_hline(y=lower_threshold, line_dash="dash", line_color="blue", 
                  annotation_text="10th Percentile", annotation_position="bottom left")
    fig.add_hline(y=upper_threshold, line_dash="dash", line_color="red", 
                  annotation_text="90th Percentile", annotation_position="top left")

    fig.update_layout(xaxis_title="Date", yaxis_title=selected_variable.capitalize())
    return fig


@callback(
    Output("correlation-crop-dropdown", "options"),
    Output("correlation-crop-dropdown", "value"),
    Input("correlation-county-dropdown", "value")
)
def update_crop_dropdown(selected_county):
    df_county = merged_yearly[merged_yearly['County'] == selected_county]
    crops = df_county['Crop Name'].unique()
    crop_options = [{"label": crop, "value": crop} for crop in crops]
    return crop_options, crops[0]

@callback(
    Output("correlation-matrix-plot", "figure"),
    Input("correlation-county-dropdown", "value"),
    Input("correlation-crop-dropdown", "value")
)
def plot_correlation_matrix(selected_county, selected_crop):
    df_filtered = merged_yearly[(merged_yearly['County'] == selected_county) &
                                (merged_yearly['Crop Name'] == selected_crop)]

    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for this selection")
        return fig
    
    correlation_matrix = df_filtered[correlation_columns].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        )
    )
    fig.update_layout(
        title=f"Correlation Matrix for {selected_crop} in {selected_county}",
        xaxis=dict(tickmode="array", tickvals=list(range(len(correlation_columns))), ticktext=correlation_columns),
        yaxis=dict(tickmode="array", tickvals=list(range(len(correlation_columns))), ticktext=correlation_columns)
    )
    return fig
