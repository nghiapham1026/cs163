import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Function to convert matplotlib figures to base64
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')

def render_weather_data_page(df):
    # Get unique cities for the dropdown
    cities = df['city_name'].unique()

    return html.Div([
        dcc.Dropdown(
            id='city-dropdown',
            options=[{'label': city, 'value': city} for city in cities],
            value=cities[0],  # Default value
            clearable=False,
            style={'width': '50%'}
        ),
        html.Div(id='weather-plots')  # This will hold the generated plots
    ])

# Callback to update plots based on selected city
def register_callbacks(app, df):
    @app.callback(
        Output('weather-plots', 'children'),
        Input('city-dropdown', 'value')
    )
    def update_weather_plots(selected_city):
        # Filter the dataset for the selected city
        city_data = df[df['city_name'] == selected_city]
        
        # Ensure 'date' column is in datetime format
        city_data['date'] = pd.to_datetime(city_data['date'], errors='coerce')  # Coerce invalid parsing to NaT

        # Plotly time series plot for temperature, precipitation, and humidity
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=city_data['date'], y=city_data['temp'], mode='lines', name='Temperature', line=dict(color='orange')))
        
        if 'rain_1h' in city_data.columns:
            fig_temp.add_trace(go.Scatter(x=city_data['date'], y=city_data['rain_1h'].fillna(0), mode='lines', name='Precipitation (1h)', line=dict(color='blue')))

        if 'humidity' in city_data.columns:
            fig_temp.add_trace(go.Scatter(x=city_data['date'], y=city_data['humidity'], mode='lines', name='Humidity', line=dict(color='green')))

        fig_temp.update_layout(title=f'Weather Variables Over Time for {selected_city}', xaxis_title='Date', yaxis_title='Value', xaxis=dict(tickangle=45))

        # Create a Plotly heatmap for average monthly temperatures
        # Create pivot table for temperature heatmap
        temperature_pivot_city = city_data.pivot_table(values='temp', index=city_data['date'].dt.month, columns=city_data['date'].dt.year)

        # Generate a Plotly heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=temperature_pivot_city.values,
            x=temperature_pivot_city.columns,
            y=temperature_pivot_city.index,
            colorscale='RdBu',
            colorbar=dict(title='Temp')
        ))

        # Update layout for the heatmap
        fig_heatmap.update_layout(
            title=f'Heatmap of Average Monthly Temperatures for {selected_city}',
            xaxis_title='Year',
            yaxis_title='Month'
        )

        # Create a new Plotly figure for precipitation-only plot
        fig_precipitation = go.Figure()

        # Add the precipitation (rain_1h or rain_3h) to the precipitation plot
        if 'rain_1h' in city_data.columns:
            fig_precipitation.add_trace(go.Scatter(x=city_data['date'], y=city_data['rain_1h'].fillna(0), mode='lines', name='Precipitation (1h)', line=dict(color='blue')))
                
        fig_precipitation.update_layout(title=f'Precipitation Over Time for {selected_city}', xaxis_title='Date', yaxis_title='Precipitation (mm)', xaxis=dict(tickangle=45))

        # Return the complete set of plots (weather variables, precipitation, and heatmap)
        return html.Div([
            dcc.Graph(figure=fig_temp),      # Time series plot (temperature, precipitation, humidity)
            dcc.Graph(figure=fig_precipitation),  # Precipitation-only plot
            dcc.Graph(figure=fig_heatmap)  # Heatmap of average monthly temperatures
        ])
