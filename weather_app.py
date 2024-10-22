import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import folium
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Load the datasets
weather_data_path = 'data/weather_yearly.csv'
monthly_weather_data_path = 'data/weather_monthly.csv'

weather_data = pd.read_csv(weather_data_path)
monthly_weather_data = pd.read_csv(monthly_weather_data_path)

# Convert 'date' and 'month' columns to datetime
weather_data['date'] = pd.to_datetime(weather_data['year'].astype(str), format='%Y')
monthly_weather_data['month'] = pd.to_datetime(monthly_weather_data['month'], format='%Y-%m')

# Extract relevant columns
weather_data['month'] = weather_data['date'].dt.month
weather_data['year'] = weather_data['date'].dt.year

# Get the list of unique cities
cities = weather_data['city_name'].unique()

# Function to create a trend chart for Temperature, Humidity, and return it as a base64 string
def generate_trend_chart(city_data):
    plt.figure(figsize=(6, 3))

    # Plot temperature
    plt.plot(city_data['date'], city_data['temp'], label='Temperature (°F)', color='orange')

    # Plot humidity
    if 'humidity' in city_data.columns:
        plt.plot(city_data['date'], city_data['humidity'], label='Humidity (%)', color='green')

    # Add title and labels
    plt.title(f"Weather Trends for {city_data['city_name'].iloc[0]}")
    plt.xlabel('Year')
    plt.ylabel('Values')
    plt.legend(loc='best')

    # Save the plot to a BytesIO object and encode it to base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    base64_img = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()

    return base64_img

# Function to create a Folium map with weather trend popups
def create_map():
    map_center = [36.7783, -119.4179]
    folium_map = folium.Map(location=map_center, zoom_start=6)

    for city in weather_data['city_name'].unique():
        city_data = weather_data[weather_data['city_name'] == city]
        img_base64 = generate_trend_chart(city_data)

        # HTML popup with embedded image
        popup_html = f"""
        <html>
            <h4>Weather Trends for {city}</h4>
            <img src="data:image/png;base64,{img_base64}">
        </html>
        """

        # Add marker with the popup to the map
        folium.Marker(
            location=[city_data['lat'].iloc[0], city_data['lon'].iloc[0]],
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(folium_map)

    # Save the map as an HTML file
    map_file = "weather_map.html"
    folium_map.save(map_file)
    return map_file

# Generate the map
map_file = create_map()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Weather Data Visualization"),

    # Line Plot Section
    html.Div([
        html.H2("Weather Variables Over Time"),
        dcc.Dropdown(
            id='city-dropdown-line',
            options=[{'label': city, 'value': city} for city in cities],
            value=cities[0]  # Default value for line plot
        ),
        dcc.Graph(id='line-graph'),
    ]),

    # Heatmap Section
    html.Div([
        html.H2("Monthly Temperature Heatmap"),
        dcc.Dropdown(
            id='city-dropdown-heatmap',
            options=[{'label': city, 'value': city} for city in cities],
            value=cities[0]  # Default value for heatmap
        ),
        dcc.Graph(id='heatmap-graph'),
    ]),

    # Folium Map Section
    html.Div([
        html.H2("Weather Data Map"),
        html.Iframe(id='folium-map', srcDoc=open(map_file, 'r').read(), width='100%', height='600'),
    ])
])

# Callback for line graph
@app.callback(
    Output('line-graph', 'figure'),
    [Input('city-dropdown-line', 'value')]
)
def update_line_graph(selected_city):
    city_data = weather_data[weather_data['city_name'] == selected_city]

    traces = []

    # Temperature trace
    traces.append(go.Scatter(
        x=city_data['date'],
        y=city_data['temp'],
        mode='lines',
        name='Temperature',
        line=dict(color='orange')
    ))

    # Precipitation trace
    if 'rain_1h' in city_data.columns:
        traces.append(go.Scatter(
            x=city_data['date'],
            y=city_data['rain_1h'].fillna(0),
            mode='lines',
            name='Precipitation (1h)',
            line=dict(color='blue')
        ))

    # Humidity trace
    if 'humidity' in city_data.columns:
        traces.append(go.Scatter(
            x=city_data['date'],
            y=city_data['humidity'],
            mode='lines',
            name='Humidity',
            line=dict(color='green')
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Weather Variables Over Time for {selected_city}',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Value'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }

# Callback for heatmap
@app.callback(
    Output('heatmap-graph', 'figure'),
    [Input('city-dropdown-heatmap', 'value')]
)
def update_heatmap(selected_city):
    city_data = monthly_weather_data[monthly_weather_data['city_name'] == selected_city]

    # Create a pivot table for the heatmap (Month x Year) showing average temperature
    city_data['year'] = city_data['month'].dt.year
    city_data['month_num'] = city_data['month'].dt.month
    temperature_pivot_city = city_data.pivot_table(values='temp', index='month_num', columns='year', aggfunc='mean')

    # Create the heatmap figure
    heatmap = go.Heatmap(
        z=temperature_pivot_city.values,
        x=temperature_pivot_city.columns,
        y=temperature_pivot_city.index,
        colorscale='RdBu',
        colorbar={'title': 'Avg Temp (°F)'}
    )

    return {
        'data': [heatmap],
        'layout': go.Layout(
            title=f'Average Monthly Temperatures for {selected_city}',
            xaxis={'title': 'Year'},
            yaxis={'title': 'Month', 'dtick': 1},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
        )
    }


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run_server(debug=True, host='0.0.0.0', port=port)
