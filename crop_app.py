import dash
from dash import dcc, html
import pandas as pd
import folium
import base64
import matplotlib.pyplot as plt
from io import BytesIO

# Mock data for crops and coordinates (replace with actual data as needed)
df_crops = pd.DataFrame({
    'Region': ['Fresno', 'Fresno', 'Fresno', 'Fresno'],
    'Year': [1980, 1981, 1982, 1983],
    'Crop Name': ['ALMONDS ALL', 'ALMONDS ALL', 'TOMATOES PROCESSING', 'TOMATOES PROCESSING'],
    'Value': [14200, 17900, 1064000, 1048000]
})

county_coordinates = {
    'Fresno': [36.7378, -119.7871]  # Fresno, CA
}

crops_of_interest = ['TOMATOES PROCESSING', 'ALMONDS ALL']

# Function to generate the combined crop yield plot
def generate_combined_yield_plot(region):
    plt.figure(figsize=(6, 4))
    crop_colors = {'TOMATOES PROCESSING': 'red', 'ALMONDS ALL': 'green'}

    for crop in crops_of_interest:
        historical_data = df_crops[(df_crops['Region'] == region) & (df_crops['Crop Name'] == crop)]
        if not historical_data.empty:
            years = historical_data['Year']
            values = historical_data['Value']
            plt.plot(years, values, label=crop, color=crop_colors[crop])

    plt.title(f'Crop Yield History ({region})')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return image_base64

# Create a separate Folium map for Fresno with crop yield trend popups
def create_fresno_map():
    california_map_with_history = folium.Map(location=[36.7378, -119.7871], zoom_start=7)

    if 'Fresno' in county_coordinates:
        yield_plot_base64 = generate_combined_yield_plot('Fresno')
        img_tag = f'<img src="data:image/png;base64,{yield_plot_base64}">'

        folium.CircleMarker(
            location=county_coordinates['Fresno'],
            radius=5,
            popup=folium.Popup(f"Fresno Crop Yields<br>{img_tag}", max_width=300),
            color='green',
            fill=True,
            fill_opacity=0.6
        ).add_to(california_map_with_history)

    map_file = "fresno_map.html"
    california_map_with_history.save(map_file)
    return map_file

# Create the map
map_file = create_fresno_map()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Fresno Crop Yield Trends Map"),
    html.Iframe(id='fresno-map', srcDoc=open(map_file, 'r').read(), width='100%', height='600')
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
