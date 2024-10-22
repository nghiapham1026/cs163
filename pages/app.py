import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import dash

# Import your page modules
from home import render_home_page
from weather_data import render_weather_data_page, register_callbacks
from crop_data import render_crop_data_page
from prediction import render_prediction_page
from analysis import render_analysis_page

# Initialize the app
app = dash.Dash(__name__)

# Load the dataset
df = pd.read_csv('../data/weather.csv')  # Replace with the actual path to your dataset

# Layout of the app
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='home', children=[
        dcc.Tab(label='Home', value='home'),
        dcc.Tab(label='Weather Data', value='weather-data'),
        dcc.Tab(label='Crop Data', value='crop-data'),
        dcc.Tab(label='Prediction', value='prediction'),
        dcc.Tab(label='Analysis', value='analysis')
    ]),
    html.Div(id='tabs-content')
])

# Callback to update content based on tab selection
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'home':
        return render_home_page()
    elif tab == 'weather-data':
        return render_weather_data_page(df)
    elif tab == 'crop-data':
        return render_crop_data_page()
    elif tab == 'prediction':
        return render_prediction_page()
    elif tab == 'analysis':
        return render_analysis_page()

# Register weather page callbacks
register_callbacks(app, df)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
