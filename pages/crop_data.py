import plotly.express as px
import pandas as pd
from dash import dcc, html
import plotly.graph_objs as go

df_fresno = pd.read_csv('../data/grouped_fresno_crops.csv')

def render_crop_data_page():
    # Plot 1: Harvested Acres over time for each crop
    fig1 = px.line(df_fresno, x='Year', y='Harvested Acres', color='Crop Name',
                title="Harvested Acres Over Time for Different Crops",
                labels={'Harvested Acres': 'Harvested Acres (in acres)'})

    # Add production trace for ALMONDS ALL
    almonds_data = df_fresno[df_fresno['Crop Name'] == 'ALMONDS ALL']
    fig1.add_trace(go.Scatter(x=almonds_data['Year'], y=almonds_data['Production'], mode='lines', name='Almond Production', line=dict(dash='dash')))

    # Add production trace for TOMATOES PROCESSING
    tomatoes_data = df_fresno[df_fresno['Crop Name'] == 'TOMATOES PROCESSING']
    fig1.add_trace(go.Scatter(x=tomatoes_data['Year'], y=tomatoes_data['Production'], mode='lines', name='Tomato Production', line=dict(dash='dash')))

    # Plot 2: Price per unit and Value over time for each crop
    fig2 = px.line(df_fresno, x='Year', y='Price P/U', color='Crop Name',
                   title="Price and Value Over Time for Different Crops",
                   labels={'Price P/U': 'Price Per Unit (USD)'})

    # Add production trace for ALMONDS ALL
    almonds_data = df_fresno[df_fresno['Crop Name'] == 'ALMONDS ALL']
    fig2.add_trace(go.Scatter(x=almonds_data['Year'], y=almonds_data['Value'], mode='lines', name='Almond Value', line=dict(dash='dash')))

    # Add production trace for TOMATOES PROCESSING
    tomatoes_data = df_fresno[df_fresno['Crop Name'] == 'TOMATOES PROCESSING']
    fig2.add_trace(go.Scatter(x=tomatoes_data['Year'], y=tomatoes_data['Value'], mode='lines', name='Tomato Value', line=dict(dash='dash')))

    return html.Div([
        html.H2("Crop Data Page"),
        dcc.Graph(figure=fig1),  # Plot for Harvested Acres and Production
        dcc.Graph(figure=fig2),  # Plot for Price per Unit and Value
    ])