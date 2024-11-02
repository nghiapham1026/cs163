import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px

# Load the crops data
crops_data = pd.read_csv('./data/crops_yearly.csv')

# Get unique counties for dropdown options
counties = crops_data['County'].unique()

# Layout function for the Crop Data Visualization page
def layout():
    return html.Div([
        html.H1("Crop Data Visualization"),
        
        # Dropdown for county selection
        html.Label("Select a County:"),
        dcc.Dropdown(
            id="county-dropdown",
            options=[{"label": county, "value": county} for county in counties],
            value=counties[0],  # Default value set to the first county
            clearable=False
        ),
        
        # Yield Per Acre plot
        html.Div([
            html.H3("Yield Per Acre Over Time by Crop"),
            dcc.Graph(id="yield-plot")
        ]),
        
        # Production Per Acre plot
        html.Div([
            html.H3("Production Per Acre Over Time by Crop"),
            dcc.Graph(id="production-plot")
        ]),
        
        # Harvested Acres plot
        html.Div([
            html.H3("Harvested Acres Over Time by Crop"),
            dcc.Graph(id="harvested-acres-plot")
        ]),
    ])
    
# Callback to update all plots based on selected county
@callback(
    Output("yield-plot", "figure"),
    Output("production-plot", "figure"),
    Output("harvested-acres-plot", "figure"),
    Input("county-dropdown", "value")
)
def update_crop_plots(selected_county):
    # Filter data for the selected county
    county_data = crops_data[crops_data["County"] == selected_county]

    # Ensure 'Year' is in datetime format
    county_data["Year"] = pd.to_datetime(county_data["Year"], format='%Y')
    
    # Yield Per Acre plot
    yield_fig = px.line(county_data, x="Year", y="Yield Per Acre", color="Crop Name",
                        title=f"Yield Per Acre Over Time by Crop in {selected_county}")
    yield_fig.update_layout(xaxis_title="Year", yaxis_title="Yield Per Acre")

    # Production Per Acre plot
    production_fig = px.line(county_data, x="Year", y="Production Per Acre", color="Crop Name",
                             title=f"Production Per Acre Over Time by Crop in {selected_county}")
    production_fig.update_layout(xaxis_title="Year", yaxis_title="Production Per Acre")

    # Harvested Acres plot
    harvested_acres_fig = px.line(county_data, x="Year", y="Harvested Acres", color="Crop Name",
                                  title=f"Harvested Acres Over Time by Crop in {selected_county}")
    harvested_acres_fig.update_layout(xaxis_title="Year", yaxis_title="Harvested Acres")

    return yield_fig, production_fig, harvested_acres_fig
