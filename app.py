import dash
import os
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import each page layout
from pages import home, data, statistical_analysis, prediction, weather_visualization, crops_visualization, visualization

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Weather & Crop Data Analysis"

# Define layout with a sidebar for navigation
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Navigation"),
            dbc.Nav([
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Data", href="/data", active="exact"),
                dbc.NavLink("Weather Visualization", href="/weather_visualization", active="exact"),
                dbc.NavLink("Crops Visualization", href="/crops_visualization", active="exact"),
                dbc.NavLink("Statistical Analysis", href="/statistical_analysis", active="exact"),
                dbc.NavLink("Visualization", href="/visualization", active="exact"),
                dbc.NavLink("Prediction", href="/prediction", active="exact"),
            ], vertical=True, pills=True),
        ], width=2),

        dbc.Col([
            dcc.Location(id="url"),  # Tracks the current URL
            html.Div(id="page-content")  # Page content loaded here based on URL
        ], width=10)
    ])
], fluid=True)

# Callbacks for page navigation
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return home.layout()
    elif pathname == "/data":
        return data.layout()
    elif pathname == "/weather_visualization":
        return weather_visualization.layout()
    elif pathname == "/crops_visualization":
        return crops_visualization.layout()
    elif pathname == "/statistical_analysis":
        return statistical_analysis.layout()
    elif pathname == "/prediction":
        return prediction.layout()
    elif pathname == "/visualization":
        return visualization.layout()
    return "404 Page Not Found"

if __name__ == "__main__":
    # Bind to $PORT if defined, otherwise default to 8050
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)