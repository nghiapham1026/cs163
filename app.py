import dash
import os
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import each page layout
from pages import home, data, statistical_analysis, prediction, weather_visualization, crops_visualization, visualization

# Import components
from components.navbar import create_navbar
from components.header import create_header
from components.footer import create_footer

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Weather & Crop Data Analysis"

# Define layout with a navbar, header, main content, and footer
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),  # Track the current URL
    create_header(),
    create_navbar(),
    html.Div(id="page-content", className="content"),  # Page content loaded here based on URL
    create_footer(),
])

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
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)