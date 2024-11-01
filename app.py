import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import each page layout
from pages import home, data, visualization, statistical_analysis, prediction

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Weather & Crop Data Analysis"

# Define layout with a sidebar for navigation
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Navigation"),
            dbc.Nav([
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Data", href="/data", active="exact"),
                dbc.NavLink("Visualization", href="/visualization", active="exact"),
                dbc.NavLink("Statistical Analysis", href="/statistical_analysis", active="exact"),
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
    elif pathname == "/visualization":
        return visualization.layout()
    elif pathname == "/statistical_analysis":
        return statistical_analysis.layout()
    elif pathname == "/prediction":
        return prediction.layout()
    return "404 Page Not Found"

if __name__ == "__main__":
    app.run_server(debug=True)
