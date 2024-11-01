from dash import html, dcc

def layout():
    return html.Div([
        html.H1("Data"),
        html.P("Explore the dataset used in this analysis."),
        dcc.Graph(id="data-overview-plot"),
        # Callback or functions to load and display data overview plots can be added here
    ])
