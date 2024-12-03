import dash_bootstrap_components as dbc

def create_navbar():
    return dbc.NavbarSimple(
        brand="Weather & Crop Analysis",
        brand_href="/",
        color="primary",
        dark=True,
        children=[
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("Data", href="/data", active="exact"),
            dbc.NavLink("Weather Visualization", href="/weather_visualization", active="exact"),
            dbc.NavLink("Crops Visualization", href="/crops_visualization", active="exact"),
            dbc.NavLink("Statistical Analysis", href="/statistical_analysis", active="exact"),
            dbc.NavLink("Visualization", href="/visualization", active="exact"),
            dbc.NavLink("Prediction", href="/prediction", active="exact"),
        ],
    )
