from dash import html

def create_header():
    return html.Div(
        className="header-container",
        children=[
            html.H1(
                "Welcome to the Weather & Crop Data Analysis Dashboard",
                className="header-title"
            ),
            html.P(
                "Explore visualizations, perform analysis, and predict crop outcomes.",
                className="header-subtitle"
            ),
            html.P(
                "This project is part of CS 163 - Data Science Senior Project, Fall 2024",
                className="header-class-info"
            )
        ]
    )
