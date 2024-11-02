from dash import html

def layout():
    return html.Div([
        html.H1("California Crop and Weather Analysis"),
        
        # Project Objective
        html.P(
            "This project aims to analyze the correlation between extreme weather conditions "
            "and crop yield, production, and harvested acres in various California counties. "
            "By examining historical weather patterns and crop data, we seek to understand how "
            "extreme weather events affect agricultural productivity and inform decision-making "
            "in climate adaptation strategies for California’s agricultural sector."
        ),
        
        # Stock Image
        html.Img(src="https://d17ocfn2f5o4rl.cloudfront.net/wp-content/uploads/2020/02/weather-monitoring-technologies-to-save-crops-from-mother-nature_optimized_optimized-1920x600.jpg", alt="Illustration of weather impact on crops", style={
            "width": "100%", "height": "auto", "margin-top": "20px"
        })
    ])
