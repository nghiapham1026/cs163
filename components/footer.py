from dash import html

def create_footer():
    return html.Div(
        className="footer-container",
        children=[
            html.P("© 2024 Weather & Crop Analysis Dashboard", className="footer-text"),
            html.P("Designed with Dash and Plotly", className="footer-text"),
            html.Div(
                className="contact-info",
                children=[
                    html.P("Contact: Nathan Pham", className="footer-contact"),
                    html.A("LinkedIn", href="https://www.linkedin.com/in/nathan-pham-0a8a65217", target="_blank", className="footer-link"),
                    " | ",
                    html.A("GitHub", href="https://github.com/nghiapham1026", target="_blank", className="footer-link"),
                    " | ",
                    html.A("Email", href="mailto:nghiapham102603@gmail.com", className="footer-link")
                ]
            )
        ]
    )
