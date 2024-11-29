import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px

df = pd.read_csv('./data/merged_yearly.csv')

extreme_weather_features = [
    'high_temp_days', 'low_temp_days', 'heavy_rain_days',
    'snow_days', 'high_wind_days', 'low_visibility_days', 'cloudy_days'
]

def layout():
    return html.Div(
        className="visualization-container",
        children=[
            # Page title
            html.H1(
                "Visualization of The Impact of Weather on Crop Yield",
                className="page-title"
            ),
            
            # New section: Impact of Extreme Weather on Crop Yield
            html.Div(
                className="impact-weather-container",
                children=[
                    html.H1(
                        "Impact of Extreme Weather on Crop Yield",
                        className="impact-weather-title"
                    ),
                    html.Div(
                        className="impact-weather-dropdowns",
                        children=[
                            html.Div(
                                children=[
                                    html.Label(
                                        "Select Crop:",
                                        className="impact-weather-dropdown-label"
                                    ),
                                    dcc.Dropdown(
                                        id='crop-dropdown',
                                        options=[{'label': crop, 'value': crop} for crop in df["Crop Name"].unique()],
                                        value=df["Crop Name"].unique()[0],
                                        className="impact-weather-dropdown"
                                    ),
                                ],
                                className="dropdown-container"
                            ),
                            html.Div(
                                children=[
                                    html.Label(
                                        "Select Extreme Weather Variable:",
                                        className="impact-weather-dropdown-label"
                                    ),
                                    dcc.Dropdown(
                                        id='extreme-variable-dropdown',
                                        options=[{'label': var, 'value': var} for var in extreme_weather_features],
                                        value=extreme_weather_features[0],
                                        className="impact-weather-dropdown"
                                    ),
                                ],
                                className="dropdown-container"
                            ),
                        ]
                    ),
                    html.Div(
                        className="yield-comparison-graph-container",
                        children=[
                            dcc.Graph(
                                id='yield-comparison-graph',
                                className="yield-comparison-graph"
                            )
                        ]
                    )
                ]
            )
        ]
    )

@callback(
    Output('yield-comparison-graph', 'figure'),
    [Input('crop-dropdown', 'value'),
     Input('extreme-variable-dropdown', 'value')]
)
def update_graph(selected_crop, selected_var):
    # Get unique counties
    counties = df["County"].unique()

    # DataFrame to store results
    yield_comparison_list = []

    for county in counties:
        # Filter data for the current county and selected crop
        county_data = df[(df["County"] == county) & (df["Crop Name"] == selected_crop)]
        if county_data.empty:
            continue  # Skip if no data for this county and crop

        # Handle possible NaNs in the selected variable
        var_values = county_data[selected_var].dropna()
        if var_values.empty:
            continue  # Skip if variable data is missing

        # Calculate the 75th percentile threshold for the selected variable
        threshold = var_values.quantile(0.65)

        # Identify extreme years for the current variable in the current county
        extreme_years = county_data[county_data[selected_var] > threshold]["Year"].unique()

        # Calculate average crop yield for extreme years
        extreme_yield = county_data[county_data["Year"].isin(extreme_years)]["Yield Per Acre"].mean()
        if pd.isna(extreme_yield):
            extreme_yield = 0

        # Calculate average crop yield for non-extreme years
        non_extreme_yield = county_data[~county_data["Year"].isin(extreme_years)]["Yield Per Acre"].mean()
        if pd.isna(non_extreme_yield):
            non_extreme_yield = 0

        # Append to list
        yield_comparison_list.append({
            "County": county,
            "Extreme Years Yield": extreme_yield,
            "Non-Extreme Years Yield": non_extreme_yield
        })

    if not yield_comparison_list:
        return px.bar(title="No data available for the selected crop and variable.")

    # Create DataFrame
    yield_comparison_df = pd.DataFrame(yield_comparison_list)

    # Melt the DataFrame for plotting
    yield_comparison_melted = yield_comparison_df.melt(
        id_vars="County",
        value_vars=["Extreme Years Yield", "Non-Extreme Years Yield"],
        var_name="Condition",
        value_name="Yield Per Acre"
    )

    # Create the bar plot
    fig = px.bar(
        yield_comparison_melted,
        x="County",
        y="Yield Per Acre",
        color="Condition",
        barmode="group",
        title=f"Impact of {selected_var} on {selected_crop} Yield Across Counties",
        labels={"Yield Per Acre": "Yield Per Acre", "Condition": "Condition"},
        category_orders={"County": sorted(counties)}
    )

    fig.update_layout(
        xaxis_title="County",
        yaxis_title="Average Yield Per Acre",
        legend_title="Condition"
    )

    return fig