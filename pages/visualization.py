import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('./data/merged_yearly.csv')

extreme_weather_features = [
    'high_temp_days', 'low_temp_days', 'heavy_rain_days',
    'snow_days', 'high_wind_days', 'low_visibility_days', 'cloudy_days'
]

extreme_weather_vars = {
    "high_temp_days": "red",
    "low_temp_days": "blue",
    "heavy_rain_days": "green",
    "high_wind_days": "orange",
    "cloudy_days": "purple"
}

# Define county-specific thresholds
county_thresholds = {}
for county in df["County"].unique():
    county_data = df[df["County"] == county]
    thresholds = {}
    for var in extreme_weather_vars.keys():
        thresholds[var] = {
            "high": county_data[var].quantile(0.9),
        }
    county_thresholds[county] = thresholds
    
# Get unique crops
crops = df["Crop Name"].unique()

def layout():
    return html.Div(
        className="visualization-container",
        children=[
            # Page title
            html.H1(
                "Visualization of The Impact of Weather on Crop Yield",
                className="page-title"
            ),
            
            # Section: Impact of Extreme Weather on Crop Yield
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
                                        id='impact-weather-crop-dropdown',  # Updated ID
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
                                        id='impact-extreme-variable-dropdown',  # Updated ID
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
                                id='impact-yield-comparison-graph',  # Updated ID
                                className="yield-comparison-graph"
                            )
                        ]
                    ),
                ]
            ),

            # Section: Weather Impact on Crop Yield
            html.Div(
                className="weather-impact-container",
                children=[
                    html.H1("Weather Impact on Crop Yield"),
                    html.Label("Select Crop:"),
                    dcc.Dropdown(
                        id='weather-impact-crop-dropdown',  # Updated ID
                        options=[{'label': crop, 'value': crop} for crop in sorted(crops)],
                        value=sorted(crops)[0]
                    ),
                    html.Label("Select Extreme Weather Variable:"),
                    dcc.Dropdown(
                        id='weather-impact-feature-dropdown',  # Updated ID
                        options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in extreme_weather_features],
                        value=extreme_weather_features[0]
                    ),
                    html.Div(
                        className="graphs-container",
                        children=[
                            dcc.Graph(id='weather-yield-per-acre-graph'),  # Updated ID
                            dcc.Graph(id='weather-production-per-acre-graph')  # Updated ID
                        ]
                    )
                ]
            ),

            # Section: Weather Anomalies and Crop Yield
            html.Div(
                className="weather-anomalies-container",
                children=[
                    html.H1("Weather Anomalies and Crop Yield", className="section-title"),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='anomalies-county-dropdown',  # Updated ID
                                options=[{'label': county, 'value': county} for county in sorted(df["County"].unique())],
                                value=sorted(df["County"].unique())[0]
                            ),
                        ]
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='anomalies-crop-dropdown',  # Updated ID
                                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                                value=sorted(df["Crop Name"].unique())[0]
                            ),
                        ]
                    ),
                    html.Div(
                        className="graphs-container",
                        children=[
                            dcc.Graph(id='anomalies-yield-graph'),  # Updated ID
                            dcc.Graph(id='anomalies-production-graph')  # Updated ID
                        ]
                    )
                ]
            )
        ]
    )

@callback(
    Output('impact-yield-comparison-graph', 'figure'),
    [Input('impact-weather-crop-dropdown', 'value'),
     Input('impact-extreme-variable-dropdown', 'value')]
)
def update_impact_yield_graph(selected_crop, selected_var):
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

@callback(
    [Output('weather-yield-per-acre-graph', 'figure'),
     Output('weather-production-per-acre-graph', 'figure')],
    [Input('weather-impact-crop-dropdown', 'value'),
     Input('weather-impact-feature-dropdown', 'value')]
)
def update_weather_graphs(selected_crop, selected_weather_feature):
    # Filter data for selected crop
    crop_data = df[df['Crop Name'] == selected_crop]

    # Check if data is available
    if crop_data.empty:
        return {}, {}

    # Remove rows with NaN in selected weather feature
    crop_data = crop_data.dropna(subset=[selected_weather_feature, 'Yield Per Acre', 'Production Per Acre', 'County'])

    if crop_data.empty:
        return {}, {}

    # For 'Yield Per Acre' plot
    fig_yield = px.scatter(
        crop_data,
        x=selected_weather_feature,
        y='Yield Per Acre',
        color='County',
        trendline='ols',
        trendline_scope='trace',
        title=f"{selected_crop}: {selected_weather_feature.replace('_', ' ').title()} vs Yield Per Acre",
        labels={selected_weather_feature: selected_weather_feature.replace('_', ' ').title(), 'Yield Per Acre': 'Yield Per Acre'}
    )

    # For 'Production Per Acre' plot
    fig_production = px.scatter(
        crop_data,
        x=selected_weather_feature,
        y='Production Per Acre',
        color='County',
        trendline='ols',
        trendline_scope='trace',
        title=f"{selected_crop}: {selected_weather_feature.replace('_', ' ').title()} vs Production Per Acre",
        labels={selected_weather_feature: selected_weather_feature.replace('_', ' ').title(), 'Production Per Acre': 'Production Per Acre'}
    )

    # Update layouts if needed
    fig_yield.update_layout(
        xaxis_title=selected_weather_feature.replace('_', ' ').title(),
        yaxis_title='Yield Per Acre',
        legend_title='County',
        height=500
    )

    fig_production.update_layout(
        xaxis_title=selected_weather_feature.replace('_', ' ').title(),
        yaxis_title='Production Per Acre',
        legend_title='County',
        height=500
    )

    return fig_yield, fig_production

@callback(
    [Output('anomalies-yield-graph', 'figure'),
     Output('anomalies-production-graph', 'figure')],
    [Input('anomalies-county-dropdown', 'value'),
     Input('anomalies-crop-dropdown', 'value')]
)
def update_anomalies_graphs(selected_county, selected_crop):
    # Filter data for the selected county and crop
    county_crop_data = df[(df["County"] == selected_county) & (df["Crop Name"] == selected_crop)]

    if county_crop_data.empty:
        return {}, {}

    # Get thresholds for the current county
    thresholds = county_thresholds[selected_county]

    # Create the Yield Per Acre plot
    fig_yield = go.Figure()

    # Add crop yield trend line
    fig_yield.add_trace(go.Scatter(
        x=county_crop_data["Year"],
        y=county_crop_data["Yield Per Acre"],
        mode="lines+markers",
        name="Yield Per Acre",
        line=dict(color="black"),
        marker=dict(size=6)
    ))

    # Keep track of variables added to legend
    variables_plotted = set()

    # Add markers for each extreme weather variable
    for var, color in extreme_weather_vars.items():
        for _, row in county_crop_data.iterrows():
            if row[var] > thresholds[var]["high"]:
                if var not in variables_plotted:
                    showlegend = True
                    variables_plotted.add(var)
                else:
                    showlegend = False
                fig_yield.add_trace(go.Scatter(
                    x=[row["Year"]],
                    y=[row["Yield Per Acre"]],
                    mode="markers",
                    name=var.replace('_', ' ').title(),
                    marker=dict(size=12, color=color, opacity=0.8),
                    hovertext=f"{var.replace('_', ' ').title()}: {row[var]}",
                    showlegend=showlegend
                ))

    # Customize layout
    fig_yield.update_layout(
        title=f"Weather Anomalies and Yield in {selected_county} - {selected_crop}",
        xaxis_title="Year",
        yaxis_title="Yield Per Acre",
        legend_title="Legend",
        height=500
    )

    # Create the Production Per Acre plot
    fig_production = go.Figure()

    # Add crop production trend line
    fig_production.add_trace(go.Scatter(
        x=county_crop_data["Year"],
        y=county_crop_data["Production Per Acre"],
        mode="lines+markers",
        name="Production Per Acre",
        line=dict(color="black"),
        marker=dict(size=6)
    ))

    # Keep track of variables added to legend
    variables_plotted = set()

    # Add markers for each extreme weather variable
    for var, color in extreme_weather_vars.items():
        for _, row in county_crop_data.iterrows():
            if row[var] > thresholds[var]["high"]:
                if var not in variables_plotted:
                    showlegend = True
                    variables_plotted.add(var)
                else:
                    showlegend = False
                fig_production.add_trace(go.Scatter(
                    x=[row["Year"]],
                    y=[row["Production Per Acre"]],
                    mode="markers",
                    name=var.replace('_', ' ').title(),
                    marker=dict(size=12, color=color, opacity=0.8),
                    hovertext=f"{var.replace('_', ' ').title()}: {row[var]}",
                    showlegend=showlegend
                ))

    # Customize layout
    fig_production.update_layout(
        title=f"Weather Anomalies and Production in {selected_county} - {selected_crop}",
        xaxis_title="Year",
        yaxis_title="Production Per Acre",
        legend_title="Legend",
        height=500
    )

    return fig_yield, fig_production