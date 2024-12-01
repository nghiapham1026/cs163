import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv('./data/merged_yearly.csv')
techniques_df = pd.read_csv('./data/techniques.csv')
merged_data = pd.merge(techniques_df, df, on="County")

data_copy = df.copy()

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
            # Page Title
            html.H1(
                "Visualization of The Impact of Weather on Crop Yield",
                className="page-title"
            ),
            html.Hr(className="divider"),
            
            # Section 1: Impact of Extreme Weather on Crop Yield
            html.Div(
                className="section impact-weather-section",
                children=[
                    html.H2(
                        "Impact of Extreme Weather on Crop Yield",
                        className="section-title"
                    ),
                    html.P(
                        "Placeholder: Explore how extreme weather variables such as high temperatures, heavy rain, and strong winds "
                        "impact crop yields. Use the dropdowns to select specific crops and weather variables for detailed visualizations.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='impact-weather-crop-dropdown',
                                options=[{'label': crop, 'value': crop} for crop in df["Crop Name"].unique()],
                                value=df["Crop Name"].unique()[-1],
                                className="dropdown"
                            ),
                            html.Label("Select Extreme Weather Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='impact-extreme-variable-dropdown',
                                options=[{'label': var, 'value': var} for var in extreme_weather_features],
                                value=extreme_weather_features[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='impact-yield-comparison-graph',
                        className="graph"
                    )
                ]
            ),
            html.Hr(className="divider"),
            
            # Section 2: Weather Impact on Crop Yield
            html.Div(
                className="section weather-impact-section",
                children=[
                    html.H2(
                        "Weather Impact on Crop Yield",
                        className="section-title"
                    ),
                    html.P(
                        "Placeholder: Analyze how specific weather variables over time influence crop yield and production per acre. "
                        "The graphs below provide a comparison of yield and production under varying weather conditions.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='weather-impact-crop-dropdown',
                                options=[{'label': crop, 'value': crop} for crop in sorted(crops)],
                                value=sorted(crops)[0],
                                className="dropdown"
                            ),
                            html.Label("Select Extreme Weather Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='weather-impact-feature-dropdown',
                                options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in extreme_weather_features],
                                value=extreme_weather_features[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    html.Div(
                        className="graphs-container",
                        children=[
                            dcc.Graph(id='weather-yield-per-acre-graph', className="graph"),
                            dcc.Graph(id='weather-production-per-acre-graph', className="graph")
                        ]
                    )
                ]
            ),
            html.Hr(className="divider"),
            
            # Section 3: Weather Anomalies and Crop Yield
            html.Div(
                className="section weather-anomalies-section",
                children=[
                    html.H2(
                        "Weather Anomalies and Crop Yield",
                        className="section-title"
                    ),
                    html.P(
                        "Placeholder: Discover how deviations from normal weather patterns, such as anomalies in rainfall or temperature, "
                        "affect crop yields and production across counties.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='anomalies-county-dropdown',
                                options=[{'label': county, 'value': county} for county in sorted(df["County"].unique())],
                                value=sorted(df["County"].unique())[-1],
                                className="dropdown"
                            ),
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='anomalies-crop-dropdown',
                                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                                value=sorted(df["Crop Name"].unique())[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    html.Div(
                        className="graphs-container",
                        children=[
                            dcc.Graph(id='anomalies-yield-graph', className="graph"),
                            dcc.Graph(id='anomalies-production-graph', className="graph")
                        ]
                    )
                ]
            ),
            html.Hr(className="divider"),
            
            # Section 4: Crop Yield and Production by Farming Method
            html.Div(
                className="section yield-production-section",
                children=[
                    html.H2(
                        "Crop Yield and Production by Farming Method Under Different Weather Conditions",
                        className="section-title"
                    ),
                    html.P(
                        "Placeholder: Examine the relationship between farming practices and crop outcomes under various weather conditions. "
                        "Identify which methods yield the best results during extreme weather.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Extreme Weather Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='yield-production-weather-variable-dropdown',
                                options=[{'label': var.replace('_', ' ').title(), 'value': var} for var in extreme_weather_features],
                                value=extreme_weather_features[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='yield-production-graph',
                        className="graph"
                    )
                ]
            ),
            html.Hr(className="divider"),
            
            # Section 5: Weather Impact on Crop Yields by County
            html.Div(
                className="section county-impact-section",
                children=[
                    html.H2(
                        "Impact of Extreme Weather on Crop Yields",
                        className="section-title"
                    ),
                    html.P(
                        "Placeholder: Focus on specific counties to understand how localized extreme weather events impact crop yields. "
                        "This section allows granular analysis of regional effects.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id="weather-impact-county-dropdown",
                                options=[{'label': county, 'value': county} for county in sorted(data_copy["County"].unique())],
                                value="Fresno",
                                clearable=False,
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id="weather-impact-plot",
                        className="graph"
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

@callback(
    Output('yield-production-graph', 'figure'),
    [Input('yield-production-weather-variable-dropdown', 'value')]
)
def update_yield_production_graph(selected_var):
    data = merged_data.copy()

    # Categorize the weather variable into 'Low', 'Moderate', 'High'
    data['Weather Level'] = pd.cut(
        data[selected_var],
        bins=[
            data[selected_var].min() - 0.01,
            data[selected_var].quantile(0.33),
            data[selected_var].quantile(0.67),
            data[selected_var].max() + 0.01
        ],
        labels=['Low', 'Moderate', 'High'],
        include_lowest=True
    )

    # Group data for plotting
    grouped_data = data.groupby(['Farming Methods', 'Weather Level'], observed=True).agg({
        'Yield Per Acre': 'mean',
        'Production Per Acre': 'mean'
    }).reset_index()

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Yield Per Acre", "Production Per Acre"),
        shared_yaxes=False
    )

    # Define colors for weather levels
    weather_colors = {'Low': 'lightblue', 'Moderate': 'orange', 'High': 'red'}

    # Add bars for Yield Per Acre
    for level in ['Low', 'Moderate', 'High']:
        data_level = grouped_data[grouped_data['Weather Level'] == level]
        fig.add_trace(
            go.Bar(
                x=data_level['Farming Methods'],
                y=data_level['Yield Per Acre'],
                name=f"{level} Weather",
                marker_color=weather_colors[level],
            ),
            row=1, col=1
        )

    # Add bars for Production Per Acre
    for level in ['Low', 'Moderate', 'High']:
        data_level = grouped_data[grouped_data['Weather Level'] == level]
        show_legend = False if level != 'Low' else True  # Show legend only once
        fig.add_trace(
            go.Bar(
                x=data_level['Farming Methods'],
                y=data_level['Production Per Acre'],
                name=f"{level} Weather",
                marker_color=weather_colors[level],
                showlegend=show_legend,
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text=f"Crop Yield and Production by Farming Method Under Different Levels of {selected_var.replace('_', ' ').title()}",
        barmode='group',
        legend_title_text='Weather Level',
        height=600,
        width=1000
    )

    # Update axes titles
    fig.update_xaxes(title_text="Farming Methods", row=1, col=1)
    fig.update_xaxes(title_text="Farming Methods", row=1, col=2)
    fig.update_yaxes(title_text="Yield Per Acre", row=1, col=1)
    fig.update_yaxes(title_text="Production Per Acre", row=1, col=2)

    return fig

@callback(
    Output("weather-impact-plot", "figure"),
    [Input("weather-impact-county-dropdown", "value")]
)
def update_weather_impact_plot(selected_county):
    # Filter data for the selected county
    county_data = data_copy[data_copy["County"] == selected_county]

    # Initialize a list to store aggregated results
    aggregated_results = []

    # Loop through each weather variable, categorize, and calculate average yield
    for var in extreme_weather_vars:
        # Create categories for the current weather variable
        county_data['Category'] = pd.cut(
            county_data[var],
            bins=[county_data[var].min() - 0.01, county_data[var].quantile(0.33),
                  county_data[var].quantile(0.67), county_data[var].max() + 0.01],
            labels=['Low', 'Moderate', 'High'],
            include_lowest=True
        )

        # Aggregate data: Calculate average yield per category and crop name
        avg_yield = county_data.groupby(['Category', 'Crop Name'], observed=True)['Yield Per Acre'].mean().reset_index()
        avg_yield['Weather Variable'] = var.replace('_', ' ').title()

        # Calculate percentage change relative to "Low"
        low_yield = avg_yield[avg_yield['Category'] == 'Low'][['Crop Name', 'Yield Per Acre']].rename(
            columns={'Yield Per Acre': 'Low Yield'}
        )
        avg_yield = avg_yield.merge(low_yield, on='Crop Name', how='left')
        avg_yield['Percent Change'] = ((avg_yield['Yield Per Acre'] - avg_yield['Low Yield']) / avg_yield['Low Yield']) * 100
        avg_yield['Percent Change'] = avg_yield['Percent Change'].round(2)  # Round to two decimal places
        aggregated_results.append(avg_yield)

    # Combine all results into a single DataFrame
    combined_data = pd.concat(aggregated_results)

    # Create the grouped bar plot for both raw yield and percentage change
    fig = px.bar(
        combined_data,
        x="Category",
        y="Yield Per Acre",
        color="Crop Name",
        barmode="group",
        facet_col="Weather Variable",
        text="Percent Change",
        title=f"Impact of Extreme Weather on Crop Yields in {selected_county}",
        labels={
            "Yield Per Acre": "Average Yield Per Acre",
            "Category": "Weather Severity",
            "Weather Variable": "Extreme Weather Variable",
            "Percent Change": "Change (%)"
        }
    )

    # Add text annotations for percentage changes
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(
        texttemplate="%{text}%",  # Show percentage change inside bars
        textposition="outside"
    )

    # Enhance layout for better readability
    fig.update_layout(
        legend_title_text="Crop Type",
        xaxis_title="Weather Severity",
        yaxis_title="Average Yield Per Acre",
        title_x=0.5  # Center the title
    )

    return fig