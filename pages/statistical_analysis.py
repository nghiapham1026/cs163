import pandas as pd
from dash import html, dcc, callback, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Load the weather data
weather_data = pd.read_csv('./data/weather_data.csv')
merged_yearly = pd.read_csv('./data/merged_yearly.csv')
results = pd.read_csv('./data/results.csv')
summary_data = pd.read_csv('./data/hypothesis_summary.csv')
correlation_df = pd.read_csv('./data/correlation_df.csv')
results_df = pd.read_csv('./data/regression_results.csv')

# Define a threshold for strong correlations
strong_corr_threshold = 0.3

# Filter for strong positive correlations
strong_positive = correlation_df[
    (correlation_df['Correlation'] >= strong_corr_threshold)
]

# Filter for strong negative correlations
strong_negative = correlation_df[
    (correlation_df['Correlation'] <= -strong_corr_threshold)
]

# Combine strong correlations
strong_correlations = pd.concat([strong_positive, strong_negative])

correlation_columns = [
    'Yield Per Acre', 'Production Per Acre', 'Value Per Acre',
    'high_temp_days', 'low_temp_days', 'heavy_rain_days',
    'high_wind_days', 'cloudy_days', 'low_visibility_days',
    'snow_days'
]

# Define targets and predictors
targets = ['Yield Per Acre', 'Production Per Acre']
predictors = ['high_temp_days', 'low_temp_days', 'heavy_rain_days',
              'snow_days', 'high_wind_days', 'low_visibility_days',
              'cloudy_days']

def layout():
    return html.Div(
        [
            html.H1(
                "Extreme Weather Threshold Analysis",
                className="extreme-weather-title"
            ),

            # Extreme Weather Section
            html.Div(
                [
                    html.Label(
                        "Select a County (Extreme Weather):",
                        className="extreme-weather-county-label"
                    ),
                    dcc.Dropdown(
                        id="extreme-weather-county-dropdown",
                        options=[
                            {"label": county, "value": county}
                            for county in weather_data['city_name'].unique()
                        ],
                        value=weather_data['city_name'].unique()[0],
                        clearable=False,
                        className="extreme-weather-county-dropdown"
                    ),
                    html.Label(
                        "Select a Weather Variable:",
                        className="extreme-weather-variable-label"
                    ),
                    dcc.Dropdown(
                        id="extreme-weather-variable-dropdown",
                        options=[
                            {"label": "Temperature", "value": "temp"},
                            {"label": "Rain (1 Hour)", "value": "rain_1h"},
                            {"label": "Dew Point", "value": "dew_point"},
                            {"label": "Cloud Cover", "value": "clouds_all"}
                        ],
                        value="temp",
                        clearable=False,
                        className="extreme-weather-variable-dropdown"
                    ),
                    dcc.Graph(
                        id="extreme-weather-graph",
                        className="extreme-weather-graph"
                    ),
                ],
                className="extreme-weather-section",
                style={"margin-bottom": "50px"}
            ),

            html.H1(
                "County-Crop Correlation Analysis",
                className="correlation-analysis-title"
            ),

            # Correlation Matrix Section
            html.Div(
                [
                    html.Label(
                        "Select a County (Correlation Matrix):",
                        className="correlation-county-label"
                    ),
                    dcc.Dropdown(
                        id="correlation-county-dropdown",
                        options=[
                            {"label": county, "value": county}
                            for county in merged_yearly['County'].unique()
                        ],
                        value=merged_yearly['County'].unique()[0],
                        clearable=False,
                        className="correlation-county-dropdown"
                    ),
                    html.Label(
                        "Select a Crop:",
                        className="correlation-crop-label"
                    ),
                    dcc.Dropdown(
                        id="correlation-crop-dropdown",
                        options=[],  # Populated by callback based on county selection
                        value=None,
                        clearable=False,
                        className="correlation-crop-dropdown"
                    ),
                    dcc.Graph(
                        id="correlation-matrix-plot",
                        className="correlation-matrix-graph"
                    ),
                ],
                className="correlation-matrix-section"
            ),

            html.Div([
                html.Label("Select County:"),
                dcc.Dropdown(
                    id='county-dropdown3',
                    options=[{'label': 'All Counties', 'value': 'All Counties'}] +
                            [{'label': county, 'value': county} for county in correlation_df['County'].unique()],
                    value='All Counties'  # Default value
                ),
            ]),
            dcc.Graph(id='correlation-boxplot'),

            html.Div([
                html.Label("Select Weather Variable:"),
                dcc.Dropdown(
                    id='weather-variable-dropdown',
                    options=[{'label': 'All Variables', 'value': 'All Variables'}] + 
                            [{'label': var, 'value': var} for var in strong_correlations['Weather Variable'].unique()],
                    value='All Variables'
                ),
            ]),
            dcc.Graph(id='frequency-plot'),

            html.Div([
                html.Label("Select County:"),
                dcc.Dropdown(
                    id='county-dropdown',
                    options=[{'label': 'All Counties', 'value': 'All Counties'}] +
                            [{'label': county, 'value': county} for county in results_df['County'].unique()],
                    value='All Counties'  # Default value
                ),
            ]),
            dcc.Graph(id='ols-regression-plot'),

            html.Div([
                html.Label("Select County:"),
                dcc.Dropdown(
                    id='county-dropdown2',
                    options=[{'label': 'All Counties', 'value': 'All Counties'}] +
                            [{'label': county, 'value': county} for county in results_df['County'].unique()],
                    value='All Counties'  # Default option
                ),
            ]),
            dcc.Graph(id='ols-heatmap'),
        ],
        className="main-container"
    )

@callback(
    Output("extreme-weather-graph", "figure"),
    Input("extreme-weather-county-dropdown", "value"),
    Input("extreme-weather-variable-dropdown", "value")
)
def plot_extreme_weather(selected_county, selected_variable):
    county_data = weather_data[weather_data["city_name"] == selected_county]
    lower_threshold = county_data[selected_variable].quantile(0.1)
    upper_threshold = county_data[selected_variable].quantile(0.9)
    
    county_data["Extreme"] = county_data[selected_variable].apply(
        lambda x: "Low (10th percentile)" if x <= lower_threshold 
                  else ("High (90th percentile)" if x >= upper_threshold else "Normal")
    )

    county_data["date"] = pd.to_datetime(county_data["date"])
    fig = px.scatter(
        county_data, x="date", y=selected_variable, color="Extreme",
        title=f"{selected_variable.capitalize()} Extremes Over Time in {selected_county}",
        labels={"Extreme": "Extreme Level"},
        color_discrete_map={
            "Low (10th percentile)": "blue",
            "High (90th percentile)": "red",
            "Normal": "gray"
        }
    )
    fig.add_hline(y=lower_threshold, line_dash="dash", line_color="blue", 
                  annotation_text="10th Percentile", annotation_position="bottom left")
    fig.add_hline(y=upper_threshold, line_dash="dash", line_color="red", 
                  annotation_text="90th Percentile", annotation_position="top left")

    fig.update_layout(xaxis_title="Date", yaxis_title=selected_variable.capitalize())
    return fig


@callback(
    Output("correlation-crop-dropdown", "options"),
    Output("correlation-crop-dropdown", "value"),
    Input("correlation-county-dropdown", "value")
)
def update_crop_dropdown(selected_county):
    df_county = merged_yearly[merged_yearly['County'] == selected_county]
    crops = df_county['Crop Name'].unique()
    crop_options = [{"label": crop, "value": crop} for crop in crops]
    return crop_options, crops[0]

@callback(
    Output("correlation-matrix-plot", "figure"),
    Input("correlation-county-dropdown", "value"),
    Input("correlation-crop-dropdown", "value")
)
def plot_correlation_matrix(selected_county, selected_crop):
    df_filtered = merged_yearly[(merged_yearly['County'] == selected_county) &
                                (merged_yearly['Crop Name'] == selected_crop)]

    if df_filtered.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for this selection")
        return fig
    
    correlation_matrix = df_filtered[correlation_columns].corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation")
        )
    )
    fig.update_layout(
        title=f"Correlation Matrix for {selected_crop} in {selected_county}",
        xaxis=dict(tickmode="array", tickvals=list(range(len(correlation_columns))), ticktext=correlation_columns),
        yaxis=dict(tickmode="array", tickvals=list(range(len(correlation_columns))), ticktext=correlation_columns)
    )
    return fig

@callback(
    Output('correlation-boxplot', 'figure'),
    [Input('county-dropdown3', 'value')]
)
def update_correlation_boxplot(selected_county):
    # Filter data for the selected county or use all data if "All Counties" is selected
    if selected_county == 'All Counties':
        county_data = correlation_df
    else:
        county_data = correlation_df[correlation_df['County'] == selected_county]

    # If no data is available, return an empty figure
    if county_data.empty:
        return go.Figure().update_layout(
            title=f"No data available for {selected_county}",
            xaxis_title="Weather Variable",
            yaxis_title="Correlation Coefficient"
        )

    # Create the box plot
    fig = go.Figure()

    # Add box plot for correlations
    fig.add_trace(go.Box(
        x=county_data['Weather Variable'],
        y=county_data['Correlation'],
        name=f'Correlations in {selected_county}',
        boxmean=True  # Show mean line in the box plot
    ))

    # Customize the layout
    fig.update_layout(
        title=(
            'Correlation of Crop Variables with Weather Variables (All Counties)'
            if selected_county == 'All Counties'
            else f'Correlation of Crop Variables with Weather Variables in {selected_county}'
        ),
        xaxis_title='Weather Variable',
        yaxis_title='Correlation Coefficient',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

@callback(
    Output('frequency-plot', 'figure'),
    [Input('weather-variable-dropdown', 'value')]
)
def update_frequency_plot(selected_variable):
    # Filter data based on the selected variable
    if selected_variable == 'All Variables':
        filtered_data = strong_correlations
    else:
        filtered_data = strong_correlations[strong_correlations['Weather Variable'] == selected_variable]

    # Count the frequency of strong correlations per county
    county_frequency = filtered_data['County'].value_counts().reset_index()
    county_frequency.columns = ['County', 'Frequency']
    county_frequency = county_frequency.sort_values(by='Frequency', ascending=False)

    # Create the bar plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=county_frequency['County'],
        y=county_frequency['Frequency'],
        marker=dict(color='skyblue'),
        name='Frequency'
    ))

    # Customize layout
    fig.update_layout(
        title=f'Frequency of Strong Correlations by County ({selected_variable})',
        xaxis_title='County',
        yaxis_title='Number of Strong Correlations',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

@callback(
    Output('ols-regression-plot', 'figure'),
    [Input('county-dropdown', 'value')]
)
def update_ols_regression_plot(selected_county):
    # Filter data for the selected county or use all data if "All Counties" is selected
    if selected_county == 'All Counties':
        filtered_data = results_df
    else:
        filtered_data = results_df[results_df['County'] == selected_county]

    # If no data is available, return an empty figure
    if filtered_data.empty:
        return go.Figure().update_layout(
            title=f"No data available for {selected_county}",
            xaxis_title="Predictor",
            yaxis_title="P-value"
        )

    # Create the box plot
    fig = go.Figure()

    fig.add_trace(go.Box(
        x=filtered_data['Predictor'],
        y=filtered_data['P-value'],
        name=f'OLS Regression in {selected_county}',
        boxmean=True  # Show mean line in the box plot
    ))

    # Add a horizontal line for the significance threshold
    fig.add_trace(go.Scatter(
        x=filtered_data['Predictor'].unique(),
        y=[0.05] * len(filtered_data['Predictor'].unique()),
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Significance Threshold (p=0.05)'
    ))

    # Customize the layout
    fig.update_layout(
        title=(
            'Distribution of P-values per Predictor (All Counties)'
            if selected_county == 'All Counties'
            else f'Distribution of P-values per Predictor in {selected_county}'
        ),
        xaxis_title='Predictor',
        yaxis_title='P-value',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig

@callback(
    Output('ols-heatmap', 'figure'),
    [Input('county-dropdown2', 'value')]
)
def update_ols_heatmap(selected_county):
    # Filter data based on selected county
    if selected_county == 'All Counties':
        filtered_data = results_df
    else:
        filtered_data = results_df[results_df['County'] == selected_county]

    # If no data is available, return an empty figure
    if filtered_data.empty:
        return go.Figure().update_layout(
            title=f"No data available for {selected_county}",
            xaxis_title="Crop",
            yaxis_title="Predictor"
        )

    # Aggregate p-values by taking the median p-value per Predictor and Crop
    agg_p_values = filtered_data.groupby(['Predictor', 'Crop'])['P-value'].median().reset_index()

    # Pivot the DataFrame to create a heatmap structure
    p_values_pivot = agg_p_values.pivot(index='Predictor', columns='Crop', values='P-value')

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=p_values_pivot.values,
            x=p_values_pivot.columns,
            y=p_values_pivot.index,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Median P-value")
        )
    )

    # Customize layout
    fig.update_layout(
        title=(
            'Heatmap of Median P-values per Predictor and Crop (All Counties)'
            if selected_county == 'All Counties'
            else f'Heatmap of Median P-values per Predictor and Crop in {selected_county}'
        ),
        xaxis_title='Crop',
        yaxis_title='Predictor',
        xaxis=dict(tickangle=45),
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig
