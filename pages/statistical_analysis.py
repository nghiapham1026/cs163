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

correlation_columns = [
    'Yield Per Acre', 'Production Per Acre', 'Value Per Acre', 'high_temp_days',
    'low_temp_days', 'heavy_rain_days', 'high_wind_days', 'cloudy_days', 'low_visibility_days', 'snow_days'
]

# Define targets and predictors
targets = ['Harvested Acres', 'Yield Per Acre', 'Production Per Acre']
predictors = ['high_temp_days', 'low_temp_days', 'heavy_rain_days', 'snow_days',
              'high_wind_days', 'low_visibility_days', 'cloudy_days']

def layout():
    return html.Div([
        html.H1("Extreme Weather Threshold Analysis"),
        
        # Extreme Weather Section
        html.Div([
            html.Label("Select a County (Extreme Weather):"),
            dcc.Dropdown(
                id="extreme-weather-county-dropdown",
                options=[{"label": county, "value": county} for county in weather_data['city_name'].unique()],
                value=weather_data['city_name'].unique()[0],
                clearable=False
            ),
            html.Label("Select a Weather Variable:"),
            dcc.Dropdown(
                id="extreme-weather-variable-dropdown",
                options=[
                    {"label": "Temperature", "value": "temp"},
                    {"label": "Rain (1 Hour)", "value": "rain_1h"},
                    {"label": "Dew Point", "value": "dew_point"},
                    {"label": "Cloud Cover", "value": "clouds_all"}
                ],
                value="temp",
                clearable=False
            ),
            dcc.Graph(id="extreme-weather-graph"),
        ], style={"margin-bottom": "50px"}),
        
        html.H1("County-Crop Correlation Analysis"),
        
        # Correlation Matrix Section
        html.Div([
            html.Label("Select a County (Correlation Matrix):"),
            dcc.Dropdown(
                id="correlation-county-dropdown",
                options=[{"label": county, "value": county} for county in merged_yearly['County'].unique()],
                value=merged_yearly['County'].unique()[0],
                clearable=False
            ),
            html.Label("Select a Crop:"),
            dcc.Dropdown(
                id="correlation-crop-dropdown",
                options=[],  # Populated by callback based on county selection
                value=None,
                clearable=False
            ),
            dcc.Graph(id="correlation-matrix-plot"),
        ]),

        html.H1("Statistical Analysis Results"),

        # Dropdown for selecting multiple counties
        html.Label("Select Counties:"),
        dcc.Dropdown(
            id="county-dropdown",
            options=[{"label": county, "value": county} for county in results['County'].unique()],
            value=[results['County'].unique()[0]],  # Default to the first county
            multi=True,
            clearable=False
        ),

        # Dropdown for selecting the target variable
        html.Label("Select an Outcome Variable:"),
        dcc.Dropdown(
            id="target-dropdown",
            options=[{"label": target, "value": target} for target in results['Target'].unique()],
            value=results['Target'].unique()[0],
            clearable=False
        ),

        # P-Value Plot
        html.H3("P-Value Significance"),
        dcc.Graph(id="p-value-plot"),

        # Coefficient Plot
        html.H3("Coefficient Analysis"),
        dcc.Graph(id="coefficient-plot"),

        # R-Squared Plot
        html.H3("R-Squared Analysis"),
        dcc.Graph(id="r-squared-plot"),

        html.H2("Hypothesis Summary Table"),
        
        dash_table.DataTable(
            data=summary_data.to_dict("records"),
            columns=[{"name": i, "id": i} for i in summary_data.columns],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Hypothesis Conclusion} = "Reject H₀"', 'column_id': 'Hypothesis Conclusion'},
                    'backgroundColor': 'tomato',
                    'color': 'white'
                },
                {
                    'if': {'filter_query': '{Hypothesis Conclusion} = "Fail to Reject H₀"', 'column_id': 'Hypothesis Conclusion'},
                    'backgroundColor': 'lightgreen',
                    'color': 'black'
                }
            ],
            sort_action="native",
            filter_action="native",
            page_size=100  # Customize the number of rows per page
        ),

        html.H1("Statistical Analysis Results - Some Additional Plots"),
        
        # Crop selection dropdown
        html.Label("Select a Crop:"),
        dcc.Dropdown(
            id="crop-dropdown",
            options=[{"label": crop, "value": crop} for crop in results['Crop'].unique()],
            value=results['Crop'].unique()[0],  # Default to the first crop
            clearable=False
        ),

        # Coefficient Plot
        html.H3("Interactive Regression Coefficients by County"),
        dcc.Graph(id="coefficient-graph"),

        html.H1("Statistical Analysis Results"),
        
        # Dropdown for selecting a crop
        html.Label("Select a Crop:"),
        dcc.Dropdown(
            id="crop-heatmap-dropdown",
            options=[{"label": crop, "value": crop} for crop in results['Crop'].unique()],
            value=results['Crop'].unique()[0],  # Default to the first crop
            clearable=False
        ),

        # Coefficient Heatmap
        html.H3("Interactive Coefficient Heatmap by County"),
        dcc.Graph(id="coefficient-heatmap"),
        
        # Dropdown for selecting a crop
        html.Label("Select a Crop:"),
        dcc.Dropdown(
            id="crop-r2-dropdown",
            options=[{"label": crop, "value": crop} for crop in results['Crop'].unique()],
            value=results['Crop'].unique()[0],  # Default to the first crop
            clearable=False
        ),

        # R-Squared Plot
        html.H3("R-Squared Values by County and Target"),
        dcc.Graph(id="r-squared-barplot"),
    ])
    
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
    Output("p-value-plot", "figure"),
    Input("county-dropdown", "value"),
    Input("target-dropdown", "value")
)
def plot_p_values(selected_counties, selected_target):
    # Filter data for selected counties and target
    filtered_data = results[(results["County"].isin(selected_counties)) & (results["Target"] == selected_target)]

    # Plot p-values, grouped by county
    fig = px.bar(
        filtered_data,
        x="Predictor",
        y="P-Value",
        color="County",
        barmode="group",
        title=f"P-Values for Selected Counties - {selected_target}",
        labels={"P-Value": "P-Value", "County": "County"}
    )
    
    # Add a significance line at p=0.05
    fig.add_hline(y=0.05, line_dash="dash", line_color="blue", 
                  annotation_text="Significance Level (0.05)", annotation_position="top left")
    fig.update_layout(xaxis_title="Predictor", yaxis_title="P-Value")
    
    return fig

@callback(
    Output("coefficient-plot", "figure"),
    Input("county-dropdown", "value"),
    Input("target-dropdown", "value")
)
def plot_coefficients(selected_counties, selected_target):
    # Filter data for selected counties, target, and significant predictors (p < 0.05)
    filtered_data = results[(results["County"].isin(selected_counties)) & 
                            (results["Target"] == selected_target) & 
                            (results["P-Value"] < 0.05)]

    # Plot coefficients, grouped by county
    fig = px.bar(
        filtered_data,
        x="Predictor",
        y="Coefficient",
        color="County",
        barmode="group",
        title=f"Significant Coefficients for Selected Counties - {selected_target}",
        labels={"Coefficient": "Coefficient", "County": "County"}
    )
    fig.update_layout(xaxis_title="Predictor", yaxis_title="Coefficient")
    
    return fig

@callback(
    Output("r-squared-plot", "figure"),
    Input("county-dropdown", "value"),
    Input("target-dropdown", "value")
)
def plot_r_squared(selected_counties, selected_target):
    # Filter data for selected counties and target
    filtered_data = results[(results["County"].isin(selected_counties)) & (results["Target"] == selected_target)]

    # Plot R-squared values for each selected county
    fig = px.bar(
        filtered_data,
        x="County",
        y="R-Squared",
        color="County",
        title=f"R-Squared Values for Selected Counties - {selected_target}",
        labels={"R-Squared": "R-Squared"}
    )
    fig.update_layout(xaxis_title="County", yaxis_title="R-Squared")
    
    return fig

@callback(
    Output("coefficient-graph", "figure"),
    Input("crop-dropdown", "value")
)
def update_coefficient_plot(selected_crop):
    # Filter data for the selected crop
    subset = results[results['Crop'] == selected_crop]

    # Create the interactive bar plot
    fig = px.bar(
        subset,
        x='Predictor',
        y='Coefficient',
        color='County',
        title=f"Interactive Regression Coefficients by County for {selected_crop}",
        labels={'Coefficient': 'Coefficient Value'},
        hover_data={'P-Value': True, 'Target': True},
        barmode='group'
    )

    # Add a horizontal line at y=0 for reference
    fig.add_shape(type='line', x0=-0.5, y0=0, x1=len(subset['Predictor'].unique()) - 0.5, y1=0,
                  line=dict(color="Gray", width=1, dash="dash"))

    fig.update_layout(
        xaxis_title="Predictor",
        yaxis_title="Coefficient",
        legend_title="County"
    )
    
    return fig

@callback(
    Output("coefficient-heatmap", "figure"),
    Input("crop-heatmap-dropdown", "value")
)
def update_coefficient_heatmap(selected_crop):
    # Filter data for the selected crop across all counties
    subset = results[results['Crop'] == selected_crop]

    # Pivot data to create heatmap format
    heatmap_data = subset.pivot_table(index='County', columns='Predictor', values='Coefficient')

    # Create the interactive heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu',  # Red to Blue color scale
        zmid=0,  # Center the color scale at 0
        colorbar=dict(title="Coefficient")
    ))

    # Update layout with title and axis labels
    fig.update_layout(
        title=f"Coefficient Heatmap for {selected_crop}",
        xaxis_title="Predictor",
        yaxis_title="County",
        height=500,  # Adjust for readability
        width=800   # Adjust for readability
    )
    
    return fig

@callback(
    Output("r-squared-barplot", "figure"),
    Input("crop-r2-dropdown", "value")
)
def update_r_squared_plot(selected_crop):
    # Filter data for the selected crop
    subset = results[results['Crop'] == selected_crop]

    # Prepare R-squared data, ensuring each (County, Target) combination is unique
    r_squared_data = subset.drop_duplicates(subset=['County', 'Target'])[['County', 'Target', 'R-Squared']]

    # Create the bar plot
    fig = px.bar(
        r_squared_data,
        x='Target',
        y='R-Squared',
        color='County',
        barmode='group',
        title=f"R-Squared Values by County for {selected_crop}",
        labels={'R-Squared': 'R-Squared Value'},
    )

    # Set y-axis limit from 0 to 1 for R-squared values
    fig.update_yaxes(range=[0, 1])

    # Update layout for legend and titles
    fig.update_layout(
        xaxis_title="Outcome Variable (Target)",
        yaxis_title="R-Squared Value",
        legend_title="County",
        height=500,
        width=800
    )
    
    return fig