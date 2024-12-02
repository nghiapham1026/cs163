import pandas as pd
from dash import html, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objs as go

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
        className="main-container",
        children=[
            # Page Title
            html.H1(
                "Visualization Dashboard: Weather and Crop Analysis",
                className="page-title"
            ),
            
            html.Hr(className="divider"),

            # Section 1: Extreme Weather Threshold Analysis
            html.Div(
                className="section extreme-weather-section",
                children=[
                    html.H2(
                        "Extreme Weather Threshold Analysis",
                        className="section-title"
                    ),
                    html.P(
                        "This section allows you to explore how extreme weather variables are determined using a percentile-based system. "
                        "By selecting a county and a specific weather variable, you can visualize the distribution of that variable over time "
                        "and identify the thresholds that define extreme conditions. The 10th and 90th percentiles are used to represent the lower "
                        "and upper extremes, respectively. Data points falling below the 10th percentile are considered 'Low (10th percentile)', "
                        "while those above the 90th percentile are labeled 'High (90th percentile)'. This analysis helps you understand the variability "
                        "of weather parameters in different regions and how they might contribute to agricultural outcomes.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label(
                                "Select a County:",
                                className="dropdown-label"
                            ),
                            dcc.Dropdown(
                                id="extreme-weather-county-dropdown",
                                options=[
                                    {"label": county, "value": county}
                                    for county in weather_data['city_name'].unique()
                                ],
                                value=weather_data['city_name'].unique()[0],
                                clearable=False,
                                className="dropdown"
                            ),
                            html.Label(
                                "Select a Weather Variable:",
                                className="dropdown-label"
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
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id="extreme-weather-graph",
                        className="graph"
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Section 2: County-Crop Correlation Analysis
            html.Div(
                className="section correlation-analysis-section",
                children=[
                    html.H2(
                        "County-Crop Correlation Analysis",
                        className="section-title"
                    ),
                    html.P(
                        "In this section, you can explore the extent of the correlation between extreme weather variables and crop outcomes for each county. "
                        "By selecting a county and a specific crop from the dropdown menus below, you can generate a correlation matrix that visualizes the relationships "
                        "between various weather factors and crop performance indicators such as yield per acre, production per acre, and harvested acres. "
                        "Correlation coefficients range from -1 to 1, where values closer to 1 or -1 indicate a strong positive or negative correlation, respectively. "
                        "A coefficient above **0.4** is considered a good correlation, suggesting a meaningful relationship between the variables. "
                        "Values near 0 indicate little to no linear relationship. "
                        "This analysis helps you identify which weather conditions significantly impact crop outcomes in different regions, "
                        "providing valuable insights for agricultural planning and risk management.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label(
                                "Select a County:",
                                className="dropdown-label"
                            ),
                            dcc.Dropdown(
                                id="correlation-county-dropdown",
                                options=[
                                    {"label": county, "value": county}
                                    for county in merged_yearly['County'].unique()
                                ],
                                value=merged_yearly['County'].unique()[0],
                                clearable=False,
                                className="dropdown"
                            ),
                            html.Label(
                                "Select a Crop:",
                                className="dropdown-label"
                            ),
                            dcc.Dropdown(
                                id="correlation-crop-dropdown",
                                options=[],  # Populated by callback based on county selection
                                value=None,
                                clearable=False,
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id="correlation-matrix-plot",
                        className="graph"
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Section 3: Weather Variable Frequency Analysis
            html.Div(
                className="section frequency-analysis-section",
                children=[
                    html.H2(
                        "Weather Variable Frequency Analysis",
                        className="section-title"
                    ),
                    html.P(
                        "This section aggregates counties based on the number of strong correlations between extreme weather variables and crop outcomes. "
                        "By selecting a weather variable from the dropdown menu, you can visualize how frequently that variable exhibits a significant correlation "
                        "with agricultural outputs across different counties. A correlation coefficient above **0.4** is considered strong and suggests a meaningful relationship. "
                        "The bar plot generated displays the number of strong correlations per county, allowing you to assess regional patterns and identify areas where weather conditions "
                        "have a pronounced impact on crop performance. This analysis reveals that the impact of weather on crops is complex and highly subjective to specific counties and crops.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Weather Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='weather-variable-dropdown',
                                options=[{'label': 'All Variables', 'value': 'All Variables'}] + 
                                        [{'label': var, 'value': var} for var in strong_correlations['Weather Variable'].unique()],
                                value='All Variables',
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='frequency-plot',
                        className="graph"
                    ),
                    html.Div(
                        className="analysis-section",
                        children=[
                            html.H3(
                                "Analysis of Correlation Patterns",
                                className="analysis-title"
                            ),
                            html.P(
                                "The bar plot above illustrates that most strong correlations are concentrated in Central California counties such as **Kings**, **Tulare**, and **Fresno**. "
                                "These regions exhibit a higher frequency of significant relationships between extreme weather variables and crop outcomes, indicating that weather conditions "
                                "in these areas have a more pronounced effect on agricultural productivity. Conversely, some counties, like **Santa Clara**, lack sufficient data to compute "
                                "correlation coefficients, leading to incomplete matrices and suggesting negligible relationships between weather variables and crop outcomes. "
                                "For instance, in the case of grapes, counties like **Santa Clara**, **Sonoma**, and **Alameda** show invalid correlations for variables like `snow_days` "
                                "and `low_visibility_days`. However, **Sonoma** still exhibits a decent correlation with `high_wind_days` (0.58), indicating that despite incomplete data, "
                                "certain weather factors can significantly impact crop yields in these regions.",
                                className="analysis-text"
                            ),
                            html.P(
                                "This analysis underscores that the impact of weather on crops is complex and highly dependent on both the county and the specific crops grown. "
                                "For example, in **Fresno**, `high_wind_days` and `low_visibility_days` have a stronger correlation with yield for **almonds** (0.6 and -0.58, respectively) "
                                "than for **grapes** (0.13 and -0.12). Similarly, for grapes, these weather variables have a stronger relationship with yield in **Tulare County** "
                                "(-0.47 and 0.61) compared to **Fresno County**, despite their geographical proximity. This suggests that local factors, such as crop type and farming practices, "
                                "influence how weather conditions affect agricultural outcomes.",
                                className="analysis-text"
                            ),
                            html.P(
                                "The counties with the highest number of strong correlations are predominantly located in Central California, with **El Dorado** in the Lake Tahoe region being a notable exception. "
                                "Interestingly, variables like `snow_days` are not good indicators of crop yield, likely because snowfall is relatively rare in most of California's agricultural regions. "
                                "Similarly, the low frequency of strong correlations for `heavy_rain_days` and `cloudy_days` is surprising, suggesting that these weather conditions may have less impact "
                                "on crop yields than expected. This could be attributed to the highly industrialized and climate-controlled nature of agriculture in Central California, where advanced farming techniques "
                                "mitigate the influence of external weather conditions.",
                                className="analysis-text"
                            )
                        ]
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Section 4: OLS Regression Analysis
            html.Div(
                className="section ols-regression-section",
                children=[
                    html.H2(
                        "OLS Regression Analysis",
                        className="section-title"
                    ),
                    html.P(
                        "In this section, we perform Ordinary Least Squares (OLS) regression analysis to validate our hypothesis that crop outcomes are determined by at least one extreme weather variable. "
                        "The null hypothesis (H₀) states that extreme weather variables do not significantly impact crop outcomes, while the alternative hypothesis (H₁) suggests that they do. "
                        "We reject H₀ if any predictor variable has a p-value below 0.05, indicating a statistically significant impact on crop outcomes.",
                        className="section-description"
                    ),
                    html.P(
                        "We define our dependent variables (targets) as **Yield Per Acre** and **Production Per Acre**, and our independent variables (predictors) as various extreme weather metrics such as `high_temp_days` and `low_visibility_days`. "
                        "The data is grouped by county and crop to conduct a localized analysis, allowing us to assess the impact of weather variables within specific regions and for specific crops. "
                        "For each County-Crop-Target combination, we perform an OLS regression and evaluate the coefficients (measure of predictor impact), p-values (statistical significance), and confidence intervals (reliability of coefficient estimates).",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='county-dropdown',
                                options=[{'label': 'All Counties', 'value': 'All Counties'}] +
                                        [{'label': county, 'value': county} for county in results_df['County'].unique()],
                                value='All Counties',
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='ols-regression-plot',
                        className="graph"
                    ),
                    html.Div(
                        className="analysis-section",
                        children=[
                            html.H3(
                                "Analysis of Regression Results",
                                className="analysis-title"
                            ),
                            html.P(
                                "The box plot above visualizes the distribution of p-values for each predictor variable across the selected county or all counties. "
                                "A p-value below the significance threshold of 0.05 (indicated by the dashed red line) suggests that we can reject the null hypothesis for that predictor, "
                                "indicating a statistically significant impact on the crop outcome. "
                                "By analyzing the p-values, we can identify which extreme weather variables significantly influence crop yields and production.",
                                className="analysis-text"
                            ),
                            html.P(
                                "Our findings indicate that in many counties, at least one extreme weather variable significantly affects crop outcomes. "
                                "For example, in **Fresno County**, predictors like `high_temp_days` and `low_visibility_days` show p-values below 0.05 for **almonds**, "
                                "implying a significant impact on yield per acre. "
                                "However, for **grapes** in the same county, these predictors may not exhibit significant p-values, highlighting the variability of weather impacts across different crops within the same region.",
                                className="analysis-text"
                            ),
                            html.P(
                                "In some counties, such as **Santa Clara**, the regression analysis may not yield significant predictors due to insufficient data or negligible relationships between weather variables and crop outcomes. "
                                "This underscores the importance of localized analysis, as the influence of extreme weather conditions can vary greatly depending on regional characteristics and agricultural practices.",
                                className="analysis-text"
                            ),
                            html.P(
                                "Overall, the regression analysis supports our hypothesis that extreme weather variables significantly impact crop outcomes in many cases. "
                                "The results highlight the complex and region-specific nature of agricultural productivity, emphasizing the need for tailored strategies to mitigate the effects of adverse weather conditions.",
                                className="analysis-text"
                            )
                        ]
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Section 5: OLS Regression Heatmap
            html.Div(
                className="section ols-heatmap-section",
                children=[
                    html.H2(
                        "OLS Regression Heatmap",
                        className="section-title"
                    ),
                    html.P(
                        "In this section, we extend our OLS regression analysis by mapping the p-values of each extreme weather variable into a heatmap for each county and crop combination. "
                        "This visualization allows us to determine which counties and crops have lower p-values, indicating a stronger statistical impact of specific weather variables on crop outcomes in those regions. "
                        "By analyzing the heatmap, we can assess the significance of each predictor across different crops and counties, gaining insights into the localized effects of extreme weather on agricultural productivity.",
                        className="section-description"
                    ),
                    html.P(
                        "Our hypothesis testing results show that **75% of all county-crop combinations succeeded in rejecting the Null Hypothesis**, meaning that at least one extreme weather variable significantly impacts crop yield or production in those regions. "
                        "Conversely, **25% failed to reject the Null Hypothesis**, indicating no significant impact detected. "
                        "These findings align with those obtained in the correlation heatmaps, reinforcing the importance of extreme weather variables in influencing crop outcomes.",
                        className="section-description"
                    ),
                    html.P(
                        "The heatmap reveals that variables like `snow_days`, `cloudy_days`, and `heavy_rain_days` generally have the highest p-values, suggesting they are less significant predictors of crop outcomes in most counties. "
                        "On the other hand, `low_visibility_days` and `high_wind_days` have the lowest p-values, indicating a stronger and more consistent impact on crops across different regions. "
                        "Variables such as `low_temp_days` and `high_temp_days` fall somewhere in between, with their significance varying by county and crop.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='county-dropdown2',
                                options=[{'label': 'All Counties', 'value': 'All Counties'}] +
                                        [{'label': county, 'value': county} for county in results_df['County'].unique()],
                                value='All Counties',
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(
                        id='ols-heatmap',
                        className="graph"
                    ),
                    html.Div(
                        className="analysis-section",
                        children=[
                            html.H3(
                                "Analysis of OLS Regression Heatmap",
                                className="analysis-title"
                            ),
                            html.P(
                                "The OLS Regression Heatmap visualizes the median p-values of each predictor variable (extreme weather metrics) across different crops. "
                                "The color intensity represents the magnitude of the p-values, with darker colors indicating lower p-values and thus stronger statistical significance. "
                                "This allows for a quick assessment of which weather variables are significant predictors for specific crops in various counties.",
                                className="analysis-text"
                            ),
                            html.P(
                                "From the heatmap, we observe that the majority of county-crop combinations have at least one extreme weather variable with a low p-value (below 0.05), reinforcing our conclusion that extreme weather significantly impacts crop outcomes in most regions. "
                                "Specifically, `low_visibility_days` and `high_wind_days` consistently show low p-values across various crops and counties, highlighting their importance as predictors. "
                                "These variables may affect pollination, evapotranspiration rates, and physical damage to crops, thereby influencing yield and production.",
                                className="analysis-text"
                            ),
                            html.P(
                                "In contrast, variables like `snow_days`, `cloudy_days`, and `heavy_rain_days` exhibit higher p-values, suggesting they are less influential on crop performance in the regions analyzed. "
                                "This could be due to the climatic characteristics of California, where snowfall is rare, and modern agricultural practices mitigate the effects of rainfall and cloud cover. "
                                "Additionally, irrigation and controlled environments may reduce the crops' dependence on natural precipitation, diminishing the impact of these weather variables.",
                                className="analysis-text"
                            ),
                            html.P(
                                "Variables such as `low_temp_days` and `high_temp_days` have p-values that vary between low and moderate, indicating that temperature extremes impact crop outcomes in some counties and for certain crops. "
                                "This variability underscores the importance of localized analysis, as the significance of temperature-related variables may depend on the specific temperature thresholds that affect different crops.",
                                className="analysis-text"
                            ),
                            html.P(
                                "Overall, the heatmap underscores the complex and localized nature of weather impacts on agriculture. "
                                "It reveals that while some weather variables have a consistent effect across multiple regions and crops, others are significant only in specific contexts. "
                                "This insight is crucial for developing targeted agricultural strategies and risk management practices tailored to the unique conditions of each county and crop type.",
                                className="analysis-text"
                            )
                        ]
                    )
                ]
            )
        ]
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
