import dash
from dash import html, dcc, Input, Output, callback, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import joblib

results_df = pd.read_csv('./data/results.csv')
all_models = joblib.load("./data/all_models.joblib")
df = pd.read_csv('./data/merged_yearly.csv')
data = df.copy()

def layout():
    return html.Div(
        className="main-container",
        children=[
            # Page Title
            html.H1(
                "Prediction Dashboard: Weather Impact on Crop Yields",
                className="page-title"
            ),

            html.Hr(className="divider"),

            # Model Workflow Description Section
            html.Div(
                className="section model-workflow-section",
                children=[
                    html.H2(
                        "Model Workflow Description",
                        className="section-title"
                    ),
                    html.P(
                        "In this section, we outline the machine learning model workflow developed to predict crop outcomes based on extreme weather variables and other influencing factors. "
                        "Our objective is to create predictive models that estimate:",
                        className="section-description"
                    ),
                    html.Ul(
                        children=[
                            html.Li("Yield Per Acre: Crop productivity efficiency."),
                            html.Li("Production Per Acre: Land usage efficiency for crop production.")
                        ],
                        className="objective-list"
                    ),
                    html.H3(
                        "Feature Engineering: Lagged Weather and Crop Features",
                        className="subsection-title"
                    ),
                    html.P(
                        "To capture delayed impacts of weather on crop outcomes, we incorporated 1-year and 2-year lagged features for extreme weather variables such as `high_temp_days`, `heavy_rain_days`, and `cloudy_days`. "
                        "We also added 1-year and 2-year lagged metrics for `Yield Per Acre` and `Production Per Acre` to account for historical crop performance influence. "
                        "Missing values were imputed with zeros for both weather and crop features to maintain data integrity.",
                        className="subsection-text"
                    ),
                    html.H3(
                        "Scaling",
                        className="subsection-title"
                    ),
                    html.P(
                        "All features were standardized using StandardScaler to ensure consistent input to the models and to improve model performance. "
                        "Data was grouped by County-Crop-Target combinations to conduct localized analyses and account for regional variations.",
                        className="subsection-text"
                    ),
                    html.H3(
                        "Model Selection",
                        className="subsection-title"
                    ),
                    html.P(
                        "We experimented with several machine learning algorithms to capture complex relationships and improve prediction accuracy:",
                        className="subsection-text"
                    ),
                    html.Ul(
                        children=[
                            html.Li("Gradient Boosting Regressor: Captures complex non-linear relationships."),
                            html.Li("Decision Tree Regressor: Provides interpretable models for feature importance."),
                            html.Li("K-Nearest Neighbors: Non-parametric approach sensitive to local data structures.")
                        ],
                        className="model-list"
                    ),
                    html.P(
                        "A time-based train-test split was applied, using the first 80% of observations for training and the remaining 20% for testing to respect the chronological order of data.",
                        className="subsection-text"
                    ),
                    html.H3(
                        "Model Evaluation Metrics",
                        className="subsection-title"
                    ),
                    html.P(
                        "Models were evaluated using the following metrics to assess performance:",
                        className="subsection-text"
                    ),
                    html.Ul(
                        children=[
                            html.Li("R-squared (R²): Measures the proportion of variance explained by the model."),
                            html.Li("Root Mean Squared Error (RMSE): Evaluates prediction accuracy in original units."),
                            html.Li("Mean Absolute Error (MAE): Quantifies average prediction errors.")
                        ],
                        className="metrics-list"
                    ),
                    html.H3(
                        "Training and Validation",
                        className="subsection-title"
                    ),
                    html.P(
                        "We applied time-based train-test splits for each County-Crop combination to ensure models were trained and tested on appropriate data subsets. "
                        "Cross-validation scores were computed for R², RMSE, and MAE to evaluate model stability and performance. "
                        "Feature impact was assessed using Recursive Feature Elimination with Cross-Validation (RFECV) to identify the most significant predictors. "
                        "Learning curves were generated to detect overfitting and underfitting issues. "
                        "All trained models and associated metadata, such as evaluation metrics and learning curves, were saved using `joblib` for future use.",
                        className="subsection-text"
                    )
                ]
            ),
            html.Hr(className="divider"),

            # Section 1: Performance Metrics
            html.Div(
                className="section performance-metrics-section",
                children=[
                    html.H2("Performance Metrics", className="section-title"),
                    html.P(
                        "In this section, we compare the performance of different machine learning models used to predict crop outcomes. "
                        "By selecting a target variable (Yield Per Acre or Production Per Acre) and a specific crop from the dropdown menus below, "
                        "you can view bar charts displaying the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each model across different counties. "
                        "These metrics help evaluate the accuracy of the models' predictions and assess their suitability for forecasting crop yields and production.",
                        className="section-description"
                    ),
                    html.P(
                        "The ideal ranges for learning have been determined based on the statistical properties of the target variables:",
                        className="section-description"
                    ),
                    html.Ul(
                        children=[
                            html.Li([
                                html.Strong("Yield Per Acre: "),
                                "The mean is 0.01 and the median is 0.00, indicating extremely small values skewed towards zero. "
                                "An acceptable MAE is set at 0.01 (the mean), and RMSE at 0.015, slightly higher to account for RMSE's sensitivity to larger errors."
                            ]),
                            html.Li([
                                html.Strong("Production Per Acre: "),
                                "The mean is 11.53 and the median is 4.01, showing right-skewed data with a wide range up to 66.80 and a standard deviation of 15.51. "
                                "An acceptable MAE is set at 1.5 (approximately 13% of the mean), and RMSE at 2, roughly 17% of the mean."
                            ])
                        ],
                        className="section-description"
                    ),
                    html.P(
                        "From the analysis of results, we observe that:",
                        className="section-description"
                    ),
                    html.Ul(
                        children=[
                            html.Li(
                                "The impact of extreme weather on crop yields and production is nonlinear, as evidenced by the better performance of nonlinear models like the Gradient Boosting Regressor."
                            ),
                            html.Li(
                                "Lagged extreme weather variables are the second most important predictors but are only half as influential as lagged crop features. This indicates that while weather impacts crops, its effect may interact with other factors."
                            ),
                            html.Li(
                                "Predictions for Production Per Acre are less accurate, possibly due to higher variability and a less consistent relationship with the predictors."
                            ),
                            html.Li(
                                "Small datasets limit the ability to capture general patterns, leading to overfitting and poor test performance. The models quickly exhaust available information, as shown by learning curves plateauing."
                            )
                        ],
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Target Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='target-dropdown',
                                options=[
                                    {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                    {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                                ],
                                value='Yield Per Acre',
                                className="dropdown"
                            )
                        ]
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='performance-crop-dropdown',
                                options=[
                                    {'label': crop, 'value': crop} for crop in results_df['Crop'].unique()
                                ],
                                value=results_df['Crop'].unique()[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(id='mae-bar-plot', className="performance-graph"),
                    dcc.Graph(id='rmse-bar-plot', className="performance-graph")
                ]
            ),

            html.Hr(className="divider"),

            # Section 2: Learning Curve Analysis
            html.Div(
                className="section learning-curve-section",
                children=[
                    html.H2("Learning Curve Analysis", className="section-title"),
                    html.P(
                        "In this section, we analyze the learning curves of different machine learning models for each county-crop combination. "
                        "By selecting a county and a crop from the dropdown menus below, you can visualize how the model's performance evolves with increasing training data. "
                        "The learning curves display the Mean Squared Error (MSE) for the training and validation sets as the training set size increases.",
                        className="section-description"
                    ),
                    html.P(
                        "The analysis of the learning curves reveals that while the models show declining MSE over time, indicating potential convergence given more data, "
                        "the performance on the test sets remains poor. The MSE declines slowly before abruptly plateauing, suggesting that the models quickly exhaust the available information in the dataset. "
                        "This behavior indicates that with the small datasets available, the models tend to capture noise and patterns specific to the training set, failing to generalize well to new data.",
                        className="section-description"
                    ),
                    html.P(
                        "Furthermore, the fact that lagged crop features contribute the most to the model's predictions suggests that the models heavily rely on historical crop data. "
                        "Lagged weather variables are the second most important predictors but are only about half as influential as the lagged crop features. "
                        "This implies that while weather impacts crops, the models depend more on historical performance, possibly due to limited data capturing the complex interactions between weather and crop outcomes.",
                        className="section-description"
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='training-county-dropdown',
                                options=[
                                    {'label': county, 'value': county} for county in results_df['County'].unique()
                                ],
                                value=results_df['County'].unique()[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Crop:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='training-crop-dropdown',
                                options=[
                                    {'label': crop, 'value': crop} for crop in results_df['Crop'].unique()
                                ],
                                value=results_df['Crop'].unique()[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(id='yield-per-acre-plot', className="training-graph"),
                    dcc.Graph(id='production-per-acre-plot', className="training-graph"),
                    html.Div(
                        className="analysis-section",
                        children=[
                            html.H3("Analysis of Learning Curves", className="analysis-title"),
                            html.P(
                                "The learning curves for the models demonstrate that although the Mean Squared Error (MSE) decreases with more training data, the decrease is gradual and eventually plateaus. "
                                "This suggests that the models are quickly reaching the limit of the information available in the data. "
                                "The poor performance on the test sets indicates that the models may be overfitting to the training data, capturing noise and patterns that do not generalize well.",
                                className="analysis-text"
                            ),
                            html.P(
                                "The reliance on lagged crop features as the most significant predictors shows that the models depend heavily on historical crop performance to make predictions. "
                                "While lagged weather variables are the second most important, their influence is significantly less, suggesting that the models are not effectively capturing the impact of weather on crop outcomes. "
                                "This could be due to the complexity of the relationships or the limited size of the dataset, which makes it challenging for the models to learn the intricate patterns.",
                                className="analysis-text"
                            ),
                            html.P(
                                "Overall, the learning curve analysis indicates that to improve model performance, larger and more comprehensive datasets are needed. "
                                "This would provide the models with more information to learn from and potentially capture the nonlinear interactions between weather variables and crop outcomes.",
                                className="analysis-text"
                            )
                        ]
                    )
                ]
            ),

            html.Hr(className="divider"),

            # Section 3: Prediction Demo
            dbc.Container(
                className="section prediction-demo-section",
                children=[
                    html.Div(
                        className="section-header",
                        children=[
                            html.H2(
                                "Crop Yield and Production Prediction Demo",
                                className="section-title"
                            ),
                            html.P(
                                "In this final section, you can interactively predict crop yield or production using our trained machine learning models. "
                                "By selecting a county, crop, target variable, and model, you can input specific feature values to generate a prediction. "
                                "Alternatively, you can randomize the input features based on historical data for a more dynamic experience. "
                                "This demo provides insights into how different factors influence crop outcomes and allows you to see the importance of each feature in the prediction process.",
                                className="section-description"
                            )
                        ]
                    ),
                    html.Br(),

                    # Input and Output Row
                    dbc.Row(
                        className="prediction-demo-row",
                        children=[
                            # Input Section
                            dbc.Col(
                                width=6,
                                className="input-section",
                                children=[
                                    html.Div(
                                        className="input-container",
                                        children=[
                                            html.Label("Select County:", className="dropdown-label"),
                                            dcc.Dropdown(
                                                id='county-dropdown2',
                                                options=[
                                                    {'label': county, 'value': county}
                                                    for county in sorted(data["County"].unique())
                                                ],
                                                value=sorted(data["County"].unique())[0],
                                                className="dropdown"
                                            ),
                                            html.Br(),

                                            html.Label("Select Crop:", className="dropdown-label"),
                                            dcc.Dropdown(
                                                id='crop-dropdown2',
                                                options=[
                                                    {'label': crop, 'value': crop}
                                                    for crop in sorted(data["Crop Name"].unique())
                                                ],
                                                value=sorted(data["Crop Name"].unique())[0],
                                                className="dropdown"
                                            ),
                                            html.Br(),

                                            html.Label("Select Target Variable:", className="dropdown-label"),
                                            dcc.RadioItems(
                                                id='target-variable',
                                                options=[
                                                    {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                                    {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                                                ],
                                                value='Yield Per Acre',
                                                className="radio-items"
                                            ),
                                            html.Br(),

                                            html.Label("Select Model:", className="dropdown-label"),
                                            dcc.RadioItems(
                                                id='model-name',
                                                options=[
                                                    {'label': model, 'value': model}
                                                    for model in ['KNN', 'DecisionTree', 'GradientBoosting']
                                                ],
                                                value='GradientBoosting',
                                                className="radio-items"
                                            ),
                                            html.Br(),

                                            html.Div(
                                                className="feature-section",
                                                children=[
                                                    html.H4("Input Features:", className="features-title"),
                                                    html.P(
                                                        "Enter values for the features used by the model to make predictions. "
                                                        "You can manually input values for each feature or click the 'Randomize Input' button to fill them with random historical data from the selected county and crop.",
                                                        className="feature-description"
                                                    ),
                                                    html.Div(id='feature-inputs', className="feature-inputs")
                                                ]
                                            ),
                                            html.Br(),

                                            html.Div(
                                                className="button-container",
                                                children=[
                                                    dbc.Button(
                                                        "Randomize Input",
                                                        id='randomize-button',
                                                        color='secondary',
                                                        className='me-2 randomize-button'
                                                    ),
                                                    dbc.Button(
                                                        "Predict",
                                                        id='predict-button',
                                                        color='primary',
                                                        className='predict-button'
                                                    )
                                                ]
                                            ),
                                            html.Br(),
                                            html.Div(id='prediction-text-output', className="prediction-output"),
                                            html.Div(
                                                id='randomized-info-output',
                                                className="randomized-info",
                                                style={"margin-top": "20px"}
                                            )
                                        ]
                                    )
                                ]
                            ),

                            # Feature Importance Graph Section
                            dbc.Col(
                                width=6,
                                className="graph-section",
                                children=[
                                    html.Div(
                                        className="graph-container",
                                        children=[
                                            html.H4("Feature Importance", className="graph-title"),
                                            html.P(
                                                "After making a prediction, a bar chart will display the importance of each feature used in the model. "
                                                "This visualization helps you understand which factors have the most significant impact on the predicted crop yield or production.",
                                                className="graph-description"
                                            ),
                                            dcc.Graph(id='feature-importance-graph', className="feature-importance-graph")
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="analysis-section",
                        children=[
                            html.H3("Prediction Demo", className="analysis-title"),
                            html.P(
                                "The Prediction Demo allows users to interact with the trained machine learning models by inputting custom values for the features or randomizing them based on historical data. "
                                "This interactive approach provides a practical understanding of how different variables affect crop outcomes in specific counties and for specific crops.",
                                className="analysis-text"
                            ),
                            html.P(
                                "The feature importance graph helps identify which factors the model considers most significant in making predictions. "
                                "Features with higher importance values have a greater impact on the outcome, indicating areas where farmers and agricultural planners might focus their attention to improve yields or production.",
                                className="analysis-text"
                            ),
                            html.P(
                                "However, it's important to note that the accuracy of predictions may be limited by the quality and quantity of data available. "
                                "As discussed in previous sections, small datasets and the complexity of agricultural systems can affect model performance. "
                                "Therefore, predictions should be interpreted cautiously and used as a supplementary tool alongside expert knowledge and other resources.",
                                className="analysis-text"
                            )
                        ]
                    )
                ]
            )
        ]
    )

# Callback for updating both plots
@callback(
    [Output('mae-bar-plot', 'figure'),
     Output('rmse-bar-plot', 'figure')],
    [Input('target-dropdown', 'value'),
     Input('performance-crop-dropdown', 'value')]
)
def update_plots(selected_target, selected_crop):
    filtered_df = results_df[
        (results_df['Target'] == selected_target) &
        (results_df['Crop'] == selected_crop)
    ]

    if filtered_df.empty:
        no_data_fig = go.Figure().update_layout(
            title=f"No data available for {selected_target} and {selected_crop}",
            xaxis_title="Model",
            yaxis_title="Metric",
        )
        return no_data_fig, no_data_fig

    # MAE Plot
    mae_fig = go.Figure()
    for model in filtered_df['Model'].unique():
        model_data = filtered_df[filtered_df['Model'] == model]
        mae_fig.add_trace(go.Bar(
            x=model_data['County'],
            y=model_data['MAE_Test'],
            name=model,
        ))
    mae_fig.update_layout(
        title=f'MAE for {selected_target} and {selected_crop} Across Models',
        xaxis_title='County',
        yaxis_title='MAE',
        barmode='group',  # Display bars side by side
        xaxis=dict(tickangle=45),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white'
    )

    # RMSE Plot
    rmse_fig = go.Figure()
    for model in filtered_df['Model'].unique():
        model_data = filtered_df[filtered_df['Model'] == model]
        rmse_fig.add_trace(go.Bar(
            x=model_data['County'],
            y=model_data['RMSE_Test'],
            name=model,
        ))
    rmse_fig.update_layout(
        title=f'RMSE for {selected_target} and {selected_crop} Across Models',
        xaxis_title='County',
        yaxis_title='RMSE',
        barmode='group',  # Display bars side by side
        xaxis=dict(tickangle=45),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white'
    )

    return mae_fig, rmse_fig

@callback(
    [Output('yield-per-acre-plot', 'figure'),
     Output('production-per-acre-plot', 'figure')],
    [Input('training-county-dropdown', 'value'),
     Input('training-crop-dropdown', 'value')]
)
def update_yield_and_production_plots(selected_county, selected_crop):
    # Filter models by county and crop
    matching_models = [
        (key, value) for key, value in all_models.items()
        if selected_county in key and selected_crop in key
    ]

    # If no models match, return empty figures
    if not matching_models:
        no_data_fig = go.Figure().update_layout(
            title=f"No data available for {selected_county} and {selected_crop}",
            xaxis_title="Training Set Size",
            yaxis_title="Score",
        )
        return no_data_fig, no_data_fig

    # Yield Per Acre Plot
    yield_fig = go.Figure()

    for model_key, model_data in matching_models:
        if 'Yield Per Acre' in model_key:
            learning_curve = model_data['learning_curve']
            train_sizes = learning_curve['train_sizes']
            test_scores = learning_curve['test_scores']

            # Add test scores for Yield Per Acre
            yield_fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_scores,
                mode='lines+markers',
                name=f'{model_key} - Test'
            ))

    yield_fig.update_layout(
        title=f'Yield Per Acre for {selected_county} and {selected_crop}',
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Production Per Acre Plot
    production_fig = go.Figure()

    for model_key, model_data in matching_models:
        if 'Production Per Acre' in model_key:
            learning_curve = model_data['learning_curve']
            train_sizes = learning_curve['train_sizes']
            test_scores = learning_curve['test_scores']

            # Add test scores for Production Per Acre
            production_fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_scores,
                mode='lines+markers',
                name=f'{model_key} - Test'
            ))

    production_fig.update_layout(
        title=f'Production Per Acre for {selected_county} and {selected_crop}',
        xaxis_title='Training Set Size',
        yaxis_title='Score',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return yield_fig, production_fig

@callback(
    [Output({'type': 'feature-input', 'index': ALL}, 'value'),
     Output('randomized-info-output', 'children')],
    [Input('randomize-button', 'n_clicks')],
    [State('county-dropdown2', 'value'),
     State('crop-dropdown2', 'value'),
     State('target-variable', 'value'),
     State('model-name', 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'id')]
)
def randomize_input(n_clicks, county, crop, target, model_name, input_ids):
    data = pd.read_csv('./data/data.csv')

    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Filter dataset for the specified county, crop, and year range
    filtered_data = data[
        (data["County"] == county) &
        (data["Crop Name"] == crop) &
        (data["Year"] >= 1983) & (data["Year"] <= 2020)
    ]

    if filtered_data.empty:
        return [None] * len(input_ids), "No data available for the selected county, crop, and year range."

    # Randomly sample a row
    sampled_row = filtered_data.sample(n=1).iloc[0]

    # Generate random values for the feature inputs based on the sampled row
    random_values = [
        sampled_row.get(id_dict['index'], 0.0) for id_dict in input_ids
    ]

    # Retrieve the actual target variable value
    actual_target_value = sampled_row["Yield Per Acre"] if target == "Yield Per Acre" else sampled_row["Production Per Acre"]

    # Prepare display information
    randomized_info = html.Div([
        html.P(f"Randomized from Year: {int(sampled_row['Year'])}"),
        html.P(f"County: {sampled_row['County']}"),
        html.P(f"Crop: {sampled_row['Crop Name']}"),
        html.P(f"Actual {target}: {actual_target_value}")
    ])

    return random_values, randomized_info

@callback(
    Output('feature-inputs', 'children'),
    [Input('county-dropdown2', 'value'),
     Input('crop-dropdown2', 'value'),
     Input('target-variable', 'value'),
     Input('model-name', 'value')]
)
def update_feature_inputs(county, crop, target, model_name):
    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key in all_models:
        # Use all features directly
        all_features = all_models[model_key].get('selected_features')
        if all_features is None:
            return html.Div("Error: Feature metadata missing in the model.")

        input_fields = [
            html.Div([
                html.Label(f"{feature.replace('_', ' ').title()}"),
                dcc.Input(
                    id={'type': 'feature-input', 'index': feature},
                    type='number',
                    placeholder=f"Enter {feature}",
                    style={'margin-bottom': '10px', 'width': '100%'}
                )
            ]) for feature in all_features
        ]
        return input_fields
    else:
        return html.Div("Model not available for the selected combination.")

@callback(
    [Output('prediction-text-output', 'children'),
     Output('feature-importance-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [State('county-dropdown2', 'value'),
     State('crop-dropdown2', 'value'),
     State('target-variable', 'value'),
     State('model-name', 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'id')]
)
def make_prediction_and_plot_importance(n_clicks, county, crop, target, model_name, input_values, input_ids):
    if n_clicks is None:
        return "Please click the Predict button.", go.Figure()

    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key not in all_models:
        return (f"Model not available for the combination: {model_name}, {county}, {crop}, {target}.", go.Figure())

    try:
        model_entry = all_models[model_key]
        model = model_entry['model']
        scaler = model_entry['scaler']
        all_features = model_entry.get('selected_features')

        if all_features is None:
            return ("Error: Missing feature metadata in the model. Please ensure 'selected_features' is saved during training.", go.Figure())

        # Check if all input values are filled
        if None in input_values or "" in input_values:
            return "Please fill in all input features before making a prediction.", go.Figure()

        # Initialize feature values for all features
        feature_values = {feature: 0.0 for feature in all_features}

        # Update feature values with user inputs
        for value, id_dict in zip(input_values, input_ids):
            feature = id_dict['index']
            if feature in feature_values:
                feature_values[feature] = float(value) if value is not None else 0.0

        # Prepare input DataFrame
        X_input = pd.DataFrame([feature_values])

        # Scale the input
        X_scaled = scaler.transform(X_input)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        # Extract feature importance
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": all_features,
                "Importance": feature_importances
            }).sort_values(by="Importance", ascending=False)

            # Create feature importance plot
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation='h',
                title="Feature Importance",
                labels={"Importance": "Importance", "Feature": "Feature"},
                height=500
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),  # Reverse y-axis for better readability
                margin=dict(l=50, r=50, t=50, b=50)
            )
        else:
            fig = go.Figure()
            fig.add_annotation(
                text="Feature importance is not available for this model.",
                showarrow=False,
                font=dict(size=16)
            )

        return (html.Div([html.H4(f"Predicted {target}: {prediction:.2f}")]), fig)

    except Exception as e:
        return (f"An error occurred during prediction: {e}", go.Figure())
