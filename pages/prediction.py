import dash
from dash import Dash, html, dcc, Input, Output, State, ALL, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import joblib
import random

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Model Metrics Comparison and Predictions"

results_df = pd.read_csv('./data/results_df.csv')

all_models = joblib.load("./data/all_models.joblib")

df = pd.read_csv('./data/merged_yearly.csv')
counties_predictions = df['County'].unique()
crops_predictions = df['Crop Name'].unique()
models = ['KNN', 'DecisionTree', 'GradientBoosting']

# Metrics and variables to visualize
metrics_list = ['RMSE', 'MAE']
variables_list = ['Yield Per Acre', 'Production Per Acre']

def layout():
    # Get unique counties and crops for the dropdown menus
    counties_metrics = results_df['County'].unique()
    crops_metrics = results_df['Crop'].unique()

    return html.Div([
        dcc.Tabs([
            # Tab for Metrics Comparison
            dcc.Tab(label="Metrics Comparison", children=[
                html.Div([
                    html.H1("Model Metrics Comparison", id='model-metrics-title', className='page-title'),

                    # Dropdown menus for Model Metrics Comparison
                    html.Div([
                        html.Label("Select County:", className='dropdown-label', htmlFor='county-dropdown-metrics'),
                        dcc.Dropdown(
                            id='county-dropdown-metrics',
                            options=[{'label': county, 'value': county} for county in counties_metrics],
                            value=counties_metrics[0],
                            className='dropdown'
                        ),
                        html.Br(),
                        html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-metrics'),
                        dcc.Dropdown(
                            id='crop-dropdown-metrics',
                            options=[{'label': crop, 'value': crop} for crop in crops_metrics],
                            value=crops_metrics[0],
                            className='dropdown'
                        )
                    ], style={'width': '50%', 'margin': 'auto'}, id='metrics-dropdown-container', className='dropdown-container'),

                    # Plots for Yield Per Acre and Production Per Acre Metrics
                    html.Div([
                        html.H2("Yield Per Acre Metrics", id='yield-metrics-title', className='section-title'),
                        dcc.Graph(id='yield-per-acre-graph', className='graph'),
                        html.H2("Production Per Acre Metrics", id='production-metrics-title', className='section-title'),
                        dcc.Graph(id='production-per-acre-graph', className='graph')
                    ], id='metrics-graphs-container', className='graphs-container'),

                    # Heatmap Section
                    html.H1("Interactive Heatmap of Metrics by County and Crop", id='heatmap-title', className='page-title'),

                    # Dropdowns for Heatmap
                    html.Div([
                        html.Label("Select Variable:", className='dropdown-label', htmlFor='variable-dropdown-heatmap'),
                        dcc.Dropdown(
                            id='variable-dropdown-heatmap',
                            options=[{'label': var, 'value': var} for var in variables_list],
                            value=variables_list[0],
                            className='dropdown'
                        ),
                        html.Label("Select Metric:", className='dropdown-label', htmlFor='metric-dropdown-heatmap'),
                        dcc.Dropdown(
                            id='metric-dropdown-heatmap',
                            options=[{'label': met, 'value': met} for met in metrics_list],
                            value=metrics_list[0],
                            className='dropdown'
                        )
                    ], style={'width': '50%', 'margin': 'auto'}, id='heatmap-dropdown-container', className='dropdown-container'),

                    # Heatmap Graph
                    dcc.Graph(id='heatmap-graph', className='graph'),

                    # New Learning Curve Section
                    html.H1("Learning Curve Analysis", id='learning-curve-title', className='section-title'),
                    html.Div([
                        html.Label("Select County:", className='dropdown-label', htmlFor='county-dropdown-learning'),
                        dcc.Dropdown(
                            id='county-dropdown-learning',
                            options=[{'label': county, 'value': county} for county in counties_metrics],
                            value=counties_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-learning'),
                        dcc.Dropdown(
                            id='crop-dropdown-learning',
                            options=[{'label': crop, 'value': crop} for crop in crops_metrics],
                            value=crops_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Model:", className='dropdown-label', htmlFor='model-dropdown-learning'),
                        dcc.Dropdown(
                            id='model-dropdown-learning',
                            options=[{'label': model, 'value': model} for model in models],
                            value=models[0],
                            className='dropdown'
                        )
                    ], style={'width': '50%', 'margin': 'auto'}, id='learning-dropdown-container', className='dropdown-container'),
                    dcc.Graph(id='learning-curve-graph', className='graph'),

                    html.H1("Actual vs. Prediction Plot", id='actual-vs-prediction-title', className='section-title'),
                    html.Div([
                        html.Label("Select County:", className='dropdown-label', htmlFor='county-dropdown-actual-pred'),
                        dcc.Dropdown(
                            id='county-dropdown-actual-pred',
                            options=[{'label': county, 'value': county} for county in counties_metrics],
                            value=counties_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-actual-pred'),
                        dcc.Dropdown(
                            id='crop-dropdown-actual-pred',
                            options=[{'label': crop, 'value': crop} for crop in crops_metrics],
                            value=crops_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Model:", className='dropdown-label', htmlFor='model-dropdown-actual-pred'),
                        dcc.Dropdown(
                            id='model-dropdown-actual-pred',
                            options=[{'label': model, 'value': model} for model in models],
                            value=models[0],
                            className='dropdown'
                        ),
                        html.Label("Select Target Variable:", className='dropdown-label', htmlFor='target-variable-actual-pred'),
                        dcc.RadioItems(
                            id='target-variable-actual-pred',
                            options=[
                                {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                            ],
                            value='Yield Per Acre',
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                        )
                    ], style={'width': '50%', 'margin': 'auto'}, id='actual-pred-dropdown-container', className='dropdown-container'),

                    dcc.Graph(id='actual-vs-prediction-graph', className='graph'),

                    html.H1("Residual Plot", id='residual-plot-title', className='section-title'),
                    html.Div([
                        html.Label("Select County:", className='dropdown-label', htmlFor='county-dropdown-residual'),
                        dcc.Dropdown(
                            id='county-dropdown-residual',
                            options=[{'label': county, 'value': county} for county in counties_metrics],
                            value=counties_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Crop:", className='dropdown-label', htmlFor='crop-dropdown-residual'),
                        dcc.Dropdown(
                            id='crop-dropdown-residual',
                            options=[{'label': crop, 'value': crop} for crop in crops_metrics],
                            value=crops_metrics[0],
                            className='dropdown'
                        ),
                        html.Label("Select Model:", className='dropdown-label', htmlFor='model-dropdown-residual'),
                        dcc.Dropdown(
                            id='model-dropdown-residual',
                            options=[{'label': model, 'value': model} for model in models],
                            value=models[0],
                            className='dropdown'
                        ),
                        html.Label("Select Target Variable:", className='dropdown-label', htmlFor='target-variable-residual'),
                        dcc.RadioItems(
                            id='target-variable-residual',
                            options=[
                                {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                            ],
                            value='Yield Per Acre',
                            labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                        )
                    ], style={'width': '50%', 'margin': 'auto'}, id='residual-dropdown-container', className='dropdown-container'),

                    dcc.Graph(id='residual-plot-graph', className='graph'),
                ])
            ]),

            # Tab for Prediction Demo
            dcc.Tab(label="Prediction Demo", children=[
                dbc.Container([
                    html.H1("Crop Yield and Production Prediction Demo"),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Select County:"),
                            dcc.Dropdown(
                                id='county-dropdown',
                                options=[{'label': county, 'value': county} for county in sorted(df["County"].unique())],
                                value=sorted(df["County"].unique())[0]
                            ),
                            html.Br(),
                            html.Label("Select Crop:"),
                            dcc.Dropdown(
                                id='crop-dropdown',
                                options=[{'label': crop, 'value': crop} for crop in sorted(df["Crop Name"].unique())],
                                value=sorted(df["Crop Name"].unique())[0]
                            ),
                            html.Br(),
                            html.Label("Select Target Variable:"),
                            dcc.RadioItems(
                                id='target-variable',
                                options=[
                                    {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                    {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                                ],
                                value='Yield Per Acre',
                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                            ),
                            html.Br(),
                            html.Label("Select Model:"),
                            dcc.RadioItems(
                                id='model-name',
                                options=[{'label': model, 'value': model} for model in ['KNN', 'DecisionTree', 'GradientBoosting']],
                                value='GradientBoosting',
                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                            ),
                            html.Br(),
                            html.H4("Input Features:"),
                            html.Div(id='feature-inputs'),
                            html.Br(),
                            dbc.Button("Randomize Input", id='randomize-button', color='secondary', className='me-2'),
                            dbc.Button("Predict", id='predict-button', color='primary'),
                            html.Br(),
                            html.Br(),
                            html.Div(id='prediction-output')
                        ], width=6),
                        dbc.Col([
                            html.H4("Feature Importance"),
                            html.Div(id='feature-importance'),
                        ], width=6)
                    ])
                ])
            ])
        ])
    ])

app.layout = layout

# Callback for Model Metrics Comparison
@callback(
    [Output('yield-per-acre-graph', 'figure'),
     Output('production-per-acre-graph', 'figure')],
    [Input('county-dropdown-metrics', 'value'),
     Input('crop-dropdown-metrics', 'value')]
)
def update_metrics_graphs(selected_county, selected_crop):
    # Filter the data based on selected county and crop
    filtered_df = results_df[
        (results_df['County'] == selected_county) &
        (results_df['Crop'] == selected_crop)
    ]

    # All unique models
    all_models = results_df['Model'].unique()

    # Function to prepare data for a given target
    def prepare_data(target):
        target_df = filtered_df[filtered_df['Target'] == target]
        rmse_values = []
        mae_values = []
        for model in all_models:
            model_data = target_df[target_df['Model'] == model]
            if not model_data.empty:
                rmse_values.append(model_data['RMSE'].values[0])
                mae_values.append(model_data['MAE'].values[0])
            else:
                rmse_values.append(None)
                mae_values.append(None)
        return rmse_values, mae_values

    # Prepare data for Yield Per Acre
    yield_rmse, yield_mae = prepare_data('Yield Per Acre')

    # Prepare data for Production Per Acre
    production_rmse, production_mae = prepare_data('Production Per Acre')

    # Create the bar charts for Yield Per Acre
    yield_fig = go.Figure(data=[
        go.Bar(name='RMSE', x=all_models, y=yield_rmse, offsetgroup=0),
        go.Bar(name='MAE', x=all_models, y=yield_mae, offsetgroup=1)
    ])
    yield_fig.update_layout(
        barmode='group',
        title='Yield Per Acre Metrics',
        xaxis_title='Model',
        yaxis_title='Metric Value',
        legend_title='Metrics'
    )

    # Create the bar charts for Production Per Acre
    production_fig = go.Figure(data=[
        go.Bar(name='RMSE', x=all_models, y=production_rmse, offsetgroup=0),
        go.Bar(name='MAE', x=all_models, y=production_mae, offsetgroup=1)
    ])
    production_fig.update_layout(
        barmode='group',
        title='Production Per Acre Metrics',
        xaxis_title='Model',
        yaxis_title='Metric Value',
        legend_title='Metrics'
    )

    return yield_fig, production_fig

# Callback for Predictions
@callback(
    [Output('prediction-graph-production', 'figure'),
     Output('residual-graph-production', 'figure'),
     Output('prediction-graph-yield', 'figure'),
     Output('residual-graph-yield', 'figure'),
     Output('metrics-display', 'children')],
    [Input('county-dropdown-prediction', 'value'),
     Input('crop-dropdown-prediction', 'value'),
     Input('model-dropdown-prediction', 'value')]
)
def update_prediction_graphs(county, crop, model_name):
    # Generate keys for model retrieval
    model_key_production = f"{model_name}_{county}_{crop}_Production Per Acre"
    model_key_yield = f"{model_name}_{county}_{crop}_Yield Per Acre"

    # Check if both model keys are available
    missing_data = False
    error_messages = []
    if model_key_production not in all_models:
        missing_data = True
        error_messages.append("Model data not available for Production Per Acre.")
    if model_key_yield not in all_models:
        missing_data = True
        error_messages.append("Model data not available for Yield Per Acre.")
    if missing_data:
        return {}, {}, {}, {}, html.Div(error_messages, id='error-messages', className='error-messages')

    # Retrieve model data for Production Per Acre
    model_data_production = all_models[model_key_production]
    predictions_production = model_data_production["predictions"]
    actuals_production = model_data_production["actuals"]
    residuals_production = model_data_production["residuals"]
    metrics_production = model_data_production["metrics"]

    # Retrieve model data for Yield Per Acre
    model_data_yield = all_models[model_key_yield]
    predictions_yield = model_data_yield["predictions"]
    actuals_yield = model_data_yield["actuals"]
    residuals_yield = model_data_yield["residuals"]
    metrics_yield = model_data_yield["metrics"]

    # Prediction graph for Production Per Acre
    prediction_fig_production = go.Figure()
    prediction_fig_production.add_trace(go.Scatter(
        y=actuals_production, mode='lines', name='Actual Production Per Acre'))
    prediction_fig_production.add_trace(go.Scatter(
        y=predictions_production, mode='lines', name='Predicted Production Per Acre'))
    prediction_fig_production.update_layout(
        title="Actual vs Predictions - Production Per Acre",
        xaxis_title="Data Points",
        yaxis_title="Production Per Acre"
    )

    # Residual graph for Production Per Acre
    residual_fig_production = go.Figure()
    residual_fig_production.add_trace(go.Scatter(
        y=residuals_production, mode='markers', name='Residuals Production Per Acre'))
    residual_fig_production.add_hline(y=0, line_dash="dash", line_color="red")
    residual_fig_production.update_layout(
        title="Residuals - Production Per Acre",
        xaxis_title="Data Points",
        yaxis_title="Residuals"
    )

    # Prediction graph for Yield Per Acre
    prediction_fig_yield = go.Figure()
    prediction_fig_yield.add_trace(go.Scatter(
        y=actuals_yield, mode='lines', name='Actual Yield Per Acre'))
    prediction_fig_yield.add_trace(go.Scatter(
        y=predictions_yield, mode='lines', name='Predicted Yield Per Acre'))
    prediction_fig_yield.update_layout(
        title="Actual vs Predictions - Yield Per Acre",
        xaxis_title="Data Points",
        yaxis_title="Yield Per Acre"
    )

    # Residual graph for Yield Per Acre
    residual_fig_yield = go.Figure()
    residual_fig_yield.add_trace(go.Scatter(
        y=residuals_yield, mode='markers', name='Residuals Yield Per Acre'))
    residual_fig_yield.add_hline(y=0, line_dash="dash", line_color="red")
    residual_fig_yield.update_layout(
        title="Residuals - Yield Per Acre",
        xaxis_title="Data Points",
        yaxis_title="Residuals"
    )

    # Metrics display
    metrics_text = html.Div([
        html.H3("Metrics Comparison", id='metrics-comparison-title', className='section-title'),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Production Per Acre"), html.Th("Yield Per Acre")]),
            html.Tr([html.Td("R² Score"),
                     html.Td(f"{metrics_production['R2']:.2f}"),
                     html.Td(f"{metrics_yield['R2']:.2f}")]),
            html.Tr([html.Td("RMSE"),
                     html.Td(f"{metrics_production['RMSE']:.2f}"),
                     html.Td(f"{metrics_yield['RMSE']:.2f}")]),
            html.Tr([html.Td("MAE"),
                     html.Td(f"{metrics_production['MAE']:.2f}"),
                     html.Td(f"{metrics_yield['MAE']:.2f}")]),
        ], id='metrics-table', className='metrics-table')
    ], id='metrics-container', className='metrics-container')

    return (prediction_fig_production, residual_fig_production,
            prediction_fig_yield, residual_fig_yield, metrics_text)

# Callback to update the heatmap based on user selection
@callback(
    Output('heatmap-graph', 'figure'),
    [Input('variable-dropdown-heatmap', 'value'),
     Input('metric-dropdown-heatmap', 'value')]
)
def update_heatmap(selected_variable, selected_metric):
    # Filter the data for the selected variable
    filtered_df = results_df[results_df['Target'] == selected_variable]

    # Create a pivot table
    pivot_table = filtered_df.pivot_table(
        index='County',
        columns='Crop',
        values=selected_metric,
        aggfunc='mean'
    )

    # Reset index to use County as a column
    pivot_table = pivot_table.reset_index()

    # Melt the DataFrame to long format
    heatmap_data = pivot_table.melt(id_vars='County', var_name='Crop', value_name=selected_metric)

    # Create the heatmap
    fig = px.density_heatmap(
        heatmap_data,
        x='Crop',
        y='County',
        z=selected_metric,
        color_continuous_scale='RdBu_r',
        text_auto=True
    )

    fig.update_layout(
        title=f"{selected_metric} for {selected_variable} by County and Crop",
        xaxis_title="Crop",
        yaxis_title="County",
        xaxis={'categoryorder':'category ascending'},
        yaxis={'categoryorder':'category ascending'}
    )

    return fig

@callback(
    Output('learning-curve-graph', 'figure'),
    [Input('county-dropdown-learning', 'value'),
     Input('crop-dropdown-learning', 'value'),
     Input('model-dropdown-learning', 'value')]
)
def update_learning_curve(selected_county, selected_crop, selected_model):
    # Initialize the figure
    fig = go.Figure()

    # Loop through targets ("Yield Per Acre" and "Production Per Acre")
    targets = ["Yield Per Acre", "Production Per Acre"]
    for target in targets:
        # Generate the model key
        model_key = f"{selected_model}_{selected_county}_{selected_crop}_{target}"

        # Check if the model data is available
        if model_key not in all_models:
            fig.add_annotation(
                text=f"No data available for {target}",
                xref="paper", yref="paper",
                x=0.5, y=1,
                showarrow=False,
                font=dict(size=12, color="red")
            )
            continue

        # Extract learning curve data
        model_data = all_models[model_key]
        train_sizes = model_data["learning_curve"]["train_sizes"]
        train_scores = model_data["learning_curve"]["train_scores"]
        val_scores = model_data["learning_curve"]["val_scores"]

        # Add training and validation scores to the plot
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores,
            mode='lines+markers',
            name=f"Training Score ({target})"
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores,
            mode='lines+markers',
            name=f"Validation Score ({target})"
        ))

    # Update layout
    fig.update_layout(
        title=f"Learning Curves for {selected_model} ({selected_county}, {selected_crop})",
        xaxis_title="Training Samples",
        yaxis_title="Score",
        legend_title="Curve",
        template="plotly_white"
    )

    return fig

@callback(
    [Output({'type': 'feature-input', 'index': ALL}, 'value')],
    [Input('randomize-button', 'n_clicks')],
    [State('county-dropdown', 'value'),
     State('crop-dropdown', 'value'),
     State('target-variable', 'value'),
     State('model-name', 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'id')]
)
def randomize_input(n_clicks, county, crop, target, model_name, input_ids):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key not in all_models:
        return [[]]  # No randomization if model is not found

    # Generate random values
    random_values = []
    for id_dict in input_ids:
        feature = id_dict['index']
        if "days" in feature:  # Extreme weather variables
            random_values.append(random.randint(10, 100))
        elif "crop" in feature or "lag" in feature:  # Crop variables
            random_values.append(round(random.uniform(0.1, 1), 2))
        else:
            random_values.append(0.0)  # Default for other features

    return [random_values]

@callback(
    Output('feature-inputs', 'children'),
    [Input('county-dropdown', 'value'),
     Input('crop-dropdown', 'value'),
     Input('target-variable', 'value'),
     Input('model-name', 'value')]
)
def update_feature_inputs(county, crop, target, model_name):
    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key in all_models:
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
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('county-dropdown', 'value'),
     State('crop-dropdown', 'value'),
     State('target-variable', 'value'),
     State('model-name', 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'value'),
     State({'type': 'feature-input', 'index': ALL}, 'id')]
)
def make_prediction(n_clicks, county, crop, target, model_name, input_values, input_ids):
    if n_clicks is None:
        return ""

    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key not in all_models:
        return f"Model not available for the combination: {model_name}, {county}, {crop}, {target}."

    try:
        model_entry = all_models[model_key]
        model = model_entry['model']
        scaler = model_entry['scaler']
        all_features = model_entry.get('selected_features')

        if all_features is None:
            return "Error: Missing feature metadata in the model. Please ensure 'selected_features' is saved during training."

        feature_values = {feature: 0.0 for feature in all_features}

        for value, id_dict in zip(input_values, input_ids):
            feature = id_dict['index']
            if feature in feature_values:
                feature_values[feature] = float(value) if value is not None else 0.0

        X_input = pd.DataFrame([feature_values])
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]

        return html.Div([html.H4(f"Predicted {target}: {prediction:.2f}")])

    except Exception as e:
        return f"An error occurred during prediction: {e}"

@callback(
    Output('feature-importance', 'children'),
    [Input('county-dropdown', 'value'),
     Input('crop-dropdown', 'value'),
     Input('target-variable', 'value'),
     Input('model-name', 'value')]
)
def update_feature_importance(county, crop, target, model_name):
    model_key = f"{model_name}_{county}_{crop}_{target}"
    if model_key not in all_models:
        return html.Div("Feature importance not available for this model.")

    model_entry = all_models[model_key]
    model = model_entry['model']
    all_features = model_entry['selected_features']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        return html.Div("Feature importance calculation not supported for this model.")

    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance")
    return dcc.Graph(figure=fig)

@callback(
    Output('actual-vs-prediction-graph', 'figure'),
    [Input('county-dropdown-actual-pred', 'value'),
     Input('crop-dropdown-actual-pred', 'value'),
     Input('model-dropdown-actual-pred', 'value'),
     Input('target-variable-actual-pred', 'value')]
)
def update_actual_vs_prediction_plot(county, crop, model_name, target_variable):
    # Generate the model key to access the stored data
    model_key = f"{model_name}_{county}_{crop}_{target_variable}"
    
    # Check if the model key exists in the all_models dictionary
    if model_key not in all_models:
        return go.Figure(layout=go.Layout(title="No data available for the selected combination."))

    # Retrieve predictions, actuals, and other data
    model_entry = all_models[model_key]
    predictions = model_entry.get('predictions', [])
    actuals = model_entry.get('actuals', [])

    # Ensure predictions and actuals are available
    if not predictions or not actuals:
        return go.Figure(layout=go.Layout(title="No prediction data available for the selected combination."))

    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals,
        y=predictions,
        mode='markers',
        name='Predicted vs Actual'
    ))

    # Add a diagonal reference line (y = x)
    max_val = max(max(actuals), max(predictions))
    min_val = min(min(actuals), min(predictions))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction (y=x)',
        line=dict(dash='dash', color='red')
    ))

    # Update layout
    fig.update_layout(
        title=f"Actual vs. Predicted {target_variable} for {crop} in {county}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        legend_title="Legend",
        height=600,
        width=800
    )

    return fig

@callback(
    Output('residual-plot-graph', 'figure'),
    [Input('county-dropdown-residual', 'value'),
     Input('crop-dropdown-residual', 'value'),
     Input('model-dropdown-residual', 'value'),
     Input('target-variable-residual', 'value')]
)
def update_residual_plot(county, crop, model_name, target_variable):
    # Generate the model key to access the stored data
    model_key = f"{model_name}_{county}_{crop}_{target_variable}"
    
    # Check if the model key exists in the all_models dictionary
    if model_key not in all_models:
        return go.Figure(layout=go.Layout(title="No data available for the selected combination."))

    # Retrieve residuals and actuals from the model entry
    model_entry = all_models[model_key]
    residuals = model_entry.get('residuals', [])
    actuals = model_entry.get('actuals', [])

    # Ensure residuals and actuals are available
    if not residuals or not actuals:
        return go.Figure(layout=go.Layout(title="No residual data available for the selected combination."))

    # Create residual plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actuals,
        y=residuals,
        mode='markers',
        name='Residuals'
    ))

    # Add a horizontal reference line at y = 0
    fig.add_trace(go.Scatter(
        x=[min(actuals), max(actuals)],
        y=[0, 0],
        mode='lines',
        name='Zero Residual',
        line=dict(dash='dash', color='red')
    ))

    # Update layout
    fig.update_layout(
        title=f"Residual Plot for {target_variable} ({county}, {crop})",
        xaxis_title="Actual Values",
        yaxis_title="Residuals (Actual - Predicted)",
        legend_title="Legend",
        height=600,
        width=800
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
