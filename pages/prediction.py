import dash
from dash import html, dcc, Input, Output, callback, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import joblib
import random

# Load data
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
                "Visualization Dashboard: Weather Impact on Crop Yields",
                className="page-title"
            ),
            
            html.Hr(className="divider"),

            # Section 1: Performance Metrics
            html.Div(
                className="section performance-metrics-section",
                children=[
                    html.H2("Performance Metrics", className="section-title"),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select Target Variable:", className="dropdown-label"),
                            dcc.Dropdown(
                                id='target-dropdown',  # Updated ID
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
                            html.Label("Select Crop (Performance Metrics):", className="dropdown-label"),
                            dcc.Dropdown(
                                id='performance-crop-dropdown',  # Updated ID
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

            # Section 2: Training Data Visualization
            html.Div(
                className="section training-data-section",
                children=[
                    html.H2("Training Data Visualization", className="section-title"),
                    html.Div(
                        className="dropdown-container",
                        children=[
                            html.Label("Select County (Training Data):", className="dropdown-label"),
                            dcc.Dropdown(
                                id='training-county-dropdown',  # Updated ID
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
                            html.Label("Select Crop (Training Data):", className="dropdown-label"),
                            dcc.Dropdown(
                                id='training-crop-dropdown',  # Updated ID
                                options=[
                                    {'label': crop, 'value': crop} for crop in results_df['Crop'].unique()
                                ],
                                value=results_df['Crop'].unique()[0],
                                className="dropdown"
                            )
                        ]
                    ),
                    dcc.Graph(id='yield-per-acre-plot', className="training-graph"),
                    dcc.Graph(id='production-per-acre-plot', className="training-graph")
                ]
            ),

            html.Hr(className="divider"),

            # Section 3: Prediction Demo
            dbc.Container(
                className="section prediction-demo-section",
                children=[
                    html.H2("Crop Yield and Production Prediction Demo", className="section-title"),
                    html.Br(),
                    dbc.Row(
                        children=[
                            dbc.Col(
                                width=6,
                                children=[
                                    html.Div(
                                        className="dropdown-container",
                                        children=[
                                            html.Label("Select County:", className="dropdown-label"),
                                            dcc.Dropdown(
                                                id='county-dropdown2',  # Updated ID
                                                options=[
                                                    {'label': county, 'value': county} for county in sorted(data["County"].unique())
                                                ],
                                                value=sorted(data["County"].unique())[0],
                                                className="dropdown"
                                            ),
                                            html.Br(),
                                            html.Label("Select Crop:", className="dropdown-label"),
                                            dcc.Dropdown(
                                                id='crop-dropdown2',  # Updated ID
                                                options=[
                                                    {'label': crop, 'value': crop} for crop in sorted(data["Crop Name"].unique())
                                                ],
                                                value=sorted(data["Crop Name"].unique())[1],
                                                className="dropdown"
                                            ),
                                            html.Br(),
                                            html.Label("Select Target Variable:", className="dropdown-label"),
                                            dcc.RadioItems(
                                                id='target-variable',  # Updated ID
                                                options=[
                                                    {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                                                    {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                                                ],
                                                value='Yield Per Acre',
                                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                                            ),
                                            html.Br(),
                                            html.Label("Select Model:", className="dropdown-label"),
                                            dcc.RadioItems(
                                                id='model-name',  # Updated ID
                                                options=[
                                                    {'label': model, 'value': model} for model in ['KNN', 'DecisionTree', 'GradientBoosting']
                                                ],
                                                value='GradientBoosting',
                                                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                                            ),
                                            html.Br(),
                                            html.H4("Input Features:", className="features-title"),
                                            html.Div(id='feature-inputs', className="feature-inputs"),
                                            html.Br(),
                                            dbc.Button("Randomize Input", id='randomize-button', color='secondary', className='me-2'),
                                            dbc.Button("Predict", id='predict-button', color='primary'),
                                            html.Br(),
                                            html.Br(),
                                            html.Div(id='prediction-output', className="prediction-output")
                                        ]
                                    )
                                ]
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
    # Filter the dataframe for the selected target and crop
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
    [Output({'type': 'feature-input', 'index': ALL}, 'value')],
    [Input('randomize-button', 'n_clicks')],
    [State('county-dropdown2', 'value'),
     State('crop-dropdown2', 'value'),
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
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('county-dropdown2', 'value'),
     State('crop-dropdown2', 'value'),
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

        return html.Div([html.H4(f"Predicted {target}: {prediction:.2f}")])

    except Exception as e:
        return f"An error occurred during prediction: {e}"