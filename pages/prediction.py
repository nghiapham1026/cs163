import dash
from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import joblib

# Load data
results_df = pd.read_csv('./data/results.csv')
all_models = joblib.load("./data/all_models.joblib")

def layout():
    return html.Div([
        # Performance Metrics Dropdowns
        html.Div([
            html.Label("Select Target Variable:"),
            dcc.Dropdown(
                id='target-dropdown',
                options=[
                    {'label': 'Yield Per Acre', 'value': 'Yield Per Acre'},
                    {'label': 'Production Per Acre', 'value': 'Production Per Acre'}
                ],
                value='Yield Per Acre'
            ),
        ]),
        html.Div([
            html.Label("Select Crop (Performance Metrics):"),
            dcc.Dropdown(
                id='performance-crop-dropdown',
                options=[{'label': crop, 'value': crop} for crop in results_df['Crop'].unique()],
                value=results_df['Crop'].unique()[0]
            ),
        ]),
        dcc.Graph(id='mae-bar-plot'),
        dcc.Graph(id='rmse-bar-plot'),

        html.Hr(),

        # Training Data Dropdowns
        html.Div([
            html.Label("Select County (Training Data):"),
            dcc.Dropdown(
                id='training-county-dropdown',
                options=[{'label': county, 'value': county} for county in results_df['County'].unique()],
                value=results_df['County'].unique()[0]
            ),
        ]),
        html.Div([
            html.Label("Select Crop (Training Data):"),
            dcc.Dropdown(
                id='training-crop-dropdown',
                options=[{'label': crop, 'value': crop} for crop in results_df['Crop'].unique()],
                value=results_df['Crop'].unique()[0]
            ),
        ]),
        dcc.Graph(id='yield-per-acre-plot'),
        dcc.Graph(id='production-per-acre-plot'),
    ])

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
