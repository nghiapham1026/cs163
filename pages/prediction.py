import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import joblib
import numpy as np

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
        html.H1("Model Metrics Comparison", style={'text-align': 'center'}),

        # Dropdown menus for Model Metrics Comparison
        html.Div([
            html.Label("Select County:"),
            dcc.Dropdown(
                id='county-dropdown-metrics',
                options=[{'label': county, 'value': county} for county in counties_metrics],
                value=counties_metrics[0]
            ),
            html.Br(),
            html.Label("Select Crop:"),
            dcc.Dropdown(
                id='crop-dropdown-metrics',
                options=[{'label': crop, 'value': crop} for crop in crops_metrics],
                value=crops_metrics[0]
            )
        ], style={'width': '50%', 'margin': 'auto'}),

        # Plots for Yield Per Acre and Production Per Acre Metrics
        html.Div([
            html.H2("Yield Per Acre Metrics", style={'text-align': 'center'}),
            dcc.Graph(id='yield-per-acre-graph'),
            html.H2("Production Per Acre Metrics", style={'text-align': 'center'}),
            dcc.Graph(id='production-per-acre-graph')
        ]),

        html.H1("Per Acre Prediction", style={'text-align': 'center'}),

        # Dropdowns for Predictions section
        html.Div([
            html.Label("Select County:"),
            dcc.Dropdown(
                id='county-dropdown-prediction',
                options=[{'label': county, 'value': county} for county in counties_predictions],
                value=counties_predictions[0]
            ),
            html.Label("Select Crop:"),
            dcc.Dropdown(
                id='crop-dropdown-prediction',
                options=[{'label': crop, 'value': crop} for crop in crops_predictions],
                value=crops_predictions[0]
            ),
            html.Label("Select Model:"),
            dcc.Dropdown(
                id='model-dropdown-prediction',
                options=[{'label': model, 'value': model} for model in models],
                value=models[0]
            )
        ], style={'width': '50%', 'margin': 'auto'}),

        # Visualization for Production Per Acre
        html.H2("Production Per Acre"),
        dcc.Graph(id='prediction-graph-production'),
        dcc.Graph(id='residual-graph-production'),

        # Visualization for Yield Per Acre
        html.H2("Yield Per Acre"),
        dcc.Graph(id='prediction-graph-yield'),
        dcc.Graph(id='residual-graph-yield'),

        # Metrics display
        html.Div(id='metrics-display', style={'text-align': 'center', 'margin-top': '20px'}),

        html.H1("Interactive Heatmap of Metrics by County and Crop", style={'text-align': 'center'}),

        # Dropdowns for Heatmap
        html.Div([
            html.Label("Select Variable:"),
            dcc.Dropdown(
                id='variable-dropdown-heatmap',
                options=[{'label': var, 'value': var} for var in variables_list],
                value=variables_list[0]
            ),
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown-heatmap',
                options=[{'label': met, 'value': met} for met in metrics_list],
                value=metrics_list[0]
            )
        ], style={'width': '50%', 'margin': 'auto'}),

        # Heatmap
        dcc.Graph(id='heatmap-graph')
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
        return {}, {}, {}, {}, html.Div(error_messages)

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
        html.H3("Metrics Comparison"),
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
        ], style={'margin': 'auto', 'border': '1px solid black'})
    ])

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

if __name__ == '__main__':
    app.run_server(debug=True)
