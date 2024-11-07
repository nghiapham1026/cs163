import plotly.express as px
import pandas as pd
from dash import html, dcc

# Sample Data - Replace this with actual results_df DataFrame
results_df = pd.read_csv('./data/results_df.csv')

def generate_metric_plots(metric):
    # Bar plot for average metric by model
    bar_fig = px.bar(
        results_df,
        x='Model',
        y=metric,
        title=f'Average {metric} by Model',
        labels={'Model': 'Model', metric: metric},
        color='Model',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    # Box plot for metric distribution by model
    box_fig = px.box(
        results_df,
        x='Model',
        y=metric,
        title=f'{metric} Distribution by Model',
        labels={'Model': 'Model', metric: metric},
        color='Model',
        color_discrete_sequence=px.colors.sequential.Viridis
    )

    return bar_fig, box_fig

def layout():
    # Generate plots for each metric
    metrics = ['R2', 'RMSE', 'MAE']
    plot_pairs = [generate_metric_plots(metric) for metric in metrics]
    
    # Create layout with plots
    return html.Div([
        html.H1("Prediction Model Performance Analysis"),

        # Section for each metric
        *[
            html.Div([
                html.H3(f"{metric} Analysis"),
                
                # Row with bar plot and box plot for each metric
                html.Div([
                    dcc.Graph(figure=bar_fig, config={'displayModeBar': False}),
                    dcc.Graph(figure=box_fig, config={'displayModeBar': False})
                ], style={"display": "flex", "gap": "20px", "justify-content": "center"})
            ]) for metric, (bar_fig, box_fig) in zip(metrics, plot_pairs)
        ]
    ])
