import plotly.express as px
import pandas as pd
from dash import html, dcc
import plotly.graph_objects as go

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

def filter_outliers(data, column, lower_percentile=5, upper_percentile=95):
    """Filter outliers based on percentile range."""
    lower_bound = data[column].quantile(lower_percentile / 100)
    upper_bound = data[column].quantile(upper_percentile / 100)
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def generate_density_plot(metric):
    fig = go.Figure()
    
    # Create a density plot for each model on the same figure
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        
        # Filter outliers for the metric
        filtered_data = filter_outliers(model_data, metric)
        
        fig.add_trace(go.Violin(
            x=filtered_data[metric],
            name=model,
            fillcolor="rgba(173, 216, 230, 0.5)",  # Light blue fill for each model
            opacity=0.6,
            box_visible=True,
            meanline_visible=True
        ))

    # Update layout for titles, labels, and size
    fig.update_layout(
        title=f"Distribution of {metric} Scores Across Models",
        xaxis_title=metric,
        yaxis_title="Density",
        violinmode="overlay",  # Overlay to simulate KDE-like effect
        height=1000,  # Increase height
        width=800   # Increase width
    )

    return fig

def layout():
    # Generate plots for each metric
    metrics = ['R2', 'RMSE', 'MAE']
    plot_pairs = [generate_metric_plots(metric) for metric in metrics]
    density_plots = [generate_density_plot(metric) for metric in metrics]
    
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
        ],
        
        # Section for density plots
        html.H2("Density Plots for Model Performance Metrics"),
        *[
            html.Div([
                dcc.Graph(figure=fig, config={'displayModeBar': False}),
            ], style={"margin-bottom": "30px"}) for fig in density_plots
        ]
    ])