import dash
from dash import Dash, html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd

# Load data
results_df = pd.read_csv('./data/results.csv')

# Layout function
def layout():
    return html.Div([
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
            html.Label("Select Crop:"),
            dcc.Dropdown(
                id='crop-dropdown',
                options=[{'label': crop, 'value': crop} for crop in results_df['Crop'].unique()],
                value=results_df['Crop'].unique()[0]
            ),
        ]),
        dcc.Graph(id='mae-bar-plot'),
        dcc.Graph(id='rmse-bar-plot'),
    ])

# Callback for updating both plots
@callback(
    [Output('mae-bar-plot', 'figure'),
     Output('rmse-bar-plot', 'figure')],
    [Input('target-dropdown', 'value'), Input('crop-dropdown', 'value')]
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