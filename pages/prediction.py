import plotly.express as px
import pandas as pd
from dash import html, dcc, Input, Output, callback, State
import plotly.graph_objects as go
import joblib

# Sample Data - Replace this with actual results_df DataFrame
results_df = pd.read_csv('./data/results_df.csv')

# Load the joblib file
all_models = joblib.load('./data/all_models.joblib')

# Parse keys to extract model, county, crop, and target
parsed_keys = []
for key in all_models.keys():
    model_name, county, crop, target = key.split("_")
    parsed_keys.append({
        "model": model_name,
        "county": county,
        "crop": crop,
        "target": target,
        "key": key
    })

# Convert to DataFrame for easier filtering
key_df = pd.DataFrame(parsed_keys)

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

        # Dropdown for selecting counties
        html.Label("Select Counties:"),
        dcc.Dropdown(
            id="county-multi-dropdown",
            options=[{"label": county, "value": county} for county in results_df['County'].unique()],
            value=[],
            multi=True,  # Allow multiple selections
            placeholder="Select one or more counties"
        ),

        # Dropdown for selecting crops
        html.Label("Select Crops:"),
        dcc.Dropdown(
            id="crop-multi-dropdown",
            options=[{"label": crop, "value": crop} for crop in results_df['Crop'].unique()],
            value=[],
            multi=True,  # Allow multiple selections
            placeholder="Select one or more crops"
        ),

        # Placeholder for comparison plots
        html.Div(id="comparison-plots-container", style={"margin-top": "20px"}),

        html.H1("Model Performance Visualization"),

        # Dropdown for selecting model
        html.Label("Select a Machine Learning Model:"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": model, "value": model} for model in key_df['model'].unique()],
            value=None,
            clearable=True
        ),

        # Dropdown for selecting county
        html.Label("Select a County:"),
        dcc.Dropdown(
            id="county-dropdown",
            options=[],  # Dynamically populated
            value=None,
            clearable=True
        ),

        # Dropdown for selecting crop
        html.Label("Select a Crop:"),
        dcc.Dropdown(
            id="crop-dropdown",
            options=[],  # Dynamically populated
            value=None,
            clearable=True
        ),

        # Dropdown for selecting target
        html.Label("Select a Target:"),
        dcc.Dropdown(
            id="target-dropdown",
            options=[],  # Dynamically populated
            value=None,
            clearable=True
        ),

        # Button to trigger plot generation
        html.Button("Show Plot", id="show-plot-button", n_clicks=0, style={"margin-top": "20px"}),

        # Placeholder for plots
        html.Div(id="plots-container", style={"margin-top": "20px"})
    ])


@callback(
    Output("county-dropdown", "options"),
    Input("model-dropdown", "value")
)
def update_county_dropdown(selected_model):
    # Filter the keys for the selected model
    filtered = key_df[key_df["model"] == selected_model]
    counties = filtered["county"].unique()
    return [{"label": county, "value": county} for county in counties]

@callback(
    Output("crop-dropdown", "options"),
    Input("county-dropdown", "value"),
    State("model-dropdown", "value")
)
def update_crop_dropdown(selected_county, selected_model):
    # Filter the keys for the selected model and county
    filtered = key_df[(key_df["model"] == selected_model) & (key_df["county"] == selected_county)]
    crops = filtered["crop"].unique()
    return [{"label": crop, "value": crop} for crop in crops]

@callback(
    Output("target-dropdown", "options"),
    Input("crop-dropdown", "value"),
    State("model-dropdown", "value"),
    State("county-dropdown", "value")
)
def update_target_dropdown(selected_crop, selected_model, selected_county):
    # Filter the keys for the selected model, county, and crop
    filtered = key_df[
        (key_df["model"] == selected_model) &
        (key_df["county"] == selected_county) &
        (key_df["crop"] == selected_crop)
    ]
    targets = filtered["target"].unique()
    return [{"label": target, "value": target} for target in targets]


@callback(
    Output("plots-container", "children"),
    Input("show-plot-button", "n_clicks"),  # Button click event
    State("model-dropdown", "value"),
    State("county-dropdown", "value"),
    State("crop-dropdown", "value"),
    State("target-dropdown", "value")
)
def update_plots(n_clicks, selected_model, selected_county, selected_crop, selected_target):
    # Only proceed if the button has been clicked at least once
    if n_clicks == 0:
        return html.Div("Select options and click 'Show Plot' to generate the plots.")

    # Construct the unique key to retrieve the model data
    model_key = f"{selected_model}_{selected_county}_{selected_crop}_{selected_target}"
    model_entry = all_models.get(model_key)

    if not model_entry:
        return html.Div("No data available for the selected combination.")

    # Extract the model and metadata
    learning_curve = model_entry.get("learning_curve")
    residuals = model_entry.get("residuals")
    predictions = model_entry.get("predictions")
    actuals = model_entry.get("actuals")

    if not all([learning_curve, residuals, predictions, actuals]):
        return html.Div("Incomplete data for the selected model.")

    # Extract learning curve data
    train_sizes = learning_curve["train_sizes"]
    train_scores = learning_curve["train_scores"]
    test_scores = learning_curve["val_scores"]  # Use "val_scores" instead of "test_scores"

    # Generate plots
    # Learning Curve
    fig_learning = go.Figure()
    fig_learning.add_trace(go.Scatter(x=train_sizes, y=train_scores, mode='lines+markers', name='Train Score'))
    fig_learning.add_trace(go.Scatter(x=train_sizes, y=test_scores, mode='lines+markers', name='Validation Score'))
    fig_learning.update_layout(title="Learning Curve", xaxis_title="Training Examples", yaxis_title="Score")

    # Residual Plot
    fig_residuals = px.scatter(
        x=predictions,
        y=residuals,
        labels={"x": "Predicted Values", "y": "Residuals"},
        title="Residual Plot"
    )
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

    # Actual vs. Predicted
    fig_actual_vs_predicted = px.scatter(
        x=actuals,
        y=predictions,
        labels={"x": "Actual Values", "y": "Predicted Values"},
        title="Actual vs. Predicted"
    )
    fig_actual_vs_predicted.add_trace(go.Scatter(x=actuals, y=actuals, mode='lines', name='Perfect Prediction'))

    # Return the plots
    return [
        dcc.Graph(figure=fig_learning),
        dcc.Graph(figure=fig_residuals),
        dcc.Graph(figure=fig_actual_vs_predicted)
    ]

@callback(
    Output("comparison-plots-container", "children"),
    Input("county-multi-dropdown", "value"),
    Input("crop-multi-dropdown", "value")
)
def update_comparison_plots(selected_counties, selected_crops):
    if not selected_counties or not selected_crops:
        return html.Div("Please select at least one county and one crop to generate the plots.")

    # Filter the DataFrame based on the selections
    filtered_data = results_df[
        (results_df['County'].isin(selected_counties)) &
        (results_df['Crop'].isin(selected_crops))
    ]

    if filtered_data.empty:
        return html.Div("No data available for the selected filters.")

    # Generate Metric Plots
    metrics = ['R2', 'RMSE', 'MAE']

    metric_figs = []
    for metric in metrics:
        # Create a bar plot comparing metric averages for selected counties and crops
        metric_fig = px.bar(
            filtered_data,
            x="Model",
            y=metric,
            color="Crop",  # Use Crop as the grouping
            barmode="group",
            facet_row="County",  # Separate rows by counties
            title=f"Comparison of {metric} by Model, County, and Crop",
            labels={"Model": "Model", metric: metric}
        )
        metric_fig.update_layout(height=400 * len(selected_counties))  # Adjust height dynamically
        metric_figs.append(html.Div([
            html.H3(f"Metric Plot: {metric}"),
            dcc.Graph(figure=metric_fig)
        ]))

    density_figs = []
    for metric in metrics:
        fig = go.Figure()

        # Create density traces for each model in the selected counties and crops
        for crop in selected_crops:
            for county in selected_counties:
                county_crop_data = filtered_data[
                    (filtered_data['County'] == county) & (filtered_data['Crop'] == crop)
                ]

                if county_crop_data.empty:
                    continue

                for model in county_crop_data['Model'].unique():
                    model_data = county_crop_data[county_crop_data['Model'] == model]

                    # Add a trace for this model
                    fig.add_trace(go.Violin(
                        y=model_data[metric],
                        name=f"{model} ({county}, {crop})",
                        box_visible=True,
                        meanline_visible=True,
                        opacity=0.6
                    ))

        # Update layout for this metric's density plot
        fig.update_layout(
            title=f"Density Plot of {metric} by County, Crop, and Model",
            yaxis_title=metric,
            legend_title="Models",
            height=500,
        )

        density_figs.append(html.Div([
            html.H3(f"Density Plot: {metric}"),
            dcc.Graph(figure=fig)
        ]))

    # Combine and structure the layout for both metric and density plots
    return html.Div([
        html.Div(metric_figs, style={"margin-bottom": "40px"}),
        html.Div(density_figs)
    ])
