import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from tqdm import tqdm
from scipy.stats import kendalltau
import re
import json


with open("AU_Interns_2024/Trend Analysis/tag_groups.json", "r") as file:
    tag_groups = json.load(file)

trend_data = pd.read_pickle("AU_Interns_2024/Trend Analysis/Tik_Tok_Grouped.pkl")
# print(trend_data.head())

print("total hashtag record {}".format(len(trend_data)))


print("Stats of the data")
print("post time span from {} to {}".format(trend_data["date"].min(), trend_data["date"].max()))


total_counts = trend_data.groupby('cluster')['count'].sum()

threshold_count = 0

top_hashtags = total_counts[total_counts > threshold_count].index

filtered_hashtags = trend_data[trend_data['cluster'].isin(top_hashtags)]

pivot_table = filtered_hashtags.pivot(index=['date'], columns='cluster', values='count').fillna(0)
# pivot_table.index = pd.to_datetime(pivot_table.index.map(lambda x: f"2022-{int(x[0]):02d}-{int(x[1]):02d}"))
pivot_table.index = pd.to_datetime(pivot_table.index)


# Get the last 5 weeks (blocks of 7 days each)
last_weeks = 5
min_mentions_per_block = 0
final = 1

last_blocks_end_date = pivot_table.index[-1] - pd.DateOffset(weeks= final)
last_blocks_start_date = last_blocks_end_date - pd.DateOffset(weeks=last_weeks)
pivot_table_filtered = pivot_table.loc[last_blocks_start_date:last_blocks_end_date]

resample_frequency = 4
resampled_pivot_table = pivot_table_filtered.resample(f'{resample_frequency}D').sum()

filtered_hashtags = resampled_pivot_table.columns[resampled_pivot_table.gt(min_mentions_per_block).all()]
resampled_pivot_table = resampled_pivot_table[filtered_hashtags]



percent_changes = resampled_pivot_table.pct_change()
percent_changes.fillna(0, inplace=True)



top_percent_number = 5


taus = {}

for column in resampled_pivot_table.columns:

    # Compute Kendall Tau correlation coefficient
    tau, p_value = kendalltau(resampled_pivot_table[column], resampled_pivot_table.index)


    if (p_value < 0.2) and (tau>0):
        taus[column] = tau


top_hashtags = sorted(taus, key=lambda k: taus[k], reverse=True)[:top_percent_number]


filtered_percent_changes = percent_changes[top_hashtags]
filtered_counts = resampled_pivot_table[top_hashtags]


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Top Hashtags Analysis"),
    
    # Options for user input
    html.Div([
        html.Label("Top Hashtag Group Number:"),
        dcc.Slider(
            id='top-percent-slider',
            min=1,
            max=10,
            step=1,
            value=top_percent_number,
            marks={i: str(i) for i in range(1, 11)},
        ),
    ]),
    html.Div([
        html.Label("Last Weeks:"),
        dcc.Slider(
            id='last-weeks-slider',
            min=1,
            max=8,
            step=1,
            value=last_weeks,
            marks={i: str(i) for i in range(1, 11)},
        ),
    ]),
    html.Div([
        html.Label("Resample Frequency (Days):"),
        dcc.Slider(
            id='resample-frequency-slider',
            min=1,
            max=14,
            step=1,
            value=resample_frequency,
            marks={i: str(i) for i in range(1, 31)},
        ),
    ]),
    html.Div([
        html.Label("Final Week (weeks back from 25 Jan):"),
        dcc.Slider(
            id='final',
            min=0,
            max=8,
            step=1,
            value=final,
            marks={i: str(i) for i in range(0, 12)},
        ),
    ]),
    
    # Counts Plot
    dcc.Graph(id='counts-plot'),

    # Percent Changes Plot
    dcc.Graph(id='percent-changes-plot'),
    
])

# Callbacks to update plots based on user input
@app.callback(
    [Output('percent-changes-plot', 'figure'),
     Output('counts-plot', 'figure')],
    [Input('top-percent-slider', 'value'),
     Input('last-weeks-slider', 'value'),
     Input('resample-frequency-slider', 'value'),
     Input('final', 'value')]
)
def update_plots(top_percent_number, last_weeks, resample_frequency,final):
    # Your existing code for data processing and filtering here
    
    # Filtering based on last weeks
    last_blocks_end_date = pivot_table.index[-1] - pd.DateOffset(weeks=final)

    last_blocks_start_date = last_blocks_end_date - pd.DateOffset(weeks=last_weeks)


    pivot_table_filtered = pivot_table.loc[last_blocks_start_date:last_blocks_end_date]

    # if final == 0:
    #     pivot_table_filtered = pivot_table.loc[last_blocks_start_date:]
    
    # Resampling
    resampled_pivot_table = pivot_table_filtered.resample(f'{resample_frequency}D').sum()
    
    filtered_hashtags = resampled_pivot_table.columns[resampled_pivot_table.gt(min_mentions_per_block).all()]
    resampled_pivot_table = resampled_pivot_table[filtered_hashtags]


    percent_changes = resampled_pivot_table.pct_change()
    percent_changes.fillna(0, inplace=True)
    
    # Remaining code for analysis and plotting

        
    taus = {}

    for column in resampled_pivot_table.columns:

        # Compute Kendall Tau correlation coefficient
        tau, p_value = kendalltau(resampled_pivot_table[column], resampled_pivot_table.index)

        # Print the results for each column
        # print(f"Column: {column}")
        # print(f"Kendall Tau: {tau}")
        # print(f"P-value: {p_value}")
        # print("\n")
        if (p_value < 0.1) and (tau>0):
            taus[column] = tau


    top_hashtags = sorted(taus, key=lambda k: taus[k], reverse=True)[:top_percent_number]
    
    filtered_percent_changes = percent_changes[top_hashtags]
    filtered_counts = resampled_pivot_table[top_hashtags]
    

    legend_labels = [', '.join(tag_groups[cluster]) for cluster in filtered_percent_changes.columns]

    percent_changes_fig = {
        'data': [
            {
                'x': filtered_percent_changes.index,
                'y': filtered_percent_changes[col],
                'type': 'line',
                'name': legend_label,  # Specify custom legend label for each trace
                'legendgroup': col  # Use the column name as the legend group identifier
            }
            for col, legend_label in zip(filtered_percent_changes.columns, legend_labels)
        ],
        'layout': {
            'title': f'Percent Changes for Top Hashtags',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Percent Change'},
            'showlegend': True
        }
    }


    counts_fig = {
        'data': [
            {
                'x': filtered_counts.index,
                'y': filtered_counts[col],
                'type': 'line',
                'name': legend_label,  # Specify custom legend label for each trace
                'legendgroup': col  # Use the column name as the legend group identifier
            }
            for col, legend_label in zip(filtered_counts.columns, legend_labels)
        ],
        'layout': {
            'title': f'Top Hashtags Counts (Resampled Every {resample_frequency} Days)',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Hashtag Count'},
            'showlegend': True
        }
    }
    
    
    return percent_changes_fig, counts_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
