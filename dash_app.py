#%%
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from sklearn.ensemble import IsolationForest

# Function to create a resampled dataset and apply Isolation Forest
def create_resampled_dataset(df, start_timestamp, end_timestamp):
    # Convert 'UPD_TIME' to datetime if not already in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['UPD_TIME']):
        df['UPD_TIME'] = pd.to_datetime(df['UPD_TIME'])
    
    # Filter data for the specified timestamp range
    df_filtered = df[(df['UPD_TIME'] >= start_timestamp) & (df['UPD_TIME'] <= end_timestamp)]
    
    # Check if there are data points in the filtered dataset
    if df_filtered.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data points are found
    

    # Set 'UPD_TIME' as the index
    df_filtered.set_index('UPD_TIME', inplace=True)
    
    # Resample the data to 3-hour intervals and interpolate missing values
    data = df_filtered.resample('10t').mean()
    data = data.interpolate(method='time')
    
    # Apply Isolation Forest
    model = IsolationForest(random_state=0, contamination=0.15)
    model.fit(data[['CF2_MNDE_V_VBT']])

    data['score'] = model.decision_function(data[['CF2_MNDE_V_VBT']])
    data['anomaly'] = model.predict(data[['CF2_MNDE_V_VBT']])

    return data

# Define the start and end timestamps
start_timestamp = pd.Timestamp('2024-06-13 10:03:00')
end_timestamp = pd.Timestamp('2024-06-25 05:05:00')

# Load the dataset from an Excel file directly
file_path = "Motor_Data"
df = pd.read_excel(file_path)

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=start_timestamp.date(),
        end_date=end_timestamp.date(),
        display_format='YYYY-MM-DD'
    ),
    html.Label("Start Time (HH:MM:SS):"),
    dcc.Input(id='start-time', type='text', value=str(start_timestamp.time())),
    html.Label("End Time (HH:MM:SS):"),
    dcc.Input(id='end-time', type='text', value=str(end_timestamp.time())),
    html.Div(id='warning-message', children='', style={'color': 'red', 'font-weight': 'bold', 'display': 'none'}),
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='hide-warning',
        interval=2*1000,  # in milliseconds
        n_intervals=0,
        max_intervals=5  # Trigger only 5 times
    )
])

@app.callback(
    [Output('live-graph', 'figure'),
     Output('warning-message', 'children'),
     Output('warning-message', 'style'),
     Output('hide-warning', 'n_intervals')],
    [Input('interval-component', 'n_intervals'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('start-time', 'value'),
     Input('end-time', 'value'),
     Input('hide-warning', 'n_intervals')],
    [State('hide-warning', 'max_intervals')]
)
def update_graph_live(n_intervals, start_date, end_date, start_time, end_time, hide_warning_intervals, max_intervals):
    start_timestamp = pd.Timestamp(f'{start_date} {start_time}')
    end_timestamp = pd.Timestamp(f'{end_date} {end_time}')
    
    # Incrementally increase the end timestamp based on n_intervals to simulate real-time data
    incremental_end_timestamp = start_timestamp + pd.Timedelta(minutes=10 * n_intervals)
    
    if incremental_end_timestamp > end_timestamp:
        incremental_end_timestamp = end_timestamp
    
    # Call the function for the specified timestamp range
    resampled_data = create_resampled_dataset(df, start_timestamp, incremental_end_timestamp)

    # Check if the resampled_data is empty
    if resampled_data.empty:
        # Create an empty figure with no data
        fig = go.Figure()
        fig.update_layout(
            title='No Data Available',
            xaxis_title='Time',
            yaxis_title='CF2_MNDE_V_VBT',
            legend_title='Legend',
        )
        return fig, '', {'display': 'none'}, 0

    # Identify outliers
    outliers = resampled_data[resampled_data['anomaly'] == -1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=resampled_data.index,
        y=resampled_data['CF2_MNDE_V_VBT'],
        mode='lines+markers',
        name='Normal'
    ))

    fig.add_trace(go.Scatter(
        x=outliers.index,
        y=outliers['CF2_MNDE_V_VBT'],
        mode='markers',
        marker=dict(color='red'),
        name='Anomaly'
    ))

    fig.update_layout(
        title='Isolation Forest Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='CF2_MNDE_V_VBT',
        legend_title='Legend',
    )

    # Check for threshold exceedance
    threshold_min, threshold_max = 0.23, 0.55
    exceeded_threshold = resampled_data[(resampled_data['CF2_MNDE_V_VBT'] < threshold_min) | (resampled_data['CF2_MNDE_V_VBT'] > threshold_max)]
    
    if not exceeded_threshold.empty and hide_warning_intervals < max_intervals:
        warning_message = f"Warning: CF2_MNDE_V_VBT has crossed the threshold frequency range at {exceeded_threshold.index[-1]}!"
        warning_style = {'color': 'red', 'font-weight': 'bold', 'display': 'block'}
        hide_warning_intervals += 1  # Increment hide warning interval count to keep it visible
    else:
        warning_message = ""
        warning_style = {'display': 'none'}
        hide_warning_intervals = 0  # Reset interval to ensure the message can reappear later

    return fig, warning_message, warning_style, hide_warning_intervals

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
