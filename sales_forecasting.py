import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

def load_data(file_path='sales_data_sample.csv'):
    """
    Load sales data from CSV and prepare it for forecasting
    """
    df = pd.read_csv(file_path, encoding='latin1')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    return df

def prepare_forecast_data(df, frequency='D', filters=None):
    """
    Prepare data for Prophet forecasting model
    
    Parameters:
    -----------
    df : pandas DataFrame
        The sales data
    frequency : str
        Time frequency for aggregation ('D' for daily, 'W' for weekly)
    filters : dict
        Dictionary with filtering conditions
        
    Returns:
    --------
    pandas DataFrame ready for Prophet
    """
    # Apply filters if provided
    if filters:
        if 'year_filter' in filters and filters['year_filter']:
            df = df[df['YEAR_ID'].isin(filters['year_filter'])]
        if 'product_line_filter' in filters and filters['product_line_filter']:
            df = df[df['PRODUCTLINE'].isin(filters['product_line_filter'])]
        if 'country_filter' in filters and filters['country_filter']:
            df = df[df['COUNTRY'].isin(filters['country_filter'])]
        if 'status_filter' in filters and filters['status_filter']:
            df = df[df['STATUS'].isin(filters['status_filter'])]
        if 'territory_filter' in filters and filters['territory_filter']:
            df = df[df['TERRITORY'].isin(filters['territory_filter']) | df['TERRITORY'].isna()]
    
    # Group by date according to the specified frequency
    if frequency == 'D':
        time_col = df['ORDERDATE'].dt.date
    elif frequency == 'W':
        # Use the start of the week for weekly aggregation
        time_col = df['ORDERDATE'].dt.to_period('W').dt.start_time.dt.date
    else:
        raise ValueError("Frequency must be 'D' or 'W'")
    
    # Aggregate sales by date
    sales_by_date = df.groupby(time_col)['SALES'].sum().reset_index()
    sales_by_date.columns = ['ds', 'y']
    
    # Convert ds to datetime
    sales_by_date['ds'] = pd.to_datetime(sales_by_date['ds'])
    
    # Store the frequency for later use
    sales_by_date.attrs['frequency'] = frequency
    
    return sales_by_date

def build_forecast_model(forecast_data, periods=30):
    """
    Build and train a Prophet forecasting model
    
    Parameters:
    -----------
    forecast_data : pandas DataFrame
        Data prepared for Prophet (must have 'ds' and 'y' columns)
    periods : int
        Number of periods to forecast
        
    Returns:
    --------
    model, forecast DataFrame
    """
    # Create and train the model with simplified parameters
    model = Prophet(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Fit the model
    model.fit(forecast_data)
    
    # Get the frequency used in data preparation
    frequency = forecast_data.attrs.get('frequency', 'D')
    
    # Create future dataframe for prediction with the correct frequency
    if frequency == 'D':
        future = model.make_future_dataframe(periods=periods, freq='D')
    elif frequency == 'W':
        future = model.make_future_dataframe(periods=periods, freq='W')
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

def create_forecast_plot(model, forecast, historical_data, forecast_column='yhat', plot_title='Sales Forecast'):
    """
    Create an interactive plotly figure for the forecast
    
    Parameters:
    -----------
    model : Prophet model
        Trained Prophet model
    forecast : pandas DataFrame
        Forecast DataFrame from Prophet
    historical_data : pandas DataFrame
        Original data used for training
    forecast_column : str
        Column to plot from forecast ('yhat', 'yhat_lower', or 'yhat_upper')
    plot_title : str
        Title for the plot
        
    Returns:
    --------
    plotly Figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='markers+lines',
            name='Historical Sales',
            line=dict(color='royalblue', width=2),
            marker=dict(size=5)
        )
    )
    
    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'],
            y=forecast[forecast_column],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        )
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(231, 234, 241, 0.5)',
            line=dict(color='rgba(231, 234, 241, 0)'),
            name='Confidence Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=plot_title,
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def get_forecast_metrics(historical_data, forecast):
    """
    Calculate forecast performance metrics
    
    Parameters:
    -----------
    historical_data : pandas DataFrame
        Original data used for training
    forecast : pandas DataFrame
        Forecast DataFrame from Prophet
        
    Returns:
    --------
    Dictionary with metrics
    """
    # Merge actual and predicted values
    evaluation_df = pd.merge(
        historical_data,
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='inner'
    )
    
    # Calculate metrics
    mse = np.mean((evaluation_df['y'] - evaluation_df['yhat'])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(evaluation_df['y'] - evaluation_df['yhat']))
    mape = np.mean(np.abs((evaluation_df['y'] - evaluation_df['yhat']) / evaluation_df['y'])) * 100
    
    # Calculate accuracy (percentage of actual values within prediction interval)
    in_interval = ((evaluation_df['y'] >= evaluation_df['yhat_lower']) & 
                  (evaluation_df['y'] <= evaluation_df['yhat_upper'])).mean() * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Interval Coverage': in_interval
    }

def create_forecast_table(forecast, periods=30):
    """
    Create a table of future forecasted values
    
    Parameters:
    -----------
    forecast : pandas DataFrame
        Forecast DataFrame from Prophet
    periods : int
        Number of future periods to show
        
    Returns:
    --------
    pandas DataFrame with forecasted values
    """
    # Get only future dates (the last 'periods' rows)
    future_forecast = forecast.iloc[-periods:].copy()
    
    # Format the DataFrame for display
    table_df = pd.DataFrame({
        'Date': future_forecast['ds'].dt.strftime('%Y-%m-%d'),
        'Forecasted Sales': future_forecast['yhat'].round(2),
        'Lower Bound': future_forecast['yhat_lower'].round(2),
        'Upper Bound': future_forecast['yhat_upper'].round(2)
    })
    
    return table_df 