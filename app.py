import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime
import sales_forecasting as sf
import requests
import json

# OpenRouter API configuration
OPENROUTER_API_KEY = "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"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Function to get LLM analysis from OpenRouter
def get_forecast_analysis(forecast_table, filters):
    """
    Get an LLM analysis of the forecast results in Bahasa Indonesia
    
    Args:
        forecast_table: DataFrame containing forecast data
        filters: Dictionary of filters applied to the data
        
    Returns:
        str: LLM analysis of forecast data in Bahasa Indonesia
    """
    # Create a summary of the filters
    filter_summary = ", ".join([
        f"Tahun: {', '.join(map(str, filters['year_filter']))}" if filters['year_filter'] else "",
        f"Lini Produk: {', '.join(filters['product_line_filter'])}" if filters['product_line_filter'] else "",
        f"Negara: {', '.join(filters['country_filter'])}" if filters['country_filter'] else "",
        f"Status: {', '.join(filters['status_filter'])}" if filters['status_filter'] else "",
        f"Teritori: {', '.join(map(str, filters.get('territory_filter', [])))}" if filters.get('territory_filter') else ""
    ])
    
    # Map forecast table columns to expected names
    date_col = 'Date' if 'Date' in forecast_table.columns else forecast_table.columns[0]
    forecast_col = 'Forecasted Sales' if 'Forecasted Sales' in forecast_table.columns else forecast_table.columns[1]
    
    try:
        # Get date information
        first_period = str(forecast_table[date_col].iloc[0])
        last_period = str(forecast_table[date_col].iloc[-1])
    except Exception as e:
        # Fallback if any error in date processing
        first_period = "Awal periode"
        last_period = "Akhir periode"
        print(f"Warning: Could not process date columns: {e}")
    
    # Calculate basic forecast statistics
    try:
        # Convert string columns to numeric if needed
        forecast_values = pd.to_numeric(forecast_table[forecast_col], errors='coerce')
        avg_forecast = forecast_values.mean()
        min_forecast = forecast_values.min()
        max_forecast = forecast_values.max()
        forecast_std = forecast_values.std()
    except Exception as e:
        # Fallback if any error in statistics calculation
        avg_forecast = min_forecast = max_forecast = forecast_std = 0
        print(f"Warning: Could not calculate forecast statistics: {e}")
    
    # Calculate projected growth rate
    try:
        if len(forecast_table) >= 2:
            # Convert string columns to numeric if needed
            forecast_values = pd.to_numeric(forecast_table[forecast_col], errors='coerce')
            first_value = forecast_values.iloc[0]
            last_value = forecast_values.iloc[-1]
            if first_value > 0:
                growth_rate = ((last_value - first_value) / first_value) * 100
                growth_info = f"Tingkat pertumbuhan yang diproyeksikan: {growth_rate:.2f}%"
            else:
                growth_info = "Tidak dapat menghitung pertumbuhan (nilai awal <= 0)"
        else:
            growth_info = "Data tidak cukup untuk menghitung pertumbuhan"
    except Exception as e:
        growth_info = "Tidak dapat menghitung pertumbuhan"
        print(f"Warning: Could not calculate growth rate: {e}")
    
    # Look for trends
    try:
        if len(forecast_table) > 5:
            # Convert string columns to numeric if needed
            forecast_values = pd.to_numeric(forecast_table[forecast_col], errors='coerce')
            first_half = forecast_values.iloc[:len(forecast_values)//2].mean()
            second_half = forecast_values.iloc[len(forecast_values)//2:].mean()
            trend_direction = "naik" if second_half > first_half else "turun"
            trend_info = f"Tren umum: {trend_direction} (rata-rata paruh pertama {first_half:.2f}, paruh kedua {second_half:.2f})"
        else:
            trend_info = "Data tidak cukup untuk menganalisis tren"
    except Exception as e:
        trend_info = "Tidak dapat menganalisis tren"
        print(f"Warning: Could not analyze trends: {e}")
    
    # Prepare the prompt for the LLM
    prompt = f"""Anda adalah seorang analis bisnis senior yang ahli dalam menganalisis data penjualan dan memberikan rekomendasi bisnis. 
Berikan analisis mendalam dan rekomendasi bisnis berdasarkan hasil peramalan penjualan berikut dalam Bahasa Indonesia yang profesional.
Format output Anda menggunakan Markdown untuk memudahkan pembacaan.

FILTER DATA:
{filter_summary}

PERIODE PERAMALAN:
- Dari: {first_period} 
- Sampai: {last_period}

STATISTIK PERAMALAN:
- Rata-rata peramalan: {avg_forecast:.2f}
- Nilai minimum: {min_forecast:.2f}
- Nilai maksimum: {max_forecast:.2f}
- Standar deviasi: {forecast_std:.2f}
- {growth_info}
- {trend_info}

Berikan analisis terstruktur yang mencakup:

1. **RINGKASAN EKSEKUTIF:**
   Berikan ringkasan singkat dengan temuan penting dalam 2-3 kalimat.

2. **ANALISIS TREN:**
   Jelaskan tren penjualan yang terlihat dari hasil peramalan secara detail.

3. **PELUANG DAN RISIKO:**
   Identifikasi potensi peluang atau risiko berdasarkan hasil peramalan.

4. **REKOMENDASI BISNIS:**
   Berikan 3-5 rekomendasi spesifik untuk meningkatkan penjualan berdasarkan data yang dianalisis.

5. **RENCANA TAKTIS:**
   Berikan saran praktis untuk tim penjualan dan pemasaran berdasarkan hasil peramalan.

Jika filter menunjukkan fokus pada produk, negara, atau teritorial tertentu, berikan rekomendasi yang spesifik untuk kategori tersebut.

Pastikan output Anda menggunakan format Markdown yang baik dengan heading, subheading, bullet points, dan penekanan teks yang sesuai.
"""

    # Make the API request to OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                headers=headers, 
                                data=json.dumps(data))
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        analysis = result['choices'][0]['message']['content']
        return analysis
    except Exception as e:
        return f"Error mendapatkan analisis LLM: {str(e)}"

# Page configuration
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# Helper function to check if data is available
def check_data(df, message="No data available for the selected filters."):
    """Display an info message if dataframe is empty and return True/False"""
    if df.empty:
        st.info(message)
        return False
    return True

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('sales_data_sample.csv', encoding='latin1')
    # Convert date column to datetime
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    
    # Calculate extra columns
    df['PROFIT'] = df['SALES'] - (df['QUANTITYORDERED'] * (df['MSRP'] - df['PRICEEACH']))
    df['PROFIT_MARGIN'] = df['PROFIT'] / df['SALES']
    
    return df

df = load_data()

# Sidebar
st.sidebar.header("Sales Dashboard")
st.sidebar.subheader('Filter Options')

# Handle NaN values in filters
def get_unique_values(dataframe, column):
    # Filter out NaN values and convert to list of strings
    unique_values = dataframe[column].dropna().unique().tolist()
    # Sort the values (they should all be strings now)
    return sorted(unique_values)

# Sidebar filters
year_filter = st.sidebar.multiselect("Year", options=sorted(df['YEAR_ID'].unique()), default=sorted(df['YEAR_ID'].unique()))
product_line_filter = st.sidebar.multiselect("Product Line", options=sorted(df['PRODUCTLINE'].unique()), default=sorted(df['PRODUCTLINE'].unique()))
country_filter = st.sidebar.multiselect("Country", options=sorted(df['COUNTRY'].unique()), default=sorted(df['COUNTRY'].unique()))
status_filter = st.sidebar.multiselect("Status", options=sorted(df['STATUS'].unique()), default=sorted(df['STATUS'].unique()))
territory_filter = st.sidebar.multiselect("Territory", options=get_unique_values(df, 'TERRITORY'), default=get_unique_values(df, 'TERRITORY'))

# Apply filters
filtered_df = df[(df['YEAR_ID'].isin(year_filter)) & 
                 (df['PRODUCTLINE'].isin(product_line_filter)) &
                 (df['COUNTRY'].isin(country_filter)) &
                 (df['STATUS'].isin(status_filter))]

# Special handling for TERRITORY to account for NaN values
if territory_filter:
    filtered_df = filtered_df[filtered_df['TERRITORY'].isin(territory_filter) | filtered_df['TERRITORY'].isna()]
   
# Main page title
st.title("Sales Performance Dashboard")
st.write("Interactive dashboard for analyzing sales data")

# Create tabs for different sections
tabs = st.tabs([
    "1. Overview", 
    "2. Temporal Analysis", 
    "3. Geographic Analysis",
    "4. Product Analysis",
    "5. Customer Analysis",
    "6. Business Analysis",
    "7. Sales Forecasting"
])

with tabs[0]:  # Tab 1: Overview
    st.header("Dashboard Overview")

    # KPI Cards in 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Check if filtered DataFrame is empty and handle accordingly
    if filtered_df.empty:
        with col1:
            st.metric(label="Total Sales", value="$0.00")
        with col2:
            st.metric(label="Total Orders", value="0")
        with col3:
            st.metric(label="Average Order Value", value="$0.00")
        with col4:
            st.metric(label="Top Product Line", value="No data")
    else:
        with col1:
            total_sales = filtered_df['SALES'].sum()
            st.metric(label="Total Sales", value=f"${total_sales:,.2f}")
        
        with col2:
            total_orders = filtered_df['ORDERNUMBER'].nunique()
            st.metric(label="Total Orders", value=f"{total_orders:,}")
        
        with col3:
            # Handle division by zero
            if total_orders > 0:
                avg_order_value = total_sales / total_orders
                st.metric(label="Average Order Value", value=f"${avg_order_value:,.2f}")
            else:
                st.metric(label="Average Order Value", value="$0.00")
        
        with col4:
            # Make sure there's data before trying to find max
            if not filtered_df.empty and filtered_df.groupby('PRODUCTLINE')['SALES'].sum().shape[0] > 0:
                top_product_line = filtered_df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
                top_pl_sales = filtered_df.groupby('PRODUCTLINE')['SALES'].sum().max()
                st.metric(label="Top Product Line", value=f"{top_product_line}", delta=f"${top_pl_sales:,.2f}")
            else:
                st.metric(label="Top Product Line", value="No data")
    
    # Alternative Dashboard Style KPI Cards in 3 columns
    st.subheader("Key Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    total_sales = filtered_df['SALES'].sum()
    total_products = filtered_df['QUANTITYORDERED'].sum()
    countries_count = filtered_df['COUNTRY'].nunique()
    
    with col1:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h3>💰 Total Sales</h3>
                <p style="font-size:24px; font-weight:bold;">${total_sales:,.2f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h3>📦 Products Sold</h3>
                <p style="font-size:24px; font-weight:bold;">{total_products:,} units</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <h3>🌍 Countries Served</h3>
                <p style="font-size:24px; font-weight:bold;">{countries_count} countries</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.subheader("Monthly Sales Trend")
    
    # Monthly sales trend
    if filtered_df.empty:
        st.info("No data available for the selected filters.")
    else:
        # Fix column naming conflict by explicitly naming the series
        monthly_sales = filtered_df.groupby([
            filtered_df['ORDERDATE'].dt.year.rename('Year'),
            filtered_df['ORDERDATE'].dt.month.rename('Month'),
            'MONTH_NAME'
        ])['SALES'].sum().reset_index()
        
        if not monthly_sales.empty:
            monthly_sales['Year-Month'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['MONTH_NAME']
            monthly_sales = monthly_sales.sort_values(['Year', 'Month'])
            
            # Original monthly trend chart
            col1, col2 = st.columns(2)
            
            with col1:
                fig_monthly_trend = px.line(
                    monthly_sales, 
                    x='Year-Month', 
                    y='SALES',
                    markers=True,
                    title="Monthly Sales Trend",
                    labels={"SALES": "Sales ($)", "Year-Month": "Month-Year"}
                )
                fig_monthly_trend.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_monthly_trend, use_container_width=True)
            
            # Create a new field for year-month for alternative view
            filtered_df['YearMonth'] = filtered_df['ORDERDATE'].dt.to_period('M').astype(str)
            
            # Group by month for trend with trendline
            monthly_trend = filtered_df.groupby('YearMonth')['SALES'].sum().reset_index()
            monthly_trend['MonthIndex'] = np.arange(len(monthly_trend))
            
            with col2:
                if not monthly_trend.empty:
                    # Calculate trendline
                    z = np.polyfit(monthly_trend['MonthIndex'], monthly_trend['SALES'], 1)
                    p = np.poly1d(z)
                    monthly_trend['Trend'] = p(monthly_trend['MonthIndex'])
                    
                    # Create interactive plot with plotly
                    fig_trend = go.Figure()
                    
                    # Add sales line
                    fig_trend.add_trace(
                        go.Scatter(
                            x=monthly_trend['YearMonth'],
                            y=monthly_trend['SALES'],
                            mode='lines+markers',
                            name='Total Sales',
                            line=dict(color='royalblue', width=2)
                        )
                    )
                    
                    # Add trendline
                    fig_trend.add_trace(
                        go.Scatter(
                            x=monthly_trend['YearMonth'],
                            y=monthly_trend['Trend'],
                            mode='lines',
                            name='Trend',
                            line=dict(color='red', width=2, dash='dash')
                        )
                    )
                    
                    fig_trend.update_layout(
                        title="Monthly Sales with Trendline",
                        xaxis_title="Month",
                        yaxis_title="Sales ($)",
                        xaxis_tickangle=-45,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No monthly sales data available for the selected filters.")
    
    # Order Status distribution and Product Line Sales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Status Distribution")
        if filtered_df.empty:
            st.info("No status data available for the selected filters.")
        else:
            status_counts = filtered_df['STATUS'].value_counts().reset_index()
            if not status_counts.empty:
                status_counts.columns = ['Status', 'Count']
                
                fig_status = px.pie(
                    status_counts, 
                    values='Count', 
                    names='Status',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_status.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("No status data available.")
    
    with col2:
        st.subheader("Product Line Distribution")
        if filtered_df.empty:
            st.info("No product line data available for the selected filters.")
        else:
            product_sales = filtered_df.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            if not product_sales.empty:
                fig_product = px.bar(
                    product_sales,
                    x='PRODUCTLINE',
                    y='SALES',
                    color='PRODUCTLINE',
                    title="Sales by Product Line"
                )
                st.plotly_chart(fig_product, use_container_width=True)
            else:
                st.info("No product line data available.")
    
    # Year-over-year Sales Growth from Alternative Dashboard
    st.subheader("Year-over-Year Sales Growth")
    
    sales_per_year = filtered_df.groupby('YEAR_ID')['SALES'].sum().reset_index()
    
    if len(sales_per_year) > 1:
        sales_per_year['% Growth'] = sales_per_year['SALES'].pct_change() * 100
        sales_per_year['% Growth'] = sales_per_year['% Growth'].round(2)
        sales_per_year.columns = ['Year', 'Total Sales', 'Growth %']
        
        # Create a custom color scale based on growth (green for positive, red for negative)
        colors = ['red' if x < 0 else 'green' for x in sales_per_year['Growth %'].fillna(0)]
        
        fig_growth = px.bar(
            sales_per_year,
            x='Year',
            y='Growth %',
            title='Sales Growth Year-over-Year',
            labels={'Growth %': 'Growth (%)', 'Year': 'Year'},
            text='Growth %'
        )
        
        fig_growth.update_traces(marker_color=colors, texttemplate='%{text:.2f}%', textposition='outside')
        fig_growth.update_layout(height=400)
        
        # Show the data table with formatted values
        growth_data = sales_per_year.copy()
        growth_data['Total Sales'] = growth_data['Total Sales'].apply(lambda x: f"${x:,.2f}")
        growth_data['Growth %'] = growth_data['Growth %'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_growth, use_container_width=True)
        with col2:
            st.dataframe(growth_data, use_container_width=True)
    else:
        st.info("Not enough yearly data for growth calculation.")

    # Top Products and Top Countries
    st.subheader("Top Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🥧 Top 5 Products")
        top_products = (
            filtered_df.groupby(['PRODUCTCODE', 'PRODUCTLINE'])['SALES']
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        
        if not top_products.empty:
            top_products['Label'] = top_products['PRODUCTCODE'] + " (" + top_products['PRODUCTLINE'] + ")"
            
            fig_top_products = px.pie(
                top_products,
                values='SALES',
                names='Label',
                title='Top 5 Products by Sales',
                hole=0.3
            )
            
            fig_top_products.update_traces(textposition='inside', textinfo='percent+label')
            fig_top_products.update_layout(height=400)
            
            st.plotly_chart(fig_top_products, use_container_width=True)
        else:
            st.info("No product data available.")
    
    with col2:
        st.markdown("#### 🌍 Top 5 Countries")
        top_countries = (
            filtered_df.groupby('COUNTRY')['SALES']
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .reset_index()
        )
        
        if not top_countries.empty:
            fig_top_countries = px.bar(
                top_countries,
                x='SALES',
                y='COUNTRY',
                orientation='h',
                title='Top 5 Countries by Sales',
                labels={'SALES': 'Sales ($)', 'COUNTRY': 'Country'},
                text_auto='.2s'
            )
            
            fig_top_countries.update_layout(height=400)
            fig_top_countries.update_yaxes(categoryorder='total ascending')
            
            st.plotly_chart(fig_top_countries, use_container_width=True)
        else:
            st.info("No country data available.")

with tabs[1]:  # Tab 2: Temporal Analysis
    st.header("Temporal Analysis")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    # Time series analysis for sales trend
    st.subheader("Sales Time Series Analysis")
    
    # Group by date
    daily_sales = filtered_df.groupby(filtered_df['ORDERDATE'].dt.date)['SALES'].sum().reset_index()
    
    if check_data(daily_sales, "No daily sales data available."):
        fig_time_series = px.line(
            daily_sales,
            x='ORDERDATE',
            y='SALES',
            title="Daily Sales Over Time",
            labels={"SALES": "Sales ($)", "ORDERDATE": "Date"}
        )
        st.plotly_chart(fig_time_series, use_container_width=True)
    
    # Heatmap of sales by month and year
    st.subheader("Sales Heatmap by Month and Year")
    
    # Create pivot table for the heatmap
    try:
        heatmap_data = filtered_df.pivot_table(
            values='SALES',
            index=filtered_df['ORDERDATE'].dt.year,
            columns=filtered_df['ORDERDATE'].dt.month_name(),
            aggfunc='sum'
        )
        
        # Check if pivot table has data
        if not heatmap_data.empty and heatmap_data.size > 0:
            # Reorder columns by month
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
            heatmap_data = heatmap_data.reindex(columns=[m for m in month_order if m in heatmap_data.columns])
            
            fig_heatmap = px.imshow(
                heatmap_data,
                text_auto='.2s',
                color_continuous_scale='YlGnBu',
                labels=dict(x="Month", y="Year", color="Sales"),
                x=heatmap_data.columns,
                y=heatmap_data.index
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Not enough data for heatmap visualization.")
    except Exception as e:
        st.info(f"Could not create heatmap: {str(e)}")
    
    # Quarterly analysis
    st.subheader("Quarterly Sales Analysis")
    
    # Create quarterly data
    quarterly_sales = filtered_df.groupby(['YEAR_ID', 'QTR_ID'])['SALES'].sum().reset_index()
    
    if check_data(quarterly_sales, "No quarterly sales data available."):
        quarterly_sales['Year-Quarter'] = quarterly_sales['YEAR_ID'].astype(str) + '-Q' + quarterly_sales['QTR_ID'].astype(str)
        
        fig_quarterly = px.bar(
            quarterly_sales,
            x='Year-Quarter',
            y='SALES',
            color='YEAR_ID',
            title="Quarterly Sales Performance",
            labels={"SALES": "Sales ($)", "Year-Quarter": "Year-Quarter"}
        )
        st.plotly_chart(fig_quarterly, use_container_width=True)
    
    # Year over year comparison
    st.subheader("Year-Over-Year Comparison")
    
    # Create YoY data
    yearly_sales = filtered_df.groupby(['YEAR_ID', 'MONTH_ID'])['SALES'].sum().reset_index()
    
    if check_data(yearly_sales, "No yearly comparison data available."):
        fig_yoy = px.line(
            yearly_sales,
            x='MONTH_ID',
            y='SALES',
            color='YEAR_ID',
            title="Year-Over-Year Monthly Sales Comparison",
            labels={"SALES": "Sales ($)", "MONTH_ID": "Month", "YEAR_ID": "Year"}
        )
        fig_yoy.update_xaxes(tickvals=list(range(1, 13)), ticktext=[calendar.month_name[i] for i in range(1, 13)])
        st.plotly_chart(fig_yoy, use_container_width=True)

with tabs[2]:  # Tab 3: Geographic Analysis
    st.header("Geographic Analysis")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    # World map for sales by country
    st.subheader("Global Sales Distribution")
    
    country_sales = filtered_df.groupby('COUNTRY')['SALES'].sum().reset_index()
    
    if check_data(country_sales, "No country sales data available."):
        fig_map = px.choropleth(
            country_sales,
            locations='COUNTRY',
            locationmode='country names',
            color='SALES',
            hover_name='COUNTRY',
            color_continuous_scale='Viridis',
            title="Sales by Country",
            labels={'SALES': 'Sales ($)'}
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Bar chart for sales by territory
    st.subheader("Sales by Territory")
    
    # Handle territory with NaN values
    territory_sales = filtered_df.dropna(subset=['TERRITORY']).groupby('TERRITORY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
    
    if check_data(territory_sales, "No territory sales data available."):
        fig_territory = px.bar(
            territory_sales,
            x='TERRITORY',
            y='SALES',
            color='TERRITORY',
            title="Sales by Territory (NA, EMEA, APAC, Japan)",
            labels={"SALES": "Sales ($)", "TERRITORY": "Territory"}
        )
        st.plotly_chart(fig_territory, use_container_width=True)
    
    # Top performing cities
    st.subheader("Top Performing Cities")
    
    city_sales = filtered_df.groupby(['CITY', 'STATE', 'COUNTRY'])['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(15)
    
    if check_data(city_sales, "No city sales data available."):
        fig_cities = px.bar(
            city_sales,
            x='CITY',
            y='SALES',
            hover_data=['STATE', 'COUNTRY'],
            color='SALES',
            title="Top 15 Cities by Sales",
            labels={"SALES": "Sales ($)", "CITY": "City"}
        )
        fig_cities.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cities, use_container_width=True)
    
    # Map top customers
    st.subheader("Customer Geographic Distribution")
    
    # Create a scatter map of customers with sales value determining size
    customer_locations = filtered_df.groupby(['CUSTOMERNAME', 'CITY', 'COUNTRY'])['SALES'].sum().reset_index()
    
    if check_data(customer_locations, "No customer location data available."):
        fig_customer_map = px.scatter_geo(
            customer_locations,
            locations='COUNTRY',
            locationmode='country names',
            hover_name='CUSTOMERNAME',
            hover_data=['CITY', 'SALES'],
            size='SALES',
            title="Customer Locations by Sales Volume",
            projection='natural earth'
        )
        st.plotly_chart(fig_customer_map, use_container_width=True)

with tabs[3]:  # Tab 4: Product Analysis
    st.header("Product Analysis")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    # Product line performance comparison
    st.subheader("Product Line Performance")
    
    product_performance = filtered_df.groupby('PRODUCTLINE').agg({
        'SALES': 'sum',
        'QUANTITYORDERED': 'sum',
        'ORDERNUMBER': 'nunique'
    }).reset_index()
    
    if check_data(product_performance, "No product performance data available."):
        # Create three charts in tabs
        product_tabs = st.tabs(["Sales", "Quantity", "Orders"])
        
        with product_tabs[0]:
            fig_product_sales = px.bar(
                product_performance.sort_values('SALES', ascending=False),
                x='PRODUCTLINE',
                y='SALES',
                color='PRODUCTLINE',
                title="Total Sales by Product Line",
                labels={"SALES": "Sales ($)", "PRODUCTLINE": "Product Line"}
            )
            st.plotly_chart(fig_product_sales, use_container_width=True)
            
        with product_tabs[1]:
            fig_product_qty = px.bar(
                product_performance.sort_values('QUANTITYORDERED', ascending=False),
                x='PRODUCTLINE',
                y='QUANTITYORDERED',
                color='PRODUCTLINE',
                title="Total Quantity Ordered by Product Line",
                labels={"QUANTITYORDERED": "Quantity", "PRODUCTLINE": "Product Line"}
            )
            st.plotly_chart(fig_product_qty, use_container_width=True)
            
        with product_tabs[2]:
            fig_product_orders = px.bar(
                product_performance.sort_values('ORDERNUMBER', ascending=False),
                x='PRODUCTLINE',
                y='ORDERNUMBER',
                color='PRODUCTLINE',
                title="Number of Orders by Product Line",
                labels={"ORDERNUMBER": "Orders", "PRODUCTLINE": "Product Line"}
            )
            st.plotly_chart(fig_product_orders, use_container_width=True)
    
    # Price vs Sales scatter plot
    st.subheader("Price vs. Sales Analysis")
    
    product_price_sales = filtered_df.groupby('PRODUCTCODE').agg({
        'PRICEEACH': 'mean',
        'SALES': 'sum',
        'PRODUCTLINE': 'first'
    }).reset_index()
    
    if check_data(product_price_sales, "No price vs sales data available."):
        fig_price_sales = px.scatter(
            product_price_sales,
            x='PRICEEACH',
            y='SALES',
            color='PRODUCTLINE',
            size='SALES',
            hover_name='PRODUCTCODE',
            title="Price vs. Total Sales by Product",
            labels={"PRICEEACH": "Average Price ($)", "SALES": "Total Sales ($)", "PRODUCTLINE": "Product Line"}
        )
        st.plotly_chart(fig_price_sales, use_container_width=True)
    
    # Quantity ordered distribution
    st.subheader("Quantity Ordered Distribution")
    
    if not filtered_df.empty:
        # Create a histogram of ordered quantities
        fig_quantity = px.histogram(
            filtered_df,
            x='QUANTITYORDERED',
            color='PRODUCTLINE',
            marginal='box',
            title="Distribution of Order Quantities",
            labels={"QUANTITYORDERED": "Quantity Ordered", "count": "Frequency"}
        )
        st.plotly_chart(fig_quantity, use_container_width=True)
    else:
        st.info("No quantity data available.")
    
    # MSRP vs actual price analysis
    st.subheader("MSRP vs. Actual Price Analysis")
    
    # Calculate the average actual price vs MSRP
    price_comparison = filtered_df.groupby('PRODUCTCODE').agg({
        'PRICEEACH': 'mean',
        'MSRP': 'first',
        'PRODUCTLINE': 'first'
    }).reset_index()
    
    if check_data(price_comparison, "No price comparison data available."):
        price_comparison['Discount'] = (price_comparison['MSRP'] - price_comparison['PRICEEACH']) / price_comparison['MSRP'] * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_price_comp = px.scatter(
                price_comparison,
                x='MSRP',
                y='PRICEEACH',
                color='PRODUCTLINE',
                title="MSRP vs. Actual Price",
                labels={"MSRP": "MSRP ($)", "PRICEEACH": "Actual Price ($)", "PRODUCTLINE": "Product Line"}
            )
            
            # Add reference line
            fig_price_comp.add_trace(
                go.Scatter(
                    x=[price_comparison['MSRP'].min(), price_comparison['MSRP'].max()],
                    y=[price_comparison['MSRP'].min(), price_comparison['MSRP'].max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='MSRP = Price'
                )
            )
            st.plotly_chart(fig_price_comp, use_container_width=True)
        
        with col2:
            fig_discount = px.box(
                price_comparison,
                x='PRODUCTLINE',
                y='Discount',
                color='PRODUCTLINE',
                title="Discount Percentage by Product Line",
                labels={"Discount": "Discount (%)", "PRODUCTLINE": "Product Line"}
            )
            st.plotly_chart(fig_discount, use_container_width=True)

with tabs[4]:  # Tab 5: Customer Analysis
    st.header("Customer Analysis")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    # Top customers by sales value
    st.subheader("Top Customers by Sales")
    
    top_customers = filtered_df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(15)
    
    if check_data(top_customers, "No customer data available."):
        fig_top_customers = px.bar(
            top_customers,
            x='CUSTOMERNAME',
            y='SALES',
            title="Top 15 Customers by Sales",
            labels={"SALES": "Sales ($)", "CUSTOMERNAME": "Customer"}
        )
        fig_top_customers.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_customers, use_container_width=True)
    
    # Customer distribution by deal size
    st.subheader("Customer Distribution by Deal Size")
    
    customer_deal_size = filtered_df.groupby(['CUSTOMERNAME', 'DEALSIZE']).size().reset_index(name='COUNT')
    
    if check_data(customer_deal_size, "No deal size data available."):
        customer_deal_size_pivot = customer_deal_size.pivot_table(
            index='DEALSIZE',
            values='COUNT',
            aggfunc='sum'
        ).reset_index()
        
        fig_deal_size = px.pie(
            customer_deal_size_pivot,
            values='COUNT',
            names='DEALSIZE',
            title="Customer Deal Size Distribution",
            hole=0.4
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_deal_size, use_container_width=True)
        
        with col2:
            # Top customers by frequency
            customer_frequency = filtered_df.groupby('CUSTOMERNAME')['ORDERNUMBER'].nunique().reset_index()
            customer_frequency.columns = ['CUSTOMERNAME', 'ORDER_COUNT']
            top_frequent = customer_frequency.sort_values('ORDER_COUNT', ascending=False).head(10)
            
            if not top_frequent.empty:
                fig_frequency = px.bar(
                    top_frequent,
                    x='CUSTOMERNAME',
                    y='ORDER_COUNT',
                    title="Top 10 Customers by Order Frequency",
                    labels={"ORDER_COUNT": "Number of Orders", "CUSTOMERNAME": "Customer"}
                )
                fig_frequency.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_frequency, use_container_width=True)
            else:
                st.info("No customer frequency data available.")
    
    # Geographic distribution of customers
    st.subheader("Geographic Distribution of Customers")
    
    customer_countries = filtered_df.groupby('COUNTRY')['CUSTOMERNAME'].nunique().reset_index()
    
    if check_data(customer_countries, "No customer geographic data available."):
        customer_countries.columns = ['COUNTRY', 'CUSTOMER_COUNT']
        
        fig_customer_countries = px.choropleth(
            customer_countries,
            locations='COUNTRY',
            locationmode='country names',
            color='CUSTOMER_COUNT',
            hover_name='COUNTRY',
            color_continuous_scale='Viridis',
            title="Number of Customers by Country",
            labels={'CUSTOMER_COUNT': 'Customer Count'}
        )
        st.plotly_chart(fig_customer_countries, use_container_width=True)

with tabs[5]:  # Tab 6: Business Analysis
    st.header("Business Analysis")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    # Deal size analysis
    st.subheader("Deal Size Analysis")
    
    deal_size_analysis = filtered_df.groupby('DEALSIZE').agg({
        'SALES': 'sum',
        'ORDERNUMBER': 'nunique',
        'QUANTITYORDERED': 'sum'
    }).reset_index()
    
    if check_data(deal_size_analysis, "No deal size analysis data available."):
        fig_deal_size = px.bar(
            deal_size_analysis,
            x='DEALSIZE',
            y='SALES',
            color='DEALSIZE',
            title="Sales by Deal Size",
            labels={"SALES": "Sales ($)", "DEALSIZE": "Deal Size"}
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_deal_size, use_container_width=True)
        
        with col2:
            # Average order value by deal size
            if 'ORDERNUMBER' in deal_size_analysis.columns and deal_size_analysis['ORDERNUMBER'].sum() > 0:
                deal_size_analysis['AVG_ORDER_VALUE'] = deal_size_analysis['SALES'] / deal_size_analysis['ORDERNUMBER']
                
                fig_aov = px.bar(
                    deal_size_analysis,
                    x='DEALSIZE',
                    y='AVG_ORDER_VALUE',
                    color='DEALSIZE',
                    title="Average Order Value by Deal Size",
                    labels={"AVG_ORDER_VALUE": "Average Order Value ($)", "DEALSIZE": "Deal Size"}
                )
                st.plotly_chart(fig_aov, use_container_width=True)
            else:
                st.info("No order value data available for deal size analysis.")
    
    # Territory performance comparison
    st.subheader("Territory Performance Comparison")
    
    # Handle NaN values in territory groupby 
    territory_df = filtered_df.dropna(subset=['TERRITORY'])
    
    if check_data(territory_df, "No territory data available."):
        territory_metrics = territory_df.groupby('TERRITORY').agg({
            'SALES': 'sum',
            'PROFIT': 'sum',
            'ORDERNUMBER': 'nunique',
            'CUSTOMERNAME': 'nunique'
        }).reset_index()
        
        if not territory_metrics.empty and len(territory_metrics) > 0:
            territory_metrics['PROFIT_MARGIN'] = territory_metrics['PROFIT'] / territory_metrics['SALES']
            territory_metrics['AVG_ORDER_VALUE'] = territory_metrics['SALES'] / territory_metrics['ORDERNUMBER']
            territory_metrics['REVENUE_PER_CUSTOMER'] = territory_metrics['SALES'] / territory_metrics['CUSTOMERNAME']
            
            # Create a radar chart
            categories = ['Sales', 'Profit', 'Orders', 'Customers', 'Avg Order Value']
            
            fig_radar = go.Figure()
            
            for i, territory in enumerate(territory_metrics['TERRITORY']):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        territory_metrics.loc[i, 'SALES'] / territory_metrics['SALES'].max(),
                        territory_metrics.loc[i, 'PROFIT'] / territory_metrics['PROFIT'].max(),
                        territory_metrics.loc[i, 'ORDERNUMBER'] / territory_metrics['ORDERNUMBER'].max(),
                        territory_metrics.loc[i, 'CUSTOMERNAME'] / territory_metrics['CUSTOMERNAME'].max(),
                        territory_metrics.loc[i, 'AVG_ORDER_VALUE'] / territory_metrics['AVG_ORDER_VALUE'].max()
                    ],
                    theta=categories,
                    fill='toself',
                    name=territory
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Territory Performance Comparison"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Insufficient territory data for performance comparison.")
    
    # Profit margin analysis
    st.subheader("Profit Margin Analysis")
    
    # Create profit metrics by product line
    profit_by_product = filtered_df.groupby('PRODUCTLINE').agg({
        'SALES': 'sum',
        'PROFIT': 'sum'
    }).reset_index()
    
    if check_data(profit_by_product, "No profit data available by product line."):
        profit_by_product['PROFIT_MARGIN'] = profit_by_product['PROFIT'] / profit_by_product['SALES']
        profit_by_product = profit_by_product.sort_values('PROFIT_MARGIN', ascending=False)
        
        fig_profit_margin = px.bar(
            profit_by_product,
            x='PRODUCTLINE',
            y='PROFIT_MARGIN',
            color='PRODUCTLINE',
            title="Profit Margin by Product Line",
            labels={"PROFIT_MARGIN": "Profit Margin (%)", "PRODUCTLINE": "Product Line"}
        )
        fig_profit_margin.update_layout(yaxis=dict(tickformat='.0%'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_profit_margin, use_container_width=True)
        
        with col2:
            fig_profit = px.scatter(
                profit_by_product,
                x='SALES',
                y='PROFIT',
                size='SALES',
                color='PRODUCTLINE',
                hover_name='PRODUCTLINE',
                text='PRODUCTLINE',
                title="Sales vs. Profit by Product Line",
                labels={"SALES": "Sales ($)", "PROFIT": "Profit ($)", "PRODUCTLINE": "Product Line"}
            )
            fig_profit.update_traces(textposition='top center')
            st.plotly_chart(fig_profit, use_container_width=True)

with tabs[6]:  # Tab 7: Sales Forecasting
    st.header("Sales Forecasting")
    
    # Check if there's data to display
    if not check_data(filtered_df):
        st.stop()  # Stop execution of this tab if no data
    
    st.write("This tab uses time series forecasting to predict future sales based on historical data.")
    
    # Create filters for forecasting options
    forecast_col1, forecast_col2 = st.columns(2)
    
    with forecast_col1:
        # Time aggregation for forecasting
        time_frequency = st.selectbox(
            "Time Aggregation",
            options=["Daily", "Weekly"],
            index=1,  # Default to Weekly
            help="The time frequency for aggregating sales data"
        )
        
        # Map the selection to Prophet format
        frequency_map = {"Daily": "D", "Weekly": "W"}
        selected_frequency = frequency_map[time_frequency]
    
    with forecast_col2:
        # Set appropriate max periods based on frequency
        if time_frequency == "Daily":
            max_periods = 365
            default_periods = 90
        else:  # Weekly
            max_periods = 156  # 3 years
            default_periods = 26  # 6 months
        
        # Number of periods to forecast
        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=3,
            max_value=max_periods,
            value=default_periods,
            step=1,
            help=f"Number of future {time_frequency.lower()} periods to forecast"
        )
    
    # Initialize session state for forecast data if not exists
    if 'forecast_generated' not in st.session_state:
        st.session_state.forecast_generated = False
        st.session_state.forecast_data = None
        st.session_state.forecast_table = None
        st.session_state.model = None
        st.session_state.forecast = None
        st.session_state.metrics = None
    
    # Create a button to trigger forecasting
    if not st.session_state.forecast_generated:
        forecast_button = st.button("Generate Forecast", type="primary")
    else:
        forecast_button = st.button("Regenerate Forecast", type="primary")
    
    if forecast_button:
        with st.spinner("Generating forecast..."):
            try:
                # Prepare filters dictionary
                filters = {
                    'year_filter': year_filter,
                    'product_line_filter': product_line_filter,
                    'country_filter': country_filter,
                    'status_filter': status_filter,
                    'territory_filter': territory_filter
                }
                
                # Prepare data for forecasting
                forecast_data = sf.prepare_forecast_data(
                    df=filtered_df,
                    frequency=selected_frequency,
                    filters=filters
                )
                
                if len(forecast_data) >= 2:  # Need at least 2 data points to forecast
                    # Build the forecast model
                    model, forecast = sf.build_forecast_model(
                        forecast_data=forecast_data,
                        periods=forecast_periods
                    )
                    
                    # Store in session state
                    st.session_state.forecast_generated = True
                    st.session_state.forecast_data = forecast_data
                    st.session_state.model = model
                    st.session_state.forecast = forecast
                    st.session_state.filters = filters
                    
                    # Create forecast table for display
                    forecast_table = sf.create_forecast_table(forecast, periods=min(30, forecast_periods))
                    st.session_state.forecast_table = forecast_table
                    
                    # Calculate metrics
                    metrics = sf.get_forecast_metrics(forecast_data, forecast)
                    st.session_state.metrics = metrics
                else:
                    st.error("Not enough data for forecasting. Please adjust your filters to include more data points.")
                    st.session_state.forecast_generated = False
            
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.info("If you're seeing an error about Prophet, you might need to install it with: `pip install prophet`")
                st.session_state.forecast_generated = False
    
    # Display forecast results if available
    if st.session_state.forecast_generated:
        # Display forecast plot
        st.subheader("Sales Forecast")
        forecast_title = f"Sales Forecast ({time_frequency} Frequency, {forecast_periods} Periods Ahead)"
        forecast_fig = sf.create_forecast_plot(
            model=st.session_state.model,
            forecast=st.session_state.forecast,
            historical_data=st.session_state.forecast_data,
            plot_title=forecast_title
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Display metrics
        st.subheader("Forecast Metrics")
        metrics = st.session_state.metrics
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        
        with metric_col2:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        
        with metric_col3:
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        
        with metric_col4:
            st.metric("Interval Coverage", f"{metrics['Interval Coverage']:.2f}%")
        
        # Display forecast table
        st.subheader("Forecasted Values")
        st.dataframe(st.session_state.forecast_table, use_container_width=True)
        
        # Download link for forecast data
        csv = st.session_state.forecast_table.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=csv,
            file_name="sales_forecast.csv",
            mime="text/csv"
        )
        
        # LLM Analysis section
        st.subheader("Analisis Penjualan dengan AI")
        show_analysis = st.checkbox("Tampilkan analisis AI (dalam Bahasa Indonesia)", value=False, key="show_llm_analysis")
        
        if show_analysis:
            # Create a placeholder for the spinner and the analysis
            analysis_placeholder = st.empty()
            
            with analysis_placeholder.container():
                with st.spinner("Menghasilkan analisis dengan AI... (mungkin memerlukan waktu beberapa detik)"):
                    analysis = get_forecast_analysis(st.session_state.forecast_table, st.session_state.filters)
                    
                    # Display the analysis as markdown
                    st.markdown("## Hasil Analisis Forecasting")
                    st.markdown(analysis)
                    st.markdown("---")
                    st.markdown("*Analisis dibuat oleh AI menggunakan data peramalan penjualan*")
                
                # Add a download button for the analysis
                st.download_button(
                    label="Unduh Analisis",
                    data=analysis,
                    file_name="analisis_penjualan.txt",
                    mime="text/plain"
                )
        else:
            st.info("Centang kotak di atas untuk menghasilkan analisis AI terhadap data peramalan dalam Bahasa Indonesia.")
    
    else:
        # Display instructions when first loading the page
        st.info("Adjust the forecasting parameters above and click 'Generate Forecast' to see predictions.")
        
        # Placeholder forecast image
        st.image("https://miro.medium.com/max/700/1*LwDY-mD1u6qJhwR4MnIJZQ.png", 
                caption="Example of a time series forecast (click Generate Forecast to create your own)")

# Add footer
st.markdown("---")
st.markdown("Sales Data Analysis Dashboard | Created with Streamlit")
