import pytz
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px 
import plotly.graph_objects as go
from datetime import datetime, timedelta

class web_functions:
    def __init__(self):
        self.emiten_code_data = pd.read_csv('emiten_code.csv')
        jkt = pytz.timezone('Asia/Jakarta')
        current_hour = datetime.now(jkt).hour
        current_day = datetime.now(jkt).strftime("%A")

        # forecast for today's price if it is still under 3 pm and a weekday
        self.forecast_today = (current_hour < 15) & (current_day not in ['Saturday', 'Sunday'])

    def _set_figure(self, fig, title, title_size=28, font_size=20):
        """set the visualization style
        """
        fig.update_layout(title=title ,title_font_size=title_size)
        fig.update_layout(
            font=dict(
                family="Courier",
                size=font_size, 
                color="black"
            ))
        fig.update_xaxes(linewidth=2, tickfont_size=20, title_font_size=25)
        fig.update_yaxes(tickfont_size=20,title_font_size=25)

        return fig
    
    def visualize_price(self, emiten_data, start_date=datetime.now() - timedelta(days=7)):
        """visualize daily stock price using a candlestick
        """
        data_to_plot = emiten_data.loc[emiten_data.index >= start_date, :]
        fig = go.Figure(data=[go.Candlestick(x = data_to_plot.index,
                                low = data_to_plot['Low'],
                                high = data_to_plot['High'],
                                close = data_to_plot['Close'],
                                open = data_to_plot['Open'])])
        
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)

        return fig
    
    def visualize_volume(self, emiten_data, start_date=datetime.now() - timedelta(days=7)):
        """visualize daily volume using a bar chart
        """
        data_to_plot = emiten_data.loc[emiten_data.index >= start_date, :]
        fig = go.Figure(data=[go.Bar(x=data_to_plot.index,
                        y=data_to_plot['Volume'])])
        
        fig.update_layout(showlegend=False)

        return fig
    
    def add_center_text(self, text, style):
        """create a text placed right on the center of the streamlit web interface
        """
        st.markdown(f"<{style} style='text-align: center; color: white;'>{text}</{style}>", unsafe_allow_html=True)

        return
    
    def choose_emiten(self):
        """function for selecting the desired emiten to be analyzed
        """
        emiten_name = st.selectbox('Choose an Emiten', self.emiten_code_data['Nama Perusahaan'].unique())
        emiten = self.emiten_code_data.loc[self.emiten_code_data['Nama Perusahaan']==emiten_name, 'Kode'].values[0]
        emiten_name = self.emiten_code_data.loc[self.emiten_code_data['Kode'] == emiten, 'Nama Perusahaan'].values[0].replace('.', '')

        return (emiten, emiten_name)

    def visualize_forecast(self, forecast, actual, emiten_data):
        """create a visualiztion of the stock price forecast
        """
        dates = emiten_data['Date'].values[-len(forecast)+1:]
        if self.forecast_today == False:
            tomorrow_date = np.datetime64(datetime.now(self.jkt) +  timedelta(days=1))
            dates = np.concatenate((dates, [tomorrow_date]))
            actual = np.concatenate((actual, [np.nan]))
        
        fig = px.line(x=dates, y=forecast)
        fig.add_scatter(x=dates, y=actual, mode='lines')
        
        return fig
