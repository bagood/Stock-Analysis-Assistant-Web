import yfinance as yf
import streamlit as st
from datetime import datetime

from webFunctions import web_functions
from stockPriceForecast_5Days import SPF_5Days
from prepareDataset_5Days import PD_5Days

wf = web_functions()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Stock Analysis Assistant', page_icon=":reminder_ribbon:", layout='centered')

with st.container():
    wf.add_center_text('Stock Analysis Assistant', 'h1')
    emiten, emiten_name = wf.choose_emiten()

with st.container():
    wf.add_center_text(f'{emiten_name}\'s Stock Data', 'h2')
    emiten_data = yf.download(emiten.upper() + '.JK', start=datetime(2023, 1, 1), end=datetime.now())

with st.container():
    pd_5days = PD_5Days(emiten_data)
    feature_data = pd_5days.create_feature_data()
    target_data = pd_5days.create_target_data()
    spf_5days = SPF_5Days(feature_data, target_data)

with st.container():
    start_date = st.date_input('Choose a Starting Date')
    start_date = datetime(start_date.year, start_date.month, start_date.day, 0, 0, 0)

with st.container():
    wf.add_center_text('Price', 'h3')
    fig_price = wf.visualize_price(emiten_data, start_date) 
    st.plotly_chart(fig_price, use_container_width = True)

with st.container():
    wf.add_center_text(f'Trade Volume', 'h3')
    fig_volume = wf.visualize_volume(emiten_data, start_date)
    st.plotly_chart(fig_volume, use_container_width = True)