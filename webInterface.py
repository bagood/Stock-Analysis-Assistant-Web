import streamlit as st
from datetime import datetime

from webFunctions import web_functions
from modelFunctions import model_functions
from prepareDatasetFunctions import prepare_dataset_functions

wf = web_functions()
mf = model_functions()
pdf = prepare_dataset_functions()

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Stock Analysis Assistant', page_icon=":reminder_ribbon:", layout='centered')

with st.container():
    wf.add_center_text('Stock Analysis Assistant', 'h1')
    emiten, emiten_name = wf.choose_emiten()

with st.container():
    wf.add_center_text(f'{emiten_name}\'s Stock Data', 'h2')
    emiten_data = pdf.scrape_stock_price(emiten, start=datetime(2023, 1, 1), end=datetime.now())
    emiten_feature_data = pdf.create_feature_data(emiten_data)
    emiten_target_data = pdf.create_target_data(emiten_data, emiten_feature_data)

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

# with st.container():
#     try:
#         ### Parameters
#         epochs = 100
#         window_size = 5
#         split_index = round(len(emiten_feature_data) * 0.75)
#         train_feature, train_target, test_feature, test_target = mf.slice_dataset(emiten_feature_data, emiten_target_data, split_index)
#         train_feature = mf.window_feature_data(window_size, train_feature)
#         test_feature = mf.window_feature_data(window_size, test_feature, True)
#         train_target = mf.adjust_target_data(window_size, train_target)
#         test_target = mf.adjust_target_data(window_size, test_target)
#         forecast, actual = mf.model_forecast(emiten, test_feature, test_target)
#         wf.add_center_text(f'Price Forcast', 'h3')
#         fig_forecast = wf.visualize_forecast(forecast, actual, emiten_data)
#         st.plotly_chart(fig_forecast, use_container_width = True)

#     except:
#         wf.add_center_text(f'Sorry, the model is currently unavailable', 'h3')