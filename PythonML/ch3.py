##### 제 3장. 기본 데시보드 만들기 #####

import pandas as pd
from global_functions import DataAnalyzer

file_path = r'C:\Users\jstco\Downloads\PythonML-main\PythonML-main\chapter03'

m_store = pd.read_csv(file_path + '/m_store.csv')
m_area = pd.read_csv(file_path + '/m_area.csv')

order_data = pd.read_csv(file_path + '/tbl_order_202104.csv')
order_data = pd.merge(order_data, m_store, on = 'store_id', how = 'left')
order_data = pd.merge(order_data, m_area, on = 'area_cd', how = 'left')

takeout_mapping = {0: 'delevery', 1: 'takeout'}
status_mapping = {0: '주문접수', 1: '결제완료', 2: '배달완료', 9: '주문취소'}

order_data['takeout_name'] = order_data['takeout_flag'].map(takeout_mapping)
order_data['status_name'] = order_data['status'].map(status_mapping)

# print(order_data.head())

# from ipywidgets import Dropdown
#
# def order_by_store(val):
#     clear_output()
#     display(dropdown)
#     pick_data = order_data.loc[(order_data['store_name'] == val['new']) &
#                                (order_data['status'].isin([1, 2]))]
#     display(pick_data.head())
#
# store_list = m_store['store_name'].tolist()
#
# dropdown = Dropdown(options = store_list, description = '지역 선택 : ')
# dropdown.observe(order_by_store, names = 'value')
# display(dropdown)

### pycharm에서 제공되지 않음. ###