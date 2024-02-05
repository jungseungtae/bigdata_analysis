
##### 제 1장. 분석을 위한 준비 테크닉 #####

import pandas as pd
from global_functions import DataAnalyzer



### 1. 데이터 불러오기
# file_path = r'C:\Users\jstco\Downloads\PythonML-main\PythonML-main\chapter01'
# m_store = pd.read_csv(file_path + '/m_store.csv')
#
# analyzer = DataAnalyzer(m_store)
# analyzer.summarize_basic()

# m_area = pd.read_csv(file_path + '/m_area.csv')
# # print(m_area.head())
#
# tbl_order_4 = pd.read_csv(file_path + '/tbl_order_202104.csv')
# # print(tbl_order_4.head())
# # print(len(tbl_order_4))
#
#
# ### 2. 데이터 결합하기
# tbl_order_5 = pd.read_csv(file_path + '/tbl_order_202105.csv')
# # print(len(tbl_order_5))
#
# order_all = pd.concat([tbl_order_4, tbl_order_5], ignore_index= True)
# # print(len(order_all))
# # print(order_all.head())
#
# import os
#
# ## 현재 경로 찾기
# # current_dir = os.getcwd()
# # print(current_dir)
# file_list = os.listdir(file_path)
# # print(file_list)
#
# ## 파일 찾기
# tbl_order_file = os.path.join(file_path, 'tbl_order_*.csv')
# # print(tbl_order_files)
#
# import glob
# tbl_order_files = glob.glob(tbl_order_file)
# # print(tbl_order_files)
#
# ### 3. 여러 데이터 결합하기
# ## 데이터 처리 테스트
# order_all = pd.DataFrame()
# # file = tbl_order_files[0]
# # order_data = pd.read_csv(file)
# # print(f'{file} : {len(order_data)}')
#
# ## 모든 데이터 결합
# for file in tbl_order_files:
#     order_data = pd.read_csv(file)
#     # print(f'{file} : {len(order_data)}')
#     order_all = pd.concat([order_all, order_data],
#                           ignore_index = True)
#
# # print(order_all)
#
# ### 4. 통계량 확인
# null_sum = order_all.isnull().sum()
# # print(null_sum)
#
# # analyzer = DataAnalyzer(order_all)
# # analyzer.summarize_basic()
#
# order_all['total_amount'].describe()
# # print(order_all)
#
# ### 5. 데이터 전처리 및 결합(조인)
# order_data = order_all.loc[order_all['store_id'] != 999]
# # print(order_data)
#
# order_data = pd.merge(order_data,
#                       m_store,
#                       on = 'store_id',
#                       how = 'left')
# # print(order_data)
#
# order_data = pd.merge(order_data,
#                       m_area,
#                       on = 'area_cd',
#                       how = 'left')
# # print(order_data)
#
# # print(order_data.dtypes)
#
# ### 6. 코드이름 설정
# order_data['takeout_name'] = ''
# order_data.loc[order_data['takeout_flag'] == 0, 'takeout_name'] = 'delivery'
# order_data.loc[order_data['takeout_flag'] == 1, 'takeout_name'] = 'takeout'
#
# # print(order_data)
#
# order_data['status_name'] = ''
# # order_data.loc[order_data['status'] == 0, 'status_name'] = '주문 접수'
# # order_data.loc[order_data['status'] == 1, 'status_name'] = '결제 완료'
# # order_data.loc[order_data['status'] == 2, 'status_name'] = '배달 완료'
# # order_data.loc[order_data['status'] == 9, 'status_name'] = '주문 취소'
#
# status_mapping = {0: '주문 접수', 1: '결제 완료', 2: '배달 완료', 9: '주문 취소'}
# order_data['status_name'] = order_data['status'].map(status_mapping)
#
# # print(order_data['status_name'])
#
# ### 7. 분석 기초 테이블 출력
# output_dir = os.path.join(file_path, 'output_data')
# os.makedirs(output_dir, exist_ok = True)

# output_file = os.path.join(output_dir, 'order_data.csv')
# order_data.head(500).to_csv(output_file, index = False, encoding = 'UTF-8')
# print('output complete')

# ================================================================= #
## 마무리 정리하기 ##
import os
import glob

# 파일 불러오기
m_store = pd.read_csv(file_path + '/m_store.csv')
m_area = pd.read_csv(file_path + '/m_area.csv')

tbl_order_file = os.path.join(file_path, 'tbl_order_*.csv')
tbl_order_files = glob.glob(tbl_order_file)
print(tbl_order_files)

# 데이터 결합
order_all = pd.DataFrame()

for file in tbl_order_files:
    order_data = pd.read_csv(file)
    print(f'{file} : {len(order_data)}')
    order_all = pd.concat([order_all, order_data], ignore_index=True)

order_data = order_all.loc[order_all['store_id'] != 999]

order_data = pd.merge(order_data, m_store, on='store_id', how='left')
order_data = pd.merge(order_data, m_area, on='area_cd', how='left')

## 데이터 이름 변환
order_data['takeout_name'] = ''
order_data['status_name'] = ''

takeout_mapping = {0: 'delivery', 1: 'takeout'}
status_mapping = {0: '주문접수', 1: '결제완료', 2: '배달완료', 9: '주문취소'}

order_data['takeout_name'] = order_data['takeout_flag'].map(takeout_mapping)
order_data['status_name'] = order_data['status'].map(status_mapping)

## 파일 저장
output_dir = os.path.join(file_path, 'output_data2')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'order_data3.csv')
order_data.head(500).to_csv(output_file, index=False, encoding='utf-8-sig')
print('output completed')
