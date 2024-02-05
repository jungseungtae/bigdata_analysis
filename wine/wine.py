import csv
import os
import pandas as pd

# 경로 설정
os.chdir(r'C:\Users\jstco\Downloads\wine')  # 사용자의 실제 경로로 변경

# 데이터 파일 읽기
with open('wine.data', 'r') as f:
   reader = csv.reader(f, delimiter=',')
   data = list(reader)[1:]  # 첫 번째 행 제거

# 이름 파일 읽기
with open('wine.names', 'r') as f:
   reader = csv.reader(f, delimiter=',')
   names = list(reader)[0]  # 첫 번째 행만 가져오기

# DataFrame 생성
df = pd.DataFrame(data, columns=names)

# 처음 5개 행 출력
print(df.head())