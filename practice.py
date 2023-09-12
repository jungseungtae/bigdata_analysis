import seaborn as sns
import os

# tips = sns.load_dataset('tips')
# print(tips.head(5))

import openpyxl
import win32com.client as wc

file_path = r'C:\Users\jstco\OneDrive\바탕 화면\통합문서1.xlsx'

workbook = openpyxl.load_workbook(file_path)