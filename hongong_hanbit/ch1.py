import pandas as pd
from global_functions import DataAnalyzer
import matplotlib.pyplot as plt

file_path = r'C:\Users\jstco\Downloads\hg-mldl-master\hg-mldl-master\fish.csv'

fish = pd.read_csv(file_path)
analysis = DataAnalyzer(fish)
# analysis.summarize_basic()

# Index(['Species', 'Weight', 'Length', 'Diagonal', 'Height', 'Width'], dtype='object')

# length = fish['Length']
# weight = fish['Weight']
#
# plt.scatter(length, weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# species = fish['Species'].unique()
# print(species)
fish_species = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']

fish_data_dict = {}

for species in fish_species:
    subset = fish[fish['Species'] == species]
    fish_data_dict[species] = {
        'length' : subset['Length'].tolist(),
        'weight' : subset['Weight'].tolist()
    }

bream_length = fish_data_dict['Bream']['length']
bream_weight = fish_data_dict['Bream']['weight']

# print(f'Bream Length : {bream_length}')
# print(f'Bream Weight : {bream_weight}')

