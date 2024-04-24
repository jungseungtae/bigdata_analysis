from matplotlib import pyplot as plt

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    ## 데이터탐색
    def summarize_basic(self):

        methods = ['info', 'describe', 'head']
        attributes = ['columns', 'dtypes', 'shape']

        for method in methods:
            print(f"{'-' * 10} {method.upper()} {'-' * 50}")
            print(getattr(self.data, method)())
            print('\n')

        for attr in attributes:
            print(f"{'-' * 10} {attr.upper()} {'-' * 50}")
            print(getattr(self.data, attr))
            print('\n')

        print(f"{'-' * 10} ISNULL {'-' * 50}")
        print(self.data.isnull().sum())


    # 데이터 평균('집계기준', '집계값)
    def groupby_means(self, criteria, value):
        print(self.data.groupby(criteria)[value].mean())

    def groupby_mean_plot(self, criteria, value):
        plot = self.data.groupby(criteria)[value].mean()
        plot.plot()
        plt.show()