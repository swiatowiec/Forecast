class Dataset:
    def __init__(self, df):
        self._data = df

    def get_data(self):
        return self._data