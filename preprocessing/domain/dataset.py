from enum import Enum
from typing import List

class Dataset:
    def __init__(self, df):
        self._data = df

    def get_data(self):
        return self._data

    def fulfill_missing_values(self,
                               filler,
                               columns_to_fulfill: List[str]):
        self._data[columns_to_fulfill] = filler.fill(
            self._data[columns_to_fulfill])


class FulfillmentMode(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    LAST_VALUE = 'last_value'