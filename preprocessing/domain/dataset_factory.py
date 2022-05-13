from infrastructure.dataset_repository import DatasetRepository
from domain.dataset import Dataset
import pandas as pd

class DatasetFactory:
    def __init__(self, dataset_repository: DatasetRepository):
        self._dataset_repository = dataset_repository

    def create_from_files(self, input_dir_path):
        df = self._dataset_repository.read_dataset(input_dir_path)
        df.columns = ['date', 'rainfall', 'depth_to_groundwater',
                      'temperature', 'temperature', 'river_hydrometry']
        return Dataset(df)

    def create_from_dict(measurements):
        df = pd.DataFrame(measurements)
        return Dataset(df)