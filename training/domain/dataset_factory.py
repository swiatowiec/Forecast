from training.infrastructure.file_manager import FileManager
from domain.dataset import Dataset


class DatasetFactory:
    def __init__(self, file_manager: FileManager):
        self._file_manager = file_manager

    def create_from_files(self, train_dir_path, n_splits):
        train_df = self._file_manager.read_dataset(
            input_dir_path=train_dir_path)
        return Dataset(train_df, n_splits)