from infrastructure.dataset_repository import DatasetRepository
from domain.dataset_factory import DatasetFactory

class PreprocessingFitTransformFacade:

    def __init__(self,
                 dataset_repository: DatasetRepository,
                 ):
        self._dataset_repository = dataset_repository

    def fit_transform(self, args):
        dataset_factory = DatasetFactory(self._dataset_repository)
        dataset = dataset_factory.create_from_files(
            input_dir_path=args.input_dir_path)