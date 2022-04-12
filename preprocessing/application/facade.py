from infrastructure.dataset_repository import DatasetRepository
from domain.dataset_factory import DatasetFactory
from domain.service import PreprocessingOptions

class PreprocessingFitTransformFacade:

    def __init__(self,
                 dataset_repository: DatasetRepository,
                 ):
        self._dataset_repository = dataset_repository

    def fit_transform(self, args):
        dataset_factory = DatasetFactory(self._dataset_repository)
        dataset = dataset_factory.create_from_files(
            input_dir_path=args.input_dir_path)

        preprocessing_options = PreprocessingOptions(fulfillment_mode=args.fulfillment_mode,
                                            columns_to_fulfill=args.columns_to_fulfill,
                                            )

        metadata = self._preprocessing_service.preprocess(dataset=dataset,
                                                    preprocessing_options=preprocessing_options,
                                                )
        self._dataset_repository.save_dataset(
            dataset, output_dir_path=args.output_dir_path)

