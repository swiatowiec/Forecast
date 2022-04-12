from infrastructure.dataset_repository import DatasetRepository
from domain.dataset_factory import DatasetFactory
from domain.service import PreprocessingOptions, PreprocessingService
from infrastructure.metadata import MetadataRepository
from pydantic import BaseModel
from typing import List

class PreprocessingFitTransformArgs(BaseModel):
    input_dir_path: str
    output_dir_path: str
    fulfillment_mode: str
    columns_to_fulfill: List[str]

class PreprocessingFitTransformFacade:
    def __init__(self,
                 dataset_repository: DatasetRepository,
                 preprocessing_service: PreprocessingService,
                 metadata_repository: MetadataRepository,
                 ):
        self._dataset_repository = dataset_repository
        self._preprocessing_service = preprocessing_service
        self._metadata_repository = metadata_repository

    def fit_transform(self, args: PreprocessingFitTransformArgs):
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

        self._metadata_repository.save_metadata(
            metadata=metadata.to_dict(), run_name=args.run_name)

