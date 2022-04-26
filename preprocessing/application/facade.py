from infrastructure.dataset_repository import DatasetRepository
from domain.dataset_factory import DatasetFactory
from domain.service import PreprocessingOptions, PreprocessingService
from infrastructure.metadata import MetadataRepository
from pydantic import BaseModel
from typing import List, Dict
from domain.service import Metadata

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

class PreprocessingTransformFacade:
    def __init__(self,
                 metadata_repository: MetadataRepository,
                 preprocessing_service: PreprocessingService,
                 dataset_repository: DatasetRepository,
                 ):
        self._metadata_repository = metadata_repository
        self._preprocessing_service = preprocessing_service
        self._dataset_repository = dataset_repository

    def transform(self,
                  measurements: List[Dict],
                  run_name: str):

        measurements_series = DatasetFactory.create_from_dict(measurements)

        metadata_do = Metadata.from_dict(self._metadata_repository.get_metadata(
            run_name=run_name))

        preprocessing_options = PreprocessingOptions(fulfillment_mode=metadata_do.filler_metadata['filler_type'], 
                                                    columns_to_fulfill=list(metadata_do.filler_metadata[
                                                                                        'filler_value'].keys()),
                                                    )

        self._preprocessing_service.preprocess(dataset=measurements_series,
                        preprocessing_options=preprocessing_options,
                        metadata_do=metadata_do,
                        )                              