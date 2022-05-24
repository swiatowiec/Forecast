from domain.facade import PreprocessingFitTransformFacade, PreprocessingFitTransformArgs
from infrastructure.dataset_repository import DatasetRepository
from domain.service import PreprocessingService
from infrastructure.metadata import MetadataRepository
from config import PreprocessingConfig


if __name__ == '__main__':
    config = PreprocessingConfig('config.yaml')
    args = PreprocessingFitTransformArgs(input_dir_path=config.input_dir_path,
                                         output_dir_path=config.output_dir_path,
                                         fulfillment_mode=config.fulfillment_mode,
                                        columns_to_fulfill=config.columns_to_fulfill.split(
                                             ',') if config.columns_to_fulfill else None)

    dataset_repository = DatasetRepository()
    preprocessing_service = PreprocessingService()
    metadata_repository = MetadataRepository(config.metadata_dir_path)

    preprocess = PreprocessingFitTransformFacade(
        dataset_repository=dataset_repository,
        preprocessing_service=preprocessing_service,
        metadata_repository=metadata_repository)

    preprocess.fit_transform(args)