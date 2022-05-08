import argparse
from domain.facade import PreprocessingFitTransformFacade, PreprocessingFitTransformArgs
from infrastructure.dataset_repository import DatasetRepository
from domain.service import PreprocessingService
from infrastructure.metadata import MetadataRepository
from config import PreprocessingConfig


#dev version
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_path', type=str, default='data/input')
    parser.add_argument('--output_dir_path', type=str,
                        default='data/preprocessed')
    parser.add_argument('--metadata_dir_path', type=str,
                        default='preprocessing/artifacts/metadata')
    parser.add_argument('--fulfillment_mode', type=str, help='')
    parser.add_argument('--columns_to_fulfill', type=str, help='')

    arg = parser.parse_args()
    args = PreprocessingFitTransformArgs(input_dir_path=arg.input_dir_path,
                                         output_dir_path=arg.output_dir_path,
                                         fulfillment_mode=arg.fulfillment_mode,
                                        columns_to_fulfill=arg.columns_to_fulfill.split(
                                             ',') if arg.columns_to_fulfill else None)

#prod version
    config = PreprocessingConfig('config.yaml')
    args =  args = PreprocessingFitTransformArgs(input_dir_path=config.input_dir_path,
                                         output_dir_path=config.output_dir_path,
                                         fulfillment_mode=config.fulfillment_mode,
                                        columns_to_fulfill=config.columns_to_fulfill.split(
                                             ',') if config.columns_to_fulfill else None)

    dataset_repository = DatasetRepository()
    preprocessing_service = PreprocessingService()
    metadata_repository = MetadataRepository(arg.metadata_dir_path)

    preprocess = PreprocessingFitTransformFacade(
        dataset_repository=dataset_repository,
        preprocessing_service=preprocessing_service,
        metadata_repository=metadata_repository)

    preprocess.fit_transform(args)