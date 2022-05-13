from dependency_injector import providers, containers
from domain.facade import PreprocessingTransformFacade
from domain.service import PreprocessingService
from infrastructure.dataset_repository import DatasetRepository
from infrastructure.metadata import MetadataRepository


class TransformContainer(containers.DeclarativeContainer):
    # config
    config = providers.Configuration()

    # services
    preprocessing_service = providers.Singleton(PreprocessingService)
    metadata_repository = providers.Singleton(MetadataRepository)

    dataset_repository = providers.Singleton(
        DatasetRepository)

    # facade
    preprocessing_transform_facade = providers.Singleton(
        PreprocessingTransformFacade,
        metadata_repository=metadata_repository,
        preprocessing_service=preprocessing_service,
        dataset_repository=dataset_repository,)