from dependency_injector import providers, containers
from domain.service import ModelService
from infrastructure.model_manager import ModelManager
from domain.facade import TrainFacade
from training.infrastructure.file_manager import FileManager


class TrainContainer(containers.DeclarativeContainer):
    # Config
    config = providers.Configuration()

    # Services
    service = providers.Singleton(ModelService)

    # Infrastructure
    model_manager = providers.Singleton(ModelManager)
    file_manager = providers.Singleton(FileManager)

    # Facade
    training_facade = providers.Singleton(TrainFacade,
                                          service=service,
                                          file_manager=file_manager,
                                          model_manager=model_manager)