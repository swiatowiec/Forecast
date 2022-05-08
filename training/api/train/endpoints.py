from dependency_injector.wiring import Provide, inject
from training.containers_config import TrainContainer


@inject
def perform_training(train_dir_path,
                     n_splits,
                     look_back,
                     run_name,
                     artifacts_dir_path,
                     facade=Provide[TrainContainer.training_facade]):
    facade.train(train_dir_path=train_dir_path,
                 n_splits=n_splits,
                 look_back=look_back,
                 run_name=run_name,
                 artifacts_dir_path=artifacts_dir_path,
                 )