import argparse
from training.api.train.endpoins import perform_training
from training.containers_config import TrainContainer
import sys
from config import TrainingConfig

if __name__ == '__main__':
    container = TrainContainer()
    container.wire(modules=[sys.modules[__name__]])

    config = TrainingConfig('config.yaml')
    perform_training(train_dir_path=config.train_dir_path,
                    artifacts_dir_path=config.artifacts_dir_path,
                     n_splits=config.n_splits,
                     look_back=config.look_back,
                     run_name=config.run_name,
                     )