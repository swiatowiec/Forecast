import argparse
from training.api.train.endpoins import perform_training
from training.containers_config import TrainContainer
import sys

if __name__ == '__main__':
    container = TrainContainer()
    container.wire(modules=[sys.modules[__name__]])

    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--train_dir_path', type=str, default='data/train')
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--look_back', type=int, default=5)
    parser.add_argument('--run_name', type=str, default='lookback_test')
    parser.add_argument('--artifacts_dir_path', type=str,
                        default='data/artifacts')
    args = parser.parse_args()

    perform_training(train_dir_path=args.train_dir_path,
                     n_splits=args.n_splits,
                     look_back=args.look_back,
                     run_name=args.run_name,
                     artifacts_dir_path=args.artifacts_dir_path,
                     )