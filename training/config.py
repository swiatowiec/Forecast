import yaml

class TrainingConfig:
    def __init__(self, config_file):
        f = open(config_file, 'r')
        params = yaml.safe_load(f)
        self.train_dir_path=params['training']['train_dir_path']
        self.n_splits=params['training']['n_splits']
        self.look_back=params['training']['look_back']
        self.run_name=params['training']['run_name']
        self.artifacts_dir_path=params['training']['artifacts_dir_path']