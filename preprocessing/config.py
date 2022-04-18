import yaml

class PreprocessingConfig:
    def __init__(self, config_file):
        f = open(config_file, 'r')
        params = yaml.safe_load(f)
        self.input_dir_path=params['preprocessing']['input_dir_path']
        self.output_dir_path=params['preprocessing']['output_dir_path']
        self.metadata_dir_path=params['preprocessing']['metadata_dir_path']
        self.fulfillment_mode=params['preprocessing']['fulfillment_mode']
        self.columns_to_fulfill=params['preprocessing']['columns_to_fulfill']