import pandas as pd
import glob
import json
import pickle


class FileManager:
    def read_dataset(self, input_dir_path):
        path_to_files = self.read_path = glob.glob(
            '{}/*.csv'.format(input_dir_path))
        for path in path_to_files:
            return pd.read_csv(path)

    def save_to_json(self, json_content, output_file):
        with open(output_file, 'w') as fd:
            return json.dump(json_content, fd)

    def read_from_json(self, file):
        with open(file, 'r') as fd:
            return json.load(fd)

    def save_to_pickle(self, obj, output_file):
        f = open(output_file, "wb")
        pickle.dump(obj, f)
        f.close()