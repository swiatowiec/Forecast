import glob
import pandas as pd

class DatasetRepository:
    def read_dataset(self, input_dir_path):
        usecols = ['Date',
                   'Rainfall_Bastia_Umbra',
                   'Depth_to_Groundwater_P25',
                   'Temperature_Bastia_Umbra',
                   'Volume_C10_Petrignano',
                   'Hydrometry_Fiume_Chiascio_Petrignano',
                   ]
        path_to_files = self.read_path = glob.glob(
            '{}/*.csv'.format(input_dir_path))
        for path in path_to_files:
            return pd.read_csv(path, usecols=usecols)