from abc import ABC
import os.path
import json
from typing import Dict
from pathlib import Path


class AbstractMetadataRepository(ABC):
    def __init__(self, target_dir_path, metadata_file_name):
        self._target_dir_path = target_dir_path
        self._metadata_file_name = metadata_file_name

    def _get_file_name(self):
        return self._metadata_file_name

    def _get_full_path_to_file(self, run_name):
        return os.path.join(self._target_dir_path,
                            run_name,
                            self._get_file_name())

    def get_metadata(self, run_name: str):
        path = self._get_full_path_to_file(run_name=run_name)
        with open(path, 'r') as j:
            return json.loads(j.read())

    def save_metadata(self, metadata: Dict, run_name: str):
        path = self._get_full_path_to_file(run_name=run_name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as file:
            json.dump(metadata, file)


class MetadataRepository(AbstractMetadataRepository):
    def __init__(self,
                 target_dir_path: str = 'data/artifacts/metadata',
                 metadata_file_name: str = 'metadata.json'):
        super().__init__(target_dir_path=target_dir_path,
                         metadata_file_name=metadata_file_name)