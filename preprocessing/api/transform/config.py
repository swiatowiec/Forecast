from functools import lru_cache
from pydantic import BaseSettings


class PreprocessTransformServerSettings(BaseSettings):
    host: str
    port: int


@lru_cache()
def get_preprocess_transform_server_settings():
    return PreprocessTransformServerSettings(host="127.0.0.1", port=8000)