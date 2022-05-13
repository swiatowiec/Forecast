from fastapi import FastAPI
from preprocessing.api.transform import endpoints
from preprocessing.api.transform.config import get_preprocess_transform_server_settings
from preprocessing.containers_config import TransformContainer

settings = get_preprocess_transform_server_settings()

container = TransformContainer()
container.wire(modules=[endpoints])


app = FastAPI()
app.container = container
app.include_router(endpoints.router)