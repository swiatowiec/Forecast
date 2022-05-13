import uvicorn
from fastapi.logger import logger as fastapi_logger
from api.transform.config import get_preprocess_transform_server_settings

if __name__ == "__main__":
    settings = get_preprocess_transform_server_settings()
    fastapi_logger.info('****************** Starting Server *****************')
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s"
    log_config["formatters"]["default"][
        "fmt"] = "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s"
    uvicorn.run(
        "preprocessing.api.transform.main:app",
        host=settings.host,
        port=settings.port,
        log_config=log_config
    )
