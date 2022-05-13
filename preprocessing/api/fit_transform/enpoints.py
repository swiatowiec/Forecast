from dependency_injector.wiring import Provide, inject
from containers_config import FitTransformContainer
from domain.facade import PreprocessingFitTransformArgs


@inject
def perform_fit_transform(args: PreprocessingFitTransformArgs,
                          facade=Provide[FitTransformContainer.preprocessing_fit_transform_facade]):
    facade.fit_transform(args=args)