from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime
from preprocessing.containers_config import TransformContainer
from domain.facade import PreprocessingTransformFacade

router = APIRouter(prefix="/preprocessing")


class Measurement(BaseModel):
    run_name: str
    date: datetime
    rainfall: float
    depth_to_groundwater: float
    temperature: float
    temperature: float
    river_hydrometry: float


class PreprocessTransformResponse(BaseModel):
    measurements: List[Measurement]


class PreprocessTransformRequest(BaseModel):
    measurements: List[Measurement]


@router.post("/transform",
             summary="Preprocess measurements",
             description="Preprocess measurements",
             response_model=PreprocessTransformResponse)
@inject
async def transform(request: PreprocessTransformRequest,
                    run_name: str,
                    preprocessing_transform_facade: PreprocessingTransformFacade = Depends(
                        Provide[TransformContainer.preprocessing_transform_facade])) -> PreprocessTransformResponse:
    results = preprocessing_transform_facade.transform(measurements=[m.dict() for m in request.measurements],
                                                       run_name=run_name,
                                                       )
    return PreprocessTransformResponse(measurements=[Measurement.parse_obj(r) for r in results])