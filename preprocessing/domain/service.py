from typing import List
from pydantic import BaseModel
from domain.dataset import FulfillmentMode

class PreprocessingOptions(BaseModel):
    fulfillment_mode: FulfillmentMode
    columns_to_fulfill: List[str]

class PreprocessingService:
    def preprocess(self,
                   dataset,
                   preprocessing_options: PreprocessingOptions,
    ):
        if dataset is None:
            raise Exception("The dataset is empty")