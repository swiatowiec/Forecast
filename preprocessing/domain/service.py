from typing import List
from pydantic import BaseModel
from domain.dataset import FulfillmentMode
from domain.filler import MeanValueFiller, MedianValueFiller, LastValueFiller

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
        filler_metadata = None #TODO
        filler = self._determine_filler(
            fulfillment_mode=preprocessing_options.fulfillment_mode, filler_metadata=filler_metadata)
        def _determine_filler(self, fulfillment_mode: FulfillmentMode,
                    filler_metadata):
            if fulfillment_mode == FulfillmentMode.MEAN:
                return MeanValueFiller(metadata=filler_metadata)
            if fulfillment_mode == FulfillmentMode.MEDIAN:
                return MedianValueFiller(metadata=filler_metadata)
            elif fulfillment_mode == FulfillmentMode.LAST_VALUE:
                return LastValueFiller(metadata=filler_metadata)

        dataset.fulfill_missing_values(
            filler=filler, columns_to_fulfill=preprocessing_options.columns_to_fulfill)
            
        #TODO save metadata