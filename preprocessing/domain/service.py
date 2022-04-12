from typing import List, Dict, Optional
from pydantic import BaseModel
from domain.dataset import FulfillmentMode
from domain.filler import MeanValueFiller, MedianValueFiller, LastValueFiller

class PreprocessingOptions(BaseModel):
    fulfillment_mode: FulfillmentMode
    columns_to_fulfill: List[str]

class FillerMetadata(BaseModel):
    filler_type: str
    filler_value: Dict[str, float]

class Metadata(BaseModel):
    filler_metadata: FillerMetadata

    def to_dict(self):
        return {
            'filler_metadata': {'filler_type': self.filler_metadata['filler_type'].value, 'filler_value': self.filler_metadata['filler_value']}}

class PreprocessingService:
    def preprocess(self,
                   dataset,
                   preprocessing_options: PreprocessingOptions,
                   metadata: Optional[Metadata],
    ):
        if dataset is None:
            raise Exception("The dataset is empty")
        filler_metadata = metadata.filler_metadata if metadata is not None else None
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

        return Metadata(filler_metadata=filler.metadata())
