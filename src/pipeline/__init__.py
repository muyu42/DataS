from .score_pipeline import score_pipeline_deita
from .score_pipeline import score_pipeline_ifd
from .score_pipeline import score_pipeline_lenth
from .score_pipeline import score_pipeline_rw

from .base import PipelineRegistry
from typing import Callable

PipelineRegistry.register("score_pipeline_deita", score_pipeline_deita.deita)
PipelineRegistry.register("score_pipeline_rw", score_pipeline_rw.ScorePipeline)
PipelineRegistry.register("score_pipeline_ifd", score_pipeline_ifd.ScorePipeline)
PipelineRegistry.register("score_pipeline_lenth", score_pipeline_lenth.ScorePipeline)

class Pipeline:
    
    def __new__(cls, name, **kwargs) -> Callable:
        
        PipelineClass = PipelineRegistry.get_pipeline(name)
        return PipelineClass(name, **kwargs)