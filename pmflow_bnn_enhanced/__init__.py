from .version import __version__
from .pmflow import (
    ParallelPMField, 
    VectorizedLateralEI, 
    AdaptiveScheduler, 
    vectorized_pm_plasticity,
    # Enhanced v0.3.0 features for Lilith
    MultiScalePMField,
    AttentionGatedPMField,
    EnergyBasedPMField,
    contrastive_plasticity,
    batch_plasticity_update,
    hybrid_similarity,
)
from .bnn import TemporalPipelineBNN, MultiGPUPMBNN, PMBNNAlwaysPlasticV2
from .factory import get_model_v2, get_performance_config, benchmark_temporal_parallelism, validate_embarrassingly_parallel_scaling
from .utils import create_training_config, get_hardware_info, optimize_for_device
from .evaluation import PMFlowEvaluator, create_meaningful_benchmark_suite
from .baselines import CNNBaseline, GRUBaseline, MLPBaseline, PMFlowCNN

__all__ = [
    '__version__',
    'ParallelPMField',
    'VectorizedLateralEI', 
    'AdaptiveScheduler',
    'vectorized_pm_plasticity',
    # Enhanced v0.3.0
    'MultiScalePMField',
    'AttentionGatedPMField',
    'EnergyBasedPMField',
    'contrastive_plasticity',
    'batch_plasticity_update',
    'hybrid_similarity',
    # BNN models
    'TemporalPipelineBNN',
    'MultiGPUPMBNN',
    'PMBNNAlwaysPlasticV2',
    # Factory
    'get_model_v2',
    'get_performance_config',
    'benchmark_temporal_parallelism',
    'validate_embarrassingly_parallel_scaling',
    # Utils
    'create_training_config',
    'get_hardware_info',
    'optimize_for_device',
    # Evaluation
    'PMFlowEvaluator',
    'create_meaningful_benchmark_suite',
    # Baselines
    'CNNBaseline',
    'GRUBaseline',
    'MLPBaseline',
    'PMFlowCNN',
]