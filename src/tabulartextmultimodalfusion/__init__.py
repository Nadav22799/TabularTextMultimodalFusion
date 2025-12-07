"""
TabularTextMultimodalFusion - A framework for multimodal fusion of tabular and text data.
"""

__version__ = "0.1.5"

# Import main modules for easy access
from . import models
from . import dataset
from . import optimization
from . import settings

__all__ = [
    'models',
    'dataset',
    'optimization',
    'settings',
]
