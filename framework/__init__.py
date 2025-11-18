"""
Modular Training Framework for Surgical Action Recognition
Provides unified interface for training, testing, evaluation, visualization, and reporting
"""

__version__ = "1.0.0"
__author__ = "Modular Training Framework"

from .config.base_config import ExperimentConfig
from .core.trainer import Trainer
from .core.tester import Tester
from .core.evaluator import Evaluator
from .core.experiment import ExperimentManager
from .visualization.training_viz import TrainingVisualizer
from .reporting.report_generator import ReportGenerator

__all__ = [
    'ExperimentConfig',
    'Trainer',
    'Tester',
    'Evaluator',
    'ExperimentManager',
    'TrainingVisualizer',
    'ReportGenerator',
]
