"""High-level jobs for the project."""

# %% IMPORTS

from qto_categorizer_ml.jobs.inference import InferenceJob
from qto_categorizer_ml.jobs.training import TrainingJob
from qto_categorizer_ml.jobs.transition import TransitionJob
from qto_categorizer_ml.jobs.tuning import TuningJob

# %% TYPES

JobKind = TuningJob | TrainingJob | TransitionJob | InferenceJob

# %% EXPORTS

__all__ = ["InferenceJob", "TrainingJob", "TransitionJob", "TuningJob", "JobKind"]
