
from dataclasses import dataclass

import enum

class FeatureSet(enum.Enum):
    small: str = 'small'
    medium: str = 'medium'
    all: str = 'all'