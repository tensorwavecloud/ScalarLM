from enum import Enum


class TrainingJobStatus(str, Enum):
    QUEUED = "QUEUED"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
