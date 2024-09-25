from enum import Enum


class PipelineLevel(Enum):
    SEGMENTATION = 'Segmentation'
    FEATURE_EXTRACTION = 'FeatureExtraction'
    DIMENSIONALITY_REDUCTION = 'DimensionalityReduction'
    CLUSTERING = 'Clustering'


class WorkItem:
    """
    WorkItem tracks the progress of an object moving through pipelines.
    
    Attributes:
    ----------
    status : str or None
        Current level of the work item in the pipeline.
    body : dict
        Holds key-value pairs containing information about the work item.
    levels : set
        A set of valid levels in the pipeline.
    """

    def __init__(self, status=None, body=None):
        self._status = status
        self.body = body if body is not None else {}
        self.levels = {level.value for level in PipelineLevel}

    @property
    def status(self):
        """Get the current status."""
        return self._status

    @status.setter
    def status(self, level):
        """Set the status if the level is valid, else raise ValueError."""
        if level not in self.levels:
            raise ValueError(f'Invalid level: {level}. Available levels are: {self.levels}')
        self._status = level

    def update_status(self, level):
        """Update the status to the specified level."""
        self.status = level

    def set_attribute(self, key, value):
        """Set a key-value pair in the body dictionary."""
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self.body[key] = value

    def get_attribute(self, key, default=None):
        """
        Get the value for the specified key from the body dictionary.
        If the key is not present, return the default value.
        """
        return self.body.get(key, default)

    def __repr__(self):
        return f"<WorkItem(status={self.status}, body={self.body})>"
