import numpy as np
import base64
from .logging import log_message

class Segment:
    def __init__(self, vector, path, url):
        """
        Initialize a Segment object with a 128-float vector and a 2D numpy array.
        """
        if len(vector) != 128:
            raise ValueError("Vector must be 128 floats long.")
        self.vector = vector
        self.path = self.encode_path(path)
        self.url = url

    def encode_path(self, path):
        """
        Convert numpy array to bytes and then encode it to base64 string.
        """
        # Convert numpy array to bytes
        path_bytes = path.tobytes()
        # Encode the bytes to base64 string
        path_base64 = base64.b64encode(path_bytes).decode('utf-8')
        log_message('info', f'{path_base64=}')
        return path_base64

    @classmethod
    def get_path(cls, encoded_path):
        """
        Decode the base64 string back into a numpy array.
        """
        # Decode the base64 string into bytes
        path_bytes = base64.b64decode(encoded_path)
        # Convert the bytes back into a numpy array
        path = np.frombuffer(path_bytes, dtype=np.float64)  # Use the original dtype of the array
        return path
       
    def to_dict(self):
        """
        Convert the Segment object to a dictionary for MongoDB storage.
        """
        return {
            "vector": self.vector,
            "path": self.get_path(self.path).tolist(),
            "url": self.url
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Segment object from a dictionary retrieved from MongoDB.
        """
        path = cls.get_path(data["path"])
        url = data["url"]
        return cls(vector=data["vector"], path=path, url=url)
