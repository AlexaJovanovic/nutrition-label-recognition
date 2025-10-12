from abc import ABC, abstractmethod
import numpy as np

class ImagePreprocessor(ABC):
    """
    Abstract image preprocessing pipeline interface.
    Implementations can include thresholding, denoising, rotation correction, etc.
    """
    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Perform preprocessing and return the processed image.
        """
        pass
