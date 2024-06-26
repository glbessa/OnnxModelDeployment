import onnxruntime
from abc import abstractmethod, ABCMeta

from utils import load_model

class BaseModel(metaclass=ABCMeta):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.model = load_model(onnx_path)

    @abstractmethod
    def __call__(self, inputs):
        """
            Run inference using model.
            Here it'll preprocess the inputs, run inference and postprocess the outputs.
        """
        raise NotImplementedError 

    @abstractmethod
    def preprocess(self, inputs):
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self, outputs):
        raise NotImplementedError