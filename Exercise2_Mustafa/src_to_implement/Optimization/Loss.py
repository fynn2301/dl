import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor

        label_index_array = np.where(label_tensor == 1, True, False)
        prediction_tensor_filtered = prediction_tensor[label_index_array]
        loss = np.sum(-np.log(prediction_tensor_filtered + np.finfo(float).eps))

        return loss

    def backward(self, label_tensor):
        error_tensor = np.divide(-label_tensor, self.prediction_tensor + np.finfo(float).eps)
        return error_tensor

