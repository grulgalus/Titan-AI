import os
os.environ["DEV"] = "CPU"

from tinygrad.tensor import Tensor
from tinygrad.device import Device

# Nejmodernější způsob TinyGradu pro Termux:
Device.DEFAULT = "CPU"

class TitanLLM:
    def __init__(self):
        # Parametry sítě pro predikci znaků (8 znaků kontext)
        self.w1 = Tensor.uniform(8 * 256, 128)
        self.b1 = Tensor.zeros(128)
        self.w2 = Tensor.uniform(128, 256)
        self.b2 = Tensor.zeros(256)

    def forward(self, x):
        h = x.dot(self.w1).add(self.b1).relu()
        out = h.dot(self.w2).add(self.b2)
        return out

    def get_weights(self):
        return {"w1": self.w1.numpy(), "b1": self.b1.numpy(), "w2": self.w2.numpy(), "b2": self.b2.numpy()}

    def load_weights(self, state):
        self.w1.assign(state["w1"])
        self.b1.assign(state["b1"])
        self.w2.assign(state["w2"])
        self.b2.assign(state["b2"])
