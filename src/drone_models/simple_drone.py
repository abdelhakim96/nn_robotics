import torch
from src.drone_model import DroneModel

class SimpleDrone(DroneModel):
    def __init__(self):
        mass = 1.0  # Example mass
        inertia = torch.tensor([0.1, 0.1, 0.1])  # Example inertia
        super().__init__(mass, inertia)
