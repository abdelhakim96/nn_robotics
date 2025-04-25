import importlib
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.func import jvp
from torch.nn import Linear, Softplus, Sequential, Parameter, Module, LayerNorm
from torch.nn.init import xavier_uniform_, zeros_

from conflictfree.grad_operator import ConFIGOperator
from conflictfree.utils import get_gradient_vector, apply_gradient_vector
from conflictfree.length_model import TrackHarmonicAverage
# we go with this length_model because it works the best with more losses

operator = ConFIGOperator(length_model=TrackHarmonicAverage()) # initialize operator

from data.data_utility import TrajectoryDataset
from src.drone_models.new_drone import NewDrone

class DNN(Module):
    def __init__(self, N_in, N_out, N_h, N_layer, activation=Softplus):
        super().__init__()
        layers = []
        for i in range(N_layer):
            if i == 0:
                layers.append(Linear(N_in, N_h[i]))
            else:
                layers.append(Linear(N_h[i-1], N_h[i]))
            layers.append(activation())
        layers.append(Linear(N_h[-1], N_out))
        self.net = Sequential(*layers)

    def forward(self, z):
        return self.net(z)

def convert_input_data(X: Tensor, U: Tensor, time: Tensor):
    """
    Converts the inputs for the model by concatenating state, control input, and time tensors.

    Args:
        X (Tensor): State tensor of shape [N_batch, N_seq, N_x].
        U (Tensor): Control input tensor of shape [N_batch, N_seq, N_u].
        time (Tensor): Time tensor of shape [N_batch, N_seq, 1].

    Returns:
        Tuple[Tensor, int, int, int]: Flattened input tensor Z, batch size N_batch, sequence length N_seq, number of state variables N_x.
    """
    N_batch, N_seq, N_x = X.shape
    # Ensure time has the correct shape [N_batch, N_seq, 1] before concatenating
    if time.dim() == 2: # If time is [N_batch, N_seq], add the last dimension
        time = time.unsqueeze(-1)
    elif time.dim() != 3 or time.shape[-1] != 1:
         raise ValueError(f"Time tensor must have shape [N_batch, N_seq, 1], but got {time.shape}")

    # Slice X to remove the last 3 features (distance)
    X_sliced = X[:, :, :12]
    Z = torch.cat((X_sliced, U, time), dim=2) # Remove extra unsqueeze
    N_in = Z.shape[2]
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, N_batch, N_seq, N_x

def convert_input_collocation(X: Tensor, U: Tensor, t_coll: Tensor):
    """
    Converts inputs for evaluating the model at collocation points.

    Args:
        X (Tensor): State tensor of shape [N_batch, N_seq, N_x]
        U (Tensor): Control input tensor of shape [N_batch, N_seq, N_u]
        t_coll (Tensor): Collocation time tensor of shape [N_batch, N_seq, N_coll]

    Returns:
        Tuple[Tensor, Tensor]: Flattened input tensor Z, expanded control input U_coll
    """
    N_coll = t_coll.shape[2]

    # Expand X and U to match the collocation points
    X_coll = X.unsqueeze(2).expand(-1, -1, N_coll, -1)  # Shape: [N_batch, N_seq, N_coll, N_x]
    X_coll = X_coll[:, :, :, :12] # Slice X_coll to remove the last 3 features (distance)
    U_coll = U.unsqueeze(2).expand(-1, -1, N_coll, -1).contiguous()  # Shape: [N_batch, N_seq, N_coll, N_u]

    # Concatenate inputs
    Z = torch.cat((X_coll, U_coll, t_coll.unsqueeze(3)), dim=3)  # Shape: [N_batch, N_seq, N_coll, N_in]
    N_in = Z.shape[3]

    # Flatten Z
    Z = Z.view(-1, N_in)
    Z.requires_grad_()
    return Z, U_coll

def convert_output_data(X_hat, N_batch, N_seq, N_out):
    """
    Reshapes the model's output back into trajectories.

    Args:
        X_hat (Tensor): Flattened model output tensor of shape [N_batch * N_seq, N_out].
        N_batch (int): Batch size.
        N_seq (int): Sequence length.
        N_out (int): Number of output variables (states).

    Returns:
        Tensor: Reshaped output tensor of shape [N_batch, N_seq, N_out].
    """
    return X_hat.view(N_batch, N_seq, N_out)

def compute_time_derivatives(Z_1, N_in, model):
    """
    Computes time derivatives using forward-mode automatic differentiation (jvp).

    Args:
        Z_1 (Tensor): Input tensor.
        N_in (int): Total input size.
        model (Module): Neural network model.

    Returns:
        Tuple[Tensor, Tensor]: Model output and its time derivative.
    """
    # Direction vector for time input
    v = torch.zeros_like(Z_1)
    v[:, N_in-1] = 1.0  # Set 1.0 for time input derivative
    
    X_2_hat, dX_2_hat_dt = jvp(model, (Z_1,), (v,))
    return X_2_hat, dX_2_hat_dt  # Shape: [N_total, N_x]

def compute_physics_loss(X_2_hat_coll_flat, dX2_hat_dt_flat, U_coll_flat):
    """
    Computes the physics-based loss at collocation points.

    Args:
        X_2_hat_coll_flat (Tensor): Flattened predicted states.
        dX2_hat_dt_flat (Tensor): Flattened time derivatives of predicted states.
        U_coll_flat (Tensor): Flattened control inputs.

    Returns:
        Tensor: Physics loss.
    """
    # Extract state and control input
    state = X_2_hat_coll_flat
    control_input = U_coll_flat

    # Load the new drone model with mass and inertia
    # Using default mass from NewDrone and placeholder inertia
    # Ensure inertia tensor is on the same device as the state tensor
    drone_model = NewDrone(mass=3.8, inertia=torch.eye(3, device=X_2_hat_coll_flat.device))

    # Compute the predicted state derivative using the new drone model
    next_state = drone_model.forward(state, control_input)

    # Compute the mean squared residuals
    l_phy = ((dX2_hat_dt_flat - next_state)**2).mean()

    return l_phy

def data_loss_fn(model, X_1, U, time, device, N_out, noise_level = 0.0): # Added N_out
        """
        Calculates the 1-step ahead prediction loss.
        """
        N_batch, N_seq, N_x = X_1.shape
        Z_1, _, _, _ = convert_input_data(X_1[:, :-1, :] + torch.normal(0, noise_level, X_1[:, :-1, :].shape, device=device), U[:, :-1, :], time[:, :-1].unsqueeze(-1))
        Z_1 = Z_1.to(device)
        X_2_hat = model(Z_1)
        X_2_hat = convert_output_data(X_2_hat, N_batch, N_seq - 1, N_out)
        X_2_hat = X_2_hat[:, :, :12]
        
        l_data = mse_loss(X_2_hat, X_1[:, 1:, :12]) # output, target
        return l_data
    

def rollout_loss_fn(model, X_0, U, time, N_roll: int, device, t_coll, pinn, N_out, noise_level = 0.0, model_name = 'drone'): # Added N_out
    """
    Computes the rollout loss over multiple steps.

    Args:
        model (Module): Neural network model.
        X_0 (Tensor): Initial states.
        U (Tensor): Control inputs.
        time (Tensor): Time tensor.
        N_roll (int): Number of rollout steps.
        device (torch.device): Device to run computations on.
        t_coll (Tensor): Collocation times.
        pinn (bool): Flag to include physics-informed loss.
        noise_level (float): Noise level for data augmentation.
        model_name (str): Name of the model to use for physics loss calculation.

    Returns:
        Tuple[Tensor, Tensor]: Rollout loss and physics loss.
    """
    N_seq = X_0.shape[1]
    N_seq_slice = N_seq - N_roll
    X_hat = X_0[:, :N_seq_slice, :]    
    l_roll = 0
    l_phy = 0

    # Ensure time tensor is 3D before the loop
    if time.dim() == 2:
        time = time.unsqueeze(-1) # Shape becomes [N_batch, N_seq, 1]
    elif time.dim() != 3 or time.shape[-1] != 1:
         raise ValueError(f"Rollout time tensor must have shape [N_batch, N_seq] or [N_batch, N_seq, 1], but got {time.shape}")

    for i in range(N_roll):
        # Slice the 3D time tensor correctly
        time_slice = time[:, i:i+N_seq_slice, :]
        Z_0, N_batch, N_seq, N_x = convert_input_data(X_hat + torch.normal(0, noise_level, X_hat.shape, device=device), U[:, i:i+N_seq_slice, :], time_slice)
        X_hat = model(Z_0)
        X_hat = convert_output_data(X_hat, N_batch, N_seq, N_out) # Use N_out
        l_roll += mse_loss(X_hat, X_0[:, i+1:i+1+N_seq_slice, :12]) # Target still sliced to 12
        if pinn:
             # Ensure t_coll also has the right shape for physics_loss_fn if needed
             # Assuming t_coll is [N_batch, N_seq-1, N_coll] based on previous errors
             # physics_loss_fn expects X, U, t_coll
             # X_hat is [N_batch, N_seq_slice, N_x]
             # U slice is [N_batch, N_seq_slice, N_u]
             # t_coll slice should match X_hat and U slice length (N_seq_slice)
             # If original t_coll is N_seq-1, the slice t_coll[:, i:i+N_seq_slice-1, :] might be needed?
             # Let's assume for now physics_loss_fn handles the shapes correctly internally or t_coll shape matches X/U.
             # Slice t_coll to match the sequence length of X_hat (N_seq_slice)
             t_coll_slice = t_coll[:, i:i+N_seq_slice, :]

             # Check if the sliced length matches X_hat's length before calling physics loss
             if t_coll_slice.shape[1] == X_hat.shape[1]:
                 l_phy += physics_loss_fn(model, X_hat, U[:, i:i+N_seq_slice, :], t_coll_slice, device, noise_level, model_name)
             elif t_coll_slice.shape[1] > 0:
                 # This case indicates a potential logic error or unexpected data shape mismatch
                 print(f"Warning: Mismatch in sequence length during rollout physics loss calculation. X_hat: {X_hat.shape[1]}, t_coll_slice: {t_coll_slice.shape[1]}")
                 pass # Skip physics loss for this step due to mismatch
             else:
                 # Handle case where t_coll slice is empty, maybe skip physics loss for this step
                 pass


    l_roll /= N_roll
    l_phy /= N_roll
    
    return l_roll, l_phy
        
def initial_condition_loss(model, X_1, U, time, device):
    """
    Calculates the initial condition prediction loss.
    """
    Z_1, N_batch, N_seq_1, N_x = convert_input_data(X_1, U, torch.zeros_like(time).unsqueeze(-1))
    Z_1 = Z_1.to(device)
    
    X_1_hat = model(Z_1)
    X_1_hat = convert_output_data(X_1_hat, N_batch, N_seq_1, N_x)
    
    l_ic = mse_loss(X_1_hat, X_1) # output, target
    
    return l_ic
        
def physics_loss_fn(model, X_1, U, t_coll, device, noise_level=0.0, model_name='drone'):
    """
    Computes the physics-based loss.

    Args:
        model (Module): Neural network model.
        X_1 (Tensor): States.
        U (Tensor): Control inputs.
        t_coll (Tensor): Collocation times.
        device (torch.device): Device to run computations on.
        noise_level (float): Noise level for data augmentation.
        model_name (str): Name of the model to use for physics loss calculation.

    Returns:
        Tensor: Physics loss.
    """
    N_x = X_1.shape[2]
    N_u = U.shape[-1]
    N_in = 12 + N_u + 1 # Update N_in to reflect the sliced state size
            
    # Prepare collocation inputs
    Z_1_coll, U_coll = convert_input_collocation(X_1 + torch.normal(0, noise_level, X_1.shape, device=device), U, t_coll)
    Z_1_coll = Z_1_coll.to(device)
    U_coll_flat = (U_coll.to(device)).view(-1, N_u)
    X_2_hat_coll_flat, dX2_hat_dt_flat = compute_time_derivatives(Z_1_coll, N_in, model)
    
    # Load the drone model with mass and inertia
    # Using default mass from NewDrone and placeholder inertia
    # Ensure inertia tensor is on the same device as the state tensor
    drone_model = NewDrone(mass=3.8, inertia=torch.eye(3, device=X_1.device)) # Use X_1's device

    # Extract state and control input
    state = X_2_hat_coll_flat
    control_input = U_coll_flat

    # Compute the predicted state derivative using the drone model
    next_state = drone_model.forward(state, control_input) # Shape: [N_total, 9]

    # Compute the mean squared residuals, comparing only the first 9 derivatives
    # dX2_hat_dt_flat has shape [N_total, 12] (output of NN)
    # next_state has shape [N_total, 9] (output of drone_model)
    l_phy = ((dX2_hat_dt_flat[:, :9] - next_state)**2).mean()

    return l_phy

def get_data_sets(N_batch=32, data_path='training_set', train_path='training_set', dev_path='dev_set', test_1_path='test_set_interp', test_2_path='test_set_extrap'):
    """
    Loads the training, development, and test datasets.

    Args:
        N_batch (int): Batch size.
        data_path (str): Path to the data.
        train_path (str): Path to training data.
        dev_path (str): Path to development data.
        test_1_path (str): Path to the first test data.
        test_2_path (str): Path to the second test data.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, development, and test sets.
    """
    training_data = TrajectoryDataset(data_path)
    train_dataloader = DataLoader(training_data, batch_size=N_batch, shuffle=True)
    dev_data = TrajectoryDataset(dev_path)
    dev_dataloader = DataLoader(dev_data, batch_size=N_batch, shuffle=True)
    test_1_data = TrajectoryDataset(test_1_path)
    test_1_dataloader = DataLoader(test_1_data, batch_size=N_batch, shuffle=True)
    test_2_data = TrajectoryDataset(test_2_path)
    test_2_dataloader = DataLoader(test_2_data, batch_size=N_batch, shuffle=True)
    return train_dataloader, dev_dataloader, test_1_dataloader, test_2_dataloader
