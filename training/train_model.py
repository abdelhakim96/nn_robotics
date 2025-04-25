import os
import sys
sys.path.append('.')
import time
import torch
import numpy as np
from torch.nn import Softplus
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import models.model_utility
import importlib
import argparse
importlib.reload(models.model_utility)

# Use absolute import now that training/ and models/ are packages
from models.model_utility import (
    get_data_sets,
    DNN,
    convert_input_data,
    # train, # Defined locally
    # test_dev_set # Defined locally
    physics_loss_fn, # Assuming this is needed by train/test_dev_set
    data_loss_fn,    # Assuming this is needed by train/test_dev_set
    rollout_loss_fn  # Assuming this is needed by train/test_dev_set
)

# Set seed for reproducibility
torch.manual_seed(0)
torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
alpha = 0.6 # filter constant
batch_size = 3
tb_idx = 1 # Reset or increment tensorboard index for new experiment
exp_name = "drone_training" # experiment name for drone model

epochs = 1200
lr_0 = 8e-3
lr_factor = 0.5
lr_patience = 1200
lr_thr = 1e-4
lr_min = 1e-5   
pinn = True
rollout = True
activation = Softplus
noise_level = 0.0 # std of noise level


def load_model_data(data_dir):
    training_data = torch.load(os.path.join(data_dir, "training_set.pt"))
    dev_data = torch.load(os.path.join(data_dir, "dev_set.pt"))
    X_train = training_data['X']
    U_train = training_data['U']
    t_coll_train = training_data['t_coll']
    time_train = training_data['time']
    X_dev = dev_data['X']
    U_dev = dev_data['U']
    t_coll_dev = dev_data['t_coll']
    time_dev = dev_data['time']
    return X_train, U_train, t_coll_train, time_train, X_dev, U_dev, t_coll_dev, time_dev

# Define train function locally
def train(model, dataloader, optimizer, epoch, device, writer, pinn, rollout, noise_level, N_out): # Added N_out
    model.train()
    total_loss = 0
    for batch_idx, (X, U, t_coll, time) in enumerate(dataloader):
        X, U, t_coll, time = X.to(device), U.to(device), t_coll.to(device), time.to(device)
        
        optimizer.zero_grad()
        
        l_data = data_loss_fn(model, X, U, time, device, N_out, noise_level) # Pass N_out
        l_phy = 0
        l_roll = 0

        if pinn:
            # Assuming physics_loss_fn is defined in model_utility and imported
            # X has shape [batch, N, N_x], U has shape [batch, N, N_u]
            # t_coll has shape [batch, N-1, N_coll_pts] (as per data generation)
            # We need X and U up to the second-to-last time step (0 to N-2, length N-1)
            # to match the N-1 intervals represented by t_coll.
            if X.shape[1] > 1 and t_coll.shape[1] == X.shape[1] - 1:
                # Use states/controls from index 0 to N-2 (length N-1)
                # Use the full t_coll tensor (length N-1)
                l_phy = physics_loss_fn(model, X[:, :-1, :], U[:, :-1, :], t_coll, device, noise_level)
            else:
                # Handle potential shape mismatch or edge cases (e.g., sequence length <= 1)
                print(f"Warning: Skipping physics loss due to unexpected shapes. X: {X.shape}, t_coll: {t_coll.shape}")
                l_phy = 0 # Assign zero loss if shapes don't match expectation

        if rollout:
             # Assuming rollout_loss_fn is defined in model_utility and imported
             # Determine N_roll, maybe half the sequence length? Or a fixed value? Let's use a fixed value for now.
             N_roll = 10 # Example value, adjust as needed
             if X.shape[1] > N_roll:
                 l_roll, l_phy_roll = rollout_loss_fn(model, X, U, time, N_roll, device, t_coll, pinn, N_out, noise_level) # Pass N_out
                 l_phy = (l_phy + l_phy_roll) / 2 if pinn else l_phy # Average physics loss if rollout also calculates it

        # Combine losses (adjust weighting as needed)
        loss = l_data + (l_phy if pinn else 0) + (l_roll if rollout else 0)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    return avg_loss

# Define test_dev_set function locally
def test_dev_set(model, dataloader, epoch, device, writer, N_out): # Added N_out
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (X, U, t_coll, time) in enumerate(dataloader):
            X, U, t_coll, time = X.to(device), U.to(device), t_coll.to(device), time.to(device)
            
            # Calculate validation loss (e.g., using data_loss_fn)
            # You might want a different loss combination for validation
            l_data = data_loss_fn(model, X, U, time, device, N_out, 0.0) # Pass N_out, No noise for validation
            loss = l_data # Simple validation loss, adjust if needed
            
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Loss/dev', avg_loss, epoch)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train dynamics learning model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="drone",
        help="Name of the model (bluerov or drone)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="drone_data",
        help="Path to the data directory",
    )
    args = parser.parse_args()

    # Load data
    X_train, U_train, t_coll_train, time_train, X_dev, U_dev, t_coll_dev, time_dev = load_model_data(
        args.data_dir
    )
    # Removed print statements


    train_dataset = TensorDataset(X_train, U_train, t_coll_train, time_train)
    dev_dataset = TensorDataset(X_dev, U_dev, t_coll_dev, time_dev)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    X_0, U_0, _, _ = next(iter(train_dataloader))

    # Define network parameters based on model
    if args.model_name == "bluerov":
        N_x = 9  # BlueROV states
        N_out = 9
        N_u = 4
    elif args.model_name == "drone":
        N_x = 12 # Drone states (pos, att, lin_vel, ang_vel)
        N_out = 12
        N_u = 4 # Drone controls (phi_cmd, theta_cmd, psi_cmd, Fz_cmd)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    N_coll = 10  # Dummy value, not directly used in network size
    N_in = N_x + N_u + 1 # State + Control + Time
    N_h = [32, 32, 32, 32]
    N_layer = len(N_h)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f"runs/{exp_name}_{tb_idx}_{timestamp}") 

    # Initialize model, optimizer, and scheduler
    model = DNN(N_in=N_in, N_out=N_out, N_h=N_h, N_layer=N_layer, activation=activation).to(device)
    optimizer = AdamW(model.parameters(), lr=lr_0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=lr_factor,
        patience=lr_patience,
        threshold=lr_thr,
        min_lr=lr_min
    )

    # Prepare for training
    Z_0, _, _, _ = convert_input_data(
        X_0.to(device),
        U_0.to(device),
        torch.zeros_like(X_0[:, :, :1]).to(device)
    )
    Z_0.requires_grad_()

    # Create directory for saving models
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    # Define base model name (removed gradient_method)
    base_model_name = f"{exp_name}_{tb_idx}"
    model_path = os.path.join(model_dir, base_model_name)

    l_dev_best = np.float32('inf')
    l_dev_smooth = 1.0  # Initial value of smoothed l_dev
    
    try:
        for epoch in trange(epochs):
            # Training step
            l_train = train(
                model,
                train_dataloader,
                optimizer,
                epoch,
                device,
                writer,
                pinn=pinn,
                rollout=rollout,
                noise_level=noise_level,
                N_out=N_out # Pass N_out
            )
            # Validation step
            l_dev = test_dev_set(
                model,
                dev_dataloader,
                epoch,
                device,
                writer, # Added missing writer argument
                N_out=N_out # Pass N_out
            )
            
            # Update smoothed validation loss
            l_dev_smooth = alpha*l_dev_smooth + (1 - alpha)*l_dev
            
            # Save the best model
            if l_dev < l_dev_best:
                # Define best model filename (removed gradient_method)
                model_path_best = os.path.join(
                    model_dir,
                    f"{base_model_name}_best_dev_l_{epoch}"
                )
                torch.save(model.state_dict(), model_path_best)
                l_dev_best = l_dev

            # Scheduler step
            scheduler.step(l_dev_smooth)
                
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
            writer.flush() 
        torch.save(model, model_path)    
        writer.add_hparams(
            {
                "Number of collocation points": N_coll,
                "PINN": pinn,
                "Rollout": rollout,
                "lr_0": lr_0,
                "lr_decay": lr_factor,
                "lr_patience": lr_patience,
                "lr_thr": lr_thr,
                "N_batch": batch_size,
                "epochs": int(epoch+1),
                "Number of states (x)": N_x,
                "Number of inputs (u)": N_u,
                "Total input size": N_in,
                "Number of hidden layers": N_layer,
                "Hidden layers size": f"{N_h}",
                "Activation function": f"{activation.__name__}",
                "Noise level": noise_level,
                # Removed "Gradient Method": gradient_method,
            },
            {
                "final_train_loss": l_train,
                "final_dev_loss": l_dev
            }
        )
        writer.close()
    except Exception as e:
        print(f"An error occured: {e}")
        writer.close()
        raise

if __name__ == "__main__":
    main()
