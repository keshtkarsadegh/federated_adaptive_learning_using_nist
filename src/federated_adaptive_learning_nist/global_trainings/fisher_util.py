import os

import torch

"""
Utilities for computing and storing Fisher information matrices and 
corresponding model parameters, typically used in Elastic Weight 
Consolidation (EWC) or similar continual learning regularization methods.

Functions:
    - compute_and_save_fisher_and_params: computes Fisher diagonals and saves them with model params
    - load_fisher_and_params: loads previously saved Fisher information and parameters
"""


def compute_and_save_fisher_and_params(model, dataloader, criterion, device, fisher_path):
    """
    Compute and save Fisher information and model parameters.

    Runs the model in evaluation mode over the provided dataloader, computes
    the diagonal Fisher information matrix (squared gradients of the loss),
    and saves both the Fisher values and a snapshot of the current model
    parameters to disk.

    Args:
        model (torch.nn.Module): The model to analyze.
        dataloader (torch.utils.data.DataLoader): Data used to compute gradients.
        criterion (torch.nn.Module): Loss function.
        device (torch.device or str): Device to perform computations on.
        fisher_path (str): Directory path where results will be saved.

    Saves:
        fisher.pt         : dict mapping parameter names -> Fisher diagonals.
        global_params.pt  : dict mapping parameter names -> parameter tensors.
"""


    model.eval()

    fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters() if p.requires_grad}
    global_params = {n: p.clone().detach().to(device) for n, p in model.named_parameters() if p.requires_grad}

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2)

    for n in fisher:
        fisher[n] /= len(dataloader)

    # Save both to a file
    os.makedirs(fisher_path, exist_ok=True)
    torch.save(fisher, os.path.join(fisher_path, "fisher.pt"))
    torch.save(global_params, os.path.join(fisher_path, "global_params.pt"))

def load_fisher_and_params(fisher_path, device):
    """
    Load Fisher information and model parameters from disk.

    Args:
        fisher_path (str): Directory containing 'fisher.pt' and 'global_params.pt'.
        device (torch.device or str): Device to map loaded tensors to.

    Returns:
        tuple:
            fisher (dict): Fisher information per parameter.
            global_params (dict): Saved model parameters.
    """

    fisher = torch.load(os.path.join(fisher_path, "fisher.pt"), map_location=device)
    global_params = torch.load(os.path.join(fisher_path, "global_params.pt"), map_location=device)
    return fisher, global_params