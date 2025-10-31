import os

import torch



def compute_and_save_fisher_and_params(model, dataloader, criterion, device, fisher_path):
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
    fisher = torch.load(os.path.join(fisher_path, "fisher.pt"), map_location=device)
    global_params = torch.load(os.path.join(fisher_path, "global_params.pt"), map_location=device)
    return fisher, global_params



def save_teacher_model(model, path):
    torch.save(model.state_dict(), path)
def load_teacher_model(model_class, path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
