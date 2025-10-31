import copy
import json
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.federated_adaptive_learning_nist.constants import BATCH_SIZE

from pathlib import Path
import random
from typing import List, Tuple



from torch.utils.data import ConcatDataset, DataLoader

def combine_loaders(loader_dicts):

    datasets = []
    for loader_dict in loader_dicts:
        # Each dict has one key-value pair
        datasets.append(loader_dict["test"].dataset)

    combined_dataset = ConcatDataset(datasets)

    return DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )



def get_two_non_overlapping_writer_sets(nist_root: str, k: int, n: int, seed: int = 42) -> Tuple[List[str], List[str]]:

    base_path = Path(nist_root)
    all_writers = []

    for hsf_dir in base_path.glob("hsf_*"):
        for writer_dir in hsf_dir.glob("f*_*"):
            all_writers.append(writer_dir.name)

    all_writers = sorted(set(all_writers))
    total_needed = k + n

    if total_needed > len(all_writers):
        raise ValueError(f"Cannot select {k} + {n} = {total_needed} writers from {len(all_writers)} total writers.")

    random.seed(seed)
    sampled = random.sample(all_writers, total_needed)
    random.shuffle(sampled)

    selected_writer_ids = sampled[:k]
    non_selected_writer_ids = sampled[k:]

    return selected_writer_ids, non_selected_writer_ids



class NISTDigitDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_nist_data_by_writer(
    nist_root,
    digits_labels_json,
    non_writers_ids=None,
    writer_ids=None,
    seed=42,
    train_rate=0.6,
    eval_rate=0.2,
    is_stratified=False
):
    nist_root = Path(nist_root)

    # ✅ Load labels from JSON
    with open(digits_labels_json, "r") as f:
        label_dict = json.load(f)

    # ✅ Normalize full paths and group them by writer ID
    writer_samples = defaultdict(list)
    for rel_path, label in label_dict.items():
        full_path = nist_root / rel_path
        parts = rel_path.split("/")
        if len(parts) < 3:
            continue
        writer_id = parts[2]  # e.g., f0302_47
        writer_samples[writer_id].append((full_path, label))

    all_writer_ids = sorted(writer_samples.keys())
    random.seed(seed)

    # ✅ Determine selected vs. global writers
    if writer_ids is not None and non_writers_ids is not None:
        selected_writers = writer_ids
        global_writers = non_writers_ids
    else:
        raise ValueError("You must provide both `writer_ids` and `non_writers_ids`.")

    # ✅ Image transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ✅ Stratified split (preserve label balance)
    def stratified_split(data):
        label_map = defaultdict(list)
        for path, label in data:
            label_map[label].append(path)

        train, val, test = [], [], []
        for label, imgs in label_map.items():
            random.shuffle(imgs)
            n = len(imgs)
            n_train = int(n * train_rate)
            n_val = int(n * eval_rate)

            train.extend((img, label) for img in imgs[:n_train])
            val.extend((img, label) for img in imgs[n_train:n_train + n_val])
            test.extend((img, label) for img in imgs[n_train + n_val:])
        return train, val, test

    # ✅ Random split (no label balancing)
    def random_split(data):
        random.shuffle(data)
        n = len(data)
        n_train = int(n * train_rate)
        n_val = int(n * eval_rate)
        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]
        return train, val, test

    # ✅ Choose split method
    split_fn = stratified_split if is_stratified else random_split

    # ✅ Helper to make DataLoaders
    def make_loader(data):
        image_paths = [p for p, _ in data]
        labels = [l for _, l in data]
        dataset = NISTDigitDataset(image_paths, labels, transform)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ✅ Process global writers (non-selected)
    global_data = []
    for wid in global_writers:
        global_data.extend(writer_samples[wid])
    global_train, global_val, global_test = split_fn(global_data)

    global_loaders = {
        "train": make_loader(global_train),
        "val": make_loader(global_val),
        "test": make_loader(global_test),
    }
    global_test_loader = {"writer_id": "global","test": make_loader(global_test)}

    # ✅ Process each selected writer individually
    writer_loaders = []
    clients_test_loaders=[]
    for wid in selected_writers:
        writer_data=None
        writer_data = copy.deepcopy(writer_samples[wid])
        train_data, val_data, test_data = split_fn(writer_data)
        writer_loaders.append({
            "writer_id": wid,
            "train": make_loader(train_data),
            "val": make_loader(val_data),
            "test": make_loader(test_data),
        })
        clients_test_loaders.append({"writer_id": wid, "test": make_loader(test_data)})
    all_test_loaders = copy.deepcopy(clients_test_loaders)
    all_test_loaders.append(global_test_loader)

    all_test_loaders=combine_loaders(all_test_loaders)
    clients_test_loaders=combine_loaders(clients_test_loaders)

    return global_loaders, writer_loaders, all_test_loaders, clients_test_loaders


import matplotlib.pyplot as plt


def show_dataloader_images(dataloader, num_batches=1):
    """
    Display images and labels from a PyTorch dataloader.
    Assumes each item is (image_tensor, label).
    """
    batch_count = 0
    for images, labels in dataloader:
        batch_size = images.size(0)

        # Denormalize (since you used Normalize((0.5,), (0.5,)))
        images = images * 0.5 + 0.5  # Undo normalization for visualization

        plt.figure(figsize=(batch_size * 2, 3))
        for i in range(batch_size):
            img = images[i].squeeze().cpu()  # grayscale
            label = labels[i].item()

            plt.subplot(1, batch_size, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f"Label: {label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        batch_count += 1
        if batch_count >= num_batches:
            break
