from typing import Tuple, List

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import copy
import json
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import random
from torch.utils.data import  DataLoader

from src.federated_adaptive_learning_nist.nist_logger import NistLogger

project_results_dir=Path(__file__).resolve().parents[2]

class NistPath:
    base_path = project_results_dir
    nist_root_path =  project_results_dir/"data"
    by_write_data_path = nist_root_path /"by_write"
    digits_labels_json = by_write_data_path / "digits_labels.json"

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

class NistDataset:
    def __init__(self, ):
        self.nist_data_path=NistPath()

    def get_sample_count(self, writer_id):
        """
        Return the number of training samples available for this writer/client.
        Used in aggregation weighting.
        """
        _, _, trn= self.build_dataset(writer_id, train_rate=0.0, eval_rate=0.0, batch_size=64)
        return len(trn.dataset)

    def visualize_dataset_samples(self,data_loader, class_names=None, num_batches=1):

        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            images = images[:8]  # Show up to 8 samples per batch
            labels = labels[:8]

            fig, axs = plt.subplots(1, len(images), figsize=(15, 3))
            for i, (img, label) in enumerate(zip(images, labels)):
                img = img.squeeze()  # Remove channel dim (1, H, W) → (H, W)
                img = TF.to_pil_image((img * 0.5 + 0.5).clamp(0, 1))  # Undo normalization

                axs[i].imshow(img, cmap='gray')
                lbl = class_names[label.item()] if class_names else str(label.item())
                axs[i].set_title(f"Label: {lbl}")
                axs[i].axis('off')
            plt.show()
            NistLogger.info(f"Plotted the samples from dataset")
    def build_dataset(self,
            writers,
            seed=42,
            train_rate=0.6,
            eval_rate=0.2,
            is_stratified=True,
            batch_size=64
    ):
        if isinstance(writers,str):
            writers = [writers]

        # ✅ Load labels from JSON
        with open( self.nist_data_path.digits_labels_json, "r") as f:
            label_dict = json.load(f)

        # ✅ Normalize full paths and group them by writer ID
        writer_samples = defaultdict(list)
        for rel_path, label in label_dict.items():
            full_path =  self.nist_data_path.nist_root_path / rel_path
            parts = rel_path.split("/")
            if len(parts) < 3:
                continue
            writer_id = parts[2]  # e.g., f0302_47
            writer_samples[writer_id].append((full_path, label))

        random.seed(seed)

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
            if not data:
                return None
            image_paths = [p for p, _ in data]
            labels = [l for _, l in data]
            dataset = NISTDigitDataset(image_paths, labels, transform)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ✅ Process global writers (non-selected)
        data = []
        for wid in writers:
            data.extend(writer_samples[wid])
        if not data:
            return None,None, None
        train, val, test = split_fn(data)
        return make_loader(train), make_loader(val), make_loader(test)


def split_local_global_writers(nist_data_path: str, global_size: float, seed: int = 42) -> Tuple[List[str], List[str]]:

    base_path = Path(nist_data_path)
    all_writers = []

    for hsf_dir in base_path.glob("hsf_*"):
        for writer_dir in hsf_dir.glob("f*_*"):
            all_writers.append(writer_dir.name)
    random.seed(seed)
    all_writers = sorted(set(all_writers))
    random.shuffle(all_writers)


    # all_writers is your list of unique writer names
    local_writers, global_writers = train_test_split(
        all_writers, test_size=global_size, random_state=42
    )


    return local_writers, global_writers
