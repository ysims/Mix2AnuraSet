import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import random
import numpy as np

class DiffusionDataset(Dataset):
    def __init__(self, diffusion_model, dataloader, prototypes, data_path, threshold=300):
        """
        Args:
            diffusion_model: The diffusion model to generate samples.
            feature_size: The size of the feature vector.
            threshold: Classes with fewer samples than the threshold will be used to generate data.
            data_path: Path to the dataset to find underrepresented classes.
        """

        # Get the classes from the metadata
        metadata = os.path.join(data_path, "metadata.csv")
        metadata_df = pd.read_csv(metadata)
        classes = metadata_df.columns[8:50]

        # Find the classes with fewer samples than the threshold
        # Only consider classes in the train set (row[7] == 'train')
        train_metadata = metadata_df[metadata_df['subset'] == 'train']
        # Ignore classes with no data
        # python list of zeros equal to the number of classes
        class_counts = np.zeros(len(classes), dtype=int)
        class_counts_r = train_metadata.iloc[:, 8:50].sum().values
        underrepresented_classes = [
            i for i, count in enumerate(class_counts) if count < threshold
        ]
        class_counts[class_counts_r == 0] = threshold
        self.labels = []
        self.data = []

        diffusion_model.eval()
        # Continue until all classes have at least `threshold` samples
        while (class_counts < threshold).any():
            for feature, label in dataloader.dataset:
                # Ignore samples with no classes
                if label.sum() == 0:
                    continue
                classes_in_label = label.nonzero(as_tuple=True)[0]
                # Only generate if any class in this label is underrepresented
                if any(class_counts[c] < threshold for c in classes_in_label):
                    with torch.no_grad():
                        feature = feature.unsqueeze(0)
                        noisy_sample = diffusion_model.distort(feature.to("cuda"), 0.1)
                        # Get a random prototype from positive classes
                        prototype = random.choice(
                            [prototypes[c] for c in classes_in_label if class_counts[c] < threshold]
                        ).unsqueeze(0).to("cuda")
                        sample = diffusion_model(noisy_sample, prototype)
                    self.data.append(sample.squeeze().cpu())
                    self.labels.append(label.detach().clone().cpu())
                    for c in classes_in_label:
                        class_counts[c] += 1
                # Stop early if all classes are filled
                if not (class_counts < threshold).any():
                    break

        # Convert to tensor
        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

        print(class_counts)
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __add__(self, other):
        self.data = torch.cat((self.data, other.data))
        self.labels = torch.cat((self.labels, other.labels))
        return self
