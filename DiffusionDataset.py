import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import random
import numpy as np

class DiffusionDataset(Dataset):
    def __init__(self, diffusion_model, dataloader, prototypes, data_path, encoder, transform=None, threshold=300):
        """
        Args:
            diffusion_model: The diffusion model to generate samples.
            dataloader: The original dataloader
            prototypes: The class prototypes
            data_path: Path to the dataset to find underrepresented classes.
            encoder: The encoder model to extract features
            transform: Transform to apply to raw audio before encoding
            threshold: Classes with fewer samples than the threshold will be used to generate data.
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
        encoder.eval()
        
        # Continue until all classes have at least `threshold` samples
        while (class_counts < threshold).any():
            for input_audio, labels, _ in dataloader:
                # Process each sample in the batch
                for i in range(input_audio.size(0)):
                    sample_audio = input_audio[i:i+1]  # Keep batch dimension
                    sample_label = labels[i]
                    
                    # Ignore samples with no classes
                    if sample_label.sum() == 0:
                        continue
                    classes_in_label = sample_label.nonzero(as_tuple=True)[0]
                    
                    # Only generate if any class in this label is underrepresented
                    if any(class_counts[c] < threshold for c in classes_in_label):
                        with torch.no_grad():
                            # Move input to device and apply transform
                            sample_audio = sample_audio.to("cuda")
                            if transform is not None:
                                sample_audio = transform(sample_audio)
                            
                            # Extract features using the encoder
                            features = encoder(sample_audio)
                            
                            # Generate synthetic features using diffusion
                            noisy_sample = diffusion_model.distort(features, 0.1)
                            # Get a random prototype from positive classes
                            prototype = random.choice(
                                [prototypes[c] for c in classes_in_label if class_counts[c] < threshold]
                            ).unsqueeze(0).to("cuda")
                            sample = diffusion_model(noisy_sample, prototype)
                        self.data.append(sample.squeeze().cpu())
                        self.labels.append(sample_label.detach().clone().cpu())
                        for c in classes_in_label:
                            class_counts[c] += 1
                    
                    # Stop early if all classes are filled
                    if not (class_counts < threshold).any():
                        break
                
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
