import torch
from tqdm import tqdm
import numpy as np
from Diffusion import Diffusion

def generate_prototypes(encoder, dataloader, num_classes=42, transform=None):
    """
    Generate prototypes for each class from the training data.
    
    Args:
        encoder: Trained encoder model
        dataloader: Training dataloader
        num_classes: Number of classes (default 42 for AnuraSet)
        transform: Transform to apply to input data before encoding
    
    Returns:
        prototypes: Tensor of shape (num_classes, feature_dim) containing class prototypes
    """
    encoder.eval()
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    
    print("Collecting features for prototype generation...")
    with torch.no_grad():
        for input, labels, _ in tqdm(dataloader):
            input = input.to(next(encoder.parameters()).device)
            
            # Apply transform if provided
            if transform is not None:
                input = transform(input)
                
            features = encoder(input)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all features and labels
    train_audio_np = torch.cat(all_features, dim=0).numpy()
    train_labels_np = torch.cat(all_labels, dim=0).numpy()
    
    # Generate prototypes
    prototypes = []
    
    for class_idx in range(num_classes):
        indices = np.where(train_labels_np[:, class_idx] > 0)[0]
        if len(indices) > 0:
            class_prototype = train_audio_np[indices].mean(axis=0)
        else:
            class_prototype = np.zeros(train_audio_np.shape[1], dtype=np.float32)
        prototypes.append(class_prototype.astype(np.float32))

    # Convert prototypes to a PyTorch tensor
    prototypes = torch.tensor(np.stack(prototypes, axis=0), dtype=torch.float32)
    
    print(f"Generated prototypes with shape: {prototypes.shape}")
    return prototypes

def compute_mmd(x, y, sigma=1.0):
    def rbf(x1, x2):
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        dist_sq = (diff ** 2).sum(2)
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

def variance_loss(gen, real):
    var_gen = gen.var(dim=0)
    var_real = real.var(dim=0)
    return torch.nn.functional.mse_loss(var_gen, var_real)


def train_diffusion(encoder, dataloader, args, transform=None):
    feat_dim = 960
    model = Diffusion(input_dim=feat_dim, hidden_dim=256).to(args.device)
    model.train()
    encoder.eval()

    optimiser = torch.optim.Adam(model.parameters(), lr=args.diff_lr, weight_decay=args.diff_weight_decay)

    for epoch in range(args.diff_epochs):
        epoch_loss = 0.0
        for _, (input, _, _) in enumerate(tqdm(dataloader)):
            input = input.to(args.device)

            # Apply transform if provided
            if transform is not None:
                input = transform(input)

            # Get the features from the encoder
            with torch.no_grad():
                features = encoder(input)

            # Distort the features
            distorted_features = model.distort(features, epoch / args.diff_epochs)

            # Forward pass through the diffusion model
            generated = model(distorted_features, features)

            # Compute loss (e.g., MSE loss)
            loss = torch.nn.functional.mse_loss(generated, features)
            # Mean Maximum Discrepancy (MMD) Loss
            loss += 0.5 * compute_mmd(generated, features)  # MMD loss
            # Variance Loss
            loss += 0.1 * variance_loss(generated, features)  # Variance loss
            # Cosine Similarity Loss            
            cos_sim = torch.nn.functional.cosine_similarity(generated, features, dim=1)
            cos_loss = 1.0 - cos_sim.mean()
            loss += 2.0 * cos_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{args.diff_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')

    return model

def generate_prototypes_from_features(train_audio_np, train_labels_np, num_classes=42):
    """
    Generate prototypes for each class from pre-computed features.
    
    Args:
        train_audio_np: Numpy array of features
        train_labels_np: Numpy array of labels
        num_classes: Number of classes (default 42 for AnuraSet)
    
    Returns:
        prototypes: Tensor of shape (num_classes, feature_dim) containing class prototypes
    """
    # Generate prototypes
    prototypes = []
    
    for class_idx in range(num_classes):
        indices = np.where(train_labels_np[:, class_idx] > 0)[0]
        if len(indices) > 0:
            class_prototype = train_audio_np[indices].mean(axis=0)
        else:
            class_prototype = np.zeros(train_audio_np.shape[1], dtype=np.float32)
        prototypes.append(class_prototype.astype(np.float32))

    # Convert prototypes to a PyTorch tensor
    prototypes = torch.tensor(np.stack(prototypes, axis=0), dtype=torch.float32)
    
    print(f"Generated prototypes with shape: {prototypes.shape}")
    return prototypes