from tqdm import tqdm
import numpy as np
import torch
import math
import os
import pickle

from train_diffusion import train_diffusion, generate_prototypes
from DiffusionDataset import DiffusionDataset

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CachedDiffusionComponents:
    """Cache diffusion model, prototypes, and synthetic dataset to avoid recomputation"""
    
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_path(self, component_name, args_hash):
        return os.path.join(self.cache_dir, f"{component_name}_{args_hash}.pkl")
    
    def get_args_hash(self, args):
        """Create a hash of relevant args to determine cache validity"""
        relevant_args = {
            'diff_epochs': args.diff_epochs,
            'diff_lr': args.diff_lr,
            'diff_weight_decay': args.diff_weight_decay,
            'threshold': args.threshold,
            'sr': args.sr,
            'model': args.model
        }
        return hash(str(sorted(relevant_args.items())))
    
    def save_components(self, diffusion_model, prototypes, diffusion_dataset, args):
        """Save all components to cache"""
        args_hash = self.get_args_hash(args)
        
        # Save diffusion model
        torch.save(diffusion_model.state_dict(), 
                  self.get_cache_path("diffusion_model", args_hash))
        
        # Save prototypes
        torch.save(prototypes, 
                  self.get_cache_path("prototypes", args_hash))
        
        # Save diffusion dataset
        dataset_data = {
            'data': diffusion_dataset.data,
            'labels': diffusion_dataset.labels
        }
        torch.save(dataset_data, 
                  self.get_cache_path("diffusion_dataset", args_hash))
    
    def load_components(self, args, device):
        """Load components from cache if available"""
        args_hash = self.get_args_hash(args)
        
        # Check if all cache files exist
        diffusion_path = self.get_cache_path("diffusion_model", args_hash)
        prototypes_path = self.get_cache_path("prototypes", args_hash)
        dataset_path = self.get_cache_path("diffusion_dataset", args_hash)
        
        if not (os.path.exists(diffusion_path) and 
                os.path.exists(prototypes_path) and 
                os.path.exists(dataset_path)):
            return None, None, None
        
        try:
            # Load diffusion model (need to recreate the model structure)
            from Diffusion import Diffusion
            diffusion_model = Diffusion(input_dim=960, hidden_dim=256).to(device)
            diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device))
            
            # Load prototypes
            prototypes = torch.load(prototypes_path, map_location=device)
            
            # Load dataset data
            dataset_data = torch.load(dataset_path, map_location=device)
            
            # Create a simple dataset class for cached data
            class CachedDiffusionDataset:
                def __init__(self, data, labels):
                    self.data = data
                    self.labels = labels
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx], self.labels[idx]
            
            diffusion_dataset = CachedDiffusionDataset(dataset_data['data'], dataset_data['labels'])
            
            print(f"Loaded cached diffusion components with {len(diffusion_dataset)} synthetic samples")
            return diffusion_model, prototypes, diffusion_dataset
        
        except Exception as e:
            print(f"Error loading cached components: {e}")
            return None, None, None

def setup_diffusion_components(encoder, train_dataloader, train_transform, args, use_cache=True):
    """Setup diffusion components with optional caching"""
    cache_manager = CachedDiffusionComponents() if use_cache else None
    
    # Try to load from cache first
    if cache_manager:
        diffusion_model, prototypes, diffusion_dataset = cache_manager.load_components(args, args.device)
        if diffusion_model is not None:
            return diffusion_model, prototypes, diffusion_dataset
    
    # If cache miss or no caching, compute from scratch
    print('Training diffusion model...')
    diffusion_model = train_diffusion(encoder, train_dataloader, args, train_transform)
    
    print('Generating prototypes...')
    prototypes = generate_prototypes(encoder, train_dataloader, 42, train_transform)
    
    print('Creating diffusion dataset...')
    diffusion_dataset = DiffusionDataset(diffusion_model, train_dataloader, prototypes, 
                                       args.rootdir, encoder, train_transform, args.threshold)
    
    # Save to cache
    if cache_manager:
        cache_manager.save_components(diffusion_model, prototypes, diffusion_dataset, args)
    
    return diffusion_model, prototypes, diffusion_dataset

def train_optimized(encoder, projector, data_loader, transform, loss_fn, optimiser, scaler, args, 
                   diffusion_dataset=None, use_synthetic=True):
    """Optimized training function with reduced overhead"""
    
    num_batches = len(data_loader)
    encoder.train()
    projector.train()

    loss_total = 0.0
    
    # Main training loop
    for _, (input, target, _) in enumerate(tqdm(data_loader, desc="Training")):
        input, target = input.to(args.device), target.to(args.device)
        
        # Apply transforms and mixing in a single autocast context
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = transform(input)
            
            if args.mix == 'mixup':   
                mixed_x, y_a, y_b, lam = mixup_data(x, target)
                prediction = projector(encoder(mixed_x))
                loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
                
            elif args.mix == 'manmixup':   
                z = encoder(x)
                mixed_z, y_a, y_b, lam = mixup_data(z, target)
                prediction = projector(mixed_z)
                loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
                
            elif args.mix == 'multimix':
                z = encoder(x)
                mixed_z, mixed_y, lam = multimix_data(z, target, args.mixnum, args.alpha1, args.alpha2)
                prediction = projector(mixed_z)
                loss = multimix_criterion(loss_fn, prediction, mixed_y, lam)
                
            elif args.mix == 'mix2':
                p = np.random.random()
                z = encoder(x)
                
                if p < 0.5:  # 50% manmixup
                    mixed_z, y_a, y_b, lam = mixup_data(z, target)
                    prediction = projector(mixed_z)
                    loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
                elif p < 0.75:  # 25% multimix
                    mixed_z, mixed_y, lam = multimix_data(z, target, args.mixnum, args.alpha1, args.alpha2)
                    prediction = projector(mixed_z)
                    loss = multimix_criterion(loss_fn, prediction, mixed_y, lam)
                else:  # 25% mixup
                    mixed_x, y_a, y_b, lam = mixup_data(x, target)
                    prediction = projector(encoder(mixed_x))
                    loss = mixup_criterion(loss_fn, prediction, y_a, y_b, lam)
            else:
                prediction = projector(encoder(x))
                loss = loss_fn(prediction, target)
    
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        loss_total += loss.item()  

    # Use synthetic data if available and requested
    if use_synthetic and diffusion_dataset is not None and len(diffusion_dataset) > 0:
        diffusion_loader = torch.utils.data.DataLoader(
            diffusion_dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=min(args.workers, 2),  # Reduce workers for synthetic data
            pin_memory=True
        )
        
        for _, (synthetic_features, target) in enumerate(tqdm(diffusion_loader, desc="Synthetic")):
            synthetic_features, target = synthetic_features.to(args.device), target.to(args.device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                prediction = projector(synthetic_features)
                loss = loss_fn(prediction, target)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            loss_total += loss.item()

    loss_total /= num_batches   
    return loss_total

# Keep the original mixing functions for compatibility
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample(((x.size(0)), 1))
        lam = lam.to(x.device)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    l = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    return l.mean()

def multimix_data(z, y, num_mixed_examples, alpha1=1.0, alpha2=1.0, use_cuda=True):
    '''Returns mixed inputs, multiple pairs of targets, and multiple lambda values for each example in the batch'''
    batch_size = z.size()[0]

    alpha_values = np.random.uniform(alpha1, alpha2, size=batch_size)
    lambda_vectors = torch.from_numpy(np.random.dirichlet(alpha_values, num_mixed_examples)).float()

    if use_cuda:
        lambda_vectors = lambda_vectors.cuda()

    mixed_z = torch.matmul(lambda_vectors, z)
    mixed_y = torch.matmul(lambda_vectors, y)

    return mixed_z, mixed_y, lambda_vectors

def multimix_criterion(criterion, pred, mixed_y, lambda_vectors):
    '''Compute the loss for each mixed example and lambda vector'''
    mixed_loss = torch.sum(lambda_vectors.unsqueeze(2) * criterion(pred, mixed_y), dim=1)
    return torch.mean(mixed_loss)
