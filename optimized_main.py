import os

import torch

from torch import nn
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking
from transforms import MinMaxNorm
from torchmetrics.classification import MultilabelF1Score

from dataset import AnuraSet
from models import mobilenetv3

from optimized_train import train_optimized, adjust_learning_rate, setup_diffusion_components
from val import validate

from transforms import TimeShift
from models import model_dim

from args import args

import warnings
warnings.filterwarnings("ignore")

NUM_CLASSES = 42

root_dir = args.rootdir

# Pre-compute transforms to avoid repeated creation
resamp = Resample(orig_freq=22050, new_freq=args.sr)
min_max_norm = MinMaxNorm()
mel_spectrogram = MelSpectrogram(n_fft=512, hop_length=128, n_mels=128)
time_mask = TimeMasking(time_mask_param=args.tmask)
freq_mask = FrequencyMasking(freq_mask_param=args.fmask)
tshift = TimeShift(Tshift=376) #376 = length of timesteps for 3s melspectrogram with the config above

train_transform = nn.Sequential(
    resamp,
    mel_spectrogram,
    AmplitudeToDB(),
    min_max_norm,              
    tshift,
    *[freq_mask for _ in range(args.fstripe)],
    *[time_mask for _ in range(args.tstripe)],
).to(args.device)

val_transform = nn.Sequential(
    resamp,
    mel_spectrogram,
    AmplitudeToDB(),
    min_max_norm,
).to(args.device)

ANNOTATIONS_FILE = os.path.join(root_dir, 'metadata.csv')
AUDIO_DIR = os.path.join(root_dir, 'audio')

training_data = AnuraSet(
    annotations_file=ANNOTATIONS_FILE, 
    audio_dir=AUDIO_DIR, 
    train=True
)
print(f"There are {len(training_data)} samples in the training set.")

val_data = AnuraSet(
    annotations_file=ANNOTATIONS_FILE, 
    audio_dir=AUDIO_DIR, 
    train=False
)
print(f"There are {len(val_data)} samples in the test set.")

# Optimize dataloader settings for better performance
train_dataloader = DataLoader(training_data, 
    batch_size=args.bs,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=args.workers,
    persistent_workers=True if args.workers > 0 else False,  # Keep workers alive
    prefetch_factor=2 if args.workers > 0 else 2,  # Prefetch more batches
)

val_dataloader = DataLoader(val_data, 
    batch_size=args.bs,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=args.workers,
    persistent_workers=True if args.workers > 0 else False,
)

encoder = mobilenetv3()
projector = nn.Linear(model_dim[args.model], NUM_CLASSES)

encoder.to(args.device)
projector.to(args.device)    
trainable_params = list(encoder.parameters()) + list(projector.parameters())

loss_fn = nn.BCEWithLogitsLoss()
optimiser = torch.optim.AdamW(trainable_params, lr=args.lr)
metric_fn = MultilabelF1Score(num_labels=NUM_CLASSES).to(args.device)

scaler = torch.cuda.amp.GradScaler()

save_path = os.path.join(root_dir, 'ckpt')
os.makedirs(save_path, exist_ok=True)
pt_filepath = os.path.join(save_path, 'model.pt')    

best_score = None

# Setup diffusion components (will be initialized after warmup)
diffusion_model = None
prototypes = None
diffusion_dataset = None
diffusion_trained = False

# Compile models for better performance (PyTorch 2.0+)
try:
    encoder = torch.compile(encoder)
    projector = torch.compile(projector)
    print("Models compiled for better performance")
except:
    print("Model compilation not available, continuing without compilation")

print('Starting optimized training with proper warmup')
for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}")

    adjust_learning_rate(optimiser, epoch, args)

    # Train diffusion model after encoder has learned meaningful features
    if (not diffusion_trained and epoch >= args.warmup_epochs and 
        args.mix in ['mix2', 'multimix', 'manmixup']):
        print(f'Encoder has warmed up after {args.warmup_epochs} epochs. Setting up diffusion components...')
        diffusion_model, prototypes, diffusion_dataset = setup_diffusion_components(
            encoder, train_dataloader, train_transform, args, use_cache=args.use_cache
        )
        diffusion_trained = True
        print('Diffusion training complete. Continuing with augmented training...')

    # Use optimized training function
    use_synthetic = diffusion_trained and (epoch % args.synthetic_freq == 0)
    loss_train = train_optimized(
        encoder, projector, train_dataloader, train_transform, 
        loss_fn, optimiser, scaler, args, 
        diffusion_dataset=diffusion_dataset,
        use_synthetic=use_synthetic
    )
    
    metric_val, f1_freq, f1_common, f1_rare = validate(
        encoder, projector, val_dataloader, val_transform, metric_fn, args.device
    )

    if best_score is None:
        best_score = metric_val
        best_freq = f1_freq; best_common = f1_common; best_rare = f1_rare
        best_encoder_state = encoder.state_dict()
        best_projector_state = projector.state_dict()

        if args.save:
            torch.save(nn.Sequential(encoder, projector), pt_filepath)
        
    if metric_val > best_score:
        best_score = metric_val
        best_freq = f1_freq; best_common = f1_common; best_rare = f1_rare
        best_encoder_state = encoder.state_dict()
        best_projector_state = projector.state_dict()

        if args.save:
            torch.save(nn.Sequential(encoder, projector), pt_filepath)

    print(f"Loss train: {loss_train:.4f}\tMacro F1-score: {metric_val:.4f}\tFreq: {f1_freq:.4f}\tCommon: {f1_common:.4f}\tRare: {f1_rare:.4f}")

print(f"Best scores:\nMacro F1-score: {best_score:.4f}\tFreq: {best_freq:.4f}\tCommon: {best_common:.4f}\tRare: {best_rare:.4f}")    
print("Finished training")
