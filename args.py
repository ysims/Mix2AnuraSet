import argparse

parser = argparse.ArgumentParser()

# Generic
parser.add_argument("--rootdir", type=str, default='') #root directory for the dataset
parser.add_argument("--device", type=str, default='cuda') #device to train on
parser.add_argument("--sr", type=int, default=16000) #sampling rate
parser.add_argument("--workers", type=int, default=4) #number of workers
parser.add_argument("--model", type=str, default='mobilenetv3') #model architecture
parser.add_argument("--save", action='store_true') #save ckpt

# Training
parser.add_argument("--momentum", type=float, default=0.9) #SGD momentum
parser.add_argument("--bs", type=int, default=128) #batch size for representation learning
parser.add_argument("--mixnum", type=int, default=128) #number of generated samples in multimix
parser.add_argument("--epochs", type=int, default=100) #nb of epochs to train the feature extractor on the training set
parser.add_argument("--lr", type=float, default=1e-2) #learning rate 
parser.add_argument("--wd", type=float, default=1e-6) #weight decay
parser.add_argument("--outdim", type=int, default=128) #output dimension of projector

# Mixing parameters
parser.add_argument("--mix", type=str, default='mix2') #mixing method in ['', 'mixup', 'manmixup', 'multimix', 'mix2']
parser.add_argument("--alpha", type=float, default=1.0) #mixup parameter
parser.add_argument("--alpha1", type=float, default=0.2) #lower bound of uniform for alpha values #0.2 is not bad, reach high acc but doesnt look stable, 0.4 is also good
parser.add_argument("--alpha2", type=float, default=0.2) #upper bound of uniform for alpha values #1.0

# Data Augmentation
parser.add_argument("--fmask", type=int, default=10) #fmax
parser.add_argument("--tmask", type=int, default=30) #tmax
parser.add_argument("--fstripe", type=int, default=3) #fstripe
parser.add_argument("--tstripe", type=int, default=6) #tstripe

# Diffusion parameters
parser.add_argument("--diff_epochs", type=int, default=50) #number of epochs for diffusion training
parser.add_argument("--diff_lr", type=float, default=1e-3) #learning rate for diffusion model
parser.add_argument("--diff_weight_decay", type=float, default=1e-6) #weight decay for diffusion model
parser.add_argument("--threshold", type=int, default=300) #threshold for underrepresented classes (reduced from 2000)
parser.add_argument("--warmup_epochs", type=int, default=70) #epochs to train encoder before diffusion training

# Optimization parameters
parser.add_argument("--use_cache", action='store_true', default=True) #cache diffusion components
parser.add_argument("--synthetic_freq", type=int, default=5) #use synthetic data every N epochs
parser.add_argument("--reduce_diff_epochs", action='store_true') #reduce diffusion epochs for speed

args = parser.parse_args()