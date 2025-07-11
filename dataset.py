from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os

class AnuraSet(Dataset):
    def __init__(self, annotations_file, audio_dir, train=True):

        if isinstance(annotations_file, str):
            df = pd.read_csv(annotations_file)
        else:
            df = annotations_file.copy()
        
        if train:
            df = df[df["subset"]=="train"]
        else:
            df = df[df["subset"]=="test"]

        self.annotations = df
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, _ = torchaudio.load(audio_sample_path)

        return signal, torch.Tensor(label), index
    
    def _get_audio_sample_path(self, index):
        index_row = self.annotations.iloc[index]
        fname = index_row['fname']
        start_second = index_row['min_t']
        final_second = index_row['max_t']
        path = os.path.join(self.audio_dir, fname.split('_')[0], fname+'_'+str(start_second)+'_'+str(final_second)+'.wav')
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 8:]