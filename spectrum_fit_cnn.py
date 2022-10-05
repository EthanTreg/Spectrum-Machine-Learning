import os
import torch
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import spectrum_data


class SpectrumDataset(Dataset):
    def __init__(self, data_dir, annotations_file=None, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform

        self.spectra_files = next(os.walk(self.data_dir + '/spectra'))[2]

    def __len__(self):
        return len(self.spectra_files)

    def __getitem__(self, idx):
        spectrum = torch.from_numpy(spectrum_data(self.data_dir, self.spectra_files[idx]))

        if self.transform:
            spectrum = self.transform(spectrum)

        if self.annotations_file:
            label = self.annotations_file[idx]

            if self.target_transform:
                label = self.target_transform(label)

            return spectrum, label

        return spectrum


train_data = SpectrumDataset('./data/training')
test_data = SpectrumDataset('./data/test')

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
print(next(iter(train_dataloader)))
print(next(iter(test_dataloader)))
