# TODO: Does log x affect results?
import os
from time import time
import xspec
import torch
import numpy as np
from torch import nn
from torch import optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split


class SpectrumDataset(Dataset):
    """
    A dataset object containing spectrum data for PyTorch training

    Attributes
    ----------
    spectra : tensor
        Spectra dataset
    params : tensor
        Parameters for each spectra if supervised
    names : ndarray
        Names of each spectrum
    """
    def __init__(self, data_dir: str, labels_file: str):
        """
        Parameters
        ----------
        data_dir : str
            Directory of the spectra dataset
        labels_file : str
            Path to the labels file, if none, then an unsupervised approach is used
        """
        self.spectra = torch.from_numpy(np.load(data_dir))

        labels = np.loadtxt(labels_file, skiprows=6, dtype=str)
        self.names = labels[:, 6]
        self.params = torch.from_numpy(labels[:, 9:].astype(float))

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        """
        Gets the spectrum data for a given index
        If supervised learning return target parameters of spectrum otherwise returns spectrum name

        Parameters
        ----------
        idx : int
            Index of the target spectrum

        Returns
        -------
        (tensor, tensor, str)
            Spectrum data, target parameters and spectrum name
        """
        return self.spectra[idx], self.params[idx], self.names[idx]


class PyXspecLoss(nn.Module):
    """
    Custom loss function using PyXspec fit statistics

    Attributes
    ----------
    device : device
        Device type for PyTorch to use
    model : string
        Model to use for PyXspec
    fix_params : ndarray
        Parameter number & value of fixed parameters
    param_limits : tensor
        Parameter lower & upper limits

    Methods
    -------
    forward(params)
        Calculates PyXspec loss from fit statistic
    backward(grad)
        Returns upstream gradient to allow backpropagation
    """
    def __init__(self, device: torch.device, model: str, fix_params: np.ndarray):
        """
        Parameters
        ----------
        device : device
            Device type for PyTorch to use
        model : string
            Model to use for PyXspec
        fix_params : ndarray
            Parameter number & value of fixed parameters
        """
        super().__init__()
        self.device = device
        self.fix_params = fix_params
        self.model = xspec.Model(model)
        self.param_limits = torch.empty((0, 2)).to(device)
        xspec.AllModels.setEnergies('0.002 500 1000 log')
        xspec.Fit.statMethod = 'pgstat'

        # Generate parameter limits
        for j in range(self.model.nParameters):
            if j + 1 not in fix_params[:, 0]:
                limits = torch.tensor(self.model(j + 1).values).to(device)[[2, 5]]
                self.param_limits = torch.vstack((self.param_limits, limits))

    def forward(self, params: Tensor) -> Tensor:
        """
        Forward function of PyXspec loss
        Parameters
        ----------
        params : tensor
            Parameter predictions from CNN
        Returns
        -------
        tensor
            Loss value
        """
        params = params.detach().cpu().numpy()

        # Merge fixed & free parameters
        fix_params_index = self.fix_params[0] - np.arange(self.fix_params.shape[1])
        params = np.insert(params, fix_params_index, self.fix_params[1])

        # Update model parameters
        self.model.setPars(params.tolist())

        # Calculate fit statistic loss
        return torch.tensor(xspec.Fit.statistic, requires_grad=True).to(self.device)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagation function returning upstream gradient

        Parameters
        ----------
        grad : Tensor
            Upstream gradient
        Returns
        -------
        tensor
            Gradient of PyXspec loss function
        """
        return grad * torch.ones(self.param_limits.size(0), requires_grad=True).to(self.device)


class CNN(nn.Module):
    """
    Constructs a CNN network to predict model parameters from spectrum data

    Attributes
    ----------
    model : XspecModel
        PyXspec model
    conv1 : nn.Sequential
        First convolutional layer
    conv2 : nn.Sequential
        Second convolutional layer
    out : nn.Linear
        Fully connected layer to produce parameter predictions

    Methods
    -------
    forward(x)
        Forward pass of CNN
    """
    def __init__(self, model: PyXspecLoss, spectra_size: int, parameters: int):
        """
        Parameters
        ----------
        model : XspecModel
            PyXspec model
        spectra_size : int
            number of data points in spectra
        parameters : int
            Number of parameters in model
        """
        super().__init__()
        self.model = model
        test_tensor = torch.empty((1, 1, spectra_size))

        # Construct layers in CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        conv_output = self.conv2(self.conv1(test_tensor)).shape
        self.out = nn.Linear(in_features=conv_output[1] * conv_output[2], out_features=parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN taking spectrum input and producing parameter predictions output

        Parameters
        ----------
        x : tensor
            Input spectrum

        Returns
        -------
        tensor
            Parameter predictions
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def progress_bar(i: int, total: int):
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    """
    length = 50
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t', end='')

    if i == total:
        print()


# TODO: Is calculating loss manually faster
# TODO: Does this work with PyTorch?
# TODO: Load multiple spectra at once
def unsupervised_loss(
        device: torch.device,
        names: list[str],
        model: PyXspecLoss,
        params: torch.Tensor,
        weight: float = 1) -> torch.Tensor:
    """
    Custom loss function using PyXspec to calculate statistic for Poisson data
    and Gaussian background (PGStat) and L1 loss for parameters that exceed limits

    params
    ----------
    device : device
        Device type for PyTorch to use
    names : list[string]
        Spectra names
    model : XspecModel
        PyXspec model
    params : tensor
        Output from CNN of parameter predictions between 0 & 1
    weight : float, default = 1
        relative weight of parameter loss to PyXspec loss

    Returns
    -------
    tensor
        Average PGStat loss
    """
    spectra = ''
    loss = torch.tensor(0.).to(device)

    # Loop through each spectrum in the batch
    for i, name in enumerate(names):
        # Parameter L1 loss function if parameter exceeds limit
        param_loss = torch.sum(torch.abs(
            nn.ReLU()(model.param_limits[:, 0] - params[i]) +
            nn.ReLU()(params[i] - model.param_limits[:, 1])
        ))

        # Limit parameters
        param_min = model.param_limits[:, 0] + 1e-6 * (
                model.param_limits[:, 1] - model.param_limits[:, 0])
        param_max = model.param_limits[:, 1] - 1e-6 * (
                model.param_limits[:, 1] - model.param_limits[:, 0])

        params = torch.clip(params, min=param_min, max=param_max)

        xspec.Spectrum(name)

        loss += model(params[i]) + weight * param_loss

        spectra += f'{i + 1}:{i + 1} ' + name + ' '

        progress_bar(i, len(names))

        xspec.AllData.clear()
    # xspec.AllData(spectra)

    # Average loss of batch
    return loss / len(names)


def train(
        device: torch.device,
        num_epochs: int,
        loader: DataLoader,
        cnn: CNN,
        optimizer: torch.optim.Optimizer,
        supervised: bool = True,
        dirs: list[str] = None):
    """
    Trains the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    device : device
        Device type for PyTorch to use
    num_epochs : int
        Number of epochs to train
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    optimizer : optimizer
        Optimisation method to use for training
    supervised : boolean, default = true
        Whether to use supervised learning
    dirs : list[string], default = None
        Directory of project & dataset files
    """
    print_num_epoch = 1
    t_initial = time()
    cnn.train()

    # Train for each epoch
    for epoch in range(num_epochs):
        for i, data in enumerate(loader):
            # Fetch data for training
            spectra, target_params, names = data
            spectra = spectra.to(device)
            target_params = target_params.to(device)

            # Pass data through CNN
            spectra = torch.unsqueeze(spectra.to(device=device, dtype=torch.float), dim=1)
            output = cnn(spectra)

            # Calculate loss
            if supervised:
                loss = nn.CrossEntropyLoss()(output, target_params)
            else:
                os.chdir(dirs[1])
                loss = unsupervised_loss(device, names, cnn.model, output)
                os.chdir(dirs[0])

            # Optimise CNN
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'{loss.item():.3e}')
            if (i + 1) % int(len(loader) / print_num_epoch) == 0 or i + 1 == len(loader):
                print(f'Epoch [{epoch + 1}/{num_epochs}]\t'
                      f'Step [{i + 1}/{len(loader)}]\t'
                      f'Loss: {loss.item() / spectra.shape[0]:.3e}\t'
                      f'Time: {time() - t_initial:.1f}')


def test(device: torch.device, loader: DataLoader, cnn: CNN, dirs: list[str]):
    """
    Tests the CNN on spectra data either supervised or unsupervised

    Parameters
    ----------
    device : device
        Device type for PyTorch to use
    loader : DataLoader
        PyTorch DataLoader that contains data to train
    cnn : CNN
        Model to use for training
    dirs : list[string]
        Directory of project & dataset files
    """
    loss = 0
    cnn.eval()

    with torch.no_grad():
        for data in loader:
            # Fetch data for testing
            spectra, _, names = data
            spectra = spectra.to(device)

            # Pass data through CNN
            spectra = torch.unsqueeze(spectra.to(device=device, dtype=torch.float), dim=1)
            output = cnn(spectra)

            # Calculate loss
            os.chdir(dirs[1])
            loss += unsupervised_loss(device, names, cnn.model, output)
            os.chdir(dirs[0])

    print(f'Average loss: {loss / len(loader):.2e}')


def main():
    """
    Main function for spectrum machine learning
    """
    # Initialize variables
    num_epochs = 20
    val_frac = 0.1
    torch.manual_seed(3)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = '../../Documents/Nicer_Data/ethan'
    spectra_path = './data/preprocessed_spectra.npy'
    labels_path = './data/nicer_bh_specfits_simplcut_ezdiskbb_freenh.dat'
    fix_params = np.array([[4, 0], [5, 100]])

    # Xspec initialization
    xspec.Xset.chatter = 0
    xspec.Xset.logChatter = 0
    xspec.AllModels.lmod('simplcutx', dirPath='../../Documents/Xspec_Models/simplcutx')

    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    # Fetch dataset & create train & val data
    dataset = SpectrumDataset(spectra_path, labels_path)
    val_amount = int(len(dataset) * val_frac)

    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_amount, val_amount])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, **kwargs)

    # Initialise the CNN
    model = PyXspecLoss(device, 'tbabs(simplcutx(ezdiskbb))', fix_params)
    cnn = CNN(model, dataset[0][0].size(dim=0), dataset[0][1].size(dim=0))
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # Train & validate CNN
    test(device, val_loader, cnn, [root_dir, data_dir])
    train(
        device,
        num_epochs,
        train_loader,
        cnn,
        optimizer,
        supervised=False,
        dirs=[root_dir, data_dir]
    )
    test(device, val_loader, cnn, [root_dir, data_dir])


if __name__ == '__main__':
    main()
