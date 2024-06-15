import os
from multiprocessing import Pool, context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mne.filter import filter_data
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import AmplitudeToDB, Spectrogram


def downsample(x, sf, new_sf):
    num = int(new_sf / sf * x.shape[-1])
    return resample(x, num, axis=-1)


class AccusleepDataset(Dataset):
    def __init__(
        self,
        root,
        raw_sf,
        model_sf,
        raw_epoch_len,
        model_epoch_len,
        bandpass_freqs=None,
        context_window=1,
        random_shift=False,
        data_transform=None,
        target_transform=None,
        run_preproc=False,
        num_workers=0,
    ):
        # Build the paths
        all_paths = Path(root).rglob("*")
        all_dirs = sorted([folder for folder in all_paths if folder.is_dir()])
        if not all_dirs:
            all_dirs = [root]
        assert (
            float(raw_sf * raw_epoch_len).is_integer()
            and float(model_sf * model_epoch_len).is_integer()
        )

        # Save attributes
        self.window_size = int(model_sf * model_epoch_len)
        self.raw_sf = raw_sf
        self.model_sf = model_sf
        self.raw_epoch_len = raw_epoch_len
        self.model_epoch_len = model_epoch_len
        assert context_window >= 1
        self.context_window = context_window
        self.random_shift = random_shift
        self.data_transform = data_transform
        self.target_transform = target_transform

        self.data = []
        self.labels = []
        index_map = []
        dir_map = []
        weight = torch.tensor([0, 0, 0])
        i_map = 0

        # Load data
        if run_preproc:
            if num_workers > 1:
                pool = Pool(num_workers)
                pool.starmap(
                    self.preprocess,
                    [
                        (
                            dir,
                            raw_sf,
                            model_sf,
                            bandpass_freqs,
                            raw_epoch_len,
                            model_epoch_len,
                        )
                        for dir in all_dirs
                    ],
                )
            else:
                for dir in all_dirs:
                    self.preprocess(
                        dir,
                        raw_sf,
                        model_sf,
                        bandpass_freqs,
                        raw_epoch_len,
                        model_epoch_len,
                    )

        for dir in all_dirs:
            try:
                data, labels = self.get_files(dir, self.window_size)
            except FileNotFoundError:
                continue
            self.data.append(data)
            self.labels.append(labels)

            # Count labels for class weighting
            _, counts = labels.unique(sorted=True, return_counts=True)
            weight += counts

            # Mapping from "global index" to list and tensor indices
            dir_map.append(np.repeat(i_map, len(labels) - context_window + 1))
            index_map.append(np.arange(context_window - 1, len(labels)))
            i_map += 1
        if len(dir_map) == 0:
            raise FileNotFoundError

        # Build index mapping
        self.weight = weight.sum() / (weight * 3)
        dir_map = np.concatenate(dir_map)
        index_map = np.concatenate(index_map)
        self.index_map = np.column_stack((dir_map, index_map))

    def preprocess(
        self,
        data_dir,
        raw_sf,
        model_sf,
        bandpass_freqs,
        raw_epoch_len,
        model_epoch_len,
    ):
        eeg_file = data_dir / "EEG.mat"
        emg_file = data_dir / "EMG.mat"
        labels_file = data_dir / "labels.mat"

        if not (eeg_file.is_file() and emg_file.is_file() and labels_file.is_file()):
            return
        # load files
        eeg = loadmat(eeg_file)
        emg = loadmat(emg_file)
        labels = loadmat(labels_file)

        eeg = np.squeeze(eeg["EEG"])
        emg = np.squeeze(emg["EMG"])
        data = np.stack((eeg, emg), axis=0)
        labels = np.squeeze(labels["labels"])

        if bandpass_freqs:
            eeg = filter_data(
                data.astype("float64"),
                raw_sf,
                bandpass_freqs[0],
                bandpass_freqs[1],
                verbose=False,
            )
        if raw_sf != model_sf:
            data = downsample(data, raw_sf, model_sf)

        num_samples = int(data.shape[1] // (model_sf * raw_epoch_len))
        assert num_samples == labels.shape[0]

        # Scale data
        data = RobustScaler().fit_transform(data.T).T

        if raw_epoch_len != model_epoch_len:
            labels_new_len = int(labels.shape[0] * raw_epoch_len // model_epoch_len)
            data = data[: int(labels_new_len * model_sf * model_epoch_len)]

        # Re-map the label values
        accusleep_dict = {
            1: 2,  # REM
            2: 0,  # Wake
            3: 1,  # NREM
        }
        labels = np.vectorize(accusleep_dict.get)(labels)

        if raw_epoch_len != model_epoch_len:
            labels_new = np.zeros(labels_new_len, dtype=int)
            for i in range(len(labels_new)):
                labels_new[i] = labels[int(round(i * model_epoch_len / raw_epoch_len))]
            labels = labels_new

        np.save(data_dir / "data_preproc.npy", data)
        np.save(data_dir / "labels_preproc.npy", labels)

    def get_files(self, data_dir, window_size):
        """Retrieves preprocessed EEG, EMG, and label data

        Args:
            data_dir (str): directory in which data is stored
            homemade (bool): homemade or Accusleep data
            eeg_idx (int): desired eeg probe index in raw homemade data array
            emg_idx (int): desired emg probe index in raw homemade data array
            a

        Raises:
            FileNotFoundError: raised if EEG.mat, EMG.mat, or labels.mat do not exist within target_folder

        Returns:
            eeg_array, emg_array, label_array (torch.Tensor, torch.Tensor, torch.Tensor): EEG, EMG, and label data
            as numpy arrays
        """
        data_file = data_dir / "data_preproc.npy"
        labels_file = data_dir / "labels_preproc.npy"

        if not data_file.is_file() or not labels_file.is_file():
            raise FileNotFoundError
        # load files
        data = np.load(data_file)
        labels = np.load(labels_file)

        # Check lengths
        num_samples = data.shape[1] / window_size
        assert num_samples >= labels.shape[0]
        data = data[:, : labels.shape[0] * window_size]

        return (
            torch.tensor(data, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return self.index_map.shape[0]

    def __getitem__(self, idx):
        dir_idx, adjusted_idx = self.index_map[idx]
        label = self.labels[dir_idx][adjusted_idx]
        if self.random_shift and adjusted_idx < len(self.labels[dir_idx]) - 1:
            offset = np.random.randint(self.window_size)
        else:
            offset = 0
        low = (adjusted_idx - self.context_window + 1) * self.window_size + offset
        high = (adjusted_idx + 1) * self.window_size + offset
        data = self.data[dir_idx][:, low:high]

        if self.data_transform:
            data = self.data_transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label


class MatiasDataset(Dataset):
    def __init__(
        self,
        root,
        raw_sf,
        model_sf,
        raw_epoch_len,
        model_epoch_len,
        bandpass_freqs=None,
        context_window=1,
        eeg_idx=0,
        emg_idx=5,
        random_shift=False,
        data_transform=None,
        target_transform=None,
        run_preproc=False,
        num_workers=0,
    ):
        # Build the paths
        root = Path(root)
        all_paths = list(root.glob("**/*"))
        all_dirs = sorted([folder for folder in all_paths if folder.is_dir()])
        if not all_dirs:
            all_dirs = [root]
        if not (
            float(raw_sf * raw_epoch_len).is_integer()
            and float(model_sf * model_epoch_len).is_integer()
        ):
            raise ValueError(
                "Sampling frequencies and epoch lengths must multiply to be integers"
            )

        # Save attributes
        self.window_size = int(model_sf * model_epoch_len)
        self.raw_sf = raw_sf
        self.model_sf = model_sf
        self.raw_epoch_len = raw_epoch_len
        self.model_epoch_len = model_epoch_len
        if context_window < 1:
            raise ValueError("Context window must be at least 1")
        self.context_window = context_window
        self.random_shift = random_shift
        self.data_transform = data_transform
        self.target_transform = target_transform

        self.data = []
        self.labels = []
        index_map = []
        dir_map = []
        weight = torch.tensor([0, 0, 0])
        i_map = 0

        if run_preproc:
            if num_workers > 1:
                pool = Pool(num_workers)
                pool.starmap(
                    self.preprocess,
                    [
                        (
                            dir,
                            raw_sf,
                            model_sf,
                            bandpass_freqs,
                            raw_epoch_len,
                            model_epoch_len,
                        )
                        for dir in all_dirs
                    ],
                )
            else:
                for dir in all_dirs:
                    self.preprocess(
                        dir,
                        raw_sf,
                        model_sf,
                        bandpass_freqs,
                        raw_epoch_len,
                        model_epoch_len,
                    )

        # Load data
        for dir in all_dirs:
            try:
                data, labels = self.get_files(dir, eeg_idx, emg_idx, self.window_size)
            except FileNotFoundError:
                continue
            self.data.append(data)
            self.labels.append(labels)

            # Count labels for class weighting
            _, counts = labels.unique(sorted=True, return_counts=True)
            weight += counts

            # Mapping from "global index" to list and tensor indices
            dir_map.append(np.repeat(i_map, labels.shape[0] - self.context_window + 1))
            index_map.append(np.arange(self.context_window - 1, labels.shape[0]))
            i_map += 1
        if len(dir_map) == 0:
            raise FileNotFoundError("No data found in the specified directory")

        # Build index mapping
        self.weight = weight.sum() / (weight * 3)
        dir_map = np.concatenate(dir_map)
        index_map = np.concatenate(index_map)
        self.index_map = np.column_stack((dir_map, index_map))

    def preprocess(
        self,
        data_dir,
        raw_sf,
        model_sf,
        bandpass_freqs,
        raw_epoch_len,
        model_epoch_len,
    ):
        data_dir = Path(data_dir)
        data_file = data_dir / "data.csv"
        labels_file = data_dir / "labels.csv"

        if not data_file.is_file() or not labels_file.is_file():
            raise FileNotFoundError(
                f"Data or labels file not found in {data_dir} while preprocessing"
            )
        # load files
        data_df = pd.read_csv(data_file)
        labels_df = pd.read_csv(labels_file)

        data = data_df.values.T.astype("float64")
        labels = labels_df["behavior"].values

        if bandpass_freqs:
            data = filter_data(
                data,
                raw_sf,
                bandpass_freqs[0],
                bandpass_freqs[1],
                verbose=0,
            )
        if raw_sf != model_sf:
            data = downsample(data, raw_sf, model_sf)

        num_samples = int(data.shape[1] // (model_sf * raw_epoch_len))
        if num_samples != labels.shape[0]:
            raise ValueError("Data and labels have different lengths")

        # Scale EEG and EMG
        data = RobustScaler().fit_transform(data.T).T

        accusleep_dict = {
            "Wake": 0,
            "NREM": 1,
            "REM": 2,
        }
        # re-map the label values
        labels = np.vectorize(accusleep_dict.get)(labels).astype(int)

        if raw_epoch_len != model_epoch_len:
            labels_new_len = int(labels.shape[0] * raw_epoch_len // model_epoch_len)
            labels_new = np.zeros(labels_new_len, dtype=int)
            for i in range(len(labels_new)):
                labels_new[i] = labels[int(round(i * model_epoch_len / raw_epoch_len))]
            labels = labels_new
            data = data[:, : int(labels_new_len * model_sf * model_epoch_len)]

        np.save(data_dir / "data_preproc.npy", data)
        np.save(data_dir / "labels_preproc.npy", labels)

    def get_files(self, data_dir, eeg_idx, emg_idx, window_size):
        """Retrieves preprocessed EEG, EMG, and label data

        Args:
            data_dir (str): directory in which data is stored
            homemade (bool): homemade or Accusleep data
            eeg_idx (int): desired eeg probe index in raw homemade data array
            emg_idx (int): desired emg probe index in raw homemade data array
            a

        Raises:
            FileNotFoundError: raised if EEG.mat, EMG.mat, or labels.mat do not exist within target_folder

        Returns:
            eeg_array, emg_array, label_array (torch.Tensor, torch.Tensor, torch.Tensor): EEG, EMG, and label data
            as numpy arrays
        """
        data_dir = Path(data_dir)
        data_file = data_dir / "data_preproc.npy"
        labels_file = data_dir / "labels_preproc.npy"

        if not data_file.is_file() or not labels_file.is_file():
            raise FileNotFoundError(
                f"Data or labels file not found in {data_dir} while getting files"
            )
        # load files
        data = np.load(data_file)
        labels = np.load(labels_file)

        data = data[[eeg_idx, emg_idx]]

        # Check lengths
        num_samples = data.shape[1] / window_size
        assert num_samples >= labels.shape[0]

        data = data[:, : labels.shape[0] * window_size]
        return (
            torch.tensor(data, dtype=torch.float),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return self.index_map.shape[0]

    def __getitem__(self, idx):
        dir_idx, adjusted_idx = self.index_map[idx]
        label = self.labels[dir_idx][adjusted_idx]

        if self.random_shift and adjusted_idx < len(self.labels[dir_idx]) - 1:
            offset = np.random.randint(self.window_size)
        else:
            offset = 0

        low = (adjusted_idx - self.context_window + 1) * self.window_size + offset
        high = (adjusted_idx + 1) * self.window_size + offset
        data = self.data[dir_idx][:, low:high]

        if self.data_transform:
            data = self.data_transform(data)
        if self.target_transform:
            label = self.target_transform(label)

        return data, label


class OneHot(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, label):
        return torch.nn.functional.one_hot(label, num_classes=self.num_classes)


def get_dataloaders(config):
    data_transform = []
    if config.conjunction.data_transform is not None:
        for key, kwargs in config.conjunction.data_transform.items():
            if key == "spectrogram":
                data_transform.append(Spectrogram(**kwargs))
            elif key == "amplitude_to_db":
                data_transform.append(AmplitudeToDB(**kwargs))
            else:
                raise ValueError(f"Transform {key} not supported.")
    config.conjunction.data_transform = (
        nn.Sequential(*data_transform) if data_transform else None
    )

    target_transform = []
    if config.conjunction.target_transform is not None:
        for key, kwargs in config.conjunction.target_transform.items():
            if key == "one_hot":
                target_transform.append(OneHot(**kwargs))
            else:
                raise ValueError(f"Transform {key} not supported.")
    config.conjunction.target_transform = (
        nn.Sequential(*target_transform) if target_transform else None
    )

    train_dataset = AccusleepDataset(**config.accusleep_train, **config.conjunction)
    test_dataset = AccusleepDataset(**config.accusleep_test, **config.conjunction)
    val_dataset = MatiasDataset(**config.matias, **config.conjunction)
    train_loader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.test_batch_size, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.val_batch_size, shuffle=False
    )
    return train_loader, test_loader, val_loader
