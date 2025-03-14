from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import polars as pl
import numpy as np


class EMNIST(Dataset):

    def __init__(self, root, selection='emnist-balanced', split='train'):
        super(EMNIST, self).__init__()

        allowed_selections = [
            'emnist-balanced',
            'emnist-byclass',
            'emnist-bymerge',
            'emnist-digits',
            'emnist-letters',
        ]

        split_selection = ['train', 'test']

        if selection not in allowed_selections:
            raise ValueError(
                f'Selection {selection} is not valid. Choose from: {" | ".join(allowed_selections)}'
            )

        if split not in split_selection:
            raise ValueError(
                f'Split {split} is not valid. Choose from: {" | ".join(split_selection)}'
            )

        root_path = Path(root)
        if not root_path.exists():
            raise ValueError(f'Directory {root} does not exist.')

        data_path = root_path / f"{selection}-{split}.parquet"
        mapping_path = root_path / f"{selection}-mapping.parquet"

        if not data_path.exists():
            raise ValueError(f'File {data_path} not found.')

        if not mapping_path.exists():
            raise ValueError(f'File {mapping_path} not found.')

        dataset = pl.read_parquet(data_path)

        self.mapping = pl.read_parquet(mapping_path)[:, 1].to_numpy()

        self.labels = dataset[:, 0].to_numpy()
        self.data = dataset[:, 1:].to_numpy().astype('float32').reshape(
            -1, 28, 28, 1).transpose([0, 2, 1, 3])

        self.ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, ))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.ts(self.data[index]), torch.tensor(self.labels[index],
                                                       dtype=torch.long)


def get_emnist(root, batch_size, split, num_workers, shuffle, drop_last):

    return DataLoader(
        EMNIST(root=root, split=split),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        prefetch_factor=2,
        persistent_workers=True,
    )


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import timeit
#     from torchvision.utils import make_grid

#     def visualize_dataset():
#         ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')
#         print(f"Dataset size: {ds.__len__()}")

#         # Print statistics about labels
#         unique_labels = np.unique(ds.labels)
#         min_label = np.min(ds.labels)
#         max_label = np.max(ds.labels)
#         print(f"Label range: {min_label} to {max_label}")
#         print(f"Number of unique labels: {len(unique_labels)}")
#         print(f"Unique labels: {unique_labels}")

#         # Get some samples
#         num_samples = 25
#         fig, axes = plt.subplots(5, 5, figsize=(10, 10))
#         axes = axes.flatten()

#         samples = []
#         labels_list = []

#         for i in range(num_samples):
#             img_tensor, label = ds[i]
#             samples.append(img_tensor)
#             labels_list.append(label)

#             # Convert tensor for display
#             img_np = img_tensor.squeeze().numpy()
#             # Denormalize if needed
#             img_np = (img_np * 0.5 + 0.5)

#             # Plot in the grid
#             axes[i].imshow(img_np, cmap='gray')
#             axes[i].set_title(f"Label: {label}")
#             axes[i].axis('off')

#         plt.tight_layout()
#         plt.savefig('emnist_samples.png')
#         plt.show()

#         # Display as a single grid
#         grid_img = make_grid(samples, nrow=5, normalize=True)
#         plt.figure(figsize=(10, 10))
#         plt.imshow(grid_img.permute(1, 2, 0))
#         plt.title('EMNIST Samples')
#         plt.axis('off')
#         plt.savefig('emnist_grid.png')
#         plt.show()

#         # Print some sample tensors and labels
#         print(f"Sample tensor shape: {samples[0].shape}")
#         print(f"Sample labels: {labels_list[:10]}")

#     # print("TIME:", timeit.timeit("visualize_dataset()", globals=globals(), number=1))

#     def pros():
#         ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')
#         print(ds.__len__())
#         print(ds.__getitem__(0))

#     print("TIME:", timeit.timeit("pros()", globals=globals(), number=1))
