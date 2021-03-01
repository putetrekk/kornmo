import h5py
import numpy as np
import random
import os
import copy
from typing import List


class MaskDataset:
    def __init__(self, file: str):
        self.filename = file
        self.labels = self.__load_labels()

    def get_iterator(self, shuffle=False):
        labels = copy.copy(self.labels)
        if shuffle:
            random.shuffle(labels)

        mask_factory = lambda orgnr, year: self.get_mask(orgnr, year)

        return MaskDatasetIterator(mask_factory, labels)

    def __load_labels(self):
        if not os.path.exists(self.filename):
            with h5py.File(self.filename, "a") as file:
                file.create_group("masks")

        labels = []

        def visit_func(name, object):
            if not isinstance(object, h5py.Dataset):
                return
            labels.append(name)

        with h5py.File(self.filename, "r+") as file:
            file.visititems(visit_func)
        return labels

    def get_mask(self, orgnr, year):
        label = f"masks/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.filename, "r+") as file:
                return file[label][()]

    def del_masks(self, orgnr, year):
        label = f"masks/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.filename, "r+") as file:
                del file[label]
                self.labels.remove(label)
                print(f"Deleted mask dataset '{label}'.")
        else:
            print(f"mask dataset '{label}' is already deleted.")

    def store_masks(self, mask, farmer_id, year):
        with h5py.File(self.filename, "a") as file:
            label = f"masks/{farmer_id}/{year}"
            # Delete existing masks (if they exist)
            if label in file:
                del file[f"masks/{farmer_id}/{year}"]
            else:
                self.labels.append(label)

            file.create_dataset(
                name=f"masks/{farmer_id}/{year}",
                data=mask,
                compression="gzip",
                compression_opts=2,
            )

    def contains(self, farmer_id, year):
        return f"masks/{farmer_id}/{year}" in self.labels

    @staticmethod
    def __extract_orgnr_year(label):
        parts = label.split("/")
        return parts[1], parts[2]


class MaskDatasetIterator:
    def __init__(self, get_masks, labels: List[str]):
        self.get_masks = get_masks
        self.labels = labels
        self.n = len(labels)

    def __iter__(self):
        i = 0
        while i < self.n:
            orgnr, year = self.labels[i].split("/")[1:3]
            yield orgnr, year, self.get_masks(orgnr, year)
            i += 1

    def __getitem__(self, key):
        # If key is a slice, eg. [0:10], we return a new iterator over the sequence
        if isinstance(key, slice):
            labels_slice = self.labels[key]
            it = MaskDatasetIterator(self.get_masks, labels_slice)
            return it

        # It's just an index
        elif isinstance(key, int):
            orgnr, year = self.labels[key].split("/")[1:3]
            return orgnr, year, self.get_masks(orgnr, year)

        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key)}")
