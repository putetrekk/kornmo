import h5py
import numpy as np
import random
import os
import copy
from typing import List


class SentinelDataset:
    '''
    When storing images it is much more efficient to store integer values instead of floats,
    however, values must be mapped from 0...1 to real numbers to store any valuable data,
    using this value as a multiplier.
    '''
    INT_SCALE = 255

    def __init__(self, file: str):
        self.filename = file
        self.labels = self.__load_labels()
    
    def get_iterator(self, shuffle=False):
        labels = copy.copy(self.labels)
        if shuffle:
            random.shuffle(labels)

        return SentinelDatasetIterator(self, labels)

    def get_iterators(self, val_split=0.2, shuffle=True):
        labels = copy.copy(self.labels)
        if shuffle:
            random.shuffle(labels)

        num = len(self.labels)
        split = int(val_split * num)
        train_labels = labels[split:]
        val_labels = labels[:split]
        
        training_iterator = SentinelDatasetIterator(self, train_labels)
        validation_iterator = SentinelDatasetIterator(self, val_labels)
        return training_iterator, validation_iterator

    def __load_labels(self):
        if not os.path.exists(self.filename):
            with h5py.File(self.filename, "a") as file:
                file.create_group("images")

        labels = []
        def visit_func(name, object):
            if not name.startswith("images") or not isinstance(object, h5py.Dataset):
                return
            labels.append(name)

        with h5py.File(self.filename, "r+") as file:
            file.visititems(visit_func)
        return labels

    def get_images(self, orgnr, year):
        label = f"images/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.filename, "r+") as file:
                return file[label][()] / self.INT_SCALE

    def del_images(self, orgnr, year):
        label = f"images/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.filename, "r+") as file:
                del file[label]
                self.labels.remove(label)
                print(f"Deleted image dataset '{label}'.")
        else:
            print(f"Image dataset '{label}' is already deleted.")
    
    def store_images(self, images, farmer_id, year, compression:int=2):
        # Convert image values to integer type
        data = (np.array(images) * self.INT_SCALE).astype(int)

        with h5py.File(self.filename, "a") as file:
            label = f"images/{farmer_id}/{year}"
            # Delete existing images (if they exist)
            if label in file:
                del file[f"images/{farmer_id}/{year}"]
            else:
                self.labels.append(label)
            
            c_args = {}
            if compression > 0:
                c_args = {'compression': 'gzip', 'compression_opts': compression}
            
            file.create_dataset(name=f"images/{farmer_id}/{year}", data=data, **c_args)

    def contains(self, farmer_id, year):
        return f"images/{farmer_id}/{year}" in self.labels
    
    def copy_to(self, output_file: str, compression:int=0):
        from tqdm import tqdm
        
        c_args = {}
        if compression > 0:
            c_args = {'compression': 'gzip', 'compression_opts': compression}

        with h5py.File(output_file, "a") as out_file:
            with h5py.File(self.filename, "r+") as file:
                for label in tqdm(self.labels):
                    data = file[label][()]
                    out_file.create_dataset(name=label, data=data, **c_args)

    @staticmethod
    def __extract_orgnr_year(label):
        parts = label.split("/")
        return parts[1], parts[2]


class SentinelDatasetIterator:
    def __init__(self, dataset: SentinelDataset, labels: List[str]):
        self.dataset = dataset
        self.filename = dataset.filename
        self.labels = labels
        self.n = len(labels)

    def __iter__(self):
        with h5py.File(self.filename, "r+") as file:
            i = 0
            while i < self.n:
                orgnr, year = self.labels[i].split("/")[1:3]
                img_array:np.ndarray = file[f"images/{orgnr}/{year}"][()]
                yield orgnr, year, img_array / SentinelDataset.INT_SCALE
                i += 1
    
    def __getitem__(self, key):
        # If key is a slice, eg. [0:10], we return a new iterator over the sequence
        if isinstance(key, slice):
            labels_slice = self.labels[key]
            it = SentinelDatasetIterator(self.dataset, labels_slice)
            return it
        
        # It's just an index
        elif isinstance(key, int):
            orgnr, year = self.labels[key].split("/")[1:3]
            return orgnr, year, self.dataset.get_images(orgnr, year)
        
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key)}")
