import h5py
import numpy as np
import random
import os
from copy import copy
from typing import List, Callable
from inspect import signature


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
        labels = copy(self.labels)
        if shuffle:
            random.shuffle(labels)

        return SentinelDatasetIterator(self, labels)

    def get_iterators(self, val_split=0.2, shuffle=True):
        labels = copy(self.labels)

        if shuffle:
            random.shuffle(labels)

        num = len(labels)
        split = int(val_split * num)
        train_labels = labels[split:]
        val_labels = labels[:split]
        
        training_iterator = SentinelDatasetIterator(self, labels=train_labels)
        validation_iterator = SentinelDatasetIterator(self, labels=val_labels)

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


class SentinelImageSeriesSource:
    def __init__(self, dataset: h5py.Dataset, orgnr, year, transformations: List[Callable] = []) -> None:
        self.image_dataset = dataset
        self.__orgnr = orgnr
        self.__year = year
        self.__transformations = transformations
        self.__mask: Union[np.ndarray, None] = None

    def __getitem__(self, key):
        images:np.ndarray = self.image_dataset[key] / SentinelDataset.INT_SCALE

        for transform in self.__transformations:
            sig = signature(transform)
            if len(sig.parameters) == 3:
                images = transform(self.__orgnr, self.__year, images)
            else:
                images = transform(images)

        return images
        

class SentinelDatasetIterator:
    def __init__(self, dataset, labels:List[str]=None, tuples=None):
        self.dataset = dataset
        self.filename = dataset.filename

        if tuples is not None:
            self.tuples = tuples
        
        else:
            labels = list(map(lambda l: l.split("/")[1:3], labels))
            self.tuples = [(orgnr, year, []) for orgnr, year in labels]

    def transform(self, transformation: Callable):
        new_tuples = []
        for orgnr, year, t_list in self.tuples:
            new_tuples.append((orgnr, year, t_list + [transformation]))
        
        return SentinelDatasetIterator(self.dataset, tuples=new_tuples)
    
    def augment(self, transformations: List[Callable]):
        tuples = copy(self.tuples)

        for transformation in transformations:
            new_tuples = []
            for orgnr, year, t_list in self.tuples:
                new_tuples.append((orgnr, year, t_list + [transformation]))
            
            tuples = tuples + new_tuples
        
        random.shuffle(tuples)

        return SentinelDatasetIterator(self.dataset, tuples=tuples)

    def __iter__(self):
        with h5py.File(self.filename, "r+") as file:
            for orgnr, year, transformations in self.tuples:

                img_dataset:h5py.Dataset = file[f"images/{orgnr}/{year}"]
                yield orgnr, year, SentinelImageSeriesSource(img_dataset, orgnr, year, transformations)

    def __getitem__(self, key):
        # If key is a slice, eg. [0:10], we return a new iterator over the sequence
        if isinstance(key, slice):
            tuples = self.tuples[key]
            it = SentinelDatasetIterator(self.dataset, tuples=tuples)
            return it
        
        # It's just an index
        elif isinstance(key, int):
            orgnr, year, transformation_idx = self.tuples[key]
            imgs = self.dataset.get_images(orgnr, year)
            
            transformation = self.transformations[transformation_idx]
            if transformation is not None:
                return orgnr, year, transformation(imgs)
                
            return orgnr, year, imgs
        
        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key)}")
    

    def __len__(self):
        return len(self.tuples)