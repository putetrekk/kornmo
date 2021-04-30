from operator import itemgetter
import h5py
import numpy as np
import random
import errno
import os
import itertools
from copy import copy
from typing import Any, Dict, List, Callable, Tuple, Union
from inspect import signature


class SentinelDataset:
    '''
    When storing images it is much more efficient to store integer values instead of floats,
    however, values must be mapped from 0...1 to real numbers to store any valuable data,
    using this value as a multiplier.
    '''
    INT_SCALE = 255

    def __init__(self, file: str, create_if_missing=False):
        self.filename = file
        self.labels = self.__load_labels(create_if_missing)
    
    def __len__(self):
        return len(self.labels)
    
    def __iter__(self):
        it = SentinelDatasetIterator.from_dataset(self)
        for x in it:
            yield x
    
    def __getitem__(self, key):
        it = SentinelDatasetIterator.from_dataset(self)
        return it[key]

    def to_iterator(self):
        return SentinelDatasetIterator.from_dataset(self)

    def __load_labels(self, create_if_missing=False):
        if not os.path.exists(self.filename):
            if not create_if_missing:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.filename)
            
            with h5py.File(self.filename, "a") as file:
                file.create_group("images")

        labels = []
        def visit_func(name, object):
            if not name.startswith("images") or not isinstance(object, h5py.Dataset):
                return
            labels.append(name)

        with h5py.File(self.filename, "r") as file:
            file.visititems(visit_func)
        return labels

    def get_images(self, orgnr, year, raw=False) -> Union[np.ndarray, None]:
        label = f"images/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.filename, "r") as file:
                if raw:
                    return file[label][()]
                else:
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
        from tqdm.autonotebook import tqdm
        
        c_args = {}
        if compression > 0:
            c_args = {'compression': 'gzip', 'compression_opts': compression}

        with h5py.File(output_file, "a") as out_file:
            with h5py.File(self.filename, "r") as file:
                for label in tqdm(self.labels):
                    data = file[label][()]
                    out_file.create_dataset(name=label, data=data, **c_args)

    @staticmethod
    def combine_datasets(datasets, output_file: str, compression: int=0):
        from tqdm.autonotebook import tqdm
        
        c_args = {}
        if compression > 0:
            c_args = {'compression': 'gzip', 'compression_opts': compression}

        with h5py.File(output_file, "a") as out_file:
            for dataset in datasets:
                assert isinstance(dataset, SentinelDataset)

                with h5py.File(dataset.filename, "r") as file:
                    for label in tqdm(dataset.labels, desc=f"Copying {dataset.filename}..."):
                        data = file[label][()]
                        out_file.create_dataset(name=label, data=data, **c_args)

    @staticmethod
    def __extract_orgnr_year(label):
        parts = label.split("/")
        return parts[1], parts[2]


class SentinelImageSeriesSource:
    def __init__(self, source: Union[h5py.Dataset, np.ndarray], orgnr, year, transformations: List[Callable], data) -> None:
        self.image_source = source
        self.__orgnr = orgnr
        self.__year = year
        self.__transformations = transformations
        self.__data = data

    def __getitem__(self, key):
        images:np.ndarray = self.image_source[key] / SentinelDataset.INT_SCALE

        for transform in self.__transformations:
            sig = signature(transform)
            if len(sig.parameters) == 4:
                images = transform(self.__orgnr, self.__year, images, self.__data)
            elif len(sig.parameters) == 3:
                images = transform(self.__orgnr, self.__year, images)
            else:
                images = transform(images)

        return images
        

class SentinelDatasetIterator:
    @staticmethod
    def from_dataset(dataset: SentinelDataset):
        tuples = map(lambda l: l.split("/")[1:3], dataset.labels)
        tuples = [(orgnr, year, [], {}) for orgnr, year in tuples]
        return SentinelDatasetIterator(dataset=dataset, tuples=tuples)

    def __init__(self, 
                 dataset:SentinelDataset=None,
                 source=None,
                 tuples: List[Tuple[str, str, List[Callable], Dict[Any, Any]]]=None,
                 transformations: List[Callable]=None,
                 shuffle=None):

        if source is not None:  # Make a copy of the supplied iterator before applying optional arguments
            assert isinstance(source, SentinelDatasetIterator)

            self.__dataset = dataset or copy(source.__dataset)
            self.__transformations = copy(source.__transformations) if transformations is None else transformations
            self.__tuples = copy(source.__tuples) if tuples is None else tuples
            self.__shuffle = source.__shuffle if shuffle is None else shuffle
        
        else:  # Create a fresh instance with optionally provided arguments
            assert isinstance(dataset, SentinelDataset)

            self.__dataset = dataset
            self.__transformations: List[Callable] = transformations or []
            self.__tuples = tuples or []
            self.__shuffle = shuffle or False

    def split(self, split_ratio=0.8, shuffle=True, rand_seed=None):
        it = SentinelDatasetIterator(source=self)
        if shuffle:
            random.Random(rand_seed).shuffle(it.__tuples)

        n = int(len(self) * split_ratio)
        return it[:n], it[n:]
    
    def filter(self, predicate: Union[Callable[[str, str, SentinelImageSeriesSource, Dict[Any, Any]], bool]]):
        '''
        Return an iterator with the elements of this iterator filtered by the supplied predicate.
        Expect this to take a while if accessing the image source.
        '''

        it = self(shuffle=False, skip_transform=True)
        mask = map(lambda tuple: predicate(*tuple), it)
        tuples = itertools.compress(self.__tuples, mask)
        return SentinelDatasetIterator(source=self, tuples=list(tuples))

    def apply(self, transformation: Callable[[Any], Tuple]):
        '''
        Apply a function to every element in the collection, optionally reshaping the output.
        '''

        transformations = self.__transformations + [transformation]
        return SentinelDatasetIterator(source=self, transformations=transformations)

    def transform(self, transformation: Union[Callable[[SentinelImageSeriesSource], Any], Callable[[str, str, SentinelImageSeriesSource, Dict[Any, Any]], Any]]):
        '''
        Apply a transformation that will be performed on the image source.
        The transformation function must take either a single parameter (img_source), or three parameters (orgnr, year, and img_source),
        and return a numpy array (an image).
        '''

        new_tuples = []
        for orgnr, year, t_list, data in self.__tuples:
            new_tuples.append((orgnr, year, t_list + [transformation], data))
        
        return SentinelDatasetIterator(source=self, tuples=new_tuples)
        
    def augment(self, transformations: List[Union[Callable[[SentinelImageSeriesSource], Any], Callable[[str, str, SentinelImageSeriesSource], Any]]], keep_original=True):
        '''
        Apply multiple transformations that will be performed on the image source, generating more output images.
        Each transformation function must take either a single parameter (img_source), or three/four parameters (orgnr, year, img_source, [data]),
        and return a numpy array (an image).
        '''
        tuples = []

        for orgnr, year, t_list, data in self.__tuples:
            if keep_original:
                tuples.append((orgnr, year, t_list, data))
            
            for transformation in transformations:
                tuples.append((orgnr, year, t_list + [transformation], data))
        
        return SentinelDatasetIterator(source=self, tuples=tuples)

    def shuffled(self, should_shuffle=True):
        '''
        Return a shuffled dataset.
        A shuffled dataset will shuffle itself for each call to the iterator.
        '''
        return SentinelDatasetIterator(source=self, shuffle=should_shuffle)
    
    def with_data(self, func: Callable[[str, str, Dict], Dict], show_progress=False):
        '''
        Add key/value pairs of data that accompanies each image in the iterator.
        Supply a function that takes three arguments (orgnr, year, dict) and returns a dict that will be merged with any previous values.
        Return multiple dictionaries (tuple or list) to create multiple copies of an image, with different data dict.
        '''

        new_tuples = []

        tuples = self.__tuples
        if show_progress:
            from tqdm.autonotebook import tqdm
            tuples = tqdm(tuples)

        for orgnr, year, t_list, data in tuples:
            new_data = func(orgnr, year, dict(data))
            if isinstance(new_data, (tuple, list)):
                for d in new_data:
                    new_tuples.append((orgnr, year, t_list, {**data, **d}))
            else:
                new_tuples.append((orgnr, year, t_list, {**data, **new_data}))
        
        return SentinelDatasetIterator(source=self, tuples=new_tuples)

    def __process_tuple(self, tuple, image_source):
        orgnr, year, data  = itemgetter(0, 1, 3)(tuple)
        
        output = (orgnr, year, image_source, data)
        for transformation in self.__transformations:
            output = transformation(*output)

        return output

    def __call__(self, shuffle=None, skip_transform=False):
        '''
        Returns a copy of the iterator with optional changes.

        :param shuffle: Whether to shuffle before iterating. Overides previous settings.
        :param transform: Whether to skip the applied transformation functions. Does not affect augmentations and image transforms.
        '''
        
        kwargs = {}
        if shuffle is not None:
            kwargs["shuffle"] = shuffle
        if skip_transform == True:
            kwargs["transformations"] = []

        return SentinelDatasetIterator(source=self, **kwargs)

    def __iter__(self):
        if self.__shuffle:
            random.shuffle(self.__tuples)
        
        filename = self.__dataset.filename
        with h5py.File(filename, "r") as file:
            for tuple in self.__tuples:
                orgnr, year = tuple[0:2]
                img_source = SentinelImageSeriesSource(file[f"images/{orgnr}/{year}"], *tuple)
                yield self.__process_tuple(tuple, img_source)

    def __getitem__(self, key):
        tuples = self.__tuples
        if self.__shuffle:
            random.shuffle(tuples)
        
        # If key is a slice, eg. [0:10], we return a new iterator over the sequence
        if isinstance(key, slice):
            tuples = tuples[key]
            return SentinelDatasetIterator(source=self, tuples=tuples)
        
        # It's just an index
        elif isinstance(key, int):
            tuple = tuples[key]

            orgnr, year = tuple[0:2]
            images = self.__dataset.get_images(orgnr, year, raw=True)
            assert images is not None

            img_source = SentinelImageSeriesSource(images, *tuple)
            return self.__process_tuple(tuple, img_source)

        else:
            raise TypeError(f"Indices must be integers or slices, not {type(key)}")
    
    def __len__(self):
        return len(self.__tuples)

    def as_array(self):
        return np.asarray(self)