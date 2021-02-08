import h5py
import numpy as np
import random


class SentinelDataset:
    FILE = "data/sentinelhub/images_raw.h5"

    '''
    When storing images it is much more efficient to store integer values instead of floats,
    however, values must be mapped from 0...1 to real numbers to store any valuable data,
    using this value as a multiplier.
    '''
    INT_SCALE = 255

    def __init__(self):
        self.labels = self.__load_labels()

    def __load_labels(self):
        labels = []
        def visit_func(name, object):
            if not isinstance(object, h5py.Dataset):
                return
            labels.append(name)

        with h5py.File(self.FILE, "r+") as file:
            file.visititems(visit_func)
        return labels

    def get_image_samples(self, num=10):
        if num > len(self.labels):
            num = len(self.labels)
        
        sample_labels = random.sample(self.labels, num)
        
        samples = np.zeros(shape=(num, 3), dtype=object)
        idx = 0
        with h5py.File(self.FILE, "r+") as file:
            for label in sample_labels:
                images = file[label][()] / self.INT_SCALE
                orgnr, year = self.__extract_orgnr_year(label)
                samples[idx] = (orgnr, year, images)
                idx += 1
        
        return samples

    def get_images(self, orgnr, year):
        label = f"images/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.FILE, "r+") as file:
                return file[label][()] / self.INT_SCALE

    def del_images(self, orgnr, year):
        label = f"images/{orgnr}/{year}"
        if label in self.labels:
            with h5py.File(self.FILE, "r+") as file:
                del file[label]
                self.labels.remove(label)
                print(f"Deleted image dataset '{label}'.")
        else:
            print(f"Image dataset '{label}' is already deleted.")
    
    def store_images(self, images, farmer_id, year):
        # Convert image values to integer type
        data = (np.array(images) * self.INT_SCALE).astype(int)

        with h5py.File(self.FILE, "a") as file:
            label = f"images/{farmer_id}/{year}"
            # Delete existing images (if they exist)
            if label in file:
                del file[f"images/{farmer_id}/{year}"]
            else:
                self.labels.append(label)
            
            file.create_dataset(
                name=f"images/{farmer_id}/{year}",
                data=data,
                compression="gzip",
                compression_opts=2,
            )

    def contains(self, farmer_id, year):
        return f"images/{farmer_id}/{year}" in self.labels

    @staticmethod
    def __extract_orgnr_year(label):
        parts = label.split("/")
        return parts[1], parts[2]
