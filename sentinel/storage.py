import numpy as np
import h5py

'''
When storing images it is much more efficient to store integer values instead of floats,
however, values must be mapped from 0...1 to real numbers to store any valuable data,
using this value as a multiplier.
'''
int_scale = 255

filename = "data/images_raw.h5"


def store_images(images, farmer_id, year, scale=int_scale):
    # Convert image values to integer type
    data = (np.array(images)*scale).astype(int)

    file = h5py.File(filename, "a")
    try:
        # Delete existing images (if they exist)
        if f"images/{farmer_id}/{year}" in file:
            del file[f"images/{farmer_id}/{year}"]
        
        file.create_dataset(
            name=f"images/{farmer_id}/{year}",
            data=data,
            compression="gzip",
            compression_opts=2,
        )
    finally:
        file.close()

def get_images(farmer_id, year, scale=int_scale):
    if not h5py.is_hdf5(filename):
        return False
    
    file = h5py.File(filename, "r+")
    try:
        if f"images/{farmer_id}/{year}" in file:
            dset = file[f"images/{farmer_id}/{year}"]
            return np.array(dset) / scale
    finally:
        file.close()

def if_exists(farmer_id, year):
    if not h5py.is_hdf5(filename):
        return False

    file = h5py.File(filename, "r+")
    try:
        return f"images/{farmer_id}/{year}" in file
    finally:
        file.close()
