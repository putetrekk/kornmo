import matplotlib.pyplot as plt
import numpy as np
import math


true_color = lambda x: [x[3], x[2], x[1]]

ndvi = lambda B: (B[7] - B[3]) / (B[7] + B[3])

L = 0.6
savi = lambda B: (B[7] - B[3]) / (B[7] + B[3] + L) * (1.0 + L)

infrared_agriculture = lambda B: [B[3], B[7], B[1]]

false_color_infrared = lambda B: [B[7], B[3], B[2]]

def highlight_optimized_natural_color(B):
    g = 0.6
    B = [B[3] * g, B[2] * g, B[1] * g]
    return [B[0]**(1/3) - 0.035, B[1]**(1/3) - 0.035, B[2]**(1/3) - 0.035]


def to_rgb(img, map_func=true_color):
    '''
    Apply a mapping function to convert a multiband image into rgb components.

    true color: to_rgb(img, lambda x: [x[3], x[2], x[1]])
    '''
    shape = img.shape
    newImg = []
    for row in range(0, shape[0]):
        newRow = []
        for col in range(0, shape[1]):
            newRow.append(map_func(img[row][col]))
        newImg.append(newRow)
    return np.array(newImg)


def plot_image(image, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting RGB images.
    """
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    
    if factor == 'auto':
        factor = 1 / np.percentile(image, 99)

    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    
    ax.set_xticks([])
    ax.set_yticks([])


def plot_image_grid(images, ncols, nrows, factor=1.0, clip_range = None, **kwargs):
    """
    Utility function for plotting a grid of RGB images
    """
    subplot_kw = {'xticks': [], 'yticks': [], 'frame_on': False}

    aspect_ratio = len(images[0][0]) / len(images[0])

    _, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows),
                            subplot_kw=subplot_kw)

    for idx, image in enumerate(images):
        ax = axs
        if ncols == 1 or nrows == 1:
            ax = axs[idx]
        else:
            ax = axs[idx // ncols][idx % ncols]
        
        img_factor = 1 / np.percentile(image, 99) if factor == 'auto' else factor
        
        if clip_range is not None:
            ax.imshow(np.clip(image * img_factor, *clip_range), **kwargs)
        else:
            ax.imshow(image * img_factor, **kwargs)

    plt.tight_layout()

#
# The following methods for calculating a bounding box in WGS84 is taken from:
# https://stackoverflow.com/a/238558
#


def deg2rad(degrees):
    return math.pi*degrees/180.0


# radians to degrees
def rad2deg(radians):
    return 180.0*radians/math.pi


# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]


# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )


# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lng = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    lat_min = lat - halfSide/radius
    lat_max = lat + halfSide/radius
    lng_min = lng - halfSide/pradius
    lng_max = lng + halfSide/pradius

    return (rad2deg(lng_min), rad2deg(lat_min),rad2deg(lng_max), rad2deg(lat_max))
