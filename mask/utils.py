import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mask.geo_point_translator import GeoPointTranslator
from PIL import Image, ImageDraw
from sentinel.utils import highlight_optimized_natural_color, to_rgb, plot_image


def add_mask_as_channel_to_image(mask, img):
    return np.concatenate((img, mask[:, :, np.newaxis]), axis=2)


def add_mask_as_channel(mask, image_source):
    # Reshape the mask to 3 dimensions
    mask = mask.reshape(100, 100, 1)

    if image_source.ndim == 4:
        # Duplicate the mask for each image
        n_imgs = image_source.shape[0]
        mask = np.tile(mask, (n_imgs, 1, 1, 1))
        
        return np.concatenate((image_source, mask), axis=3)

    # Add the mask as the last channel in each image
    return np.concatenate((image_source, mask), axis=2)


def apply_mask_to_image_series(mask, image_series):
    return image_series * mask.reshape(1, 100, 100, 1)


def apply_mask_to_image(mask, image):
    return image * mask.reshape(100, 100, 1)


def plot_masked_rgb_image(mask, single_image, transparency=0.0):
    mask = mask.astype(float)
    mask[mask == 0] = transparency

    rgb_image = to_rgb(single_image, highlight_optimized_natural_color)
    masked_image = apply_mask_to_image(mask, rgb_image)
    plot_image(masked_image)
    plt.show()

def generate_mask_image(bbox, gaard):
    # Replace (100, 100 by shapes): image.shape[1], image.shape[0]
    # if the image size overall changes
    y_max = 100
    x_max = 100
    mask_img = Image.new('1', (x_max, y_max), 0)
    bounds = bbox.geometry
    geo_translator = GeoPointTranslator(bounds)

    shapes = []

    if gaard.geometry.geom_type == 'Polygon':
        shapes.append(gaard.geometry.exterior.coords[:])
    else:
        # Multipolygon, handle each shape separately.
        [shapes.append(polygon.exterior.coords[:]) for polygon in gaard.geometry]

    for shape in shapes:
        field_polygon = []
        for point in shape:
            xy = geo_translator.lat_lng_to_screen_xy(point[1], point[0])
            x = xy['x']
            y = xy['y']
            field_polygon.append((x, y_max - y))
        ImageDraw.Draw(mask_img).polygon(field_polygon, outline=1, fill=1)

    return mask_img


def generate_mask_from_fields(bbox, fields_df):
    y_max = 100
    x_max = 100
    mask_img = Image.new('1', (x_max, y_max), 0)
    bounds = bbox.geometry
    geo_translator = GeoPointTranslator(bounds)

    shapes = []

    for index, field in fields_df.iterrows():
        if field.geometry.geom_type == 'Polygon':
            shapes.append(field.geometry.exterior.coords[:])
        else:
            # Multipolygon, handle each shape separately.
            [shapes.append(polygon.exterior.coords[:]) for polygon in field.geometry]


    for shape in shapes:
        field_polygon = []
        for point in shape:
            xy = geo_translator.lat_lng_to_screen_xy(point[1], point[0])
            x = xy['x']
            y = xy['y']
            field_polygon.append((x, y_max - y))
        ImageDraw.Draw(mask_img).polygon(field_polygon, outline=1, fill=1)

    return mask_img


def get_sentinel_shapes(bb_path, farms_path):
    import geopandas as gpd
    matrikkel_shp_gpd = gpd.read_file(bb_path)
    matrikkel_file_df = pd.DataFrame(matrikkel_shp_gpd)

    farm_shp_gpd = gpd.read_file(farms_path)
    farm_shp_df = pd.DataFrame(farm_shp_gpd)

    return matrikkel_file_df, farm_shp_df


def get_matrikkel_by_orgnr(orgnr, matrikkel_df):
    return matrikkel_df[matrikkel_df['orgnr'] == orgnr]


def mask_to_pil_image(mask_array):
    # For inspecting the mask array.
    return Image.fromarray(mask_array * 255)
