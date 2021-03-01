import numpy as np
import geopandas as gpd
import pandas as pd
from mask.geo_point_translator import GeoPointTranslator
from PIL import Image, ImageDraw


def add_mask_as_channel(mask, image_series):
    imgs_w_mask_channel = np.zeros((30, 100, 100, 13))
    for idx, img in enumerate(image_series):
        imgs_w_mask_channel[idx] = np.concatenate((img, mask[:, :, np.newaxis]), axis=2)
    return imgs_w_mask_channel


def apply_mask_to_image_series(mask, image_series):
    imgs_w_mask = np.zeros((30, 100, 100, 12))
    for idx, img in enumerate(image_series):
        imgs_w_mask[idx] = apply_mask_to_image(mask, img)
    return imgs_w_mask


def apply_mask_to_image(mask, image):
    channels = image.shape[2]
    new_img_array = np.empty(image.shape, dtype='float32')
    new_img_array[:, :, :channels] = image[:, :, :channels]
    for i in range(0, channels):
        new_img_array[:, :, i] = new_img_array[:, :, i] * mask

    return new_img_array


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
