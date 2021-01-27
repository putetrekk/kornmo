from sentinelhub import SHConfig, MimeType, CRS, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions
from .sentinel_evalscripts import true_color
import datetime


# In case you put the credentials into the configuration file you can leave this unchanged
CLIENT_ID = ''
CLIENT_SECRET = ''

# Add your credentials into the configuration file using the following command:
# sentinelhub.config --sh_client_id <your client id> --sh_client_secret <your client secret>

config = SHConfig()
if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

if config.sh_client_id == '' or config.sh_client_secret == '':
    print("Warning! To use Sentinel Hub services, please provide the credentials (client ID and client secret).")

def download_images_in_range(bbox, resolution, start, end, num, evalscript=true_color, maxcc=None, size=None):
    slots = []
    start = datetime.datetime(*start)
    end = datetime.datetime(*end)

    n_chunks = num + 1
    tdelta = (end - start) / (n_chunks-1)
    edges = [(start + i*tdelta).date().isoformat() for i in range(n_chunks)]
    slots.extend([(edges[i], edges[i+1]) for i in range(len(edges)-1)])

    return download_image_series(bbox, resolution, slots, evalscript, maxcc, size)


def download_image_series(bbox, resolution, slots, evalscript=true_color, maxcc=None, size=None):
    bbox = BBox(bbox=bbox, crs=CRS.WGS84)
    if size is None:
        size = bbox_to_dimensions(bbox, resolution=resolution)

    def create_request(time_interval):
        return SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order='leastCC',
                    maxcc=maxcc,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=config,
        )

    # create a list of requests
    list_of_requests = [create_request(slot) for slot in slots]
    list_of_requests = [request.download_list[0]
                        for request in list_of_requests]

    # download data with multiple threads
    imgs = SentinelHubDownloadClient(config=config).download(
        list_of_requests, max_threads=16)

    
    return imgs, slots
