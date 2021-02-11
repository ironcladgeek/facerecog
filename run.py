import requests
from pymongo import MongoClient
from pathlib import Path
import pprint
from pipeline import producer, aligner
import torch

print('Is GPU in:')
print(torch.cuda.is_available())

# Create directory for downloading gallery and probe images
DATA_DIR = Path('downloads')
PROBE_DIR = DATA_DIR / 'probe'
GALLERY_DIR = DATA_DIR / 'gallery'
PROBE_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# connect to MongoDB to store results in one database
# which named your team_name and authorized with your team_pass
client = MongoClient(host='localhost', port=27017)      # TODO: change localhost to 'mongodb'
db = client.grinders
db.authenticate('grinders', '09172226951')
collection = db.team_name
collection.remove({})

# read the list of probe images
probe_directory = "http://localhost/images/probe/"      # TODO: change localhost to 'nginx'
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("The number of probe images is equal to " + str(len(probe_images)))

# read the list of gallery images
gallery_directory = "http://localhost/images/gallery/"  # TODO: change localhost to 'nginx'
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("The number of gallery images is equal to " + str(len(gallery_images)))

print('Downloading probe images ... ', end='')
for img_fn in probe_images:
    img_blob = requests.get(probe_directory + img_fn).content
    with open(str(PROBE_DIR / f'{img_fn}'), 'wb') as f:
        f.write(img_blob)
print('Done\n')

print('Downloading gallery images ... ', end='')
for img_fn in gallery_images:
    img_blob = requests.get(gallery_directory + img_fn).content
    with open(str(GALLERY_DIR / f'{img_fn}'), 'wb') as f:
        f.write(img_blob)
print('Done\n')

# get gallery/probe dataframes that contain images embeddings
gallery_df = producer(GALLERY_DIR, do_masking=True, do_augs=True)
probe_df = producer(PROBE_DIR)
print(f'gallery_df shape: {gallery_df.shape}')
print(f'probe_df shape: {probe_df.shape}\n')

# get similarities for probe images
# resulting dict is something like this:
# {'probe1': [{"Gallery2": 0.8505}, {"Gallery3": 0.5130}, {"Gallery1": 0.0134}, ...],
#  'probe2': [{"Gallery2": 0.8505}, {"Gallery3": 0.5130}, {"Gallery1": 0.0134}, ...],}
similarity_dct = aligner(gallery_df, probe_df)

# create a candidate list for each probe and insert it in the database
print('Writing results to DB ...')
for k, v in similarity_dct.items():
    candidate_list = {k: v}
    inserted_id = collection.insert_one(candidate_list).inserted_id

# print to make sure that result has been saved correctly
# for result_list in collection.find():
    # pprint.pprint(result_list)
