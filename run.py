import requests
from pymongo import MongoClient
from pathlib import Path
import pprint
from pipeline import producer, aligner
from utils import format_time
import time
import torch

s = time.time()
print(f'Is GPU available: {torch.cuda.is_available()}\n')

# Create directory for downloading gallery and probe images
DATA_DIR = Path('downloads')
PROBE_DIR = DATA_DIR / 'probe'
GALLERY_DIR = DATA_DIR / 'gallery'
PROBE_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

# connect to MongoDB to store results in one database
# which named your team_name and authorized with your team_pass
client = MongoClient(host='mongodb', port=27017)
db = client.grinders
db.authenticate('grinders', '09172226951')
collection = db.grinders
collection.remove({})

print('Geeting images ...')
print('*' * 50)
# read the list of probe images
probe_directory = "http://nginx/images/probe/"
probe_images = requests.get(probe_directory + "images.txt").text.split()
print("Number of probe images: " + str(len(probe_images)))

# read the list of gallery images
gallery_directory = "http://nginx/images/gallery/"
gallery_images = requests.get(gallery_directory + "images.txt").text.split()
print("Number of gallery images: " + str(len(gallery_images)))

print('Downloading probe images ... ', end='')
for img_fn in probe_images:
    img_blob = requests.get(probe_directory + img_fn).content
    with open(str(PROBE_DIR / f'{img_fn}'), 'wb') as f:
        f.write(img_blob)
print('Done')

print('Downloading gallery images ... ', end='')
for img_fn in gallery_images:
    img_blob = requests.get(gallery_directory + img_fn).content
    with open(str(GALLERY_DIR / f'{img_fn}'), 'wb') as f:
        f.write(img_blob)
print('Done\n')

# get gallery/probe dataframes that contain images embeddings
print('Processing gallery images ...')
print('*' * 50)
gallery_df = producer(GALLERY_DIR, do_masking=True, do_augs=True, batch_size=8)
print('Processing probe images ...')
print('*' * 50)
probe_df = producer(PROBE_DIR, batch_size=8)

print(f'gallery_df shape: {gallery_df.shape}')
print(f'probe_df shape: {probe_df.shape}\n')

# get similarities for probe images
# resulting dict is something like this:
# {'probe1': [{"Gallery2": 0.8505}, {"Gallery3": 0.5130}, {"Gallery1": 0.0134}, ...],
#  'probe2': [{"Gallery2": 0.8505}, {"Gallery3": 0.5130}, {"Gallery1": 0.0134}, ...],}
print('Calculating similarities ...')
print('*' * 50)
similarity_dct = aligner(gallery_df, probe_df)

# create a candidate list for each probe and insert it in the database
print('\nWriting results to DB ...')
for k, v in similarity_dct.items():
    candidate_list = {k: v}
    inserted_id = collection.insert_one(candidate_list).inserted_id

# print to make sure that result has been saved correctly
print('\nThe last record in DB is:')
print('*' * 50)
res = [result_list for result_list in collection.find()]
pprint.pprint(res[-1])

elapsed = format_time(time.time() - s)
print(f'\nTotal application run time: {elapsed}')
