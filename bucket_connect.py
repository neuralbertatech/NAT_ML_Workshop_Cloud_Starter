'''
https://pypi.org/project/google-cloud-storage/
https://pypi.org/project/firebase-admin/ # takes a while
'''

from google.cloud import storage
from firebase_admin import credentials, firestore
import csv

storage_cli = storage.Client()
bucket = storage_cli.get_bucket('ml-workshop-123')
db = firestore.Client()

blob_oddball_path = bucket.get_blob('eeg1.csv')
oddball_data = blob_oddball_path.download_as_string()
# oddball_data = oddball_data.decode('utf-8')

f = open('eeg1.csv', 'wb')
w = csv.writer(f, delimiter = ',')
w.writerows([x.split(',') for x in oddball_data])
f.close()