from google.cloud import storage

storage_cli = storage.Client()
bucket = storage_cli.get_bucket('ml-workshop-123')

blob_oddball_path = bucket.get_blob('eeg1.csv')
oddball_data = blob_oddball_path.download_as_string()
print(oddball_data[0:1000])

