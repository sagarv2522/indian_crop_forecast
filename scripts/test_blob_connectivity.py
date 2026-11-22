"""Lightweight Azure Blob Storage connectivity test across all containers.
Uploads, reads back, and deletes a temporary test blob in each container.
"""

import os
import uuid
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not CONN_STR:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")

CONTAINERS = ["crops-data", "crops-raw", "crops-models", "crops-monitoring"]

print("=" * 60)
print("Azure Blob Storage Multi-Container Connectivity Test")
print("=" * 60)

service_client = BlobServiceClient.from_connection_string(CONN_STR)
account_name = service_client.account_name
print(f"Target Storage Account: {account_name}\n")

all_passed = True

for container_name in CONTAINERS:
    print(f"Testing container: [{container_name}] ... ", end="", flush=True)
    try:
        container_client = service_client.get_container_client(container_name)
        if not container_client.exists():
            print("FAILED (container does not exist)")
            all_passed = False
            continue

        test_blob_name = f"_connectivity_test_{uuid.uuid4().hex[:8]}.txt"
        test_payload = b"healthcheck: azure blob storage connection OK"

        # 1. Upload
        blob_client = container_client.get_blob_client(test_blob_name)
        blob_client.upload_blob(test_payload, overwrite=True)

        # 2. Read back
        downloaded = blob_client.download_blob().readall()
        assert downloaded == test_payload, "Downloaded content did not match payload"

        # 3. Delete
        blob_client.delete_blob()

        print("PASSED (Upload -> Read -> Delete OK)")

    except Exception as e:
        print(f"FAILED ({e})")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL 4 CONTAINERS PASSED CONNECTIVITY VERIFICATION!")
else:
    print("SOME CONTAINERS ENCOUNTERED ISSUES")
print("=" * 60)
