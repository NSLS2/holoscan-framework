
import requests
from argparse import ArgumentParser

parser = ArgumentParser(description="Detector trigger script")
parser.add_argument(
    "-n",
    "--nimages",
    type=int,
    default=30,
    help=("number of images to emit"),
)

args = parser.parse_args()

REST = "http://0.0.0.0:8000"

print(f"{'-' * 20} Configure number of images {'-' * 20}")
nimages = {"value": args.nimages}
r = requests.put(f"{REST}/detector/api/1.8.0/config/nimages", json=nimages)
print(r.text)

print(f"{'-' * 20} Add user data {'-' * 20}")
user_data = {
    "value": {
        "id": "my_sample",
        "grid_scan_id": "flat",
        "zmq_consumer_mode": "spotfinder",
        "number_of_columns": 5,
        "number_of_rows": 6,
    }
}
r = requests.put(f"{REST}/stream/api/1.8.0/config/header_appendix", json=user_data)
print(r.text)

print(f" {'-' * 20} Arm detector {'-' * 20}")
r = requests.put(f"{REST}/detector/api/1.8.0/command/arm")
print("sequence id:", r.json()["sequence id"])

print(f"{'-' * 20} Trigger detector {'-' * 20}")
r = requests.put(f"{REST}/detector/api/1.8.0/command/trigger")
print(r)


print(f"{'-' * 20} Disarm detector {'-' * 20}")
r = requests.put(f"{REST}/detector/api/1.8.0/command/disarm")
print(r)
