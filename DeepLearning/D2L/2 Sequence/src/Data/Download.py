import os
import hashlib
import re
import requests
import tarfile, zipfile
from Config import cfgG

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB = {
    "time_machine" : (DATA_URL + "timemachine.txt", "090b5e7e70c295757f55df93cb0a180b9691891a"),
    "fra-eng" : (DATA_URL + "fra-eng.zip", "94646ad1522d915e7b0f9296181140edcf86a4f5")
}

def download(url, folder=cfgG.data_dir, sha1_hash=None):
    if not url.startswith("http"):
        url, sha1_hash = DATA_HUB[url] # For back compatability
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split("/")[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def read_time_machine():
    with open(download("time_machine"), "r") as f:
        lines = f.readlines()
    return [re.sub("[^A-Za-z]+", " ", line).strip().lower() for line in lines]


def read_data_nmt():
    data_dir = download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), "r", encoding="utf-8") as f:
        return f.read()
