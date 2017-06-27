import time
import copy
import os
import sys
import tarfile
import re

from urllib.request import urlretrieve

VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
PASD_URL="https://codalabuser.blob.core.windows.net/public/%s"
JSON_REGEX='trainval_.*'

    
progress = 0
json_regex = re.compile(JSON_REGEX)

def printProgress(count, blockSize, totalSize):
    global progress
    
    prev_progress = progress
    progress = int(count * blockSize / totalSize * 100)
    if progress > prev_progress:
        print("Download %d%% complete." % progress)

        
if len(sys.argv) < 3 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("usage: python download.py <dataset> <folder>\n" \
          + "Download PASCAL in Detail data to <folder>.\n" \
          + "<dataset> options: 'pascal' to download VOCdevkit,\n"
          + "trainval_preview1 to download trainval_preview1.json.")
    exit(1)

if not os.path.isdir(sys.argv[2]):
    print("%s is not a directory." % sys.argv[2])
    exit(1)


rootdir = sys.argv[2]

if sys.argv[1].lower() == 'pascal':
    filepath = os.path.join(rootdir, "VOCtrainval_03-May-2010.tar")

    print("Downloading VOCdevkit 2010 to %s." % filepath)

    if os.path.exists(filepath):
        print("Tar file appears to already be downloaded. Using it.")
    else:
        urlretrieve(VOC_URL, filepath + '.download', reporthook=printProgress)
        os.rename(filepath + '.download', filepath)
        print("Download complete!")


    print("Unpacking. This may take a few minutes...")
    if os.path.exists(os.path.join(rootdir, 'VOCdevkit')):
        print("VOCdevkit directory already present. Aborting unpacking.")
    else:
        tar = tarfile.open(filepath)
        tar.extractall(path=rootdir)
        tar.close()

    print("Cleaning up.")
    if os.path.exists(os.path.join(rootdir, 'VOCdevkit')):
        os.remove(filepath)
elif json_regex.match(sys.argv[1].lower()):
    filename = sys.argv[1].lower() + '.json'
    filepath = os.path.join(rootdir, filename)
    url = PASD_URL % filename

    if os.path.exists(filepath):
        print("%s already exists. Aborting." % filepath)
        exit(1)

    print("Downloading %s to %s from:\n%s" % (filename, filepath, url))

    urlretrieve(url, filepath + '.download', reporthook=printProgress)
    os.rename(filepath + '.download', filepath)
    print("Download complete!")

