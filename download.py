import time
import copy
import os
import sys
import tarfile

from urllib.request import urlretrieve

VOC_URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
    
progress = 0

def printProgress(count, blockSize, totalSize):
    global progress
    
    prev_progress = progress
    progress = int(count * blockSize / totalSize * 100)
    if progress > prev_progress:
        print("Download %d%% complete." % progress)

        
if len(sys.argv) < 2 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print("usage: python download.py <folder>\n" \
        + "Download VOCdevkit 2010 to <folder>.")
    exit(1)

if not os.path.isdir(sys.argv[1]):
    print("%s is not a directory." % sys.argv[1])
    exit(1)


rootdir = sys.argv[1]
filepath = os.path.join(rootdir, "VOCtrainval_03-May-2010.tar")

print("Downloading VOCdevkit 2010 to %s." % filepath)

if os.path.exists(filepath):
    print("Tar file appears to already be downloaded. Using it.")
else:
    urlretrieve(VOC_URL, filepath + '.download', reporthook=printProgress)
    os.rename(filepath + '.download', filepath)
    print("Download complete!")

print("\n\n")

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
