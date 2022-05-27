import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

def moveFile(numFile,source,destination,folderName):
        listFiles = os.listdir(source)
        random.shuffle(listFiles)
        os.mkdir(os.path.join(destination,folderName))
        print('successfuly create directory %s',os.path.join(source,folderName)) \
            if os.path.isdir(os.path.join(destination,folderName)) \
                else print("failed create directory %s",folderName)
        for i in range(numFile):
            os.rename(os.path.join(source,listFiles[i]),os.path.join(destination,folderName,listFiles[i]))
        print('finish moving file into %s',folderName)
            
def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.
    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    
    # make directory for train/validation/test, and move files into them
    moveFile(60,source,destination,'train')
    moveFile(30,source,destination,'val')
    moveFile(10,source,destination,'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    print(os.getcwd())
    args = parser.parse_args(['--source','/app/project/nd013-c1-vision-starter/data/processed',\
        '--destination','/app/project/nd013-c1-vision-starter/data'])
    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)
    # split("data/processed","data")