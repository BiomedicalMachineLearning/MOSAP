import os
from PIL import Image
import glob
import io

def get_files_in_dir_recursively(directory, postfix='json'):
    """ list all the files in the directory with postfix recursively """
    files = [file for file in glob.glob(os.path.join(directory, '**/*.{0}'.format(postfix)), recursive=True)]
    return files