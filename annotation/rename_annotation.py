import argparse
import json
import os

import numpy as np
from tqdm import tqdm
import re

from utils import parse_annotation

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join('/Users/UnicornKing/20180101_120040'))
parser.add_argument('--from_label', default='souce_dumpling_2')
parser.add_argument('--to_label', default='food_sauce_2')
args = parser.parse_args()

if __name__ == '__main__':
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    for file, annot in tqdm(annotations[1:]):
        for shape in annot['shapes']:
            if shape['label'] == args.from_label:
                shape['label'] = args.to_label
        annot_path = os.path.join(args.data_dir, file)
        annot_str = json.dumps(annot, indent=4)
        with open(annot_path, 'w') as f:
            f.write(annot_str)