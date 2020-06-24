import argparse
import json
import os

import numpy as np
from tqdm import tqdm
import re

from utils import parse_annotation

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.path.join('/Users/UnicornKing/20180101_120040'))
args = parser.parse_args()

if __name__ == '__main__':
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    prev_file, prev_annot = annotations[0]
    for file, annot in tqdm(annotations[1:]):
        prev_labels = set([shape['label'] for shape in prev_annot['shapes']])
        labels = set([shape['label'] for shape in annot['shapes']])
        for label in prev_labels.intersection(labels):
            if re.search(r'_\d+', label) is None:
                continue
            prev_points = [shape['points'] for shape in prev_annot['shapes'] if shape['label'] == label][0]
            points = [shape['points'] for shape in annot['shapes'] if shape['label'] == label][0]

            prev_file_index = int(os.path.splitext(prev_file)[0])
            file_index = int(os.path.splitext(file)[0])
            diff = file_index - prev_file_index - 1

            interp_x1s = np.linspace(prev_points[0][0], points[0][0], diff).reshape((-1, 1))
            interp_y1s = np.linspace(prev_points[0][1], points[0][1], diff).reshape((-1, 1))
            interp_x2s = np.linspace(prev_points[1][0], points[1][0], diff).reshape((-1, 1))
            interp_y2s = np.linspace(prev_points[1][1], points[1][1], diff).reshape((-1, 1))

            interp_rects = np.column_stack([interp_x1s, interp_y1s, interp_x2s, interp_y2s]).astype(int)

            for idx, rect in enumerate(interp_rects):
                x1, y1, x2, y2 = rect.tolist()
                interp_file_index = prev_file_index + idx + 1
                interp_file = '{:08d}.json'.format(interp_file_index)
                interp_path = os.path.join(args.data_dir, interp_file)
                if os.path.exists(interp_path):
                    interp_annot = parse_annotation(interp_path)
                    interp_annot['shapes'].append({
                        "label": label,
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle",
                        "line_color": None,
                        "fill_color": None,
                    })
                else:
                    interp_annot = annot.copy()
                    interp_annot['shapes'] = [{
                        "label": label,
                        "points": [[x1, y1], [x2, y2]],
                        "shape_type": "rectangle",
                        "line_color": None,
                        "fill_color": None,
                    }]
                interp_annot_str = json.dumps(interp_annot, indent=4)
                with open(interp_path, 'w') as f:
                    f.write(interp_annot_str)

        prev_annot = annot
        prev_file = file
