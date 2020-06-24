import json
import os


def parse_annotation(path):
    with open(path, 'r') as f:
        annotation = json.loads(f.read())
        return annotation


def make_dir_if_needed(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
