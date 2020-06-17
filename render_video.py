import argparse
import os
import re

import cv2
import torchvision

from ops.models import TSN
from ops.transforms import *
from utils import make_dir_if_needed, parse_annotation
import torch
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', default='~/20180101_120040')
parser.add_argument('--num_segments', default=8, type=int)
parser.add_argument('--model',
                    default='checkpoint/TSM_nandos_RGB_resnet50_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar')
parser.add_argument('--device', default='cpu')

actions = list({'food_flip': 0,
                'food_drizzle': 1,
                'food_place': 2,
                'food_remove': 3,
                'fries_cooking': 4,
                'fries_serving': 5,
                'food_packaging': 6,
                'package_serving': 7,
                'food_sauce': 8,
                'food_warm_start': 9,
                'food_warm_end': 10,
                'burger_serving': 11}.keys())


def put_text(image, text, x, y, color, scale=5, thickness=2):
    font = cv2.FONT_HERSHEY_PLAIN
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
    image = cv2.rectangle(image, (x - 2, y - text_height - 4), (x + text_width, y), color, cv2.FILLED)
    image = cv2.putText(image, text, (x, y - 2), font, font_scale, (255, 255, 255), thickness)
    return image


if __name__ == '__main__':
    args = parser.parse_args()

    make_dir_if_needed(args.output_image_dir)
    annotations = {os.path.splitext(file)[0]: parse_annotation(os.path.join(args.data_dir, file)) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json'}

    model = TSN(len(actions), args.num_segments, 'RGB').to(args.device)

    r = np.random.randint(0, 255, (len(actions), 1))
    g = np.random.randint(0, 255, (len(actions), 1))
    b = np.random.randint(0, 255, (len(actions), 1))
    colors = np.column_stack([r, g, b]).tolist()

    cache = {}
    for file, annot in annotations:
        for shape in annot['shapes']:
            label = shape['label']
            if re.sub(r'_\d+', '', label) not in actions:
                continue

            image_path = os.path.join(args.frames_dir, file + '.jpg')
            image = cv2.imread(image_path)

            cache.setdefault(label, [])

            cache[label].append({'image': image, 'points': shape['points']})

            if len(cache[label]) == args.num_segments:

                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    GroupScale(int(model.scale_size)),
                    GroupCenterCrop(model.crop_size),
                    Stack(roll=False),
                    ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                    IdentityTransform(),
                ])

                rois = []

                for item in cache[label]:
                    x1, y1 = item['points'][0]
                    x2, y2 = item['points'][1]
                    roi = item['image'][y1:y2, x1:x2]
                    roi = transforms(roi)
                    rois.append(roi)

                input_ = torch.tensor(rois).to(args.device)

                output = model(input_)[0].argmax().cpu().numpy()

                label = actions[output]
                color = colors[output]

                for item in cache[label]:
                    x1, y1 = item['points'][0]
                    x2, y2 = item['points'][1]

                    image = item['image']
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)

                    image = put_text(image, label, x1, y1, color)



                cache[label].pop(0)
