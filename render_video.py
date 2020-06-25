import argparse
import os
import re

import cv2
import torchvision

from ops.models import TSN
from ops.transforms import *
from utils import make_dir_if_needed, parse_annotation, resize_image
import torch
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
import pandas as pd
np.random.seed(45)

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', default='/Users/UnicornKing/20180101_120040')
parser.add_argument('--num_segments', default=8, type=int)
parser.add_argument('--model',
                    default='checkpoint/TSM_nandos_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.pth.tar')
parser.add_argument('--meta_output_path',
                    default='out.csv')
parser.add_argument('--output_path',
                    default='out.avi')
parser.add_argument('--device', default='cpu')
parser.add_argument('--fps', default=10, type=int)

action_to_idx = {'food_flip': 0,
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
                 'burger_serving': 11}
idx_to_action = {v: k for k, v in action_to_idx.items()}


def put_text(image, text, x, y, color, scale=5, thickness=2):
    font = cv2.FONT_HERSHEY_PLAIN
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=scale, thickness=thickness)[0]
    image = cv2.rectangle(image, (x - 2, y - text_height - 4), (x + text_width, y), color, cv2.FILLED)
    image = cv2.putText(image, text, (x, y - 2), font, scale, (255, 255, 255), thickness)
    return image


if __name__ == '__main__':
    args = parser.parse_args()

    arch = 'resnet50'
    tsn = TSN(len(action_to_idx), args.num_segments, 'RGB',
                base_model=arch,
                consensus_type='avg',
                dropout=0.5,
                img_feature_dim=256,
                partial_bn=False,
                pretrain='imagenet',
                is_shift=True,
                shift_div=8,
                shift_place='blockres',
                fc_lr5=False,
                temporal_pool=False,
                non_local=False).to(args.device)
    model = torch.nn.DataParallel(tsn, device_ids=None).to(args.device)
    sd = torch.load(args.model, map_location=torch.device(args.device))['state_dict']
    model.load_state_dict(sd)
    model.eval()

    meta = pd.DataFrame(columns=['action', 'time_start', 'time_end'])

    r = np.random.randint(0, 255, (len(action_to_idx), 1))
    g = np.random.randint(0, 255, (len(action_to_idx), 1))
    b = np.random.randint(0, 255, (len(action_to_idx), 1))
    colors = np.column_stack([r, g, b]).tolist()

    annot_cache = {}
    video_writer = cv2.VideoWriter('out.avi', 0, args.fps, (1200, 675))
    for i, frame in tqdm(enumerate(sorted(glob.glob(os.path.join(args.frames_dir, '*.jpg')))[:1000][::2])):
        frame_name = os.path.splitext(frame)[0]
        annot_name = frame_name + '.json'
        annot_path = os.path.join(args.frames_dir, annot_name)
        image_path = os.path.join(args.frames_dir, frame)
        image = cv2.imread(image_path)
        if os.path.exists(annot_path):
            annot = parse_annotation(annot_path)
            for shape in annot['shapes']:
                label = shape['label']
                if re.sub(r'_\d+', '', label) not in action_to_idx:
                    continue

                annot_cache.setdefault(label, [])

                annot_cache[label].append({'image': image, 'points': shape['points']})

                if len(annot_cache[label]) == args.num_segments:

                    if label not in meta['action'].values:
                        meta = meta.append({
                            'action': label,
                            'time_start': i / args.fps,
                            'time_end': i / args.fps
                        }, ignore_index=True)
                    else:
                        meta.loc[meta['action'] == label, 'time_end'] = i / 10

                    transforms = torchvision.transforms.Compose([
                        GroupToPIL(),
                        GroupScale(int(tsn.scale_size)),
                        GroupCenterCrop(tsn.crop_size),
                        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
                        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
                        GroupNormalize(tsn.input_mean, tsn.input_std),
                    ])

                    rois = []

                    for item in annot_cache[label]:
                        x1, y1 = item['points'][0]
                        x2, y2 = item['points'][1]
                        roi = item['image'][y1:y2, x1:x2]
                        rois.append(roi)

                    rois = transforms(rois).to(args.device)

                    output = model(rois)
                    output = int(output[0].detach().argmax().cpu().numpy())

                    label_name = idx_to_action[output]
                    color = colors[output]

                    x1, y1 = shape['points'][0]
                    x2, y2 = shape['points'][1]

                    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=3)

                    image = put_text(image, label_name, x1, y1, color, 3)

                    annot_cache[label].pop(0)
            image = resize_image(image, 1200)
            video_writer.write(image)
        else:
            image = resize_image(image, 1200)
            video_writer.write(image)
    video_writer.release()
    meta.to_csv(args.meta_output_path, index=False)

