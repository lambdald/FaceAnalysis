'''
Author: lidong
Date: 2021-06-25 15:27:50
LastEditors: lidong
LastEditTime: 2021-07-02 10:47:15
Description: 
https://github.com/yiminglin-ai/widerface-coco-convertor
'''
import argparse
import json
import os
import os.path as osp
import sys
from PIL import Image


def parse_wider_gt(dets_file_name, isEllipse=False):
    # -----------------------------------------------------------------------------------------
    '''
      Parse the FDDB-format detection output file:
        - first line is image file name
        - second line is an integer, for `n` detections in that image
        - next `n` lines are detection coordinates
        - again, next line is image file name
        - detections are [x y width height score]
      Returns a dict: {'img_filename': detections as a list of arrays}
    '''
    fid = open(dets_file_name, 'r')

    # Parsing the FDDB-format detection output txt file
    img_flag = True
    numdet_flag = False
    start_det_count = False
    det_count = 0
    numdet = -1

    det_dict = {}
    img_file = ''

    for line in fid:
        line = line.strip()

        if line == '0 0 0 0 0 0 0 0 0 0':
            if det_count == numdet - 1:
                start_det_count = False
                det_count = 0
                img_flag = True  # next line is image file
                numdet_flag = False
                numdet = -1
                det_dict.pop(img_file)
            continue

        if img_flag:
            # Image filename
            img_flag = False
            numdet_flag = True
            # print('Img file: ' + line)
            img_file = line
            det_dict[img_file] = []  # init detections list for image
            continue

        if numdet_flag:
            # next line after image filename: number of detections
            numdet = int(line)
            numdet_flag = False
            if numdet > 0:
                start_det_count = True  # start counting detections
                det_count = 0
            else:
                # no detections in this image
                img_flag = True  # next line is another image file
                numdet = -1

            # print 'num det: ' + line
            continue

        if start_det_count:
            # after numdet, lines are detections
            detection = [float(x) for x in line.split()]  # split on whitespace
            det_dict[img_file].append(detection)
            # print 'Detection: %s' % line
            det_count += 1

        if det_count == numdet:
            start_det_count = False
            det_count = 0
            img_flag = True  # next line is image file
            numdet_flag = False
            numdet = -1

    return det_dict

def parse_widerface_annot_file(data_path, dets_file_name) -> dict:
    '''
      Parse the WiderFace annotation file:
        The format of txt ground truth.
        File name
        Number of bounding box
        x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
    '''
    fid = open(dets_file_name, 'r')

    face_det_dict = {}
    n_bbox_to_read = 0
    img_name = ''
    for line in fid:
        line = line.strip()
        # print(img_name, n_bbox_to_read, line)
        if n_bbox_to_read == 0:
            img_name = line.strip()
            face_det_dict[img_name] = []
            n_bbox_to_read = -1
        elif n_bbox_to_read == -1:
            n_bbox_to_read = int(line)
            if n_bbox_to_read==0:
                n_bbox_to_read = 1
        else:
            bbox_info = [float(x) for x in line.split()]  # split on whitespace

            face_det_dict[img_name].append(bbox_info[:4] + [0])
            n_bbox_to_read -= 1

    face_det_dict = clean_dataset(data_path, face_det_dict)
    return face_det_dict


def clean_dataset(data_path, face_det_dict):
    """删除空数据

    Args:
        face_det_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    new_dict ={}
    for img_name, bboxs in face_det_dict.items():

        img_path = os.path.join(data_path, img_name)
        if not os.path.exists(img_path):
            continue
        new_bbox = []
        for box in bboxs:

            if sum(box[:4]) != 0 and box[2] != 0 and box[3] != 0:
                # box[2] += box[0]
                # box[3] += box[1]
                new_bbox.append(box)

        if len(new_bbox):
            new_dict[img_name] = new_bbox
    return new_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '-d', '--datadir', help="dir to widerface", default='data/widerface', type=str)

    parser.add_argument(
        '-s', '--subset', help="which subset to convert", default='all', choices=['all', 'train', 'val'], type=str)

    parser.add_argument(
        '-o', '--outdir', help="where to output the annotation file, default same as data dir", default=None)
    return parser.parse_args()


def convert_wider_annots(args):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    subset = ['train', 'val'] if args.subset == 'all' else [args.subset]
    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
    else:
        args.outdir = args.datadir

    categories = [{"id": 0, "name": 'face'}]
    for sset in subset:
        print(f'Processing subset {sset}')
        out_json_name = osp.join(args.outdir, f'wider_face_{sset}_annot_coco_style.json')
        data_dir = osp.join(args.datadir, f'WIDER_{sset}', 'images')
        img_id = 0
        ann_id = 0
        cat_id = 0

        ann_dict = {}
        images = []
        annotations = []
        ann_file = os.path.join(args.datadir, 'wider_face_split', f'wider_face_{sset}_bbx_gt.txt')
        # wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]
        wider_annot_dict = parse_widerface_annot_file(data_dir, ann_file)  # [im-file] = [[x,y,w,h], ...]

        for filename in wider_annot_dict.keys():
            if len(images) % 100 == 0:
                print("Processed %s images, %s annotations" % (
                    len(images), len(annotations)))

            image = {}
            image['id'] = img_id
            img_id += 1
            im = Image.open(os.path.join(data_dir, filename))
            image['width'] = im.height
            image['height'] = im.width
            image['file_name'] = filename
            images.append(image)

            for gt_bbox in wider_annot_dict[filename]:
                # gt_bbox = [x0,y0,w,h] + [attributes]
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat_id  # 0:"face" for WIDER
                ann['iscrowd'] = 0
                ann['area'] = gt_bbox[2] * gt_bbox[3]
                ann['bbox'] = gt_bbox[:4]   # [x,y,width,height]
                ann['attributes'] = gt_bbox[4:]

                if ann['area'] < 1:
                    continue
                annotations.append(ann)

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(out_json_name, 'w', encoding='utf8') as outfile:
            json.dump(ann_dict, outfile, indent=4, sort_keys=True)

if __name__ == '__main__':
    convert_wider_annots(parse_args())