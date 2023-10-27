# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import os.path as osp
import os
from torch import nn
import torch

from mmocr.models.common.dictionary.dictionary import Dictionary
from mmocr.apis.inferencers import MMOCRInferencer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='results/',
        help='Output directory of results.')
    parser.add_argument(
        '--det',
        type=str,
        default=None,
        help='Pretrained text detection algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected det model. '
        'If it is not specified and "det" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--rec',
        type=str,
        default=None,
        help='Pretrained text recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected recog model. '
        'If it is not specified and "rec" is a model name of metafile, the '
        'weights will be loaded from metafile.',
        nargs='+')
    parser.add_argument(
        '--kie',
        type=str,
        default=None,
        help='Pretrained key information extraction algorithm. It\'s the path'
        'to the config file or the model name defined in metafile.')
    parser.add_argument(
        '--kie-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected kie model. '
        'If it is not specified and "kie" is a model name of metafile, the '
        'weights will be loaded from metafile.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    parser.add_argument(
        '--save_pred',
        action='store_true',
        help='Save the inference results to out_dir.')
    parser.add_argument(
        '--save_vis',
        action='store_true',
        help='Save the visualization results to out_dir.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'det', 'det_weights', 'rec', 'rec_weights', 'kie', 'kie_weights',
        'device'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)
    args = []
    for rec_weight in init_args['rec_weights']:
        init_args__ = init_args
        init_args__['rec_weights'] = rec_weight
        args.append((init_args__, call_args))
    return args


def main():
    args = parse_args()
    results = []

    out_dir = args[0][1]['out_dir']

    i = 0
    for init_args, call_args in args:
        i += 1
        model_name = 'id_' + str(i) + '_' +  osp.basename(init_args['rec_weights'])[:-4]
        call_args['out_dir'] = out_dir+'/'+model_name
        ocr = MMOCRInferencer(**init_args)
        result = ocr(**call_args)
        results.append(result['predictions'])
    n = len(results[0])
    m = len(results)

    dict_vn = Dictionary( dict_file=
    'configs/textrecog/abinet/../../../dicts/vietnamese.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=False,
    with_unknown=False)

    results_ensemble = []
    softmax = nn.Softmax()
    for i in range(n):
        img_path = results[0][i]['img_path'][0]
        frequency = defaultdict(int)
        for j in range(m):
            item = results[j][i]
            text = item['rec_texts'][0]
            frequency[text] += 1
        ensemble_probs = np.zeros_like(results[0][0]['probs'])
        total_score = 0
        for j in range(m):
            item = results[j][i]
            text = item['rec_texts'][0]
            ocr = min(item['rec_score_char'][0])
            score = (ocr**4) * frequency[text]
            total_score += score
            probs = item['probs']
            ensemble_probs += probs*score
        ensemble_probs /= total_score

        #decode
        index = np.argmax(ensemble_probs, axis=-1)
        value = np.max(ensemble_probs, axis=-1)
        p = 1.00
        text_ensemble = ''
        for idx, v in zip(index, value):
            c = dict_vn.dict[idx]
            if 'EOS' in c:
                break
            text_ensemble += c
            p = min(p, v)
        img_name = osp.basename(img_path)
        results_ensemble.append(f'{img_name} {text_ensemble} {p}')

    # save result
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, 'prediction.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_ensemble))

if __name__ == '__main__':
    main()
