import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert json to txt.')
    parser.add_argument(
        '--src',
        help='path to directory store json files')
    parser.add_argument(
        '--img_src',
        help='path to directory store images')
    parser.add_argument(
        '--dst',
        help='path to directory store result txt')
    args = parser.parse_args()
    return args

def json2txt(src, img_src, dst):
    print(dst)
    os.makedirs(dst, exist_ok=True)
    jsonfiles = os.listdir(src)
    
    imagefullnames = os.listdir(img_src)
    dict_fullname = {imagefullname[:-4]: imagefullname for imagefullname in imagefullnames}
    
    imagenames = []
    texts = []
    for jsonfile in tqdm(jsonfiles):
        path = os.path.join(src, jsonfile)
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        imagename = jsonfile[:-5]
        text = data['rec_texts'][0]
        imagenames.append(dict_fullname[imagename])
        if len(text) == 0:
            print(dict_fullname[imagename])
            text = 'n'
        texts.append(text)
        
    
    result = [(imagename + ' ' + text) for imagename, text in zip(imagenames, texts)]
    result = '\n'.join(result)
    with open(os.path.join(dst, 'prediction.txt'), 'w') as f:
        f.write(result)
        
def main():
    args = parse_args()
    json2txt(args.src, args.img_src, args.dst)

if __name__ == "__main__":
    main()