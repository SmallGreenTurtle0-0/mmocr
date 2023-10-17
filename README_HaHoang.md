# Installation
```bash
mkvirtualenv -p python3.8 mmocr
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip3 install openmim
# git clone https://github.com/SmallGreenTurtle0-0/mmocr.git
cd mmocr
pip install -r requirements.txt
pip install --no-cache-dir -e .
pip install -r requirements/albu.txt

# if mmengine has bug
python -m pip install pip~=19.0
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
pip install 'mmdet>=3.0.0rc0'
```
# Convert txt to json

Bời format của file annotation của mmocr hơi khác. 
Nên cần chuyển sang dạng của mmocr.
Sau đâu là các chuyển. Đọc thêm ở [Dataset guide](https://github.com/SmallGreenTurtle0-0/mmocr/blob/b18a09b2f063911a2de70f477aa21da255ff505d/docs/en/migration/dataset.md?plain=1#L3)

```bash
python tools/dataset_converters/textrecog/data_migrator.py ${IN_PATH} ${OUT_PATH} --format ${txt, jsonl, lmdb}
```

| ARGS     | Type                   | Description                                       |
| -------- | ---------------------- | ------------------------------------------------- |
| in_path  | str                    | （Required）Path to the old annotation file.      |
| out_path | str                    | （Required）Path to the new annotation file.      |
| --format | 'txt', 'jsonl', 'lmdb' | Specify the format of the old dataset annotation. |

# Train
Lưu ý sửa path đến data và annotation ở trong file config.
```bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/textrecog/abinet/abinet-vision_20e_st-an_mj_naver_custom.py
```

