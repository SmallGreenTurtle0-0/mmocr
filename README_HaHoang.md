# Installation
```bash
pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc1'
pip install 'mmdet>=3.0.0rc0'

git clone https://github.com/open-mmlab/mmocr.git /mmocr
cd mmocr
pip install -r requirements.txt
pip install --no-cache-dir -e .
pip install -r requirements/albu.txt
```
# Convert txt to json
[Dataset guide](https://github.com/SmallGreenTurtle0-0/mmocr/blob/b18a09b2f063911a2de70f477aa21da255ff505d/docs/en/migration/dataset.md?plain=1#L3)

```bash
python tools/dataset_converters/textdet/data_migrator.py ${IN_PATH} ${OUT_PATH}
```

| ARGS     | Type                             | Description                                                                                                                                                      |
| -------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| in_path  | str                              | （Required）Path to the old annotation file.                                                                                                                     |
| out_path | str                              | （Required）Path to the new annotation file.                                                                                                                     |
| --task   | 'auto', 'textdet', 'textspotter' | Specifies the compatible task for the output dataset annotation. If 'textdet' is specified, the text field in coco format will not be dumped. The default is 'auto', which automatically determines the output format based on the the old annotation files. |
