naver_textrecog_data_root = '/mnt/disk1/mbbank/OCR/DATA/data_quangnd/new_train'

naver_textrecog_train = dict(
    type='OCRDataset',
    data_root=naver_textrecog_data_root,
    ann_file='/mnt/disk1/mbbank/OCR/DATA/team/train.json',
    test_mode=False,
    pipeline=None)

naver_textrecog_test = dict(
    type='OCRDataset',
    data_root=naver_textrecog_data_root,
    ann_file='/mnt/disk1/mbbank/OCR/DATA/team/val.json',
    test_mode=True,
    pipeline=None)
