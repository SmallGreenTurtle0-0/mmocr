INPUT=/mnt/disk1/mbbank/OCR/DATA/data_quangnd/test
WEIGHT=/mnt/disk1/mbbank/OCR/CODE/mmocr/work_dirs/abinet_case_unchanged/best_NAVER_recog_word_acc_epoch_28.pth
CUDA_VISIBLE_DEVICES=1 python tools/infer.py $INPUT \
--rec ABINet \
--rec-weight $WEIGHT \
--out-dir /mnt/disk1/mbbank/OCR/CODE/mmocr/publictest_infer_case_unchanged \
--save_pred 