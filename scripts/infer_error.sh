INPUT=/home/vht/hahoang/HackthonOCR/DATA/error
WEIGHT=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged/best_NAVER_recog_word_acc_epoch_28.pth
WEIGHT1=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged_fold1/best_NAVER_recog_word_acc_epoch_34.pth
WEIGHT2=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged_fold2/best_NAVER_recog_word_acc_epoch_32.pth
WEIGHT3=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged_fold3/best_NAVER_recog_word_acc_epoch_36.pth
WEIGHT4=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged_fold4/best_NAVER_recog_word_acc_epoch_30.pth
WEIGHT5=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/abinet_case_unchanged_fold5/best_NAVER_recog_word_acc_epoch_34.pth

CUDA_VISIBLE_DEVICES=0 python tools/infer_ensemble.py $INPUT \
 --rec ABINet \
 --rec-weight $WEIGHT $WEIGHT1 $WEIGHT2 $WEIGHT3 $WEIGHT4 $WEIGHT5\
 --out-dir /home/vht/hahoang/HackthonOCR/DATA/infer/infer_error \
 --save_pred \
 --batch-size 128

