INPUT=/home/vht/hahoang/HackthonOCR/DATA/images
WEIGHT1=/home/vht/hahoang/HackthonOCR/CODE/mmocr/work_dirs/svtr/epoch_30.pth
CUDA_VISIBLE_DEVICES=1 python tools/infer_ensemble.py $INPUT \
 --rec SVTR \
 --rec-weight $WEIGHT1 \
 --out-dir /home/vht/hahoang/HackthonOCR/DATA/infer_svtr_private01 \
 --save_pred \
 --batch-size 2