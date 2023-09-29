python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/thin.yaml my_dataset/eval3_s/ /mnt/c/Users/User/Desktop/lama/eval/thin/in/ --ext jpg


python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/med.yaml my_dataset/eval3_s/ /mnt/c/Users/User/Desktop/lama/eval/med/in/ --ext jpg
python3 bin/gen_mask_dataset.py $(pwd)/configs/data_gen/big.yaml my_dataset/eval3_s/ /mnt/c/Users/User/Desktop/lama/eval/big/in/ --ext jpg
python bin/predict.py model.path=/content/lama/LaMa_models/big-lama-with-discr/ indir=/mnt/c/Users/User/Desktop/lama/eval/med/in/ outdir=/mnt/c/Users/User/Desktop/lama/eval/med/out_lm/ model.checkpoint=best.ckpt
python bin/predict.py model.path=/content/lama/LaMa_models/big-lama-with-discr/ indir=/mnt/c/Users/User/Desktop/lama/eval/big/in/ outdir=/mnt/c/Users/User/Desktop/lama/eval/big/out_lm/ model.checkpoint=best.ckpt
python -m pytorch_fid --batch-size=1 ~/lama/my_dataset/train/ /mnt/c/Users/User/Desktop/lama/eval/med/out_lm/
python -m pytorch_fid --batch-size=1 ~/lama/my_dataset/train/ /mnt/c/Users/User/Desktop/lama/eval/big/out_lm/



python3 bin/evaluate_predicts.py \
$(pwd)/configs/eval2_gpu.yaml \
$(pwd)/my_dataset/eval/random_medium_512/ \
$(pwd)/inference/my_dataset/random_medium_512 \
$(pwd)/inference/my_dataset/random_medium_512_metrics.csv