dir='11_9'
mkdir $dir
CUDA_VISIBLE_DEVICES=6 \
python train.py 2>&1|tee $dir/one-4000.txt &
