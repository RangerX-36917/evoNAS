dir='11_23'
mkdir $dir
CUDA_VISIBLE_DEVICES=7 \
python train.py 2>&1|tee $dir/one-5000.txt &
