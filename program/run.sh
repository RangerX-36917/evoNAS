dir='data/11_28_A'
mkdir $dir
CUDA_VISIBLE_DEVICES=0,1 \
python NAS_main.py 2>&1|tee -a $dir/simple_log.txt &
