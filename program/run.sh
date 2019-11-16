dir='11_9'
mkdir $dir
CUDA_VISIBLE_DEVICES=4 \
python NAS_main.py 2>&1|tee $dir/simple_log.txt &
