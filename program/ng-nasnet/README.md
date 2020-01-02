# Training with DDP
Training:
```
./train_ddp.sh
```
Change `CUDA_VISIBLE_DEVICES` and `--nproc_per_node=#GPUs` in `train_ddp.ssh` to deploy it to specified GPUs

Change `config.json` to control the model and training parameters. When setting `start_droppath_rate` and `end_droppath_rate` to 0, the model will replace droppath to dropblock. 