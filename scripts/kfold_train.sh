for i in {0..4}
do
   python train.py --model effectortransformer --data_dir data --batch_size 32 --lr 5e-5 --weight_decay 4e-5 --dropout_rate 0.4 --num_layers 1 --num_heads 4 --warm_epochs 1 --patience 5 --lr_scheduler cosine --lr_decay_steps 30 --kfold 5 --fold_num $i --log_dir runs/attempt_cv
done