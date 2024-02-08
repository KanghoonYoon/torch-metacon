## Initial Setting
savegraph=True
seeds="0 1 2 3 4"

## Change Here
gpu=2
dataset='citeseer' 


if [ $dataset == 'cora' ]; then
    inner_train_iters=100
    lr=0.01
    momentum=0.9
    droprate2=0.2
    coef2=1.0
    vic_coef1=1.0
    vic_coef2=2.0
    vic_coef3=2.0
    model="metacon_d"

elif [ $dataset == 'citeseer' ]; then
    inner_train_iters=100
    lr=0.01
    momentum=0.9
    droprate2=0.2
    coef2=1.0
    vic_coef1=1.0
    vic_coef2=1.0
    vic_coef3=1.0
    model="metacon_d"

elif [ $dataset == 'polblogs' ]; then
    inner_train_iters=500
    lr=0.01
    momentum=0.9
    droprate2=0.2
    coef2=0.5
    vic_coef1=1.0
    vic_coef2=1.0
    vic_coef3=2.0
    model="metacon_d"

elif [ $dataset == 'cora_ml' ]; then
    inner_train_iters=60
    lr=0.1
    momentum=0.9
    droprate2=0.2
    coef2=0.5
    vic_coef1=1.0
    vic_coef2=2.0
    vic_coef3=4.0
    model="metacon_d"
fi



for seed in $seeds
do
    CUDA_VISIBLE_DEVICES=$gpu python test/test_metacon_d.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 \
    --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 \
    --droprate1 0.0 --droprate2 $droprate2 --vic_coef1 $vic_coef1 --vic_coef2 $vic_coef2 --vic_coef3 $vic_coef3 --model $model --savegraph $savegraph

    CUDA_VISIBLE_DEVICES=$gpu python test/test_graphsage.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 --droprate1 0.0 --droprate2 $droprate2 --model $model 
    CUDA_VISIBLE_DEVICES=$gpu python test/test_gat.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 --droprate1 0.0 --droprate2 $droprate2 --model $model

    if [[ $seed -eq 4 ]]; then
        python result.py --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --attack_method $model
    fi
done  


