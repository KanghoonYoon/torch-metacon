## Initial Setting
savegraph=True
seeds="0 1 2 3 4"

## Change Here
gpu=0
dataset='polblogs' 


if [ $dataset == 'cora' ]; then
    inner_train_iters=100
    lr=0.01
    momentum=0.9
    droprate2=0.3
    coef2=0.4
    model="metacon"
    use_grad=True

elif [ $dataset == 'citeseer' ]; then
    inner_train_iters=100
    lr=0.01
    momentum=0.9
    droprate2=0.2
    coef2=0.6
    use_grad=False
    
elif [ $dataset == 'polblogs' ]; then
    inner_train_iters=500
    lr=0.01
    momentum=0.9
    droprate2=0.1
    coef2=0.1
    use_grad=True

elif [ $dataset == 'cora_ml' ]; then
    inner_train_iters=60
    lr=0.1
    momentum=0.9
    droprate2=0.1
    coef2=0.1
    use_grad=True
fi



for seed in $seeds
do
    CUDA_VISIBLE_DEVICES=$gpu python test/test_metacon.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 --droprate1 0.0 --droprate2 $droprate2 --model $model --savegraph $savegraph
    CUDA_VISIBLE_DEVICES=$gpu python test/test_graphsage.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 --droprate1 0.0 --droprate2 $droprate2 --model $model 
    CUDA_VISIBLE_DEVICES=$gpu python test/test_gat.py --seed $seed --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --hids 16 --lr $lr --inner_train_iters $inner_train_iters --momentum $momentum --coef1 1.0 --coef2 $coef2 --droprate1 0.0 --droprate2 $droprate2 --model $model

    if [[ $seed -eq 4 ]]; then
        python result.py --dataset $dataset --ptb_rate 0.05 --split 0.1 0.1 0.8 --attack_method $model
    fi
done  


