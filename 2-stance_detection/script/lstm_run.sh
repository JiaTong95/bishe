model_name="lstm"
num_epoch=10
batch_size=16
seed=2020
target="trump"
datasets=("SDwH")
dropouts=("0.1")
learning_rates=("1e-5")

for dataset in ${datasets[*]}
do
    for dropout in ${dropouts[*]}
    do
        for learning_rate in ${learning_rates[*]}
        do
            python train.py --model_name $model_name --dataset $dataset --target $target --dropout $dropout --num_epoch $num_epoch --batch_size $batch_size --seed $seed --learning_rate $learning_rate
        done
    done
done

rm -f ./state_dict/*