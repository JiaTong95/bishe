dataset="semeval16"
target="hc"
device="cuda:1"

num_epoch=50
batch_size=16
seed=2020
dropout=0.1

model_names=("lstm" "text_cnn" "atae_lstm" "ian" "memnet" "aoa" "bert_spc" "aen_bert")

learning_rates=("1e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3")

for model_name in ${model_names[*]}
do
    for learning_rate in ${learning_rates[*]}
    do
        python train.py --model_name $model_name --dataset $dataset --target $target --dropout $dropout --num_epoch $num_epoch --batch_size $batch_size --seed $seed --learning_rate $learning_rate --device $device
    done
done

rm -f ./state_dict/*

# nohup bash script/semeval16_hc_run.sh > logs/_semeval16_hc.out &