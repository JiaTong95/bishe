dataset="SDwH"
target="trump"
device="cuda:0"

learning_rates=("1e-5" "2e-5" "3e-5" "5e-5" "1e-4" "5e-4" "1e-3" "5e-3")
max_epochs=("10" "20" "50" "100")
batch_sizes=("16" "32" "64" "128")

for max_epoch in ${max_epochs[*]}
do
    for batch_size in ${batch_sizes[*]}
    do  
        for learning_rate in ${learning_rates[*]}
        do
            python train_2.py --dataset $dataset --target $target --device $device --learning_rate $learning_rate --max_epoch $max_epoch --batch_size $batch_size > "logs/"$learning_rate"_"$max_epoch"_"$batch_size".log" 
        done
    done
done

rm -f ./state_dict/*

# nohup bash script/PStance_bernie_run.sh > logs/_PStance_bernie.out &