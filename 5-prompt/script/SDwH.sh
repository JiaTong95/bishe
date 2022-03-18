dataset="SDwH"
device="cuda:1"
batch_size=64

for target in "trump" "biden"
do
    for label_words_id in "1" "2" "3"
    do
        for template_id in "1" "2" "3" "4"
        do
            python train.py --dataset=$dataset --target=$target --label_words_id=$label_words_id --template_id=$template_id --batch_size=$batch_size --device=$device > "logs/"$dataset"_"$target"_lid_"$label_words_id"_tid_"$template_id".log" 
        done
    done
done
