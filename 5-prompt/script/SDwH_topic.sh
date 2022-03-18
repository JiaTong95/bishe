dataset="SDwH"
device="cuda:0"
batch_size=64

topic_by="btm"
for target in "trump" "biden"
do
    for label_words_id in "1" "2" "3"
    do
        for template_id in "1" "2" "3" "4"
        do
            python train.py --topic_by=$topic_by --dataset=$dataset --target=$target --label_words_id=$label_words_id --template_id=$template_id --batch_size=$batch_size --device=$device > "logs/"$dataset"_"$target"_"$topic_by"_lid_"$label_words_id"_tid_"$template_id".log" 
        done
    done
done

topic_by="vae"
for target in "trump" "biden"
do
    for label_words_id in "1" "2" "3"
    do
        for template_id in "1" "2" "3" "4"
        do
            python train.py --topic_by=$topic_by --dataset=$dataset --target=$target --label_words_id=$label_words_id --template_id=$template_id --batch_size=$batch_size --device=$device > "logs/"$dataset"_"$target"_"$topic_by"_lid_"$label_words_id"_tid_"$template_id".log" 
        done
    done
done