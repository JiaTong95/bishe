for dataset in "PStance"
do
    for target in "trump" "biden" "bernie"
    do
        python data_processor.py --dataset=$dataset --target=$target
        python tokenize_sentence.py --dataset=$dataset --target=$target 
        python train_bert_base.py --dataset=$dataset --target=$target
    done
done

# for dataset in "semeval16"
# do
#     for target in "a" "cc" "fm" "hc" "la" 
#     do
#         python data_processor.py --dataset=$dataset --target=$target
#         python tokenize_sentence.py --dataset=$dataset --target=$target 
#         python train_bert_base.py --dataset=$dataset --target=$target
#     done
# done