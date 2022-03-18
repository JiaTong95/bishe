dataset="SDwH"
target="trump"
device="cuda:0"

for gcn_lr in "0.01" "0.02" "0.05" "0.001" "0.002" "0.005"
do
    for bert_lr in "1e-5" "2e-5" "5e-5"
    do
        python train_bert_gcn.py --gcn_lr=$gcn_lr --bert_lr=$bert_lr --device=$device --dataset=$dataset --target=$target
    done
done