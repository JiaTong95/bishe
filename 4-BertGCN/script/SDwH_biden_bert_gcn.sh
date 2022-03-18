dataset="SDwH"
target="biden"
device="cuda:1"
topic_by="btm"

python data_processor.py --dataset=$dataset --target=$target
python build_graph.py --dataset=$dataset --target=$target --topic_by=$topic_by
python tokenize_sentence.py --dataset=$dataset --target=$target
python train_bert_base.py --dataset=$dataset --target=$target --device=$device

for gcn_lr in "0.01" "0.02" "0.05" "0.001" "0.002" "0.005"
do
    for bert_lr in "1e-5" "2e-5" "5e-5"
    do
        python train_bert_gcn.py --gcn_lr=$gcn_lr --bert_lr=$bert_lr --device=$device --dataset=$dataset --target=$target
    done
done