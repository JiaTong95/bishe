dataset="semeval16"

for target in "a" "cc" "fm" "hc" "la"
do
    # Step 1 数据预处理
    python data_processor.py --dataset=$dataset --target=$target
    # Step 2 建图
    python build_graph.py --dataset=$dataset --target=$target
    for topic_by in "btm" "vae"
    do
        python build_topic_graph.py --dataset=$dataset --target=$target --topic_by=$topic_by
        # Step 3 训练
        # python train.py --dataset=$dataset --target=$target
        python train.py --dataset=$dataset --target=$target --topic_by=$topic_by
    done
done