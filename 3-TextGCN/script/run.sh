# Step 1 数据预处理
python data_processor.py --dataset=SDwH --target=trump

# Step 1.1 btm 提取主题词
python extract_topic_btm.py --dataset=SDwH --target=trump
#如果不用多线程，要跑上十几天

# Step 1.2 vae 提取主题词
python extract_topic_vae.py --dataset=SDwH --target=trump

# Step 2 建图
python build_graph.py --dataset=SDwH --target=trump
python build_topic_graph.py --dataset=SDwH --target=trump

# Step 3 训练
python train.py --dataset=SDwH --target=trump
python train.py --dataset=SDwH --target=trump --topic_by=vae

