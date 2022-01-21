源代码 https://github.com/zll17/Neural_Topic_Models
选用了其中的GSM，并对其做了微调
extract_topic_vae.py
使用VAE提取主题词

注：
1.本任务目的是为了提取N组topic_words及其对应的topic_distribution（也就是论文里提到的θ），在代码中，θ被表示为富有更多隐藏层的decoder，与原文并不是完全相符，但作者在[issue](https://github.com/zll17/Neural_Topic_Models/issues/8)中给出了合理的解释。

2.提取topic_words（原作者代码自带）
将一个n_topicxn_topic的单位矩阵送入decoder中，选出topK个对应的单词作为这n个topic的K个topic_words。理解为：这个单位矩阵视作由n_topic个θ（topic_distribution）组成的，每个向量都代表着概率为1的topic，送入decoder后，就可以得出各自topic的topic_words是什么。

3.提取topic_distribution
将一个1xbow_dim的全1矩阵送入encoder中，将得出的μ和σ标准化，得到z，再经过一层（n_topic, n_topic）的全连接层得到θ，再将得到的θ归一化，最终得到主题分布概率(topic_distribution)。这里的1xbow_dim的全1矩阵理解为：一个包含有所有单词的很长很长的句子。