源代码 https://github.com/zll17/Neural_Topic_Models
选用了其中的GSM，并对其做了微调
extract_topic_vae.py
使用VAE提取主题词

注：
1.本任务目的是为了提取N组topic_words及其对应的topic_distribution（也就是论文里提到的θ），在代码中，θ被表示为富有更多隐藏层的decoder，与原文并不是完全相符，但作者在[issue](https://github.com/zll17/Neural_Topic_Models/issues/8)中给出了合理的解释。