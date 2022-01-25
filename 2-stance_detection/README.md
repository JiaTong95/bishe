### 模型说明
| 模型 | 备注 |
| --- | --- |
| LSTM | |
| ATAE_LSTM | |
| IAN | 这个loss有问题，或者说这个模型根本就学不了 |
| MemNet | |
| AOA | |
| TextCNN | |
| Bert_Spc | [CLS]+text+[SEP]+target+[SEP] |
| AEN_Bert | |
| LCF_Bert | torch transformer版本有问题，查官方文档未果 |


训练脚本在script中

bash script/PStance_trump_run.sh
