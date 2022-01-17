import torch
import pickle
import time
import argparse
from GSM import GSM
from vae_utils import DocDataset


class Extract_topic_words:
    def __init__(self, args) -> None:
        self.taskname = args.taskname
        self.no_below = args.no_below
        self.no_above = args.no_above
        self.num_epochs = args.num_epochs
        self.n_topic = args.n_topic
        self.rebuild = args.rebuild
        self.batch_size = args.batch_size
        self.criterion = args.criterion
        self.auto_adj = args.auto_adj

    def main(self, hashtag):

        docSet = DocDataset(taskname=self.taskname, txtLines=txtLines,
                            no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=False)
        if auto_adj:
            no_above = docSet.topk_dfs(topk=20)
            docSet = DocDataset(taskname, no_below=no_below,
                                no_above=no_above, rebuild=rebuild, use_tfidf=False)

        voc_size = docSet.vocabsize
        print('voc size:', voc_size)

        model = GSM(bow_dim=voc_size, n_topic=n_topic,
                    taskname=taskname, device=device)
        model.train(train_data=docSet, batch_size=batch_size, test_data=docSet,
                    num_epochs=num_epochs, log_every=10, beta=1.0, criterion=criterion)
        model.evaluate(test_data=docSet)

        # 不知道以下代码是何意义
        txt_lst, embeds = model.get_embed(train_data=docSet, num=1000)
        with open('temp/topic_dist_gsm.txt', 'w', encoding='utf-8') as wfp:
            for t, e in zip(txt_lst, embeds):
                wfp.write(f'{e}:{t}\n')
        pickle.dump({'txts': txt_lst, 'embeds': embeds},
                    open('temp/gsm_embeds.pkl', 'wb'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('GSM topic model')
    parser.add_argument('--taskname', type=str,
                        default='cnews10k', help='Taskname e.g cnews10k')
    parser.add_argument('--no_below', type=int, default=5,
                        help='The lower bound of count for words to keep, e.g 10')
    parser.add_argument('--no_above', type=float, default=0.005,
                        help='The ratio of upper bound of count for words to keep, e.g 0.3')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
    parser.add_argument('--n_topic', type=int, default=5, help='Num of topics')
    parser.add_argument('--bkpt_continue', type=bool, default=False,
                        help='Whether to load a trained model as initialization and continue training.')
    parser.add_argument('--use_tfidf', type=bool, default=False,
                        help='Whether to use the tfidf feature for the BOW input')
    parser.add_argument('--rebuild', action='store_true',
                        help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (default=512)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
    parser.add_argument('--auto_adj', action='store_true',
                        help='To adjust the no_above ratio automatically (default:rm top 20)')

    args = parser.parse_args()

    etw = Extract_topic_words(args)
