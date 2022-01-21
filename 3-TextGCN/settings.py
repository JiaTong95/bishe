ROOT_PATH = "/extend/bishe/"
DATASET_PATH = ROOT_PATH + "0-dataset/"
TextGCN_PATH = ROOT_PATH + "3-TextGCN/"

CLEAN_CORPUS_PATH = TextGCN_PATH + "data/clean/"
BTM_PATH = TextGCN_PATH + "data/btm/"
VAE_PATH = TextGCN_PATH + "data/vae/"
GRAPH_PATH = TextGCN_PATH + "data/graph/"
WORD2ID_PATH = TextGCN_PATH + "data/word2id/"
LABEL_PATH = TextGCN_PATH + "data/labels/"



if __name__ == "__main__":
    import os
    for path in [CLEAN_CORPUS_PATH, GRAPH_PATH, WORD2ID_PATH, LABEL_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
