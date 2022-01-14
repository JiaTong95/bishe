CLEAN_CORPUS_PATH = "data/clean"
GRAPH_PATH = "data/graph"
WORD2ID_PATH = "data/word2id"
LABEL_PATH = "data/labels"
DATASET_PATH = "/extend/jt_2/0-dataset"
ROOT_PATH = "/extend/jt_2"

if __name__ == "__main__":
    import os
    for path in [CLEAN_CORPUS_PATH, GRAPH_PATH, WORD2ID_PATH, LABEL_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
