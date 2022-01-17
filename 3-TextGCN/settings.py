ROOT_PATH = "/extend/bishe"
CLEAN_CORPUS_PATH = "data/clean"
GRAPH_PATH = "data/graph"
WORD2ID_PATH = "data/word2id"
LABEL_PATH = "data/labels"
DATASET_PATH = ROOT_PATH + "/0-dataset"


if __name__ == "__main__":
    import os
    for path in [CLEAN_CORPUS_PATH, GRAPH_PATH, WORD2ID_PATH, LABEL_PATH]:
        if not os.path.exists(path):
            os.makedirs(path)
