import torch
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from transformers import InputExample

DATASET_PATH = "/extend/bishe/0-dataset/SDwH"


class Trainer:
    def __init__(self) -> None:
        pass

    # step1. 定义任务
    def define_classes(self):
        """
        # 根据你的任务和数据来定义classes 和 InputExample。
        # 以情感分类任务为例，classes包含2个label："negative"和"positive"
        """
        self.classes = ["AGAINST", "NONE", "FAVOR"]

    # step2. 定义预训练语言模型
    def load_plm(self, reload=False):
        """
        根据具体任务选择合适的预训练语言模型，这里采用的预训练模型是bert，
        因为根据prompt的设计，是想让模型输出[mask]位置的词语，属于填空问题。
        """
        import os
        import pickle
        if not reload and os.path.exists("pkl/plm.pkl"):
            # 这里是为了省去加载预训练模型的时间
            self.plm, self.tokenizer, self.model_config, self.WrapperClass = pickle.load(
                open("pkl/plm.pkl", 'rb'))
        else:
            self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(
                "bert", "bert-base-cased")
            pickle.dump((self.plm, self.tokenizer, self.model_config, self.WrapperClass),
                        open("pkl/plm.pkl", 'wb'))

    # step3. 定义prompt模板
    def define_template(self):
        """
        手动设计模板，模板放在ManualTemplate里面
        例如text = '{"placeholder":"texta"} It was {"mask"}', 
        其中text_a就是InputExample里面的输入text_a，It was {"mask"} 就是prompt。
        """
        self.promptTemplate = ManualTemplate(
            text='{"placeholder":"text_a"} It was {"mask"}',
            tokenizer=self.tokenizer,
        )

    # step4. 定义输出-label映射
    def define_mapping(self):
        """
        在情感分类里面，[Mask]位置的输出是一个单词，我们要把这些单词映射成"positive","negative"标签，
        这个过程称为"Verbalizer"，比如"bad"属于"negative"， "good", "wonderful", "great"属于"positive"。
        """
        self.promptVerbalizer = ManualVerbalizer(
            classes=self.classes,
            label_words={
                "negative": ["bad"],
                "positive": ["good", "wonderful", "great"],
            },
            tokenizer=self.tokenizer,
        )

    # step5. 组合构建为PromptModel类
    def get_PromptModel(self):
        """
        将前面几步构建的模板(promptTemplate)、预训练模型(plm)、输出映射(promptVerbalizer)组成promptModel
        """
        self.promptModel = PromptForClassification(
            template=self.promptTemplate,
            plm=self.plm,
            verbalizer=self.promptVerbalizer,
        )

    # step6. 定义dataloader
    def get_dataloader(self):
        import csv

        self.train_lines, self.test_lines = [], []
        with open(f"{DATASET_PATH}/trump_train.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                self.train_lines.append(line)
        with open(f"{DATASET_PATH}/trump_test.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for line in reader:
                self.test_lines.append(line)

        self.dataset = []
        for i, line in enumerate(self.test_lines):
            self.dataset.append(
                InputExample(
                    guid=i,
                    tweet=line["Tweet"]
                )
            )

        self.data_loader = PromptDataLoader(
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            template=self.promptTemplate,
            tokenizer_wrapper_class=self.WrapperClass,
        )

    # step7. 开始训练、测试
    def train(self):
        self.promptModel.eval()
        with torch.no_grad():
            for batch in self.data_loader:
                logits = self.promptModel(batch)
                preds = torch.argmax(logits, dim=-1)
                print(self.classes[preds])

    def main(self):
        self.define_classes()
        self.load_plm()
        self.define_template()
        self.define_mapping()
        self.get_PromptModel()
        self.get_dataloader()
        self.train()


if __name__ == "__main__":
    trainer = Trainer()
    trainer.main()
