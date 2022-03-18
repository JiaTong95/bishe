# 加topic_words
import os
from responses import target
import torch
import tqdm
import json
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample

from utils import timer, parameter_parser, cal_f1, get_topic_words, get_label_words, get_prompt_template

DATASET_PATH = "/extend/bishe/0-dataset"
if not os.path.exists("pkl"):
    os.mkdir("pkl")
G = get_topic_words()

class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt

    # 获取数据集、定义任务
    def define_task(self):
        """
        根据你的任务和数据来定义InputExample。

        手动设计模板，模板放在ManualTemplate里面
        例如text = '{"placeholder":"texta"} It was {"mask"}',
        其中text_a就是InputExample里面的输入text_a，It was {"mask"} 就是prompt。
        """
        d = {"AGAINST": 0, "NONE": 1, "FAVOR": 2, '-1': 0, '0': 1, '1': 2}
        import csv
        self.dataset = {}
        for split in ["train", "test"]:  # ["train", "validation", "test"]:
            self.dataset[split] = []
            with open(f"{DATASET_PATH}/{self.opt.dataset}/{self.opt.target}_{split}.csv", 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for line in tqdm.tqdm(reader):
                    _input_example = InputExample(guid=line["ID"],
                                                  text_a=G.main(line["Tweet"].lower(), self.opt),
                                                  text_b=line["Target"],
                                                  label=d[line["Stance"]])
                    self.dataset[split].append(_input_example)
            print(G.yes, G.no)

        self.label_words = get_label_words(self.opt.label_words_id)
        self.promptTemplate = get_prompt_template(tokenizer=self.tokenizer, tid=self.opt.template_id)

    # 定义预训练语言模型
    def load_plm(self, reload=False):
        """
        根据具体任务选择合适的预训练语言模型，这里采用的预训练模型是bert，
        因为根据prompt的设计，是想让模型输出[mask]位置的词语，属于填空问题。
        """
        import os
        import pickle
        plm_name = "bert-base-uncased"
        if not reload and os.path.exists(f"pkl/{plm_name}.pkl"):
            # 这里是为了省去加载预训练模型的时间
            self.plm, self.tokenizer, self.model_config, self.WrapperClass = pickle.load(
                open(f"pkl/{plm_name}.pkl", 'rb'))
        else:
            self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(
                "bert", plm_name)
            pickle.dump((self.plm, self.tokenizer, self.model_config, self.WrapperClass),
                        open(f"pkl/{plm_name}.pkl", 'wb'))

    # 打包

    def wrap(self):
        from openprompt.plms import MLMTokenizerWrapper
        self.wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=self.opt.max_seq_length,
                                                     decoder_max_length=3,
                                                     tokenizer=self.tokenizer,
                                                     truncate_method="head")

        # You can see what a tokenized example looks like by
        wrapped_example = self.promptTemplate.wrap_one_example(
            self.dataset['train'][0])
        print("wrapped_example:", wrapped_example)

        tokenized_example = self.wrapped_tokenizer.tokenize_one_example(
            wrapped_example, teacher_forcing=False)
        print("tokenized_example:", tokenized_example)
        print("input_ids convert to tokens:",
              self.tokenizer.convert_ids_to_tokens(tokenized_example['input_ids']))

    # 获取模型输入
    def get_model_inputs(self):
        model_inputs = {}
        for split in ["train", "test"]:  # ['train', 'validation', 'test']:
            model_inputs[split] = []
            for sample in self.dataset[split]:
                tokenized_example = self.wrapped_tokenizer.tokenize_one_example(
                    self.promptTemplate.wrap_one_example(sample), teacher_forcing=False)
                model_inputs[split].append(tokenized_example)

    # 获取dataloader
    def get_dataloader(self):
        self.train_dataloader = PromptDataLoader(dataset=self.dataset["train"],
                                                 template=self.promptTemplate,
                                                 tokenizer=self.tokenizer,
                                                 tokenizer_wrapper_class=self.WrapperClass,
                                                 max_seq_length=self.opt.max_seq_length,
                                                 decoder_max_length=3,
                                                 batch_size=self.opt.batch_size,
                                                 shuffle=True,
                                                 teacher_forcing=False,
                                                 predict_eos_token=False,
                                                 truncate_method="head")
        self.test_dataloader = PromptDataLoader(dataset=self.dataset["test"],
                                                template=self.promptTemplate,
                                                tokenizer=self.tokenizer,
                                                tokenizer_wrapper_class=self.WrapperClass,
                                                max_seq_length=self.opt.max_seq_length,
                                                decoder_max_length=3,
                                                batch_size=self.opt.batch_size,
                                                shuffle=False,
                                                teacher_forcing=False,
                                                predict_eos_token=False,
                                                truncate_method="head")

    # 定义Verbalizer
    def define_Verbalizer(self):
        """
        在情感分类里面，[Mask]位置的输出是一个单词，我们要把这些单词映射成"positive","negative"标签，
        这个过程称为"Verbalizer"，比如"bad"属于"negative"， "good", "wonderful", "great"属于"positive"。
        """
        self.promptVerbalizer = ManualVerbalizer(num_classes=3,
                                                 label_words=self.label_words,
                                                 tokenizer=self.tokenizer,
                                                 )
        print("label_words_ids:", self.promptVerbalizer.label_words_ids)
        # creating a pseudo output from the plm, and
        logits = torch.randn(2, len(self.tokenizer))
        print(self.promptVerbalizer.process_logits(
            logits))  # see what the verbalizer do

    # 组合构建为PromptModel类
    def get_PromptModel(self):
        """
        将前面几步构建的模板(promptTemplate)、预训练模型(plm)、输出映射(promptVerbalizer)组成promptModel
        """
        self.prompt_model = PromptForClassification(plm=self.plm,
                                                    template=self.promptTemplate,
                                                    verbalizer=self.promptVerbalizer,
                                                    freeze_plm=False).to(self.opt.device)

    # 开始训练、测试
    @ timer
    def train(self):
        # Now the training is standard
        from transformers import AdamW, get_linear_schedule_with_warmup
        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.prompt_model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.prompt_model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.opt.learning_rate)

        for epoch in range(self.opt.max_epoch):
            tot_loss = 0
            for step, inputs in enumerate(self.train_dataloader):
                inputs = inputs.to(self.opt.device)
                logits = self.prompt_model(inputs)
                labels = inputs['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(
                        epoch, tot_loss/(step+1)), flush=True)

    def test(self):
        allpreds, alllabels = [], []

        for step, inputs in enumerate(self.test_dataloader):
            inputs = inputs.to(self.opt.device)
            logits = self.prompt_model(inputs)
            labels = inputs['label']

            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        result = cal_f1(outputs=allpreds, targets=alllabels)
        print(result)
        self.update_result(result)

        # print("allpreds", allpreds)
        # print("alllabels", alllabels)


    def update_result(self, result):
        micro_f, f_avg, f_favor, f_against, f_none = result["micro_f"], result["macro_f"], result["f_favor"], result["f_against"], result["f_none"]

        with open(f"logs/_result.txt", "a+") as file:
            # =====更新最佳结果=====
            best_macro = {"macro_f1": float(f_avg), "micro_f1": float(micro_f), 
                            "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                            "learning_rate": self.opt.learning_rate, "num_epoch": self.opt.max_epoch,
                            "batch_size": self.opt.batch_size, "dropout": self.opt.dropout, "seed": self.opt.seed}
            best_micro = {"micro_f1": float(micro_f), "macro_f1": float(f_avg),
                            "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                            "learning_rate": self.opt.learning_rate, "num_epoch": self.opt.max_epoch,
                            "batch_size": self.opt.batch_size, "dropout": self.opt.dropout, "seed": self.opt.seed}
            if not os.path.exists('../result.json'):
                with open(f"../result.json", "w") as file:
                    json.dump({}, file)
            with open(f"../result.json", "r") as file:
                _result = json.load(file)

            if self.opt.dataset not in _result:
                _result[self.opt.dataset] = {}
            if self.opt.target not in _result[self.opt.dataset]:
                _result[self.opt.dataset][self.opt.target] = {}
            if model_name not in _result[self.opt.dataset][self.opt.target]:
                _result[self.opt.dataset][self.opt.target][model_name] = {"macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
            # 按照macro更新
            if _result[self.opt.dataset][self.opt.target][model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
                _result[self.opt.dataset][self.opt.target][model_name]["macro"] = best_macro
            # 按照micro更新
            if _result[self.opt.dataset][self.opt.target][model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
                _result[self.opt.dataset][self.opt.target][model_name]["micro"] = best_micro
            with open(f"../result.json", "w") as file:
                json.dump(_result, file, indent=2)

            update = False
            if opt.dataset not in _result:
                _result[opt.dataset] = {}
                update = True
            if opt.target not in _result[opt.dataset]:
                _result[opt.dataset][opt.target] = {}
                update = True
            if model_name not in _result[opt.dataset][opt.target]:
                _result[opt.dataset][opt.target][model_name] = {
                    "macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
                update = True
            # 按照macro更新
            if _result[opt.dataset][opt.target][model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
                _result[opt.dataset][opt.target][model_name]["macro"] = best_macro
                update = True
            # 按照micro更新
            if _result[opt.dataset][opt.target][model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
                _result[opt.dataset][opt.target][model_name]["micro"] = best_micro
                update = True

            if update:
                with open("../result.json", "w") as file:
                    json.dump(_result, file, indent=2)
                print('result updated.')
            # =====更新最佳结果=====end

    def main(self):
        self.load_plm()
        self.define_task()
        self.wrap()
        self.get_model_inputs()
        self.get_dataloader()
        self.define_Verbalizer()
        self.get_PromptModel()
        self.train()
        self.test()


if __name__ == "__main__":
    opt = parameter_parser()
    trainer = Trainer(opt)
    if opt.topic_by == "":
        model_name = "prompt"
    else:
        model_name = f"prompt+topic({opt.topic_by})"
    trainer.main()
