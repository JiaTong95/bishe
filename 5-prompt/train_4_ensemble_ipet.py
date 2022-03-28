# 按PET思路 多个模板融合 模型参数不共享
import os
import random
import torch
import tqdm
import json
import copy
from openprompt.plms import load_plm
from openprompt import PromptForClassification
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptDataLoader
from openprompt.data_utils import InputExample
from transformers import AdamW

from utils import timer, parameter_parser, cal_f1, get_topic_words, get_label_words, get_prompt_template

DATASET_PATH = "/extend/bishe/0-dataset"
if not os.path.exists("pkl"):
    os.mkdir("pkl")
G = get_topic_words()

# 打包模型
class Model_wrapper:
    def __init__(self, model,
                       tid,
                       lid,
                       template,
                       verbalizer,
                       trainset,
                       testset,
                       loss_func,
                       optimizer) -> None:
        self.model = model
        self.tid = tid
        self.lid = lid
        self.template = template
        self.verbalizer = verbalizer
        self.trainset = trainset
        self.testset = testset
        self.loss_func = loss_func
        self.optimizer = optimizer
    
    # 重置模型，使参数不共享
    def reset_model(self, plm, device):
        self.model = PromptForClassification(plm=plm,
                                             template=self.template,
                                             verbalizer=self.verbalizer,
                                             freeze_plm=False).to(device)

class Trainer:
    def __init__(self, opt) -> None:
        self.opt = opt
        # 手动定义种子
        # torch.manual_seed(self.opt.seed)
        # torch.cuda.manual_seed(self.opt.seed)

     # 定义预训练语言模型
    def load_plm(self, reload=False):
        """
        加载预训练模型
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
        print("分词器大小", self.tokenizer.vocab_size)

    # 打包
    def wrap(self):
        from openprompt.plms import MLMTokenizerWrapper
        self.wrapped_tokenizer = MLMTokenizerWrapper(max_seq_length=self.opt.max_seq_length,
                                                     decoder_max_length=3,
                                                     tokenizer=self.tokenizer,
                                                     truncate_method="head")

    

    def define_template(self):
        """
        定义多个template
        """
        self.promptTemplate_list = {1:get_prompt_template(tokenizer=self.tokenizer, tid=1),
                                    2:get_prompt_template(tokenizer=self.tokenizer, tid=2),
                                    3:get_prompt_template(tokenizer=self.tokenizer, tid=3),
                                    4:get_prompt_template(tokenizer=self.tokenizer, tid=4)}
    # 定义Verbalizer
    def define_Verbalizer(self):
        """
        定义多个verbalizer
        """
        self.promptVerbalizer_list = {1:ManualVerbalizer(num_classes=3,
                                      label_words=get_label_words(lid=1),
                                      tokenizer=self.tokenizer),
                                      2: ManualVerbalizer(num_classes=3,
                                      label_words=get_label_words(lid=2),
                                      tokenizer=self.tokenizer),
                                      3: ManualVerbalizer(num_classes=3,
                                      label_words=get_label_words(lid=3),
                                      tokenizer=self.tokenizer),
                                      4: ManualVerbalizer(num_classes=3,
                                      label_words=get_label_words(lid=4),
                                      tokenizer=self.tokenizer),}
                                      


    # 获取数据集
    def get_dataset(self):
        """
        读取数据集，做简单预处理
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
                                                  text_a=G.main(
                                                      line["Tweet"].lower(), self.opt),
                                                  text_b=line["Target"],
                                                  label=d[line["Stance"]])
                    self.dataset[split].append(_input_example)

    # 获取dataloader
    def get_dataloader(self, model_index, mode):
        trainset = self.model_list[model_index].trainset
        testset = self.model_list[model_index].testset
        
        train_size = int(len(trainset) * 0.9)
        valset = trainset[train_size:]
        trainset = trainset[:train_size]
        
        if mode == 'train':
            dataset = trainset
            shuffle = True
        if mode == 'val':
            dataset = valset
            shuffle = True
        if mode == 'test':
            dataset = testset
            shuffle = False
           
        
        return PromptDataLoader(dataset=dataset,
                                template=self.model_list[model_index].template,
                                tokenizer=self.tokenizer,
                                tokenizer_wrapper_class=self.WrapperClass,
                                max_seq_length=self.opt.max_seq_length,
                                decoder_max_length=3,
                                batch_size=self.opt.batch_size,
                                shuffle=shuffle,
                                teacher_forcing=False,
                                predict_eos_token=False,
                                truncate_method="head")

    def define_model(self, tids_lids):
        # 准备好所有模型, 打包起来
        self.model_list = []
        for model_index, (tid, lid) in enumerate(tids_lids):
            # prompt相关
            template = self.promptTemplate_list[tid]
            verbalizer = self.promptVerbalizer_list[lid]

            # 模型设置
            model = PromptForClassification(plm=self.plm,
                                            template=template,
                                            verbalizer=verbalizer,
                                            freeze_plm=False).to(self.opt.device)
            loss_func = torch.nn.CrossEntropyLoss()
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)

            # 打包
            mymodel = Model_wrapper(model=model,
                                    tid=tid,
                                    lid=lid,
                                    template=template,
                                    verbalizer=verbalizer,
                                    trainset=self.dataset["train"],
                                    testset=self.dataset["test"],
                                    loss_func=loss_func,
                                    optimizer=optimizer)
            self.model_list.append(mymodel)
        print("所有模型准备完毕")
    
    
    # 联合训练
    def train_ensemble(self, reset_model=False):        
        # 每次训练都加一部分testset数据拼在trainset上去, 分N个阶段，每次加N分之一
        self.N_stage = 3
        test_len = len(self.dataset["test"])
        each_stage_len = test_len // self.N_stage
        # 开始分阶段训练
        for stage in range(self.N_stage):
            # 分别独立训练
            all_preds = []
            for model_index in range(len(self.model_list)):
                # 模型参数重置，使参数不共享
                if reset_model:
                    self.model_list[model_index].reset_model(plm=self.plm, device=self.opt.device)
                preds = self.train_individual(model_index, stage_training=True)
                all_preds.append(preds)

            # 拼接unlabelset
            for model_index in range(len(self.model_list)):
                # 随机选一个模板预测出来的logits
                preds = random.choice(all_preds)
                # 然后在trainset后面加上随机模板预测的unlabelset, 从unlabelset的第start_i行到第end_i行
                start_i, end_i = each_stage_len * stage, each_stage_len * stage + each_stage_len
                for i in range(start_i, end_i):
                    pred = preds[i]
                    new_line = copy.deepcopy(self.dataset["test"][i])
                    new_line.label = pred
                    self.model_list[model_index].trainset.append(new_line)
            print(f"完成stage{stage+1}训练")
        
        # 最终测试
        for model_index in range(len(self.model_list)):
            # 模型参数重置，使参数不共享
            if reset_model:
                self.model_list[model_index].reset_model(plm=self.plm, device=self.opt.device)
            self.train_individual(model_index, stage_training=False)
    
    # 独立训练一个模型
    @timer
    def train_individual(self, model_index, stage_training=True):
        print(f"当前模型tid={self.model_list[model_index].tid}, lid={self.model_list[model_index].lid}")           
        train_dataloader = self.get_dataloader(model_index, 'train')
        val_dataloader = self.get_dataloader(model_index, 'val')
        test_dataloader = self.get_dataloader(model_index, 'test')

        best_macro_score, best_logits = 0, None
        for epoch in range(self.opt.max_epoch):
            tot_loss = 0
            logits_per_epoch = None 
            # model.train()
            self.model_list[model_index].model.train()
            for step, inputs in enumerate(train_dataloader):
                inputs = inputs.to(self.opt.device)
                labels = inputs['label']
                
                # 送入模型
                logits = self.model_list[model_index].model(inputs)

                if logits_per_epoch is None:
                    logits_per_epoch = logits
                else:
                    logits_per_epoch = torch.cat((logits_per_epoch, logits), 0)

                # 反向传播
                loss = self.model_list[model_index].loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                self.model_list[model_index].optimizer.step()
                self.model_list[model_index].optimizer.zero_grad()
                if step % 100 == 1:
                    print("Epoch {}, average loss: {}".format(
                        epoch, tot_loss/(step+1)), flush=True) 
        
            
            # 验证集
            val_result, _ = self.eval(model_index, val_dataloader)
            if val_result['macro_f'] > best_macro_score:
                best_macro_score = val_result['macro_f']
                test_result, best_test_labels = self.eval(model_index, test_dataloader)
                if not stage_training:
                    print(test_result)
        return best_test_labels

    def eval(self, model_index, dataloader):
        # model.eval()
        self.model_list[model_index].model.eval()
        
        allpreds, alllabels = [], []
        for step, inputs in enumerate(dataloader):
            inputs = inputs.to(self.opt.device)
            logits = self.model_list[model_index].model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        result = cal_f1(outputs=allpreds, targets=alllabels)
        return result, allpreds
    

    def main(self):
        # 准备预训练模型等
        self.load_plm()
        self.wrap()

        # 准备训练数据
        self.get_dataset()
        
        # 准备prompt
        self.define_template()
        self.define_Verbalizer()

        # 准备模型
        self.define_model(tids_lids=[(1,1), (2,3), (1,4), (4,4)])

        # 训练、测试
        self.train_ensemble(reset_model=True)


if __name__ == "__main__":
    opt = parameter_parser()
    print(opt)

    trainer = Trainer(opt)
    trainer.main()
