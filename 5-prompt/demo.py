# step1. 定义任务
# 根据你的任务和数据来定义classes 和 InputExample。
# 以情感分类任务为例，classes包含2个label："negative"和"positive"

from openprompt.data_utils import InputExample
classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Jiojio is a very beautiful girl.",
    ),
    InputExample(
        guid = 1,
        text_a = "Tongtong is a bad guy.",
    ),
]

# step2. 定义预训练语言模型
# 根据具体任务选择合适的预训练语言模型，这里采用的预训练模型是bert，因为根据prompt的设计，是想让模型输出[mask]位置的词语，属于填空问题。

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-uncased")

# step3. 定义prompt模板
# 这个例子是手动设计模板，模板放在ManualTemplate里面，text = '{"placeholder":"texta"} It was {"mask"}', 其中text_a就是InputExample里面的输入text_a，It was {"mask"} 就是prompt。

from openprompt.prompts import ManualTemplate
promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)

# step4. 定义输出-label映射
# 在情感分类里面，[Mask]位置的输出是一个单词，我们要把这些单词映射成"positive","negative"标签，这个过程称为"Verbalizer"，比如"bad"属于"negative"， "good", "wonderful", "great"属于"positive"。

from openprompt.prompts import ManualVerbalizer
promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["bad"],
        "positive": ["good", "wonderful", "great"],
    },
    tokenizer = tokenizer,
)
# step5. 组合构建为PromptModel类
# 将前面几步构建的模板(promptTemplate)、预训练模型(plm)、输出映射(promptVerbalizer)组成promptModel

from openprompt import PromptForClassification
promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
)
# step6. 定义dataloader
from openprompt import PromptDataLoader
data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer, 
        template = promptTemplate, 
        tokenizer_wrapper_class=WrapperClass,
    )
# step7. 开始训练、测试
# making zero-shot inference using pretrained MLM with prompt
import torch

promptModel.eval()
with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim = -1)
        print(classes[preds])
        # predictions would be 1, 0 for classes 'positive', 'negative'
