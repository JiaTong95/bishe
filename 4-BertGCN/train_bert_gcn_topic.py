import os
import json
from model import BertGCN
from torch.utils.data import DataLoader
# from data_load import data_load
from utils import get_args
from torch.nn.functional import nll_loss
import torch
from numpy import mean
from torch import log
from tqdm import tqdm
from torch.nn.functional import softmax
from utils import setup_seed
from torch.optim import lr_scheduler
from data_loader import DataSet_Bert_GCN as DataSet
from sklearn import metrics

model_name = "bert_gcn_topic"


def record_result(targets, outputs):
    f_against = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[0],
                                 average='macro')
    f_none = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[1],
                              average='macro')
    f_favor = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[2],
                               average='macro')
    f_avg = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[0, 1, 2],
                             average='macro')
    f_micro = metrics.f1_score(targets.cpu(), torch.argmax(outputs, -1).cpu(), labels=[0, 1, 2],
                               average='micro')

    # =====更新最佳结果=====
    best_macro = {"macro_f1": float(f_avg), "micro_f1": float(f_micro),
                  "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                  "learning_rate": f"gcn_lr={args.gcn_lr}, bert_lr={args.bert_lr}", "num_epoch": 1,
                  "batch_size": 1, "dropout": 0, "seed": args.seed}
    best_micro = {"micro_f1": float(f_micro), "macro_f1": float(f_avg),
                  "f_favor": float(f_favor), "f_against": float(f_against), "f_none": float(f_none),
                  "learning_rate": f"gcn_lr={args.gcn_lr}, bert_lr={args.bert_lr}", "num_epoch": 1,
                  "batch_size": 1, "dropout": 0, "seed": args.seed}
    if not os.path.exists('../result.json'):
        with open("../result.json", "w") as file:
            json.dump({}, file)
    with open("../result.json", "r") as file:
        _result = json.load(file)

    update = False
    if args.dataset not in _result:
        _result[args.dataset] = {}
        update = True
    if args.target not in _result[args.dataset]:
        _result[args.dataset][args.target] = {}
        update = True
    if model_name not in _result[args.dataset][args.target]:
        _result[args.dataset][args.target][model_name] = {
            "macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
        update = True
    # 按照macro更新
    if _result[args.dataset][args.target][model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
        _result[args.dataset][args.target][model_name]["macro"] = best_macro
        update = True
    # 按照micro更新
    if _result[args.dataset][args.target][model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
        _result[args.dataset][args.target][model_name]["micro"] = best_micro
        update = True

    if update:
        with open("../result.json", "w") as file:
            json.dump(_result, file, indent=2)
        print('result updated.')
    # =====更新最佳结果=====end


def train(i, dataloader, model: BertGCN, optim, features, graph, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    increase = 0
    for src, label, attention, mask, index in tqdm(dataloader):
        torch.cuda.empty_cache()
        mask = mask.to(device)
        src = src.to(device)
        attention = attention.to(device)
        label = label.to(device)
        predict = model(src, features, attention, graph, index)
        predict = predict[mask]
        label = label[mask]
        if predict.shape[0] == 0:
            continue
        loss = nll_loss(log(predict), label)
        loss.backward(retain_graph=True)
        increase += 1
        if increase % 4 == 0:
            optim.step()
            optim.zero_grad()
            increase = 0
        total += predict.shape[0]
        correct += (torch.argmax(predict, -1) == label).sum().item()
        losses.append(loss.item())
    print("training epoch {} || loss {} || accuracy {}".format(
        i, mean(losses), correct / total))


def update_features(features, dataset, model, device):
    with torch.no_grad():
        model.eval()
        for src, label, attention, mask, idx in tqdm(dataset):
            src = src.to(device)
            attention = attention.to(device)
            current_features = model.BertModel.model(
                src, attention_mask=attention).last_hidden_state[:, 0, :]
            features[idx] = current_features.detach()
    return features


def eval(i, data_loader, model: BertGCN, features, graph, usage, device, best_loss=None, best_accuracy=None,
         no_increasing=None):
    model.eval()
    if usage == 'valid':
        mask = graph.ndata['valid_mask']
    else:
        mask = graph.ndata['test_mask']
    mask = (mask == 1)
    if usage == 'test':
        model.load_state_dict(torch.load(
            f'data/pkl/{args.dataset}_{args.target}_best_Bert_GCN_topic_model.pkl'))
        features = update_features(features, data_loader, model, device)
    predict = model.BertModel.linear(features)
    graph_predict = model.gcn(graph, features)
    predict = softmax(predict[mask], -1) * (1 - model.lam) + \
        softmax(graph_predict[mask], -1) * model.lam
    label = graph.ndata['label']

    record_result(targets=label[mask], outputs=predict)

    if usage == 'test':
        model.load_state_dict(torch.load(
            f'data/pkl/{args.dataset}_{args.target}_best_accuracy.pkl'))
        features = update_features(features, data_loader, model, device)
        predict = model.BertModel.linear(features)
        graph_predict = model.gcn(graph, features)
        predict = softmax(predict[mask], -1) * (1 - model.lam) + \
            softmax(graph_predict[mask], -1) * model.lam
        label = graph.ndata['label']
        record_result(targets=label[mask], outputs=predict)
    if usage == 'valid':
        loss = nll_loss(log(predict), label[mask])
        correct = (torch.argmax(predict, -1) == label[mask]).sum().item()
        total = sum(mask).item()
        if best_loss > loss.item():
            best_loss = loss.item()
            no_increasing = 0
            torch.save(model.state_dict(
            ), f'data/pkl/{args.dataset}_{args.target}_best_Bert_GCN_topic_model.pkl')
            print("saving to file best_Bert_GCN_model.pkl")
        else:
            no_increasing += 1
        if best_accuracy < correct / total:
            best_accuracy = correct / total
            torch.save(
                model.state_dict(), f'data/pkl/{args.dataset}_{args.target}_best_accuracy.pkl')
            print("saving to file best_accuracy.pkl")
        return best_loss, best_accuracy, no_increasing


def main():
    global args
    args = get_args()

    setup_seed(args.seed)
    device = args.device
    dataset = DataSet(f"{args.dataset}_{args.target}")

    # graph 就是 A
    graph = dataset.graph.to(device)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # features 就是 X，把句子用bert表示的向量表示
    # 单词不用bert表示，全为0
    features = torch.zeros(graph.num_nodes(), 768,
                           requires_grad=False).to(device)
    model = BertGCN('data/pkl/best_pretrain_bert.pkl', dataset.label_num)
    model = model.to(device)
    optim = torch.optim.Adam(
        [{'params': model.gcn.parameters(), 'lr': args.gcn_lr},
         {'params': model.BertModel.parameters(), 'lr': args.bert_lr}])
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[30], gamma=0.1)
    features = update_features(features, data_loader, model, device)
    best_loss = 1e10
    no_increasing = 0
    best_accuracy = 0
    for i in range(20):  # epoch_num
        train(i, data_loader, model, optim, features, graph, device)
        scheduler.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            features = update_features(features, data_loader, model, device)
            best_loss, best_accuracy, no_increasing = eval(i, data_loader, model, features, graph, 'valid', device,
                                                           best_loss,
                                                           best_accuracy,
                                                           no_increasing)
        if no_increasing >= 10:
            break  # for i in range(20):
    with torch.no_grad():
        eval(0, data_loader, model, features, graph, 'test',
             device, best_loss, best_accuracy, no_increasing)


if __name__ == '__main__':
    main()
