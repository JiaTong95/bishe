import sys
import os
import json
from model import Bert_base
import argparse
from torch.optim import AdamW
from utils import get_args
from data_loader import DataSet_Bert_base as DataSet
import pickle
from torch.nn.functional import cross_entropy
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from numpy import mean
from tqdm import tqdm
# from Model import BertGCN
from utils import setup_seed
from sklearn import metrics

model_name = "bert_base"

def collate_fn(batch):
    src, label = [], []
    for s, t in batch:
        src.append(s)
        label.append(t)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    label = torch.tensor(label)
    return src, label

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
        with open(f"../result.json", "w") as file:
            json.dump({}, file)
    with open(f"../result.json", "r") as file:
        _result = json.load(file)

    update = False
    if args.dataset not in _result:
        _result[args.dataset] = {}
        update = True
    if args.target not in _result[args.dataset]:
        _result[args.dataset][args.target] = {}
        update = True
    if "bert_base" not in _result[args.dataset][args.target]:
        _result[args.dataset][args.target][model_name] = {"macro": {"macro_f1": 0}, "micro": {"micro_f1": 0}}
        update = True
    # 按照macro更新
    if _result[args.dataset][args.target][model_name]["macro"]["macro_f1"] < best_macro["macro_f1"]:
        _result[args.dataset][args.target][model_name]["macro"] = best_macro
        update = True
    # 按照micro更新
    if _result[args.dataset][args.target][model_name]["micro"]["micro_f1"] < best_micro["micro_f1"]:
        _result[args.dataset][args.target][model_name]["micro"] = best_micro
        update = True
    
    if update == True:
        with open(f"../result.json", "w") as file:
            json.dump(_result, file, indent=2)
        print('result updated.')
    # =====更新最佳结果=====end

def train(i, model, optim, data_loader, device):
    model.train()
    losses = []
    correct = 0
    total = 0
    for src, trg in tqdm(data_loader):
        optim.zero_grad()
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        loss = cross_entropy(predict, trg.long())
        loss.backward()
        optim.step()
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()
        total += predict.shape[0]
    print("train epoch {} accuracy {} || loss {}".format(i, correct / total, mean(losses)))


    

def eval(i, model, best_loss, no_increase, data_loader, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    t_targets_all, t_outputs_all = None, None
    for src, trg in tqdm(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        if t_targets_all is None:
            t_targets_all = trg
            t_outputs_all = predict
        else:
            t_targets_all = torch.cat((t_targets_all, trg), dim=0)
            t_outputs_all = torch.cat((t_outputs_all, predict), dim=0)

        loss = cross_entropy(predict, trg.long())
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()

        total += predict.shape[0]
    loss = mean(losses)
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), 'data/pkl/best_pretrain_bert.pkl')
        no_increase = 0
    else:
        no_increase += 1

    record_result(targets=t_targets_all, outputs=t_outputs_all)
    return no_increase, best_loss


def test(model, data_loader, device):
    model.eval()
    losses = []
    correct = 0
    total = 0
    t_targets_all, t_outputs_all = None, None
    for src, trg in tqdm(data_loader):
        src = src.to(device)
        trg = trg.to(device)
        predict = model(src)
        if t_targets_all is None:
            t_targets_all = trg
            t_outputs_all = predict
        else:
            t_targets_all = torch.cat((t_targets_all, trg), dim=0)
            t_outputs_all = torch.cat((t_outputs_all, predict), dim=0)

        loss = cross_entropy(predict, trg.long())
        losses.append(loss.item())
        correct += (torch.argmax(predict, -1) == trg).sum().item()
        total += predict.shape[0]
    loss = mean(losses)

    record_result(targets=t_targets_all, outputs=t_outputs_all)


def pretrain():
    global args
    args = get_args()
    setup_seed(args.seed)
    data = DataSet(f"{args.dataset}_{args.target}", 'train')
    train_loader = DataLoader(data, collate_fn=collate_fn, batch_size=20, shuffle=False)
    valid_data = DataSet(f"{args.dataset}_{args.target}", 'valid')
    val_loader = DataLoader(valid_data, collate_fn=collate_fn, batch_size=20)
    test_data = DataSet(f"{args.dataset}_{args.target}", 'test')
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=20)
    print("loading pretrained_model..")
    model = Bert_base(data.class_num)
    print("done.")
    model = model.to(args.device)
    optim = AdamW(model.parameters(), lr=args.bert_lr)
    best_loss = 1e10
    no_increasing = 0
    for i in range(50):
        train(i, model, optim, train_loader, args.device)
        with torch.no_grad():
            no_increasing, best_loss = eval(i, model, best_loss, no_increasing, val_loader, args.device)
            if no_increasing > 10:
                break
    model.load_state_dict(torch.load('data/pkl/best_pretrain_bert.pkl'))
    with torch.no_grad():
        test(model, test_loader, args.device)


if __name__ == "__main__":
    pretrain()