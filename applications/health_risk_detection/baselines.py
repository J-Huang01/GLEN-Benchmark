import torch.nn as nn
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from models import *
import argparse
import wandb
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
import torch
import torch.nn.functional as F
import pandas as pd, os
import math
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
    
def mask_label_edges(graph):
    target_edge = ('user', 'has', 'opioid_level')
    if target_edge in graph.edge_types:
        print(f"[INFO] Masking label edge: {target_edge}")
        graph[target_edge].edge_index = torch.empty((2, 0), dtype=torch.long)
    return graph

def add_user_reverse_edges(graph):
    import copy
    new_graph = copy.deepcopy(graph)
    for (src, rel, dst) in list(graph.edge_types):
        if src == 'user' and dst != 'user':
            edge_index = graph[(src, rel, dst)].edge_index
            new_graph[(dst, f"rev_{rel}", src)].edge_index = edge_index.flip(0)
    print(f"[INFO] Added reverse edges for user-related relations.")
    return new_graph


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        """
        cls_num_list: list of class sample counts, e.g., [80000, 3000, 400]
        max_m: maximum margin
        s: scaling factor
        """
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.cuda.FloatTensor(m_list)
        self.s = s

    def forward(self, x, target):
        # x: [batch_size, num_classes]
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)


def main():
    if args.use_wandb:
        config = wandb.config
        SEED = config.seed
        LR = config.lr
        DROPOUT = config.dropout
        HIDDEN_DIM = config.hidden_dim
        WEIGHT_DECAY = config.weight_decay
        MODEL_TYPE = args.model_type
        wandb.run.name = f"Run_with_{args.model_type}_{config.lr}_{config.dropout}_{config.hidden_dim}_{config.weight_decay}"
        # run.save()
    else:
        SEED = args.seed
        LR = args.lr
        DROPOUT = args.dropout
        HIDDEN_DIM = args.hidden_dim
        WEIGHT_DECAY = args.weight_decay
        MODEL_TYPE = args.model_type

   
    set_seed(SEED)

    graph = torch.load('./graphs/hetero_graph_embed.pt', weights_only=False)
    graph = mask_label_edges(graph)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_labels = graph['user'].y
    valid_labels = user_labels[user_labels >= 0] 
    num_classes = len(torch.unique(valid_labels))

    if MODEL_TYPE == 'MLP':
        for ntype in ['user', 'food', 'habit', 'ingredient', 'category']:
            if 'x' in graph[ntype]:
                x = graph[ntype].x
                print(f"{ntype} NaN count:", torch.isnan(x).sum().item())

        node_features, user_labels = process_features_for_MLP(graph)
        model = simple_MLP(num_features=node_features.size(1), num_classes=num_classes, hidden_dim=HIDDEN_DIM)

        node_features = node_features.to(device)
    elif MODEL_TYPE == 'GCN':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = GCN(num_features=node_features.size(1),
                    num_classes=num_classes,
                    hidden_dim=HIDDEN_DIM,
                    dropout=DROPOUT)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'SAGE':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = SAGE(num_features=node_features.size(1),
                     num_classes=num_classes,
                     hidden_dim=HIDDEN_DIM,
                     dropout=DROPOUT)

        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'GAT':
        node_features, edge_index, user_labels = load_data_for_GCN(graph)
        model = GAT(num_features=node_features.size(1),
                    num_classes=num_classes,
                    hidden_dim=HIDDEN_DIM,
                    dropout=DROPOUT)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
    elif MODEL_TYPE == 'RGCN':
        node_feature_dims, feature_dict, edge_index, edge_type, num_relations, user_labels = load_data_for_RGCN(graph)
        model = RGCN(num_relations, node_feature_dims, num_classes=num_classes, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        model = model.to(device)
        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        x_all_len = sum([x.shape[0] for x in feature_dict.values()])
        print("x_all length:", x_all_len)
        print("edge_index max:", edge_index.max().item())
        print("edge_index min:", edge_index.min().item())
    elif MODEL_TYPE == 'HAN':
        # graph = metapath_generation(graph, sample_ratio=0.2)
        # UFU_edge_list = generate_neighbors(graph, ('user', 'eats', 'food'), ('food', 'eaten', 'user'),
        #                                    shared_threshold=3)
        # UFU_edge_index = edge_concat(UFU_edge_list, [])
        # UHU_edge_list = generate_neighbors(graph, ('user', 'has', 'habit'), ('habit', 'from', 'user'),
        #                                    shared_threshold=3)
        # UHU_edge_index = edge_concat(UHU_edge_list, [])
        # refined_graph = HeteroData()
        # refined_graph['user'].x = graph['user'].x
        # refined_graph['user'].y = graph['user'].y
        # refined_graph[('user', 'UFU', 'user')].edge_index = UFU_edge_index
        # refined_graph[('user', 'UHU', 'user')].edge_index = UHU_edge_index
        graph = add_user_reverse_edges(graph)
        model = HAN(graph, in_channels=-1, out_channels=num_classes)

        feature_dict = graph.x_dict
        edge_dict = graph.edge_index_dict
        user_labels = graph['user'].y

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_dict = {key: x.to(device) for key, x in edge_dict.items()}
    elif MODEL_TYPE == 'HGT':
        graph = add_user_reverse_edges(graph)
        model = HGT(graph, hidden_channels=HIDDEN_DIM, out_channels=num_classes, num_heads=4, num_layers=2)

        feature_dict = graph.x_dict
        edge_dict = graph.edge_index_dict
        user_labels = graph['user'].y

        feature_dict = {key: x.to(device) for key, x in feature_dict.items()}
        edge_dict = {key: x.to(device) for key, x in edge_dict.items()}
    else:
        raise NotImplementedError

    indices = np.arange(len(user_labels))
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        indices, user_labels, test_size=0.4, stratify=user_labels, random_state=SEED
    )

    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    rus = RandomUnderSampler(random_state=SEED)
    train_indices_np = train_indices.reshape(-1, 1)
    train_indices_balanced, train_labels_balanced = rus.fit_resample(train_indices_np, train_labels.cpu().numpy())
    train_indices = torch.tensor(train_indices_balanced.flatten(), dtype=torch.long)
    train_labels = torch.tensor(train_labels_balanced, dtype=torch.long).to(device)

    print(f"After under-sampling, training size: {len(train_indices)}, class distribution: {np.bincount(train_labels.cpu().numpy())}")
  
    def check_split_distribution(labels, name="Split"):
        unique, counts = np.unique(labels.cpu().numpy(), return_counts=True)
        print(f"{name} class distribution:")
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c}")
        missing_classes = set(range(num_classes)) - set(unique.tolist())
        if missing_classes:
            print(f"Warning: {name} is missing classes: {missing_classes}")

    check_split_distribution(train_labels, "Train")
    check_split_distribution(val_labels, "Val")
    check_split_distribution(test_labels, "Test")

    class_counts = torch.bincount(user_labels.cpu())
    class_weights = len(user_labels) / (num_classes * class_counts.float())

    print("Class counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    train_indices = torch.tensor(train_indices, dtype=torch.long)
    train_labels = train_labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = LDAMLoss(cls_num_list=class_counts.tolist(), max_m=0.5, s=30)

    model = model.to(device)
    criterion = criterion.to(device)

    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    
    if MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
        with torch.no_grad(): 
          out = model(feature_dict, edge_dict)

    # Training Starts here
    best_model_state = None
    best_f1_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        if MODEL_TYPE == 'RGCN':
            out = model(feature_dict, edge_index, edge_type)
        elif MODEL_TYPE == 'MLP':
            out = model(node_features)
        elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':

            out = model(feature_dict, edge_dict)
        else:
            out = model(node_features, edge_index)
        # Compute loss only for the subset of nodes
        loss = criterion(out[train_indices], train_labels)
        loss.backward()
        optimizer.step()

        predictions = out[train_indices].max(1)[1].cpu().numpy()
        truth = train_labels.cpu().numpy()
        f1_train = f1_score(truth, predictions, average='macro') 

        with torch.no_grad():
            if MODEL_TYPE == 'RGCN':
                out = model(feature_dict, edge_index, edge_type)
            elif MODEL_TYPE == 'MLP':
                out = model(node_features)
            elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
                out = model(feature_dict, edge_dict)
            else:
                out = model(node_features, edge_index)
            val_output = out[val_indices]
            val_predictions = val_output.max(1)[1].cpu().numpy()
            val_truth = val_labels.cpu().numpy()
            # f1_val = f1_score(val_predictions, test_truth)
            f1_val = f1_score(val_truth, val_predictions, average='macro')
            if args.use_wandb:
                wandb.log({
                    'train_loss': loss,
                    'f1_val': f1_val
                })

            print(f"Epoch {epoch + 1}: Train Loss: {loss.item()}, Train F1-Score: {f1_train} Val F1-Score: {f1_val}")

            if f1_val > best_f1_val:
                best_f1_val = f1_val
                best_model_state = model.state_dict().copy()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        if MODEL_TYPE == 'RGCN':
            out = model(feature_dict, edge_index, edge_type)
        elif MODEL_TYPE == 'MLP':
            out = model(node_features)
        elif MODEL_TYPE == 'HGT' or MODEL_TYPE == 'HAN':
            out = model(feature_dict, edge_dict)
        else:
            out = model(node_features, edge_index)
        test_output = out[test_indices]
        test_probabilities = F.softmax(test_output, dim=1).cpu().numpy()

        test_predictions = test_probabilities.argmax(axis=1)  
        test_truth = test_labels.cpu().numpy()
        f1_test = f1_score(test_truth, test_predictions, average='macro')
        print('Final Result: F1 Score - {}'.format(f1_test))
        auc_test = roc_auc_score(
            test_truth, 
            test_probabilities, 
            multi_class="ovr", 
            average="macro"
        )
        gmean_test = geometric_mean_score(test_truth, test_predictions, average="macro")

        print('Final Result: AUC Score - {}'.format(auc_test))
        print('Final Result: G-Mean - {}'.format(gmean_test))
        # Calculating Accuracy, Precision, and Recall
        accuracy_test = accuracy_score(test_truth, test_predictions)
        precision_test = precision_score(test_truth, test_predictions, average="macro")
        recall_test = recall_score(test_truth, test_predictions, average="macro")
        print('Final Result: Accuracy - {}'.format(accuracy_test))
        print('Final Result: Precision - {}'.format(precision_test))
        print('Final Result: Recall - {}'.format(recall_test))
        if args.use_wandb:
            wandb.log({
                'f1_test': f1_test,
                'auc_test': auc_test,
                'gmean_test': gmean_test,
                'accuracy_test': accuracy_test,
                'precision_test': precision_test,
                'recall_test': recall_test
            })
    return {
        'f1': f1_test,
        'auc': auc_test,
        'gmean': gmean_test,
        'acc': accuracy_test,
        'prec': precision_test,
        'rec': recall_test
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Number of hidden dimension.')
    parser.add_argument('--dropout', type=float, default=0.6, 
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='whether to use wandb for a sweep.')
    parser.add_argument('--model_type', type=str, default='HAN',
                        help='The baseline model to use.')
    args = parser.parse_args()
    seeds = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    all_metrics = {'f1': [], 'auc': [], 'gmean': [], 'acc': [], 'prec': [], 'rec': []}

    for seed in seeds:
        print(f"\n{'='*30}\nRunning experiment with seed {seed}\n{'='*30}")
        args.seed = seed
        result = main()

        for key in all_metrics.keys():
            all_metrics[key].append(result[key])
    for key, values in all_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"{key.upper():<10}: {mean:.4f} ± {std:.4f}")
    if args.use_wandb:
        wandb.login(key='INPUT YOUR KEY')
        wandb.init(
            project='glen_bench',
            entity='',
            config={
                'seed': args.seed,
                'lr': args.lr,
                'dropout': args.dropout,
                'hidden_dim': args.hidden_dim,
                'weight_decay': args.weight_decay,
                'model_type': args.model_type
            }
        )
        main()
    else:
        main()
