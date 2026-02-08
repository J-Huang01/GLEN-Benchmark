import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
import torch_geometric
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import json
from pathlib import Path
import matplotlib.pyplot as plt

from RCSYS_models import (
    LightGCN, GCNModel, NGCF,
    SimGCL, SGL, LightGCL,
    RecipeRec, HFRS_DA,
    SGSL, modified_SGSL
)

from RCSYS_utils import (
    sample_mini_batch,
    split_data_new,
    get_user_positive_items,
    RecallPrecision_ATk,
    NDCGatK_r,
)

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def info_nce_loss(z1, z2, temp=0.2, max_samples=2048):
    N = z1.size(0)
    if N > max_samples:
        indices = torch.randperm(N, device=z1.device)[:max_samples]
        z1 = z1[indices]
        z2 = z2[indices]
        N = max_samples
    
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    pos_sim = torch.sum(z1 * z2, dim=1) / temp
    neg_sim = torch.matmul(z1, z2.T) / temp
    
    mask = torch.eye(N, dtype=torch.bool, device=z1.device)
    neg_sim.masked_fill_(mask, -1e9)
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(N, dtype=torch.long, device=z1.device)
    
    loss = F.cross_entropy(logits, labels)
    return loss


def bpr_loss(user_emb, pos_emb, neg_emb, l2_reg=1e-6):
    pos_score = torch.sum(user_emb * pos_emb, dim=-1)
    neg_score = torch.sum(user_emb * neg_emb, dim=-1)
    loss = -torch.mean(F.logsigmoid(pos_score - neg_score))
    reg_loss = l2_reg * (user_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / user_emb.size(0)
    return loss + reg_loss


def hetero_to_homo_edge_index(edge_index, num_users):
    user_idx = edge_index[0]
    food_idx = edge_index[1] + num_users
    return torch.stack([user_idx, food_idx], dim=0)


def train_one_epoch(model, optimizer, feature_dict, train_edge_index,
                    pos_train_edge_index, neg_train_edge_index,
                    batch_size, lambda_health=0.0, lambda_cl=0.1,
                    user_tags=None, food_tags=None, temp=0.2, max_cl_samples=2048,
                    args=None):
    model.train()

    cl_loss = torch.tensor(0.0)
    has_contrastive = False

    if isinstance(model, (SGSL, modified_SGSL)):
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model(feature_dict, train_edge_index, pos_train_edge_index, neg_train_edge_index)

    elif isinstance(model, LightGCN):
        num_users = model.num_users
        num_items = model.num_items
        num_nodes = num_users + num_items
        edge_index_homo = hetero_to_homo_edge_index(train_edge_index, num_users)
        edge_index_sparse = SparseTensor.from_edge_index(
            edge_index_homo, sparse_sizes=(num_nodes, num_nodes)
        )
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(edge_index_sparse)

    elif isinstance(model, NGCF):
        num_users = model.num_users
        num_items = model.num_items
        edge_index_homo = hetero_to_homo_edge_index(train_edge_index, num_users)
        num_nodes = num_users + num_items
        adj_t = SparseTensor.from_edge_index(edge_index_homo, sparse_sizes=(num_nodes, num_nodes))
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(adj_t)

    elif isinstance(model, (SimGCL, SGL, LightGCL)):
        num_users = model.backbone.num_users
        num_items = model.backbone.num_items
        edge_index_homo = hetero_to_homo_edge_index(train_edge_index, num_users)
        num_nodes = num_users + num_items
        adj_t = SparseTensor.from_edge_index(edge_index_homo, sparse_sizes=(num_nodes, num_nodes))
        
        outputs = model(adj_t)
        
        if len(outputs) == 8:
            users_emb_final, users_emb_0, items_emb_final, items_emb_0, z1_u, z1_i, z2_u, z2_i = outputs
            has_contrastive = True
            
            if lambda_cl > 0:
                cl_loss_u = info_nce_loss(z1_u, z2_u, temp=temp, max_samples=max_cl_samples)
                cl_loss_i = info_nce_loss(z1_i, z2_i, temp=temp, max_samples=max_cl_samples)
                cl_loss = (cl_loss_u + cl_loss_i) / 2
        else:
            users_emb_final, users_emb_0, items_emb_final, items_emb_0 = outputs[:4]

    elif isinstance(model, (RecipeRec, HFRS_DA)):
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model(feature_dict)

    else:
        feature_dict_filtered = {k: v for k, v in feature_dict.items() if k in ['user', 'food']}
        num_users = feature_dict_filtered['user'].size(0)
        edge_index_homo = hetero_to_homo_edge_index(train_edge_index, num_users)
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
            model(feature_dict_filtered, edge_index_homo)

    user_indices, pos_item_indices, neg_item_indices = sample_mini_batch(batch_size, train_edge_index)
    user_emb = users_emb_final[user_indices]
    pos_emb = items_emb_final[pos_item_indices]
    neg_emb = items_emb_final[neg_item_indices]

    if args and hasattr(args, 'model'):
        if args.model in ['GCN']:
            l2_reg = 5e-5
        elif args.model in ['GAT']:
            l2_reg = 1e-5
        elif args.model in ['SAGE']:
            l2_reg = 1e-6
        else:
            l2_reg = 1e-5
    else:
        l2_reg = 1e-5

    bpr = bpr_loss(user_emb, pos_emb, neg_emb, l2_reg=l2_reg)
    loss = bpr
    
    if has_contrastive and lambda_cl > 0:
        loss = loss + lambda_cl * cl_loss

    health_loss = torch.tensor(0.0)
    if lambda_health > 0 and user_tags is not None and food_tags is not None:
        if isinstance(model, HFRS_DA):
            if hasattr(model, 'health_regularizer'):
                health_loss = model.health_regularizer(user_tags, food_tags, train_edge_index)
            else:
                health_loss = F.l1_loss(user_tags[user_indices], food_tags[pos_item_indices])
        else:
            health_loss = F.l1_loss(user_tags[user_indices], food_tags[pos_item_indices])
        loss = loss + lambda_health * health_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        'total': loss.item(),
        'bpr': bpr.item(),
        'cl': cl_loss.item() if has_contrastive else 0.0,
        'health': health_loss.item(),
    }


@torch.no_grad()
def evaluate(model, feature_dict, eval_edge_index, 
             inference_edge_index=None,
             pos_edge_index=None, neg_edge_index=None, 
             K=20, graph=None, user_tags=None, food_tags=None, 
             neg_train_edge_index=None, device=None, 
             recsim_method="topk", top_k_foods=20, min_frequency=5, 
             is_test=False, recsim_edge_index=None):
    model.eval()
    
    edges_for_inference = inference_edge_index if inference_edge_index is not None else eval_edge_index
    
    if device is None:
        device = next(model.parameters()).device
    
    try:
        if isinstance(model, LightGCN):
            num_users = model.num_users
            num_items = model.num_items
            num_nodes = num_users + num_items
            edge_index_sparse = SparseTensor.from_edge_index(
                edges_for_inference.to(device), sparse_sizes=(num_nodes, num_nodes)
            )
            u_emb, _, i_emb, _ = model(edge_index_sparse)
            
        elif isinstance(model, (SGSL, modified_SGSL)):
            if inference_edge_index is not None:
                hetero_edges = inference_edge_index.clone()
                if hetero_edges[1].max() >= model.num_users:
                    hetero_edges[1] = hetero_edges[1] - model.num_users
            else:
                hetero_edges = eval_edge_index
            
            u_emb, _, i_emb, _ = model(feature_dict, hetero_edges, pos_edge_index, neg_edge_index)
            
        elif isinstance(model, NGCF):
            num_nodes = model.num_users + model.num_items
            adj_t = SparseTensor.from_edge_index(
                edges_for_inference.to(device), sparse_sizes=(num_nodes, num_nodes)
            )
            u_emb, _, i_emb, _ = model(adj_t)
            
        elif isinstance(model, (SimGCL, SGL, LightGCL)):
            num_nodes = model.backbone.num_users + model.backbone.num_items
            adj_t = SparseTensor.from_edge_index(
                edges_for_inference.to(device), sparse_sizes=(num_nodes, num_nodes)
            )
            outputs = model(adj_t)
            u_emb, _, i_emb, _ = outputs[:4]
            
        elif isinstance(model, (RecipeRec, HFRS_DA)):
            u_emb, _, i_emb, _ = model(feature_dict)
            
        elif hasattr(model, "model_type") and model.model_type in ["GCN", "GAT", "SAGE", "MLP"]:
            u_emb, _, i_emb, _ = model(feature_dict, edges_for_inference.to(device))
            
        else:
            u_emb, _, i_emb, _ = model(feature_dict, edges_for_inference.to(device))
            
    except Exception as e:
        print(f"Model forward error: {e}")
        u_emb, _, i_emb, _ = model(feature_dict, eval_edge_index.to(device))

    user_labels = graph['user'].y.to(device) if graph and 'y' in graph['user'] else None
    full_edge_index = recsim_edge_index if recsim_edge_index is not None else eval_edge_index

    metrics = get_metrics(
        model, user_tags, food_tags,
        eval_edge_index,
        [neg_train_edge_index] if neg_train_edge_index is not None else [],
        K, u_emb, i_emb,
        user_labels=user_labels,
        num_price_tags=3,
        num_nutrition_tags=18,
        num_poverty_tags=3,
        recsim_method=recsim_method,
        top_k_foods=top_k_foods,
        min_frequency=min_frequency,
        device=device,
        full_edge_index=full_edge_index,
        is_test=is_test
    )
    return metrics


def get_metrics(model, user_tags, food_tags, edge_index, exclude_edge_indices, k, 
                users_emb_final, items_emb_final, user_labels=None, 
                num_price_tags=3, num_nutrition_tags=18, num_poverty_tags=3,
                batch_size=1024, device="cuda", recsim_method="topk", 
                top_k_foods=50, min_frequency=5, full_edge_index=None, is_test=False):
    device = users_emb_final.device
    user_embedding = users_emb_final.to(device)
    item_embedding = items_emb_final.to(device)
    num_users = user_embedding.size(0)
    num_items = item_embedding.size(0)

    rating_mask = None
    if exclude_edge_indices:
        rating_mask = torch.zeros((num_users, num_items), dtype=torch.bool, device=device)
        for exclude_edge_index in exclude_edge_indices:
            src, dst = exclude_edge_index
            rating_mask[src, dst] = True

    recovered_emb = None
    edge_for_recsim = full_edge_index if full_edge_index is not None else edge_index
    
    if user_labels is not None:
        recovered_ids = torch.nonzero(user_labels == 2, as_tuple=True)[0]
        if recovered_ids.numel() > 0:
            device_edge = edge_for_recsim.device
            mask = torch.isin(edge_for_recsim[0].to(device=device_edge, dtype=torch.long),
                            recovered_ids.to(device=device_edge, dtype=torch.long))
            recovered_food_ids = edge_for_recsim[1][mask]
            
            if recovered_food_ids.numel() > 0:
                if recsim_method == "topk":
                    recovered_food_counts = torch.bincount(recovered_food_ids, minlength=num_items)
                    all_food_counts = torch.bincount(edge_for_recsim[1], minlength=num_items)
                    
                    tf = recovered_food_counts.float() / (recovered_food_counts.sum() + 1e-8)
                    idf = torch.log((num_items + 1.0) / (all_food_counts.float() + 1.0))
                    tfidf_scores = tf * idf
                    tfidf_scores[recovered_food_counts == 0] = 0
                    
                    valid_count = (tfidf_scores > 0).sum().item()
                    top_k = min(top_k_foods, valid_count)
                    
                    if top_k > 0:
                        top_values, top_indices = torch.topk(tfidf_scores, k=top_k)
                        if top_indices.numel() > 0:
                            recovered_emb = item_embedding[top_indices.to(device)].mean(dim=0, keepdim=True)

    recall_all, precision_all, ndcg_all = [], [], []
    H_scores, PA_scores, RecSims, avg_tags_all = [], [], [], []
    
    all_recommended_foods = set() if is_test else None
    
    num_batches = (num_users + batch_size - 1) // batch_size
    test_user_pos_items = get_user_positive_items(edge_index)

    for b in tqdm(range(num_batches), desc="Eval", disable=num_batches < 10):
        start, end = b * batch_size, min((b + 1) * batch_size, num_users)
        u_batch = user_embedding[start:end]
        rating = torch.matmul(u_batch, item_embedding.T)
        
        if rating_mask is not None:
            rating[rating_mask[start:end]] = -(1 << 10)
        
        _, topK = torch.topk(rating, k=k, dim=1)
        users = torch.arange(start, end, device=device)

        if is_test:
            all_recommended_foods.update(topK.flatten().cpu().tolist())

        r = torch.zeros(len(users), k, device=device)
        gt_list = []
        for i, u in enumerate(users):
            gt = test_user_pos_items.get(u.item(), [])
            gt_list.append(gt)
            if len(gt) > 0:
                hits = torch.isin(topK[i].to(device), torch.as_tensor(gt, device=device, dtype=torch.long))
                r[i] = hits.float()

        valid_mask = [len(gt) > 0 for gt in gt_list]
        if any(valid_mask):
            gt_valid = [gt_list[i] for i, v in enumerate(valid_mask) if v]
            r_valid = r[valid_mask]
            recall, precision = RecallPrecision_ATk(gt_valid, r_valid, k)
            ndcg = NDCGatK_r(gt_valid, r_valid, k)
        else:
            recall, precision, ndcg = 0.0, 0.0, 0.0
        
        recall_all.append(recall)
        precision_all.append(precision)
        ndcg_all.append(ndcg)

        user_batch_tags = user_tags[start:end]
        food_batch_tags = food_tags[topK]
        user_nutrition = user_batch_tags[:, :num_nutrition_tags]
        food_nutrition = food_batch_tags[:, :, :num_nutrition_tags]
        common_nutrition = (user_nutrition.unsqueeze(1) * food_nutrition).sum(dim=2) > 0
        H_scores.append(common_nutrition.float().mean().item())

        if num_poverty_tags > 0:
            user_poverty = user_batch_tags[:, -num_poverty_tags:]
            food_poverty = food_batch_tags[:, :, -num_poverty_tags:]
            poverty_match = (user_poverty.unsqueeze(1) * food_poverty).sum(dim=2) > 0
            PA_scores.append(poverty_match.float().mean().item())

        if recovered_emb is not None:
            food_mean = item_embedding[topK].mean(dim=1)
            sim = F.cosine_similarity(food_mean, recovered_emb, dim=1)
            RecSims.append(sim.mean().item())

        if is_test:
            tags_per_food = food_batch_tags.sum(dim=2)
            avg_tags_user = tags_per_food.mean(dim=1).mean().item()
            avg_tags_all.append(avg_tags_user)

        torch.cuda.empty_cache()

    if is_test:
        percent_foods = len(all_recommended_foods) / num_items
        avg_tags = np.mean(avg_tags_all)
    else:
        percent_foods = 0.0
        avg_tags = 0.0

    return {
        "recall": np.mean(recall_all),
        "precision": np.mean(precision_all),
        "ndcg": np.mean(ndcg_all),
        "H_score": np.mean(H_scores),
        "PA": np.mean(PA_scores),
        "RecSim": np.mean(RecSims) if RecSims else 0.0,
        "avg_tags": avg_tags,
        "percent_foods": percent_foods,
    }


def train_single_lr(args, lr, graph, train_edge_index, val_edge_index, test_edge_index,
                    pos_train_edge_index, neg_train_edge_index,
                    pos_val_edge_index, neg_val_edge_index,
                    pos_test_edge_index, neg_test_edge_index,
                    train_edge_index_homo, val_inference_edges, test_inference_edges,
                    val_recsim_edges, test_recsim_edges,
                    feature_dict, user_tags, food_tags, device, num_users, num_foods):
    
    if args.model == "LightGCN":
        model = LightGCN(num_users, num_foods, embedding_dim=args.hidden_dim, layers=args.layers)
    elif args.model == "GCN":
        model = GCNModel(num_users, num_foods, embedding_dim=args.hidden_dim, num_layers=args.layers, model_type='GCN')
    elif args.model == "GAT":
        model = GCNModel(num_users, num_foods, embedding_dim=args.hidden_dim, num_layers=args.layers, model_type='GAT')
    elif args.model == "SAGE":
        model = GCNModel(num_users, num_foods, embedding_dim=args.hidden_dim, num_layers=args.layers, model_type='SAGE')
    elif args.model == "NGCF":
        model = NGCF(num_users, num_foods, embedding_dim=args.hidden_dim, layers=args.layers)
    elif args.model == "SimGCL":
        model = SimGCL(num_users, num_foods, embedding_dim=args.hidden_dim, layers=args.layers)
    elif args.model == "SGL":
        model = SGL(num_users, num_foods, embedding_dim=args.hidden_dim, layers=args.layers)
    elif args.model == "LightGCL":
        model = LightGCL(num_users, num_foods, embedding_dim=args.hidden_dim, layers=args.layers)
    elif args.model == "RecipeRec":
        model = RecipeRec(graph, embedding_dim=args.hidden_dim)
    elif args.model == "HFRS_DA":
        model = HFRS_DA(graph, embedding_dim=args.hidden_dim)
    elif args.model == "SGSL":
        model = SGSL(graph, dim=args.hidden_dim, num_layer=args.layers)
    elif args.model == "modified_SGSL":
        model = modified_SGSL(graph, dim=args.hidden_dim, num_layer=args.layers)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if args.model in ['SAGE', 'GAT']:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if args.use_wandb:
        wandb.init(
            project="glen_bench",
            entity='',
            name=f"{args.model}_lr{lr}_dim{args.hidden_dim}_cl{args.lambda_cl}",
            config={**vars(args), 'current_lr': lr},
            reinit=True
        )

    best_val_recall = 0.0
    best_val_metrics = None
    
    for epoch in range(args.epochs):
        loss_dict = train_one_epoch(
            model, optimizer, feature_dict, train_edge_index,
            pos_train_edge_index, neg_train_edge_index,
            batch_size=args.batch_size,
            lambda_health=args.lambda_health,
            lambda_cl=args.lambda_cl,
            user_tags=user_tags, food_tags=food_tags,
            temp=args.temp,
            max_cl_samples=args.max_cl_samples,
            args=args
        )
        
        if args.model in ['SAGE', 'GAT']:
            scheduler.step()
        elif epoch % args.lr_decay_interval == 0 and epoch != 0:
            scheduler.step()

        if epoch % args.eval_interval == 0:
            metrics = evaluate(
                model, feature_dict,
                val_edge_index,
                inference_edge_index=val_inference_edges,
                pos_edge_index=pos_val_edge_index,
                neg_edge_index=neg_val_edge_index,
                K=args.K,
                graph=graph,
                user_tags=user_tags,
                food_tags=food_tags,
                neg_train_edge_index=neg_train_edge_index,
                device=device,
                recsim_method=args.recsim_method,
                top_k_foods=args.top_k_foods,
                min_frequency=args.min_frequency,
                is_test=False,
                recsim_edge_index=val_recsim_edges
            )

            print(f"[Epoch {epoch}] Loss: {loss_dict['total']:.4f} | Recall@{args.K}: {metrics['recall']:.4f} | NDCG@{args.K}: {metrics['ndcg']:.4f} | H: {metrics['H_score']:.4f}")
            
            if metrics['recall'] > best_val_recall:
                best_val_recall = metrics['recall']
                best_val_metrics = metrics
                print(f"New best validation recall: {best_val_recall:.4f}")
            
            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": lr,
                    "train/loss_total": loss_dict['total'],
                    "train/loss_bpr": loss_dict['bpr'],
                    "val/recall": metrics["recall"],
                    "val/precision": metrics["precision"],
                    "val/ndcg": metrics["ndcg"],
                    "val/H_score": metrics["H_score"],
                })

    test_metrics = evaluate(
        model, feature_dict,
        test_edge_index,
        inference_edge_index=test_inference_edges,
        pos_edge_index=pos_test_edge_index,
        neg_edge_index=neg_test_edge_index,
        K=args.K,
        graph=graph,
        user_tags=user_tags,
        food_tags=food_tags,
        neg_train_edge_index=neg_train_edge_index,
        device=device,
        recsim_method=args.recsim_method,
        top_k_foods=args.top_k_foods,
        min_frequency=args.min_frequency,
        is_test=True,
        recsim_edge_index=test_recsim_edges
    )

    print(f"   Test Results (LR={lr}):")
    print(f"   Recall@{args.K}: {test_metrics['recall']:.4f}")
    print(f"   Precision@{args.K}: {test_metrics['precision']:.4f}")
    print(f"   NDCG@{args.K}: {test_metrics['ndcg']:.4f}")
    print(f"   H-Score: {test_metrics['H_score']:.4f}")
    print(f"   PA: {test_metrics['PA']:.4f}")

    if args.use_wandb:
        wandb.log({
            "test/recall": test_metrics["recall"],
            "test/precision": test_metrics["precision"],
            "test/ndcg": test_metrics["ndcg"],
            "test/H_score": test_metrics["H_score"],
            "test/PA": test_metrics["PA"],
            "test/RecSim": test_metrics["RecSim"],
            "test/avg_tags": test_metrics["avg_tags"],
            "test/percent_foods": test_metrics["percent_foods"],
            "best_val_recall": best_val_recall,
        })
        
        wandb.summary.update({
            "test/recall": test_metrics["recall"],
            "test/precision": test_metrics["precision"],
            "test/ndcg": test_metrics["ndcg"],
            "test/H_score": test_metrics["H_score"],
            "test/PA": test_metrics["PA"],
            "test/RecSim": test_metrics["RecSim"],
            "test/avg_tags": test_metrics["avg_tags"],
            "test/percent_foods": test_metrics["percent_foods"],
            "best_val_recall": best_val_recall,
        })
        
        wandb.finish()

    return {
        'lr': lr,
        'best_val_recall': best_val_recall,
        'best_val_metrics': best_val_metrics,
        'test_metrics': test_metrics
    }


def main(args):
    if args.lr_list:
        lr_list = [float(lr) for lr in args.lr_list.split(',')]
    else:
        lr_list = [args.lr]

    set_seed(args.seed) 

    graph = torch.load(args.graph_path, map_location="cpu")
    num_users, num_foods = graph['user'].num_nodes, graph['food'].num_nodes
    edge_index = graph[('user', 'eat', 'food')].edge_index
    edge_label_index = graph[('user', 'eat', 'food')].edge_label_index
    feature_dict = graph.x_dict

    (
        train_edge_index, val_edge_index, test_edge_index,
        pos_train_edge_index, neg_train_edge_index,
        pos_val_edge_index, neg_val_edge_index,
        pos_test_edge_index, neg_test_edge_index
    ) = split_data_new(edge_index, edge_label_index)

    train_edge_index_homo = hetero_to_homo_edge_index(train_edge_index, num_users)
    val_edge_index_homo = hetero_to_homo_edge_index(val_edge_index, num_users)
    
    val_inference_edges = train_edge_index_homo
    test_inference_edges = torch.cat([train_edge_index_homo, val_edge_index_homo], dim=1)
    
    val_recsim_edges = train_edge_index
    test_recsim_edges = torch.cat([train_edge_index, val_edge_index], dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_dict = {k: v.to(device) for k, v in feature_dict.items()}
    user_tags, food_tags = graph['user'].tags.to(device), graph['food'].tags.to(device)

    all_results = []
    for lr in lr_list:
        result = train_single_lr(
            args, lr, graph, train_edge_index, val_edge_index, test_edge_index,
            pos_train_edge_index, neg_train_edge_index,
            pos_val_edge_index, neg_val_edge_index,
            pos_test_edge_index, neg_test_edge_index,
            train_edge_index_homo, val_inference_edges, test_inference_edges,
            val_recsim_edges, test_recsim_edges,
            feature_dict, user_tags, food_tags, device, num_users, num_foods
        )
        all_results.append(result)
        
        torch.cuda.empty_cache()

    best_result = max(all_results, key=lambda x: x['best_val_recall'])
    
    for result in all_results:
        lr = result['lr']
        val_recall = result['best_val_recall']
        test_recall = result['test_metrics']['recall']
        test_ndcg = result['test_metrics']['ndcg']
        test_h = result['test_metrics']['H_score']
        test_pa = result['test_metrics']['PA']
        
        marker = "⭐" if result == best_result else "  "
        print(f"{marker} {lr:<10.2e} {val_recall:<12.4f} {test_recall:<13.4f} {test_ndcg:<12.4f} {test_h:<13.4f} {test_pa:<12.4f}")
    
    print(f"{'-'*100}")
    print(f"Best LR: {best_result['lr']:.2e} (Val Recall: {best_result['best_val_recall']:.4f})")
    
    if args.save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"{args.model}_grid_search.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model': args.model,
                'lr_list': lr_list,
                'results': all_results,
                'best_lr': best_result['lr'],
                'best_val_recall': best_result['best_val_recall']
            }, f, indent=2)
        
        print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument("--graph_path", type=str, default="./graph/hetero_graph_bipartite.pt")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-2, help="Default learning rate (used if --lr_list not provided)")
    parser.add_argument("--lr_list", type=str, default=None, 
                       help="Comma-separated list of learning rates, e.g., '1e-2,3e-2,5e-2,1e-1,3e-1'")  
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--lr_decay_interval", type=int, default=50)
    parser.add_argument("--lambda_health", type=float, default=0.0)
    parser.add_argument("--lambda_cl", type=float, default=0.08)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max_cl_samples", type=int, default=2048)
    parser.add_argument("--recsim_method", type=str, default="topk")
    parser.add_argument("--top_k_foods", type=int, default=20)
    parser.add_argument("--min_frequency", type=int, default=5)
    parser.add_argument("--save_results", action='store_true', help="Save grid search results to JSON")
    
    args = parser.parse_args()
    main(args)