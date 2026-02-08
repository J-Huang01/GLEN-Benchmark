from RCSYS_utils import *
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SignedConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch import Tensor
import torch.optim as optim
from torch_geometric.nn import Linear
from typing import Tuple, Union, Optional
import logging
import math
from torch_sparse import matmul
from torch_geometric.utils import dropout_adj
logger = logging.getLogger(__name__)


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 64.
        layers (int, optional): Number of message passing layers. Defaults to 3.
        add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, add_self_loops: bool = False):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)
        
        self.cached_norm_adj = None
        self.cached_edge_index = None

    def forward(self, edge_index: Union[SparseTensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward propagation of LightGCN Model.
        
        Args:
            edge_index: Adjacency matrix (SparseTensor or edge index)
        
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        device = self.users_emb.weight.device

        need_recompute = (
            self.cached_norm_adj is None or 
            (isinstance(edge_index, torch.Tensor) and 
             self.cached_edge_index is not None and
             not torch.equal(edge_index, self.cached_edge_index))
        )
        
        if need_recompute:
            if not isinstance(edge_index, SparseTensor):
                num_nodes = self.num_users + self.num_items
                edge_index_sparse = SparseTensor.from_edge_index(
                    edge_index, sparse_sizes=(num_nodes, num_nodes)
                ).to(device)
            else:
                edge_index_sparse = edge_index.to(device)
            
            self.cached_norm_adj = gcn_norm(edge_index_sparse, add_self_loops=self.add_self_loops)
            
            if isinstance(edge_index, torch.Tensor):
                self.cached_edge_index = edge_index.clone()

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.layers):
            emb_k = self.propagate(self.cached_norm_adj, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        if self.layers == 1:
            layer_weights = torch.tensor([0.5, 0.5], device=embs.device)
        elif self.layers == 2:
            layer_weights = torch.tensor([0.2, 0.3, 0.5], device=embs.device)
        elif self.layers == 3:
            layer_weights = torch.tensor([0.1, 0.2, 0.3, 0.4], device=embs.device)
        elif self.layers == 4:
            layer_weights = torch.tensor([0.1, 0.15, 0.25, 0.5], device=embs.device)
        else:
            layer_weights = torch.arange(1, self.layers + 2, dtype=torch.float, device=embs.device)
            layer_weights = layer_weights / layer_weights.sum()

        emb_final = torch.sum(embs * layer_weights.view(1, -1, 1), dim=1)

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])

        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight
    
    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class GCNModel(nn.Module):
    """GCN Model for graph-based recommendation system.
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 64.
        num_layers (int, optional): Number of GCN layers. Defaults to 2.
        add_self_loops (bool, optional): Whether to add self loops. Defaults to False.
        model_type (str, optional): Type of GNN model ('GCN', 'GAT', 'SAGE', 'MLP'). Defaults to 'gcn'.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 num_layers: int = 2, add_self_loops: bool = False, model_type: str = 'gcn'):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.num_layers = embedding_dim, num_layers
        self.add_self_loops = add_self_loops
        self.model_type = model_type.upper()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['user', 'food']:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)

        if self.model_type == 'GCN':
            self.model_layers = nn.ModuleList([GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        elif self.model_type == 'GAT':
            self.model_layers = nn.ModuleList([GATConv(embedding_dim, embedding_dim, heads=1) for _ in range(num_layers)])
        elif self.model_type == 'SAGE':
            self.model_layers = nn.ModuleList([SAGEConv(embedding_dim, embedding_dim) for _ in range(num_layers)])
        elif self.model_type == 'MLP':
            self.model_layers = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(num_layers)])
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
    
    def forward(self, feature_dict: dict, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass of GCN model.
        
        Args:
            feature_dict: Dictionary containing node features
            edge_index: Edge index tensor
            
        Returns:
            Tuple of (users_emb_final, users_emb_0, items_emb_final, items_emb_0)
        """
        if self.model_type == 'GCN':
            edge_index_norm, edge_weight = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        else:
            edge_index_norm, edge_weight = edge_index, None

        device = list(feature_dict.values())[0].device
        edge_index_norm = edge_index_norm.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
            if node_type in self.lin_dict
        }
        
        emb_0 = torch.cat([feature_dict['user'], feature_dict['food']], dim=0)
        users_emb_0, items_emb_0 = torch.split(emb_0, [self.num_users, self.num_items])
        
        emb_k = emb_0

        for i, model_layer in enumerate(self.model_layers):
            if self.model_type == 'MLP':
                emb_k = model_layer(emb_k)
            elif self.model_type == 'GCN':
                emb_k = model_layer(emb_k, edge_index_norm, edge_weight)
            else:
                emb_k = model_layer(emb_k, edge_index_norm)
            
            emb_k = self.norms[i](emb_k)
            emb_k = F.relu(emb_k)
            
            emb_k = F.dropout(emb_k, p=0.1, training=self.training)

        users_emb_final, items_emb_final = torch.split(emb_k, [self.num_users, self.num_items])
        
        # L2 normalization
        users_emb_final = F.normalize(users_emb_final, p=2, dim=1)
        items_emb_final = F.normalize(items_emb_final, p=2, dim=1)
        
        return users_emb_final, users_emb_0, items_emb_final, items_emb_0


class SignedGCN(torch.nn.Module):
    """Signed Graph Convolutional Network from https://arxiv.org/abs/1808.06354
    
    Args:
        num_users (int): Number of users
        num_foods (int): Number of items
        hidden_channels (int): Hidden dimensionality
        num_layers (int): Number of layers
    """
    
    def __init__(self, num_users: int, num_foods: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_foods

        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.hidden_channels)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.hidden_channels)

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

        self.conv1 = SignedConv(hidden_channels, hidden_channels // 2, first_aggr=True)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(
                SignedConv(hidden_channels // 2, hidden_channels // 2, first_aggr=False))

        self.lin = torch.nn.Linear(2 * hidden_channels, 3)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tensor:
        """Computes node embeddings based on positive and negative edges.
        
        Args:
            pos_edge_index: The positive edge indices
            neg_edge_index: The negative edge indices
            
        Returns:
            Node embeddings
        """
        x = torch.cat([self.users_emb.weight, self.items_emb.weight])
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z

    def discriminate(self, z: Tensor, edge_index: Tensor) -> Tensor:
        """Classifies link relation between node pairs.
        
        Args:
            z: Node embeddings
            edge_index: Edge indices
            
        Returns:
            Classification result (-1, 0, or 1)
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)

        log_softmax_output = torch.log_softmax(value, dim=1)
        class_indices = torch.argmax(log_softmax_output, dim=1)

        # Map class indices: 0 -> -1, 1 -> 0, 2 -> 1
        mapping = torch.tensor([-1, 0, 1]).to(value.device)
        mapped_output = mapping[class_indices]

        return mapped_output


class MetricCalculator(nn.Module):
    """Applies a learnable transformation to node features."""
    
    def __init__(self, feature_dim: int):
        super(MetricCalculator, self).__init__()
        self.weight = nn.Parameter(torch.empty((1, feature_dim)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, node_features: Tensor) -> Tensor:
        return node_features * self.weight


class GraphGenerator(nn.Module):
    """Builds a graph based on similarity between node features from two different sets."""
    
    def __init__(self, feature_dim: int, num_heads: int = 2, similarity_threshold: float = 0.1):
        super(GraphGenerator, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.metric_layers = nn.ModuleList([MetricCalculator(feature_dim) for _ in range(num_heads)])
        self.num_heads = num_heads

    def forward(self, left_features: Tensor, right_features: Tensor, edge_index: Tensor) -> Tensor:
        """Compute a similarity matrix between left and right node features."""
        similarity_matrix = torch.zeros(edge_index.size(1)).to(edge_index.device)
        for metric_layer in self.metric_layers:
            weighted_left = metric_layer(left_features[edge_index[0]])
            weighted_right = metric_layer(right_features[edge_index[1]])
            similarity_matrix += F.cosine_similarity(weighted_left, weighted_right, dim=1)

        similarity_matrix /= self.num_heads
        return torch.where(similarity_matrix < self.similarity_threshold, 
                          torch.zeros_like(similarity_matrix), similarity_matrix)


class GraphChannelAttLayer(nn.Module):
    """Graph channel attention layer for fusing multiple graph views."""
    
    def __init__(self, num_channel: int):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, edge_mask_list: list) -> Tensor:
        edge_mask = torch.stack(edge_mask_list, dim=0)
        edge_mask = F.normalize(edge_mask, dim=1, p=1)
        
        softmax_weights = torch.softmax(self.weight, dim=0)
        weighted_edge_masks = edge_mask * softmax_weights[:, None]
        fused_edge_mask = torch.sum(weighted_edge_masks, dim=0)

        return fused_edge_mask > 0.5


class SGSL(nn.Module):
    """Structural Graph Structure Learning model."""
    
    def __init__(self, graph, embedding_dim: int, feature_threshold: float = 0.3, 
                 num_heads: int = 4, num_layer: int = 3):
        super(SGSL, self).__init__()

        self.num_users = graph['user'].num_nodes
        self.num_foods = graph['food'].num_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.feature_threshold = feature_threshold

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)
        
        self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
        self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
        self.fusion = GraphChannelAttLayer(3)
        self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)

    def forward(self, feature_dict: dict, edge_index: Tensor, 
                pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Heterogeneous feature mapping
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        # Generate feature graph
        mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
        mask_ori = torch.ones_like(mask_feature)

        # Generate semantic graph
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index)

        # Fusion with attention
        edge_mask = self.fusion([mask_ori, mask_feature, mask_semantic])

        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], 
                                        sparse_sizes=(sparse_size, sparse_size))
        
        return self.lightgcn(sparse_edge_index)


class modified_SGSL(nn.Module):
    """Modified SGSL with ablation options."""
    
    def __init__(self, graph, embedding_dim: int, feature_threshold: float = 0.3, 
                 num_heads: int = 4, num_layer: int = 3):
        super(modified_SGSL, self).__init__()

        self.num_users = graph['user'].num_nodes
        self.num_foods = graph['food'].num_nodes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layer = num_layer
        self.feature_threshold = feature_threshold

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)
        
        self.feature_graph_generator = GraphGenerator(self.embedding_dim, self.num_heads, self.feature_threshold)
        self.signed_layer = SignedGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer)
        self.fusion = GraphChannelAttLayer(2)
        self.lightgcn = LightGCN(self.num_users, self.num_foods, self.embedding_dim, self.num_layer, False)

    def forward(self, feature_dict: dict, edge_index: Tensor, 
                pos_edge_index: Tensor, neg_edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Heterogeneous feature mapping
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        # Generate graphs
        mask_feature = self.feature_graph_generator(feature_dict['user'], feature_dict['food'], edge_index)
        mask_ori = torch.ones_like(mask_feature)
        z = self.signed_layer(pos_edge_index, neg_edge_index)
        mask_semantic = self.signed_layer.discriminate(z, edge_index)

        # Fusion without feature mask (ablation)
        edge_mask = self.fusion([mask_ori, mask_semantic])

        edge_index_new = edge_index[:, edge_mask]
        sparse_size = self.num_users + self.num_foods
        sparse_edge_index = SparseTensor(row=edge_index_new[0], col=edge_index_new[1], 
                                        sparse_sizes=(sparse_size, sparse_size))
        
        return self.lightgcn(sparse_edge_index)


class ModifiedGCN(MessagePassing):
    """Modified GCN with feature projection."""
    
    def __init__(self, graph, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, add_self_loops: bool = False):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            self.lin_dict[node_type] = Linear(-1, embedding_dim)

    def forward(self, feature_dict: dict, edge_index: Tensor, 
                share_food: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward propagation with feature projection."""
        feature_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in feature_dict.items()
        }

        edge_index_mod = torch.stack([edge_index[0], edge_index[1] + self.num_users], dim=0)
        sparse_size = self.num_users + self.num_items
        edge_index_mod = SparseTensor(row=edge_index_mod[0], col=edge_index_mod[1], 
                                     sparse_sizes=(sparse_size, sparse_size))
        
        edge_index_norm = gcn_norm(edge_index_mod, add_self_loops=self.add_self_loops)
        
        user_emb = feature_dict['user']
        item_emb = feature_dict['food']

        emb_0 = torch.cat([user_emb, item_emb])
        embs = [emb_0]
        emb_k = emb_0

        for i in range(self.layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)
            
        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items])

        return users_emb_final, users_emb_final, items_emb_final, items_emb_final

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)

class NGCF(MessagePassing):
    """Neural Graph Collaborative Filtering from https://arxiv.org/abs/1905.08108
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int): Embedding dimensionality
        layers (int): Number of layers
        add_self_loops (bool): Whether to add self loops
        dropout (float): Dropout rate
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, add_self_loops: bool = False, dropout: float = 0.0):
        super().__init__(aggr='add')
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.layers = embedding_dim, layers
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        self.users_emb = nn.Embedding(num_users, embedding_dim)
        self.items_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.users_emb.weight)
        nn.init.xavier_uniform_(self.items_emb.weight)

        self.W1 = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(layers)])
        self.W2 = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(layers)])
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, adj_t: SparseTensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        device = self.users_emb.weight.device
        
        # Convert to edge_index
        row, col, _ = adj_t.coo()
        edge_index = torch.stack([row, col], dim=0).to(device)
        
        edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=self.add_self_loops)
        
        x0 = torch.cat([self.users_emb.weight, self.items_emb.weight], dim=0)
        xk = x0
        all_layers = [x0]

        for l in range(self.layers):
            from torch_scatter import scatter_add
            
            row, col = edge_index
            
            # Neighbor aggregation
            neigh = scatter_add(xk[col] * edge_weight.unsqueeze(1), row, dim=0, dim_size=xk.size(0))
            
            # Element-wise product aggregation
            prod = scatter_add((xk[col] * x0[col]) * edge_weight.unsqueeze(1), row, dim=0, dim_size=xk.size(0))
            
            xk = self.leaky(self.W1[l](neigh) + self.W2[l](prod))
            xk = F.dropout(xk, p=self.dropout, training=self.training)
            all_layers.append(xk)

        x_final = torch.mean(torch.stack(all_layers, dim=1), dim=1)
        uK, iK = torch.split(x_final, [self.num_users, self.num_items], dim=0)
        return uK, self.users_emb.weight, iK, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x)


class SimGCL(nn.Module):
    """SimGCL: Simple Graph Contrastive Learning for Recommendation
    
    Uses feature-level perturbation instead of graph augmentations.
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int): Embedding dimensionality
        layers (int): Number of layers
        add_self_loops (bool): Whether to add self loops
        eps (float): Perturbation magnitude
        temp (float): Temperature for contrastive loss
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, add_self_loops: bool = False, eps: float = 0.2, temp: float = 0.2):
        super().__init__()
        self.backbone = LightGCN(num_users, num_items, embedding_dim, layers, add_self_loops)
        self.eps = eps
        self.temp = temp

    @staticmethod
    def _perturb(x: Tensor, eps: float) -> Tensor:
        """Add normalized noise to embeddings."""
        noise = F.normalize(torch.randn_like(x), dim=1) * eps
        return x + noise

    def forward(self, adj_t: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass with two perturbed views for contrastive learning."""
        uK, u0, iK, i0 = self.backbone(adj_t)
        
        z1_u = self._perturb(uK, self.eps)
        z1_i = self._perturb(iK, self.eps)
        z2_u = self._perturb(uK, self.eps)
        z2_i = self._perturb(iK, self.eps)
        
        return (uK, u0, iK, i0, z1_u, z1_i, z2_u, z2_i)


class SGL(nn.Module):
    """Self-supervised Graph Learning for Recommendation
    
    Generates two graph views via edge dropout and performs contrastive learning.
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int): Embedding dimensionality
        layers (int): Number of layers
        drop_rate (float): Edge dropout rate
        add_self_loops (bool): Whether to add self loops
        temp (float): Temperature for contrastive loss
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, drop_rate: float = 0.2, add_self_loops: bool = False, temp: float = 0.2):
        super().__init__()
        self.backbone = LightGCN(num_users, num_items, embedding_dim, layers, add_self_loops)
        self.drop_rate = drop_rate
        self.temp = temp

    @staticmethod
    def _drop_view(adj_t: SparseTensor, drop_rate: float) -> SparseTensor:
        """Create a dropped-edge view of the graph."""
        row, col, _ = adj_t.coo()
        ei = torch.stack([row, col], dim=0)
        
        ei_dropped, _ = dropout_adj(ei, p=drop_rate, force_undirected=True)
        
        N = adj_t.size(0)
        return SparseTensor(row=ei_dropped[0], col=ei_dropped[1], sparse_sizes=(N, N))

    def forward(self, adj_t: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass with two dropped-edge views."""
        uK, u0, iK, i0 = self.backbone(adj_t)
        
        v1 = self._drop_view(adj_t, self.drop_rate)
        v2 = self._drop_view(adj_t, self.drop_rate)
        z1_u, _, z1_i, _ = self.backbone(v1)
        z2_u, _, z2_i, _ = self.backbone(v2)
        
        return (uK, u0, iK, i0, z1_u, z1_i, z2_u, z2_i)


class LightGCL(nn.Module):
    """Lightweight Graph Contrastive Learning
    
    Simplified version using feature noise instead of complex random walks.
    
    Args:
        num_users (int): Number of users
        num_items (int): Number of items
        embedding_dim (int): Embedding dimensionality
        layers (int): Number of layers
        rw_steps (int): Random walk steps (unused in this simplified version)
        add_self_loops (bool): Whether to add self loops
        temp (float): Temperature for contrastive loss
    """
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, 
                 layers: int = 3, rw_steps: int = 1, add_self_loops: bool = False, temp: float = 0.2):
        super().__init__()
        self.backbone = LightGCN(num_users, num_items, embedding_dim, layers, add_self_loops)
        self.noise_eps = 0.2
        self.temp = temp

    def forward(self, adj_t: SparseTensor) -> Tuple[Tensor, ...]:
        """Forward pass with noise-augmented views."""
        uK, u0, iK, i0 = self.backbone(adj_t)
        
        E = torch.cat([uK, iK], dim=0)
        
        z1 = F.normalize(E + torch.randn_like(E) * self.noise_eps, dim=1)
        z2 = F.normalize(E + torch.randn_like(E) * self.noise_eps, dim=1)
        
        num_users = uK.size(0)
        z1_u, z1_i = z1[:num_users], z1[num_users:]
        z2_u, z2_i = z2[:num_users], z2[num_users:]
        
        return (uK, u0, iK, i0, z1_u, z1_i, z2_u, z2_i)