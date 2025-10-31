import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GENConv, SAGEConv, GATConv, GATv2Conv, 
    GINConv, EdgeConv, TransformerConv, ChebConv, DeepGCNLayer,
    global_mean_pool, global_max_pool, global_add_pool
)
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.data import Data
from scipy.sparse import lil_matrix
import scipy.sparse as sp


class LinearConvWrapper(nn.Module):
    """
    Wrapper for Linear layer to make it compatible with graph convolution API.
    DeepGCNLayer expects layers that take (x, edge_index), but Linear only takes x.
    This wrapper ignores edge_index and applies linear transformation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index=None):
        """Forward pass - ignores edge_index."""
        return self.linear(x)
    
    def reset_parameters(self):
        """Reset parameters for Linear layer."""
        self.linear.reset_parameters()


class DeepGCN(nn.Module):
    """
    Deep GCN using DeepGCNLayer for robust training of deep architectures.
    
    Args:
        in_channels (int): input feature dim
        hidden_channels (int or list[int]): hidden dims for each hidden layer
        out_channels (int): output dim
        conv_types (list[str] or str): convolution types for hidden layers.
            Options: 'LINEAR', 'GCN', 'GEN', 'SAGE', 'GAT', 'GATv2', 'GIN', 'Edge', 'Transformer', 'Cheb'
            If a single string, it will be repeated for all hidden layers.
        final_layer_type (str): type for final layer ('Linear', 'GCN', or any conv type)
        activation (str): 'tanh' or 'relu'
        dropout (float): dropout probability (0 disables)
        block (str): skip connection type ('res+', 'res', 'dense', 'plain')
        use_bn (bool): whether to use batch normalization
        gat_heads (int): number of attention heads for GAT/GATv2/Transformer (default: 4)
        cheb_K (int): filter size for ChebConv (default: 3)
        residual (bool): if True, predict changes (Δu, Δv) instead of absolute values (default: False)
        use_global_pooling (bool): if True, apply global pooling to get graph-level representation (default: False)
        pooling_type (str): type of global pooling ('mean', 'max', 'sum', 'attention') (default: 'mean')
        graph_output_dim (int): dimension of graph-level output after pooling (default: None, uses out_channels)
        pooling_position (str): where to apply pooling - 'end' (after all layers) or 'middle' (encoder-decoder) (default: 'end')
        encoder_layers (int): number of layers before pooling when pooling_position='middle' (default: None, uses half)
        decoder_channels (list[int]): hidden dims for decoder layers after pooling (default: None, mirrors encoder)
        use_ed_skip (bool): if True, add skip connections from encoder node features to decoder input (middle pooling only)
        ed_skip_type (str): 'concat' to concatenate features, 'add' to add (with optional projection)
    """
    def __init__(
        self,
        hidden_channels,
        conv_types="GCN",
        final_layer_type="Linear",
        activation="relu",
        dropout=0.0,
        block="res+",
        use_bn=True,
        gat_heads=4,
        cheb_K=3,
        in_channels=3,
        out_channels=2,
        residual=False,
        use_global_pooling=False,
        pooling_type="mean",
        graph_output_dim=None,
        pooling_position="end",
        encoder_layers=None,
        decoder_channels=None,
        use_ed_skip=False,
        ed_skip_type="concat",
    ):
        super().__init__()
        
        self.residual = residual  # Whether to predict changes or absolute values
        self.use_global_pooling = use_global_pooling
        self.pooling_type = pooling_type.lower()
        self.pooling_position = pooling_position.lower()
        
        # Encoder-Decoder skip connections (only applicable when pooling_position='middle')
        self.use_ed_skip = bool(use_ed_skip)
        self.ed_skip_type = str(ed_skip_type).lower()
        
        # Normalize hidden_channels to a list
        if isinstance(hidden_channels, int):
            hidden_sizes = [hidden_channels]
        else:
            hidden_sizes = list(hidden_channels)

        n_hidden = len(hidden_sizes)
        
        # Normalize conv_types to list
        if isinstance(conv_types, str):
            conv_types = [conv_types] * n_hidden
        if len(conv_types) != n_hidden:
            raise ValueError("len(conv_types) must match number of hidden layers")
        
        # Activation module
        if activation == "tanh":
            act_module = nn.Tanh()
        elif activation == "relu":
            act_module = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act_module = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "elu":
            act_module = nn.ELU(inplace=True)
        elif activation == "gelu":
            act_module = nn.GELU()
        elif activation == "prelu":
            act_module = nn.PReLU()
        else:
            raise ValueError("activation not implemented.")
        
        self.dropout = float(dropout)
        self.n_hidden = n_hidden
        self.gat_heads = gat_heads
        self.cheb_K = cheb_K
        
        # Input projection to match first hidden dimension
        if hidden_sizes[0] != in_channels:
            self.input_proj = Linear(in_channels, hidden_sizes[0])
        else:
            self.input_proj = None
        
        # Determine encoder-decoder split if using middle pooling
        if self.use_global_pooling and self.pooling_position == "middle":
            # Split layers into encoder and decoder
            if encoder_layers is None:
                # Default: use half the layers as encoder
                self.n_encoder_layers = n_hidden // 2
            else:
                self.n_encoder_layers = min(encoder_layers, n_hidden)
            
            # Build encoder layers
            self.encoder_layers = nn.ModuleList()
            prev_dim = hidden_sizes[0] if self.input_proj else in_channels
            
            for i in range(self.n_encoder_layers):
                hid = hidden_sizes[i]
                ctype = conv_types[i].upper()
                conv = self._create_conv_layer(ctype, prev_dim, hid)
                # Use 'plain' block when feature dims change to avoid pre-norm mismatch
                layer_block = block
                norm_dim = hid
                if block != "plain" and prev_dim != hid:
                    layer_block = "plain"
                # For residual-style blocks, normalization happens pre-conv → use prev_dim
                if layer_block != "plain":
                    norm_dim = prev_dim
                deep_layer = DeepGCNLayer(
                    conv=conv,
                    norm=nn.BatchNorm1d(norm_dim) if use_bn else None,
                    act=act_module,
                    block=layer_block,
                    dropout=dropout,
                )
                self.encoder_layers.append(deep_layer)
                prev_dim = hid
            
            # Store encoder output dimension
            self.encoder_output_dim = prev_dim
            
            # Graph pooling dimension
            if graph_output_dim is None:
                self.graph_dim = self.encoder_output_dim
            else:
                self.graph_dim = graph_output_dim
                
            # Attention for pooling if needed
            if self.pooling_type == "attention":
                self.attention_weights = Linear(self.encoder_output_dim, 1)
            
            # Optional projection after pooling
            if graph_output_dim is not None and graph_output_dim != self.encoder_output_dim:
                self.graph_projection = Linear(self.encoder_output_dim, graph_output_dim)
            else:
                self.graph_projection = None
            
            # Build decoder layers
            self.decoder_layers = nn.ModuleList()
            
            # Determine decoder architecture
            if decoder_channels is None:
                # Default: mirror encoder in reverse
                decoder_sizes = list(reversed(hidden_sizes[:self.n_encoder_layers]))
            else:
                decoder_sizes = list(decoder_channels)
            
            # First decoder layer input dimension (graph features, plus optional skip)
            if self.use_ed_skip and self.pooling_position == "middle" and self.ed_skip_type == "concat":
                prev_dim = self.graph_dim + self.encoder_output_dim
            else:
                prev_dim = self.graph_dim
            
            for i, hid in enumerate(decoder_sizes):
                # Use remaining conv_types or default to first type
                ctype_idx = self.n_encoder_layers + i
                if ctype_idx < len(conv_types):
                    ctype = conv_types[ctype_idx].upper()
                else:
                    ctype = conv_types[0].upper()
                
                conv = self._create_conv_layer(ctype, prev_dim, hid)
                # Use 'plain' block when feature dims change
                layer_block = block
                norm_dim = hid
                if block != "plain" and prev_dim != hid:
                    layer_block = "plain"
                if layer_block != "plain":
                    norm_dim = prev_dim
                deep_layer = DeepGCNLayer(
                    conv=conv,
                    norm=nn.BatchNorm1d(norm_dim) if use_bn else None,
                    act=act_module,
                    block=layer_block,
                    dropout=dropout,
                )
                self.decoder_layers.append(deep_layer)
                prev_dim = hid
            
            # Final output layer
            ft = final_layer_type.upper()
            if ft == "LINEAR":
                self.final_layer = Linear(prev_dim, out_channels)
            else:
                self.final_layer = self._create_conv_layer(ft, prev_dim, out_channels)
            self.final_layer_type = ft
            
        else:
            # Standard architecture: all layers, optional pooling at end
            self.encoder_layers = None
            self.decoder_layers = None
            
            # Build hidden layers using DeepGCNLayer
            self.layers = nn.ModuleList()
            prev_dim = hidden_sizes[0] if self.input_proj else in_channels
            
            for i, hid in enumerate(hidden_sizes):
                ctype = conv_types[i].upper()
                
                # Create the convolution layer based on type
                conv = self._create_conv_layer(ctype, prev_dim, hid)
                
                # Wrap in DeepGCNLayer
                layer_block = block
                norm_dim = hid
                if block != "plain" and prev_dim != hid:
                    layer_block = "plain"
                if layer_block != "plain":
                    norm_dim = prev_dim
                deep_layer = DeepGCNLayer(
                    conv=conv,
                    norm=nn.BatchNorm1d(norm_dim) if use_bn else None,
                    act=act_module,
                    block=layer_block,
                    dropout=dropout,
                )
                self.layers.append(deep_layer)
                prev_dim = hid
            
            # Final layer (no skip connection or activation on output)
            ft = final_layer_type.upper()
            if ft == "LINEAR":
                self.final_layer = Linear(prev_dim, out_channels)
            else:
                self.final_layer = self._create_conv_layer(ft, prev_dim, out_channels)
            
            self.final_layer_type = ft
            
            # Global pooling setup (only for 'end' position)
            if self.use_global_pooling and self.pooling_position == "end":
                # Determine the dimension after pooling
                pool_dim = out_channels
                
                # Attention-based pooling requires learnable parameters
                if self.pooling_type == "attention":
                    self.attention_weights = Linear(pool_dim, 1)
                
                # Optional graph-level output projection
                if graph_output_dim is not None:
                    self.graph_projection = Linear(pool_dim, graph_output_dim)
                else:
                    self.graph_projection = None
        
        # Initialize weights after construction
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize all learnable parameters using Xavier uniform initialization.
        This is crucial for training stability, especially with higher dropout rates.
        
        Xavier/Glorot initialization maintains variance through layers by scaling
        weights based on fan-in and fan-out. For ReLU activations, Kaiming 
        initialization is sometimes preferred, but Xavier works well for most cases.
        """
        # Initialize input projection
        if self.input_proj is not None:
            nn.init.xavier_uniform_(self.input_proj.weight, gain=1.0)
            if self.input_proj.bias is not None:
                nn.init.zeros_(self.input_proj.bias)
        
        # Initialize encoder-decoder architecture layers
        if hasattr(self, 'encoder_layers') and self.encoder_layers is not None:
            for layer in self.encoder_layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()  # DeepGCNLayer has its own reset
        
        if hasattr(self, 'decoder_layers') and self.decoder_layers is not None:
            for layer in self.decoder_layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        # Initialize standard architecture layers
        if hasattr(self, 'layers'):
            for layer in self.layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        
        # Initialize skip fusion projection (encoder-decoder skip connections)
        if self.use_ed_skip and hasattr(self, 'ed_align'):
            nn.init.xavier_uniform_(self.ed_align.weight, gain=1.0)
            if self.ed_align.bias is not None:
                nn.init.zeros_(self.ed_align.bias)
        
        # Initialize final layer
        if hasattr(self.final_layer, 'reset_parameters'):
            self.final_layer.reset_parameters()
        elif isinstance(self.final_layer, Linear):
            nn.init.xavier_uniform_(self.final_layer.weight, gain=1.0)
            if self.final_layer.bias is not None:
                nn.init.zeros_(self.final_layer.bias)
        
        # Initialize global pooling components
        if self.use_global_pooling:
            if hasattr(self, 'attention_weights'):
                nn.init.xavier_uniform_(self.attention_weights.weight, gain=1.0)
                if self.attention_weights.bias is not None:
                    nn.init.zeros_(self.attention_weights.bias)
            
            if hasattr(self, 'graph_projection') and self.graph_projection is not None:
                nn.init.xavier_uniform_(self.graph_projection.weight, gain=1.0)
                if self.graph_projection.bias is not None:
                    nn.init.zeros_(self.graph_projection.bias)
    
    def _create_conv_layer(self, conv_type, in_dim, out_dim):
        """Create a convolution layer based on the specified type."""
        if conv_type == "LINEAR":
            # Simple linear transformation without graph convolution
            # Use wrapper to make it compatible with DeepGCNLayer
            return LinearConvWrapper(in_dim, out_dim)
        
        elif conv_type == "GCN":
            return GCNConv(in_dim, out_dim)
        
        elif conv_type == "GEN":
            return GENConv(in_dim, out_dim, aggr='softmax')
        
        elif conv_type == "SAGE":
            return SAGEConv(in_dim, out_dim)
        
        elif conv_type == "GAT":
            # For GAT, out_dim is per head, so total output is out_dim * heads
            # We adjust so final output matches out_dim
            return GATConv(in_dim, out_dim // self.gat_heads, heads=self.gat_heads)
        
        elif conv_type == "GATV2":
            return GATv2Conv(in_dim, out_dim // self.gat_heads, heads=self.gat_heads)
        
        elif conv_type == "GIN":
            # GIN requires an MLP
            nn_module = Sequential(
                Linear(in_dim, out_dim),
                ReLU(),
                Linear(out_dim, out_dim)
            )
            return GINConv(nn_module)
        
        elif conv_type == "EDGE":
            # EdgeConv concatenates node pairs, so input is 2*in_dim
            nn_module = Sequential(
                Linear(2 * in_dim, out_dim),
                ReLU(),
                Linear(out_dim, out_dim)
            )
            return EdgeConv(nn_module)
        
        elif conv_type == "TRANSFORMER":
            return TransformerConv(in_dim, out_dim // self.gat_heads, heads=self.gat_heads)
        
        elif conv_type == "CHEB":
            return ChebConv(in_dim, out_dim, K=self.cheb_K)
        
        else:
            raise ValueError(
                f"Unsupported conv type: {conv_type}. "
                f"Options: LINEAR, GCN, GEN, SAGE, GAT, GATv2, GIN, Edge, Transformer, Cheb"
            )
    
    def global_pooling(self, x, batch=None):
        """
        Apply global pooling to aggregate node features into graph-level representation.
        
        Args:
            x: Node features [num_nodes, feature_dim]
            batch: Batch vector [num_nodes] indicating which graph each node belongs to.
                   If None, assumes all nodes belong to a single graph.
                   
        Returns:
            pooled: Graph-level features [num_graphs, feature_dim] or [1, feature_dim]
        """
        if batch is None:
            # Single graph: create a batch vector of all zeros
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling_type == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            pooled = global_max_pool(x, batch)
        elif self.pooling_type == "sum":
            pooled = global_add_pool(x, batch)
        elif self.pooling_type == "attention":
            # Attention-based pooling: learn importance weights for each node
            attention_scores = self.attention_weights(x)  # [num_nodes, 1]
            attention_scores = F.softmax(attention_scores, dim=0)  # Softmax over nodes
            
            # Weighted sum
            pooled = global_add_pool(x * attention_scores, batch)
        else:
            raise ValueError(
                f"Unsupported pooling type: {self.pooling_type}. "
                f"Options: mean, max, sum, attention"
            )
        
        return pooled
    
    def forward(self, x, edge_index, bc_mask, batch=None, return_pooled=False, 
                u_scale=0.04, v_scale=0.08, f_scale=3):
        """
        Forward pass with input/output scaling.
        
        Args:
            x: Node features [num_nodes, in_channels] - typically [u, v, f]
            edge_index: Graph connectivity [2, num_edges]
            bc_mask: Boolean mask for boundary conditions
            batch: Batch vector [num_nodes] for multi-graph batching (optional)
            return_pooled: If True and use_global_pooling=True, returns both node and graph features
            u_scale: Scaling factor for u (Um), if None no scaling is applied
            v_scale: Scaling factor for v (Vm), if None no scaling is applied
            f_scale: Scaling factor for f (fm), if None no scaling is applied
            
        Returns:
            If use_global_pooling=False or return_pooled=False:
                x: Node-level output [num_nodes, out_channels]
                   If residual=False: [u_next, v_next] (absolute values)
                   If residual=True: [Δu, Δv] (changes), which are added to current [u, v]
            
            If use_global_pooling=True and return_pooled=True:
                (x, pooled): Tuple of node-level and graph-level outputs
                
            If pooling_position='middle': Always returns node-level output
        """
        # Apply input scaling: translate from [-scale, scale] to [-1, 1]
        x_scaled = x.clone()
        if u_scale is not None:
            x_scaled[:, 0] = x[:, 0] / (2 * u_scale)
        if v_scale is not None:
            x_scaled[:, 1] = x[:, 1] / (2 * v_scale)
        if f_scale is not None and x.shape[1] > 2:
            x_scaled[:, 2] = x[:, 2] / (2 * f_scale)
        
        # Store input for residual connection (use scaled values)
        if self.residual:
            # Extract current u and v (first 2 channels)
            u_current = x_scaled[:, 0:1]  # Keep dimension for broadcasting
            v_current = x_scaled[:, 1:2]
        
        # Input projection if needed
        if self.input_proj is not None:
            x_scaled = self.input_proj(x_scaled)
        
        # Branch based on architecture
        if self.use_global_pooling and self.pooling_position == "middle":
            # Encoder-decoder with pooling in middle
            
            # ENCODER: Process through encoder layers
            for layer in self.encoder_layers:
                x_scaled = layer(x_scaled, edge_index)
            
            # Save encoder node-level features for skip connection
            encoder_node_feats = x_scaled
            
            # POOLING: Aggregate to graph-level
            graph_features = self.global_pooling(x_scaled, batch)
            
            # Optional projection after pooling
            if self.graph_projection is not None:
                graph_features = self.graph_projection(graph_features)
            
            # DECODER: Broadcast graph features back to nodes
            if batch is None:
                # Single graph: broadcast to all nodes
                num_nodes = x_scaled.shape[0]
                x_scaled = graph_features.expand(num_nodes, -1)
            else:
                # Multiple graphs: broadcast to nodes of each graph
                x_scaled = graph_features[batch]  # Index graph features by batch assignment
            
            # Apply encoder-decoder skip connection, if enabled
            if self.use_ed_skip and self.pooling_position == "middle":
                if self.ed_skip_type == "concat":
                    x_scaled = torch.cat([x_scaled, encoder_node_feats], dim=1)
                elif self.ed_skip_type == "add":
                    if x_scaled.size(1) != encoder_node_feats.size(1):
                        if not hasattr(self, "ed_align"):
                            self.ed_align = Linear(encoder_node_feats.size(1), x_scaled.size(1))
                        encoder_node_feats = self.ed_align(encoder_node_feats)
                    x_scaled = x_scaled + encoder_node_feats
                else:
                    raise ValueError(f"Unsupported ed_skip_type: {self.ed_skip_type}")
            
            # Process through decoder layers
            for layer in self.decoder_layers:
                x_scaled = layer(x_scaled, edge_index)
            
            # Final layer
            if self.final_layer_type == "LINEAR":
                x_scaled = self.final_layer(x_scaled)
            else:
                x_scaled = self.final_layer(x_scaled, edge_index)
                
        else:
            # Standard architecture: pass through all layers
            for layer in self.layers:
                x_scaled = layer(x_scaled, edge_index)
            
            # Final layer
            if self.final_layer_type == "LINEAR":
                x_scaled = self.final_layer(x_scaled)
            else:
                x_scaled = self.final_layer(x_scaled, edge_index)
        
        # If residual mode, add predicted changes to current state
        if self.residual:
            # x_scaled contains [Δu, Δv] in scaled space
            delta_u = x_scaled[:, 0:1]
            delta_v = x_scaled[:, 1:2]
            
            # Add to current state: u_next = u_current + Δu
            u_next = u_current + delta_u
            v_next = v_current + delta_v
            
            # Combine back
            x_scaled = torch.cat([u_next, v_next], dim=1)
        
        # Apply output scaling: translate from [-1, 1] back to [-scale, scale]
        if u_scale is not None:
            x_scaled[:, 0] = x_scaled[:, 0] * (2 * u_scale)
        if v_scale is not None:
            x_scaled[:, 1] = x_scaled[:, 1] * (2 * v_scale)
        
        # Apply boundary conditions
        x_scaled[bc_mask] = 0.0
        
        # Apply global pooling at end if enabled
        if self.use_global_pooling and self.pooling_position == "end":
            pooled = self.global_pooling(x_scaled, batch)
            # Optional projection after pooling
            if self.graph_projection is not None:
                pooled = self.graph_projection(pooled)
            
            if return_pooled:
                return x_scaled, pooled
            else:
                return pooled
        
        return x_scaled
    
def build_laplacian_matrix(N, dx):
    """
    Build sparse Laplacian matrix for 1D chain.
    Edge list for chain: (0,1), (1,2), ..., (N-2,N-1)
    
    Returns:
        L: Sparse Laplacian matrix, shape (N, N)
    """
    # Create sparse matrix in COO format
    row = []
    col = []
    data = []

    for i in range(N):
        # Left neighbor
        if i > 0:
            row.append(i)
            col.append(i-1)
            data.append(1.0 / (dx**2))
        
        # Center (diagonal)
        row.append(i)
        col.append(i)
        data.append(-2.0 / (dx**2))

        # Right neighbor
        if i < N - 1:
            row.append(i)
            col.append(i+1)
            data.append(1.0 / (dx**2))

    # Boundary conditions: set rows for i=0 and i=N-1 to zero
    # (enforced in update function instead)

    L = sp.coo_matrix((data, (row, col)), shape=(N, N))
    
    # Convert to PyTorch sparse tensor for CUDA support
    L_coo = L.tocoo()
    indices = torch.LongTensor(np.vstack([L_coo.row, L_coo.col]))
    values = torch.FloatTensor(L_coo.data)
    L_torch = torch.sparse_coo_tensor(indices, values, torch.Size(L_coo.shape))
    
    return L_torch  # Return PyTorch sparse tensor instead of scipy CSR


def membranedisplacement(coords, t, t_f=1, amp=0.003, x0=0.5, y0=0.5, sign=-1, loc=None, num_terms=4, seed=None):
    """Simple analytic displacement + velocity field for testing.

    Returns (u, v) arrays matching X/Y shapes.
    """
    X = coords
    # Y = np.asarray(coords[:, 1])

    # smooth standing-wave component
    decay = np.exp(-t / max(1e-6, t_f))
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    coeffs = torch.randn(num_terms)  # Random coefficients
    coeffs1 = torch.randn(num_terms)

    u = np.zeros_like(X)
    v = np.zeros_like(X)
    for n in range(1, num_terms + 1):
        for m in range(1, num_terms + 1):
            kx = n * np.pi
            ky = m * np.pi
            omega = np.sqrt(kx**2 + ky**2)
            A_nm = coeffs[n - 1].item() * amp * decay / (n * m)
            B_nm = coeffs1[m - 1].item() * amp * decay / (n * m)

            u = np.add(u, A_nm * np.sin(kx * X) * np.cos(omega * t))
            v = np.add(v, 15 * B_nm * np.sin(kx * X) * np.sin(omega * t))

    # print(u.shape)
    u[X == 0] = 0
    u[X == 1] = 0
    # u[Y == 0] = 0
    # u[Y == 1] = 0

    v[X == 0] = 0
    v[X == 1] = 0
    # v[Y == 0] = 0
    # v[Y == 1] = 0

    return u, v

def membraneforce(coords, t, loc, forcing, x_f_1=None, y_f_1=None, sign=-1, seed=None, margin=0.1):
    """
    2D membrane forcing - Gaussian pulses in space and time.
    
    Parameters
    ----------
    X, Y : np.ndarray
        2D meshgrid of spatial coordinates.
    t : float
        Time value.
    t_f : float
        Final time constant.
    f_min : float
        Minimum force amplitude.
    x_f_1, y_f_1 : float
        First force center position.
    x_f_2, y_f_2 : float
        Second force center position.
    
    Returns
    -------
    f : np.ndarray
        Computed 2D force field values.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if loc == 'casual':
        x_f_1 = float(np.random.uniform(margin, 1.0 - margin))
        y_f_1 = float(np.random.uniform(margin, 1.0 - margin))

    X = coords
    h = sign*3

    # if forcing == 'casual':
    #     forcing_options = ['start', 'middle', 'end']
    #     forcing = np.random.choice(forcing_options)

    # if forcing == 'start':
    #     time = t
    # elif forcing == 'middle':
    #     time = t - 1.0
    # else:  # 'end'
    #     time = 2.0 - t
    
    time = t - 1.0

    z1 = h * np.exp(-400 * ((X - x_f_1)**2)) * \
            np.exp(-(time**2) / (2 * 0.5**2))

    return z1

def create_graph(seed=None, zeros=False, cfg=None):
    # Generate 100 nodes evenly spaced from 0 to 1 (1D chain)
    nodes = np.linspace(0, 1, 100).reshape(-1, 1)
    num_nodes = len(nodes)
    dx = 1.0 / (num_nodes - 1)

    # Elements: connect each node to the next (1D chain)
    elements = np.array([[i, i + 1] for i in range(num_nodes - 1)], dtype=int)

    # Boundary nodes: first and last node
    boundary_nodes = np.array([0, num_nodes - 1], dtype=int)

    # Create boundary condition mask (True at boundary nodes, False at internal nodes)
    bc_mask = np.zeros(num_nodes, dtype=bool)
    bc_mask[boundary_nodes] = True
    bc_mask_torch = torch.tensor(bc_mask, dtype=torch.bool)

    # Create adjacency matrix (sparse)
    adj = lil_matrix((num_nodes, num_nodes))
    for i, j in elements:
        adj[i, j] = 1
        adj[j, i] = 1  # undirected

    L = build_laplacian_matrix(num_nodes, dx)

    # edge_index: 2 x num_edges, undirected
    edge_index = np.vstack(np.nonzero(adj))
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)

    coords = torch.tensor(nodes, dtype=torch.float)

    # Get parameters from cfg if available, otherwise use defaults
    if cfg is not None:
        margin = cfg.dataset.force.margin
        sign = cfg.dataset.force.sign
        forcing = cfg.dataset.force.forcing_type
    else:
        margin = 0.1
        sign = -1
        forcing = 'casual'
    
    
    t = float(np.random.rand()*2)
    u_np, v_np = membranedisplacement(coords, t, loc='casual', sign=sign, seed=seed)
    f_np = membraneforce(coords, t, loc='casual', forcing=forcing, sign=sign, seed=seed, margin=margin)

    x = np.stack([u_np, v_np, f_np], axis=1).astype(np.float32)
    x = torch.from_numpy(x)
    if zeros:
        x = torch.zeros_like(x)

    x = x.squeeze(-1)  # Removes the last dimension if it's 1
    data = Data(x=x, edge_index=edge_index_torch, bc_mask=bc_mask_torch, coords=coords, laplacian=L, nodes=nodes, elements=elements)
    return data

class WaveGNN1D:
    """
    Graph Neural Network for 1D wave equation on a chain.
    Integrates spatial derivatives (via message passing) and time integration
    (via node update function) in a single forward pass.
    """
    def __init__(self, L, c, k, dt):
        """
        Args:
            L: Laplacian matrix (sparse)
            c: Wave speed
            k: Damping coefficient
            dt: Time step
        """
        self.c = c
        self.k = k
        self.dt = dt
        self.L = L
        self.N = L.shape[0]  # Number of nodes
    
    def compute_laplacian(self, u):
        """
        Matrix-based Laplacian computation using sparse matrix multiplication.
        
        Args:
            u: Node displacements, shape (N,)
            
        Returns:
            laplacian: Discrete Laplacian, shape (N,)
        """
        # Check if L is PyTorch sparse tensor or scipy sparse matrix
        if torch.is_tensor(self.L):
            # PyTorch sparse matrix multiplication
            if not torch.is_tensor(u):
                u = torch.from_numpy(u).float()
            # Ensure u is on same device as L
            u = u.to(self.L.device)
            laplacian = torch.sparse.mm(self.L, u.unsqueeze(1)).squeeze(1)
            laplacian = laplacian.cpu().numpy()
        else:
            # Original scipy sparse matrix multiplication
            laplacian = self.L @ u
        
        # Enforce boundary conditions (Laplacian is zero at boundaries)
        laplacian[0] = 0.0
        laplacian[-1] = 0.0
        
        return laplacian
    
    def update_rk4(self, node_features):
        """
        Node update function using RK4 time integration.
        
        Args:
            node_features: Current [u, v, f] for each node, shape (N, 3)
            
        Returns:
            new_features: Updated [u, v, f] for each node, shape (N, 3)
        """
        # Extract current state
        u = node_features[:, 0]
        v = node_features[:, 1]
        force = node_features[:, 2]
        
        # Define the derivative function
        # du/dt = v
        # dv/dt = c²·Laplacian(u) - k·v + f
        
        # k1: derivatives at t
        du_dt_1 = v
        laplacian_1 = self.compute_laplacian(u)
        dv_dt_1 = self.c**2 * laplacian_1 - self.k * v + force
        
        # k2: derivatives at t + dt/2 using k1
        u_2 = u + 0.5 * self.dt * du_dt_1
        v_2 = v + 0.5 * self.dt * dv_dt_1
        du_dt_2 = v_2
        laplacian_2 = self.compute_laplacian(u_2)
        dv_dt_2 = self.c**2 * laplacian_2 - self.k * v_2 + force
        
        # k3: derivatives at t + dt/2 using k2
        u_3 = u + 0.5 * self.dt * du_dt_2
        v_3 = v + 0.5 * self.dt * dv_dt_2
        du_dt_3 = v_3
        laplacian_3 = self.compute_laplacian(u_3)
        dv_dt_3 = self.c**2 * laplacian_3 - self.k * v_3 + force
        
        # k4: derivatives at t + dt using k3
        u_4 = u + self.dt * du_dt_3
        v_4 = v + self.dt * dv_dt_3
        du_dt_4 = v_4
        laplacian_4 = self.compute_laplacian(u_4)
        dv_dt_4 = self.c**2 * laplacian_4 - self.k * v_4 + force
        
        # Weighted average (RK4 formula)
        u_new = u + (self.dt / 6.0) * (du_dt_1 + 2*du_dt_2 + 2*du_dt_3 + du_dt_4)
        v_new = v + (self.dt / 6.0) * (dv_dt_1 + 2*dv_dt_2 + 2*dv_dt_3 + dv_dt_4)

        # Enforce boundary conditions
        u_new[0] = 0.0      # Left boundary
        u_new[-1] = 0.0     # Right boundary
        v_new[0] = 0.0      
        v_new[-1] = 0.0     
        
        # Force remains constant (will be updated externally)
        force_new = force
        
        # Stack back into features
        new_features = np.stack([u_new, v_new, force_new], axis=1)
        return new_features

    
    def forward(self, node_features):
        """
        Complete GNN forward pass: one time step from t to t+dt.
        
        Args:
            node_features: [u, v, f] for each node, shape (N, 3)
                          Can be either numpy array or PyTorch tensor
            
        Returns:
            new_features: [u_new, v_new, f_new] for each node, shape (N, 3)
                         Returns same type as input
        """
        # Check if input is a PyTorch tensor
        is_tensor = hasattr(node_features, 'detach')
        
        if is_tensor:
            # Convert to numpy for processing
            device = node_features.device
            node_features_np = node_features.detach().cpu().numpy()
        else:
            node_features_np = node_features
        
        # Update: Time integration
        new_features = self.update_rk4(node_features_np)
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            new_features = torch.tensor(new_features, dtype=torch.float32, device=device)
        
        return new_features

def create_dataset(num_graphs=64, cfg=None):
    """Create a list of Data objects (dataset) with Dirichlet BCs for graph-level batching."""
    dataset = []
    for i in range(num_graphs):
        # vary seed so graphs are different
        data = create_graph(seed=1000 + i, cfg=cfg)

        dataset.append(data)
    return dataset

def rollout_graph(seed, num_steps=200, cfg=None):
    graph_0 = create_graph(seed, cfg=cfg)
    gn = WaveGNN1D(graph_0.laplacian, cfg.dataset.c, cfg.dataset.k, cfg.dataset.dt)
    features = graph_0.x.clone()

    graphs = []
    # append initial graph
    graphs.append(Data(x=features.clone(),
                       edge_index=graph_0.edge_index,
                       bc_mask=graph_0.bc_mask,
                       coords=graph_0.coords,
                       laplacian=graph_0.laplacian,
                       nodes=graph_0.nodes,
                       elements=graph_0.elements))

    t = 0.0
    for i in range(1, num_steps):
        # forward step
        features = gn.forward(features)

        # advance time
        t += dt
        frc = cfg.dataset.force

        # compute forcing (use numpy coords for membraneforce)
        coords_np = graph_0.coords.cpu().numpy()
        f_np = membraneforce(coords_np, t, loc=frc.location, forcing=frc.forcing_type, sign=frc.sign, margin=frc.margin, seed=seed)
        f_np = np.asarray(f_np).squeeze()  # ensure shape (N,)

        # build new features tensor: [u, v, f]
        u = features[:, 0].detach()
        v = features[:, 1].detach()
        f_t = torch.from_numpy(f_np).float()
        if f_t.dim() == 0:
            f_t = f_t.unsqueeze(0)
        # ensure same device as features
        f_t = f_t.to(features.device)

        features = torch.stack([u, v, f_t], dim=1)

        # create new Data object for this timestep (keep other graph attributes same)
        graph_t = Data(x=features.clone(),
                       edge_index=graph_0.edge_index,
                       bc_mask=graph_0.bc_mask,
                       coords=graph_0.coords,
                       laplacian=graph_0.laplacian,
                       nodes=graph_0.nodes,
                       elements=graph_0.elements)
        graphs.append(graph_t)

    return graphs

if __name__ == "__main__":
    plt.close()
    data = create_graph(2)
    G = nx.Graph()
    G.add_nodes_from(range(data.coords.shape[0]))
    edges = data.edge_index.t().numpy()
    G.add_edges_from(edges)
    node_colors = ['red' if data.bc_mask[i] else 'blue' for i in range(data.coords.shape[0])]
    # For 1D chain, use x-coordinate only for layout
    pos = {i: (float(data.coords[i]), 0) for i in range(data.coords.shape[0])}
    plt.figure(figsize=(8, 2))
    nx.draw(G, pos, node_size=40, node_color=node_colors, edge_color='gray')
    plt.title("1D Chain Mesh Graph")
    plt.tight_layout()
    plt.savefig("figures/mesh_graph_1d.png")
    plt.show()

