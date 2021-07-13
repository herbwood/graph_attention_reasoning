import torch
import torch.nn as nn

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            # If we are concatenating the multiple heads
            self.n_hidden = out_features // n_heads
        else:
            # If we are averaging the multiple heads
            self.n_hidden = out_features

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):

        # Number of nodes
        n_nodes = h.shape[0]
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_repeat = g.repeat(n_nodes, 1, 1)
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)

        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        # Concatenate the heads
        if self.is_concat:
            # $$\overrightarrow{h'_i} = \Bigg\Vert_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        # Take the mean of the heads
        else:
            # $$\overrightarrow{h'_i} = \frac{1}{K} \sum_{k=1}^{K} \overrightarrow{h'^k_i}$$
            return attn_res.mean(dim=1)

class GAT(nn.Module):

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):

        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):

        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer (without activation) for logits
        return self.output(x, adj_mat)


class GCN(nn.Module):

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=5) # adjacent matrix 
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # conv1d by columns and rows 
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.conv2(self.relu(h))
        adj_matrix = self.conv1.weight
        return h, adj_matrix 


class GloRe_Unit(nn.Module):
    
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid) # 128 -> feature_dim
        self.num_n = int(1 * num_mid) # 64 -> num_nodes

        # dim reduction layer
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        
        # projection matrix layer 
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)

        # Graph Convolution Network 
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)

        # Graph Attention Network
        self.gat = GAT(self.in_features, self.n_hidden, self.n_classes, self.n_heads, self.dropout)
        
        # recover channel size 
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        # batch norm 
        self.blocker = BatchNormNd(num_in, eps=1e-04) 


    def forward(self, x):
        
        n = x.size(0)

        # hidden state 
        # feature map with reduced dim 
        # output shape : (batch size, feature_dim, channel)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # projection matrix 
        # output shape : (batch size, num_nodes, channel)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # reverse projection matrix 
        # output shape : (batch size, num_nodes, channel)
        x_rproj_reshaped = x_proj_reshaped

        # matrix multiplication 
        # hidden state x projection matrix 
        # output shape : (batch size, feature_dim, num_nodes)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # graph convolutional network
        # 1) conv1d : by columns
        # 2) element-wise addition : conv1d(x) + x
        # 3) conv1d + relu : by rows 
        # output shape : (batch size, feature_dim, num_nodes)
        x_n_rel, adj_matrix = self.gcn(x_n_state)

        #######################Graph Attention##############################

        ####################################################################
        
        # matrix multiplication 
        # graph output x reverse projection 
        # output shape : (batch size, feature_dim, num_nodes)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # resize to original feature map size
        # output shape : (batch size, feature_dim, height, width)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # final output 
        # 1) 1x1 conv2d : back to original channel
        # 2) batch norm 
        # 3) element-wise addition : with original feature map 
        out = x + self.blocker(self.conv_extend(x_state))

        return out


class GloRe_Unit_1D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_1D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv1d,
                                            BatchNormNd=nn.BatchNorm1d,
                                            normalize=normalize)

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        
        super(GloRe_Unit_3D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)

