import torch
import torch.nn as nn

class GATLayerWithIRCRWR(nn.Module):
    # These may change in the inductive setting - leaving it like this for now (not future proof)
    nodes_dim = 0      # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1       # attention head dim

    def __init__(self, num_in_features, num_out_features, num_of_heads, gamma=0.7, beta=0.5, concat=True, activation=nn.ELU(),
                 dropout_prob=0.6, random_walk_with_restart=True, add_residual_connection=True, add_skip_connection=True, bias=True):
        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.residual_connection = add_residual_connection
        self.random_walk_with_restart = random_walk_with_restart
        self.add_skip_connection = add_skip_connection
        self.gamma = gamma # RWR의 재시작 확률
        self.beta = beta # residual connection의 가중치

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the "additive" scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.
        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        # 즉 논문에서의 e_ij = LeakyReLU(a_T[ Wh_i ∣∣ Wh_j ])를 연산량을 줄이기 위해
        # e_ij​ = LeakyReLU((Wh_i​)⋅a_left​ + (Wh_j)⋅a_right​)로 바꿔 수행한다.
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if self.random_walk_with_restart:
            self.W_random_walk = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('W_random_walk', None)

        if self.residual_connection:
            self.W_residual = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('W_residual', None)

        if add_skip_connection and not (add_residual_connection or random_walk_with_restart):
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.reset_parameter()
        
    def forward(self, data):
        #
        # Linear Projection
        #

        in_nodes_features, edge_index, initial_features_for_random_walk, initial_features_for_residual_connection = data  # unpack data

        # Save the initial features for random walk
        if initial_features_for_random_walk is None and self.random_walk_with_restart:
            initial_features_for_random_walk = self.W_random_walk(in_nodes_features).reshape(-1, self.num_of_heads, self.num_out_features)
        
        # Save the initial features for residual connection
        if initial_features_for_residual_connection is None and self.residual_connection:
            initial_features_for_residual_connection = self.W_residual(in_nodes_features).reshape(-1, self.num_of_heads, self.num_out_features)

        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (노드 수, 입력 특징 수) : 각 노드의 입력 특징을 나타낸다.
        # 논문에서와 같이 모든 입력 노드 특징에 dropout을 적용한다.
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features) # 공식 GAT 구현에서도 dropout을 사용.

        # shape = (노드 수, 입력 특징 수) * (입력 특징 수, 헤드 수 * 출력 특징 수) -> (노드 수, 헤드 수, 출력 특징 수)
        # We project the input node features into NH independent output features (one for each attention head)
        # 즉 각 노드의 특징을 각 헤드마다 다른 특징으로 변환한다.
        nodes_features_proj = self.linear_proj(in_nodes_features).reshape(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj) # 공식 GAT 구현에서도 dropout을 사용.

        #
        # Edge attention calculation
        #

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (노드 수, 헤드 수, 출력 특징 수) * (1, 헤드 수, 출력 특징 수) -> (노드 수, 헤드 수, 1) -> (노드 수, 헤드 수)
        # sum은 마지막 차원을 기준으로 sum하므로 (N, NH, FOUT) -> (N, NH) 즉, 노드 수 x 헤드 수
        # 여기서 학습 가능한 파라미터인 scoring_fn_source, scoring_fn_target을 사용하여 각 노드의 특징을 계산한다.
        # [GAT에선 a 벡터를 사용하여 두 노드의 특징을 결합한 후, scoring function을 통해 attention score를 계산한다.]
        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1) # 마지막 차원을 기준으로 sum 즉, (N, NH, FOUT) -> (N, NH) 즉, 노드 수 x 헤드 수
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1) # (N, NH) 즉, 노드 수 x 헤드 수
        
        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        # 각 엣지에 대한 source, target의 attention score를 계산한다.
        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target, nodes_features_proj, edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        # shape = (엣지 수, 헤드 수, 1) -> (엣지 수, 헤드 수, 1) (unsqueeze를 통해 차원을 추가한다. 그래야 element-wise 곱을 할 수 있다.)
        # 이제 softmax를 통해 attention coefficient를 계산한다.
        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[1], num_of_nodes)
        # Add stochasticity to neighborhood aggregation
        attentions_per_edge = self.dropout(attentions_per_edge)

        #
        # Neighborhood aggregation
        #

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (엣지 수, 헤드 수, 출력 특징 수) * (엣지 수, 헤드 수, 1) -> (엣지 수, 헤드 수, 출력 특징 수) 1이 FOUT으로 브로드캐스팅된다.
        # FOUT은 출력 특징 수이다. 즉, 각 엣지의 attention score를 이용하여 각 노드의 특징을 가중합하여 계산한다.
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        # 이 부분은 각 노드의 이웃 노드의 특징을 가중합하여 계산한다.
        # shape = (노드 수, 헤드 수, 출력 특징 수)
        # Aggregate multi-head attentions and Random Walk with Restart (optional)
        out_nodes_features = self.aggregate_neighbors(initial_features_for_random_walk, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes)

        #
        # Skip connections (optional) and Initial Residual Connection (optional), concat and bias
        #

        out_nodes_features = self.skip_concat_bias(in_nodes_features, out_nodes_features, initial_features_for_residual_connection)

        return (out_nodes_features, edge_index, initial_features_for_random_walk, initial_features_for_residual_connection)
    
    def reset_parameter(self):
        """
        원래 논문에서 GAT을 구현한 코드가 TensorFlow로 되어있고, 그 코드에서는 기본 초기화 방법으로 사용했기 때문에
        Glorot (Xavier uniform) initialization을 사용한다.
        
        Tensorflow의 기본 초기화 방법은 Glorot (Xavier uniform) initialization이다.
        https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.random_walk_with_restart:
            nn.init.xavier_uniform_(self.W_random_walk.weight)
        if self.residual_connection:
            nn.init.xavier_uniform_(self.W_residual.weight)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        즉 lifts는 edge index에 따라 특정 벡터를 엣지 수만큼 복제한다.
        텐서의 차원 중 하나가 N -> E로 변한다.
        여기서 N은 노드 수, E는 엣지 수이다.

        """
        # src_nodes_index :  tensor([   0,    0,    0,  ..., 2707, 2707, 2707])
        # trg_nodes_index :  tensor([ 633, 1862, 2582,  ...,  598, 1473, 2706])
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]

        # scores_source shape before :  torch.Size([2708, 8])
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        # scores_source shape after :  torch.Size([10556, 8])
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        # nodes_features_proj shape torch.Size([2708, 8, 8])
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        # nodes_features_matrix_proj_lifted shape :  torch.Size([10556, 8, 8])

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        이웃 노드들의 attention score를 softmax를 통해 계산한다.
        """

        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16은 이론적으로 필요하지 않지만 수치적 안정성을 위해 (0으로 나누는 것을 피하기 위해) 추가했다.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        # shape = (E, NH) -> (E, NH, 1)로 만들어서 projected node features와 element-wise 곱을 할 수 있게 한다.

        return attentions_per_edge.unsqueeze(-1)
    
    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # attention head 개수만큼 브로드캐스팅한다. 여기서 브로드캐스팅이란 차원을 늘려서 연산을 수행하는 것을 의미한다.
        # E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH)
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # exp_scores_per_edge를 trg_index_broadcasted의 값을 index로 사용하여 neighborhood_sums에 더한다.
        # 그렇게 되면 각 노드의 이웃 노드들의 attention score의 합을 계산할 수 있다.
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # edge_index의 차원에 맞게 브로드캐스팅한다.
        # 모든 location의 값이 i번째 노드의 attention score의 합으로 브로드캐스팅되는 것이다.
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, initial_features, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features = torch.zeros(num_of_nodes, *nodes_features_proj_lifted_weighted.shape[1:], dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        # Random walk with Restart
        if self.random_walk_with_restart:
            out_nodes_features = (1 - self.gamma) * out_nodes_features + self.gamma * initial_features

        return out_nodes_features

    def explicit_broadcast(self, this, other):
        # 차원이 같아질 때까지 singleton 차원을 추가한다.
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1) # 가장 마지막 차원을 추가한다.

        # other 텐서와 같은 모양으로 확장하는데, 이 때 실제로 데이터를 복사하지는 않고, 필요에 따라 가상적으로 차원을 확장한다.
        return this.expand_as(other)
    
    def skip_concat_bias(self, in_nodes_features, out_nodes_features, initial_features):
        if self.add_skip_connection and not (self.residual_connection or self.random_walk_with_restart):  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.residual_connection:  # add residual connection
            if out_nodes_features.shape[-1] == initial_features.shape[-1]:  # if FIN == FOUT
                out_nodes_features += initial_features
        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.reshape(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)