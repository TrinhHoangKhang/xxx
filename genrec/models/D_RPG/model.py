import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class DPQ(nn.Module):
    '''
    Differentiable Product Quantization.
    '''
    def __init__(self, d: int, n_digits: int, codebook_size: int, v_dim: int, tokenizer):
        super().__init__()
        assert d % n_digits == 0, f"Sentence embedding dim ({d}) must be divisible by n_digits ({n_digits})"
        print(f'[MODEL] Initializing DPQ module...')
        self.d = d
        self.n_digits = n_digits
        self.codebook_size = codebook_size
        self.sub_dim = d // n_digits
        self.v_dim = v_dim

        # Learnable linear projection; warm-initialized from the FAISS OPQ transform.
        # No orthogonality constraint (unconstrained nn.Linear): y = x @ weight^T.
        self.rotation = nn.Linear(d, d, bias=False)
        if tokenizer.opq_rotation is not None:
            print(f'[MODEL] Warm-initializing rotation from FAISS OPQ transform')
            with torch.no_grad():
                self.rotation.weight.copy_(torch.from_numpy(tokenizer.opq_rotation))
        else:
            print(f'[MODEL] opq_rotation unavailable — rotation uses default init')

        # Key codebooks K: (n_digits, codebook_size, sub_dim)
        print('[MODEL] Creating K matrix')
        if tokenizer.pq_codebooks is not None:
            print(f'[MODEL] Using pre-trained PQ codebooks...')
            K_init = torch.from_numpy(tokenizer.pq_codebooks).float()
        else:
            print(f'[MODEL] Initializing random PQ codebooks...')
            K_init = torch.randn(n_digits, codebook_size, self.sub_dim) * 0.02
        self.K = nn.Parameter(K_init)

        # Value codebooks V: (n_digits, codebook_size, v_dim)
        print('[MODEL] Creating V matrix')
        if v_dim == self.sub_dim and tokenizer.pq_codebooks is not None:
            print(f'[MODEL] Using pre-trained PQ value codebooks...')
            V_init = torch.from_numpy(tokenizer.pq_codebooks).float()
        else:
            print(f'[MODEL] Initializing random PQ value codebooks...')
            V_init = torch.randn(n_digits, codebook_size, v_dim) * 0.02
        self.V = nn.Parameter(V_init)

    def forward(self, x: torch.Tensor, tau: float = 1.0, sigma: float = 1.0) -> dict:
        '''
        Args:
          x     : (B, seq_len, d)  sentence embeddings
          tau   : Gumbel-Softmax temperature (lower = harder assignments)
          sigma : Gumbel noise scale
        Returns:
          'ste'   : (B, seq_len, n_digits * v_dim)  STE output for downstream layers
          'soft'  : (B, seq_len, n_digits * v_dim)  soft reconstruction
          'hard'  : (B, seq_len, n_digits * v_dim)  hard reconstruction
          'hard_codes' : (B, seq_len, n_digits)           discrete code indices
        '''
        B, seq_len, _ = x.shape

        # 1) Rotate to PQ-aligned space: (B, seq_len, d) -> (B, seq_len, d)
        x_rot = self.rotation(x)

        # 2) Split into n_digits sub-vectors: (B, seq_len, n_digits, sub_dim)
        x_sub = x_rot.view(B, seq_len, self.n_digits, self.sub_dim)

        # 3) Assignment logits via dot product with key codebooks:
        #    (B, seq_len, n_digits, sub_dim) x (n_digits, codebook_size, sub_dim) -> (B, seq_len, n_digits, codebook_size)
        logits = torch.einsum('bsdi,dki->bsdk', x_sub, self.K)

        # 4) Soft probabilities and hard assignments (Gumbel noise only during training).
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10)) + 1e-10)
            noisy_logits = logits + sigma * gumbel
            soft_probs = F.softmax(noisy_logits / tau, dim=-1)
            hard_codes = noisy_logits.argmax(dim=-1)
        else:
            soft_probs = F.softmax(logits / tau, dim=-1)
            hard_codes = logits.argmax(dim=-1)

        # 5) Hard reconstruction: gather selected codewords from V.
        offsets = torch.arange(self.n_digits, device=hard_codes.device) * self.codebook_size
        flat_codes = hard_codes + offsets
        flat_V = self.V.view(self.n_digits * self.codebook_size, self.v_dim)
        hard = F.embedding(flat_codes, flat_V)                              # (B, seq_len, n_digits, v_dim)

        # 6) Soft reconstruction: weighted average over codewords.
        soft = torch.einsum('bsdk,dkv->bsdv', soft_probs, self.V)          # (B, seq_len, n_digits, v_dim)

        # 7) Straight-Through Estimator: forward=hard, backward through soft.
        ste = hard + soft - soft.detach()                                   # (B, seq_len, n_digits, v_dim)

        # Flatten (n_digits, v_dim) → (n_digits * v_dim).
        return {
            'ste':   ste.reshape(B, seq_len, self.n_digits * self.v_dim),
            'soft':  soft.reshape(B, seq_len, self.n_digits * self.v_dim),
            'hard':  hard.reshape(B, seq_len, self.n_digits * self.v_dim),
            'hard_codes': hard_codes,
        }


class ResBlock(nn.Module):
    '''
    Residual block: x + SiLU(Linear(x)).  Initialized as identity.
    '''
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class D_RPG(AbstractModel):
    '''
    GPT-2 Sequential Recommendation with end-to-end Differentiable PQ.
    '''

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)
        print(f'[MODEL] Initializing D_RPG model...')

        # Sentence embedding table: item_id → d-dim embedding (row 0 = padding).
        # Set freeze=False to allow fine-tuning during training.
        sent_embs_tensor = torch.from_numpy(tokenizer.sent_embs)    # (n_items, d)
        self.sent_emb_table = nn.Embedding.from_pretrained(
            sent_embs_tensor, freeze=config['freeze_sentence_embedding'], padding_idx=0
        )
        self.sent_emb_dim: int = sent_embs_tensor.shape[1]          # d

        # Differentiable PQ module.
        v_dim = config.get('dpq_v_dim', config['n_embd'] // config['n_codebook'])
        self.dpq = DPQ(
            d=self.sent_emb_dim,
            n_digits=config['n_codebook'],
            codebook_size=config['codebook_size'],
            v_dim=v_dim,
            tokenizer=tokenizer,
        )
        dpq_out_dim = config['n_codebook'] * v_dim  # equals n_embd when using default v_dim

        # Optional projection layers if DPQ output dim ≠ GPT-2 hidden dim.
        self.input_proj: nn.Module = (
            nn.Linear(dpq_out_dim, config['n_embd'])
            if dpq_out_dim != config['n_embd'] else nn.Identity()
        )
        self.output_proj: nn.Module = (
            nn.Linear(config['n_embd'], dpq_out_dim)
            if dpq_out_dim != config['n_embd'] else nn.Identity()
        )

        # GPT-2 backbone.
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        # Prediction heads: one ResBlock per digit, identical to RPG.
        self.n_digits: int = tokenizer.n_digit
        self.pred_heads = nn.Sequential(
            *[ResBlock(config['n_embd']) for _ in range(self.n_digits)]
        )

        # item_id → n_digits-token lookup (for loss & generate, same as RPG).
        self.item_id2tokens = self._map_item_tokens().to(config['device'])

        # Loss and generation parameters.
        self.temperature = config['temperature']
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        # Gumbel temperature — annealed each epoch via anneal_tau().
        self.gumbel_tau: float = config.get('quantizer_temperature', 1.0)
        self.gumbel_tau_min: float = config.get('min_quantizer_temperature', 0.1)
        self.gumbel_tau_decay: float = config.get('quantizer_temperature_decay', 0.9)

        self.sdud_lambda = config.get('sdud_lambda', 1.4)
        self.use_gumbel_noise: bool = config.get('use_gumbel_noise', True)
        self.sigma = 1.0 if self.use_gumbel_noise else 0.0
        self.running_loss = None

    def anneal_tau(self):
        # Exponential decay with floor: tau ← max(tau_min, tau * tau_decay).
        self.gumbel_tau = max(self.gumbel_tau_min, self.gumbel_tau * self.gumbel_tau_decay)

    def on_train_epoch_end(self, epoch: int) -> dict[str, float]:
        self.anneal_tau()
        return {
            "Quantizer/gumbel_tau": float(self.gumbel_tau),
            "Quantizer/sigma": float(self.sigma),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_item_tokens(self) -> torch.Tensor:
        '''
        item_id → n_digits-token semantic code (same as RPG).
        '''
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long
        )
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    def _get_all_item_embs(self) -> torch.Tensor:
        '''
        Returns normalized DPQ hard embeddings for all real items.
        Shape: (n_items - 1, dpq_out_dim), item ids 1..n_items-1 map to rows 0..n_items-2.
        '''
        all_sent = self.sent_emb_table.weight[1:].unsqueeze(0)          # (1, n_items-1, d)
        item_embs = self.dpq(all_sent, tau=self.gumbel_tau)['hard']     # (1, n_items-1, dpq_out_dim)
        return F.normalize(item_embs.squeeze(0), dim=-1)                # (n_items-1, dpq_out_dim)

    def build_ii_sim_mat(self) -> torch.Tensor:
        '''
        Build item-item cosine similarity matrix in DPQ space, mapped to [0, 1].
        '''
        n_items = self.dataset.n_items
        item_embs = self._get_all_item_embs()                           # (n_items-1, dpq_out_dim)
        item_item_sim = torch.zeros(
            (n_items, n_items), device=item_embs.device, dtype=torch.float32
        )
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            emb_i = item_embs[i_start - 1:i_end - 1]
            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                emb_j = item_embs[j_start - 1:j_end - 1]
                item_item_sim[i_start:i_end, j_start:j_end] = 0.5 * (emb_i @ emb_j.T + 1.0)
        return item_item_sim

    def build_adjacency_list(self, item_item_sim: torch.Tensor) -> torch.Tensor:
        '''
        Top-k nearest neighbors per item.
        '''
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        '''
        Build k-NN graph for graph-constrained decoding.
        '''
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, item_logits: torch.Tensor, n_return_sequences: int):
        '''
        Graph-based decoding: iterative neighbor expansion + local re-ranking.
        Args:
          item_logits : (B, n_items - 1), scores for item ids 1..n_items-1
        '''
        B = item_logits.shape[0]
        visited_nodes = {b: set() for b in range(B)}

        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (B, self.num_beams),
            dtype=torch.long,
            device=item_logits.device,
        )
        for b in range(B):
            for node in topk_nodes_sorted[b].detach().cpu().tolist():
                visited_nodes[b].add(node)

        for _ in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(B, -1)
            next_nodes = []
            for b in range(B):
                neighbors = torch.unique(all_neighbors[b])
                for node in neighbors.detach().cpu().tolist():
                    visited_nodes[b].add(node)
                scores = item_logits[b].index_select(0, neighbors - 1)
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors[idxs])
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        visited_counts = torch.FloatTensor([[len(visited_nodes[b])] for b in range(B)])
        return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1), visited_counts

    @property
    def n_parameters(self) -> str:
        total  = sum(p.numel() for p in self.parameters()      if p.requires_grad)
        dpq_p  = sum(p.numel() for p in self.dpq.parameters()  if p.requires_grad)
        gpt2_p = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        return (
            f'#DPQ parameters:   {dpq_p}\n'
            f'#GPT-2 parameters: {gpt2_p}\n'
            f'#Total trainable:  {total}\n'
        )

    def forward(self, batch: dict, return_loss: bool = True):
        '''
        B        : batch size
        seq_len  : sequence length
        d        : sentence embedding dimension
        n_embd   : GPT-2 hidden dimension
        n_digits : number of prediction heads (= tokenizer.n_digit)
        M        : supervised positions in the batch (label != -100)
        n_items  : number of real items (dataset.n_items - 1, excluding padding id 0)
        '''
        # 1) Sentence embedding lookup: (B, seq_len) → (B, seq_len, d)
        sent_embs = self.sent_emb_table(batch['input_ids'])

        # 2) Differentiable quantization → GPT-2 input: (B, seq_len, n_embd)
        dpq_out = self.dpq(sent_embs, tau=self.gumbel_tau, sigma=self.sigma)
        input_embs = self.input_proj(dpq_out['ste'])

        # 3) GPT-2 contextual encoding: (B, seq_len, n_embd)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask'],
        )

        # 4) Apply n_digits residual prediction heads:
        #    (B, seq_len, n_embd) → stack → (B, seq_len, n_digits, n_embd)
        final_states = torch.cat(
            [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
             for i in range(self.n_digits)],
            dim=-2,
        )
        outputs.final_states = final_states

        if return_loss:
            assert 'labels' in batch, 'Batch must contain labels.'

            # Select supervised positions: flatten to (B*seq_len,), keep M valid.
            label_mask = batch['labels'].view(-1) != -100

            # (B*seq_len, n_digits, n_embd)[label_mask] → (M, n_digits, n_embd) → mean → (M, n_embd)
            selected = final_states.view(
                -1, self.n_digits, self.config['n_embd']
            )[label_mask].mean(dim=1)

            # Project to DPQ retrieval space and L2-normalize: (M, dpq_out_dim)
            query = F.normalize(self.output_proj(selected), dim=-1)

            # Item candidate embeddings: (1, n_items-1, d) → DPQ → (n_items-1, dpq_out_dim)
            all_sent = self.sent_emb_table.weight[1:].unsqueeze(0)
            item_embs = self.dpq(all_sent, tau=self.gumbel_tau, sigma=self.sigma)['ste']
            item_embs = F.normalize(item_embs.squeeze(0), dim=-1)

            # Dense retrieval scores: (M, n_items-1)
            item_logits = query @ item_embs.T / self.temperature

            # Ground-truth item ids shifted to 0-based: (M,)
            gt_ids = batch['labels'].view(-1)[label_mask] - 1

            l_gen = nn.CrossEntropyLoss()(item_logits, gt_ids)
            if self.training and self.use_gumbel_noise:
                current_loss = l_gen.item()
                if self.running_loss is None:
                    self.running_loss = current_loss
                else:
                    # EMA of loss (99% history, 1% current batch).
                    self.running_loss = 0.99 * self.running_loss + 0.01 * current_loss
                # SDUD sigma update: max(0, sqrt(EMA_loss) - lambda)
                self.sigma = max(0.0, self.running_loss ** 0.5 - self.sdud_lambda)

            outputs.loss = l_gen

        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1, return_loss: bool = False):
        '''
        Predict next items.
        Returns:
          preds : (B, n_return_sequences, 1) item IDs
          loss  : scalar tensor (only when return_loss=True)
        '''

        outputs = self.forward(batch, return_loss=return_loss)

        # Gather hidden state at the last valid timestep of each sequence.
        # (B, seq_len, n_digits, n_embd) → (B, 1, n_digits, n_embd)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_digits, self.config['n_embd']
            ),
        )

        # Mean over n_digits heads, project, and normalize: (B, dpq_out_dim)
        query = F.normalize(self.output_proj(states[:, 0].mean(dim=1)), dim=-1)

        # Item embeddings in DPQ space: (n_items-1, dpq_out_dim)
        item_embs = self._get_all_item_embs()

        # Similarity scores: (B, n_items-1)
        item_logits = query @ item_embs.T / self.temperature

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            preds = self.graph_propagation(item_logits=item_logits, n_return_sequences=n_return_sequences)
        else:
            # Top-k; shift 0-based indices back to 1-based item ids.
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1  # (B, n_return_sequences)
            preds = preds.unsqueeze(-1)                                         # (B, n_return_sequences, 1)

        if return_loss:
            return preds, outputs.loss
        return preds

    def prepare_before_test(self):
        self.generate_w_decoding_graph = self.config.get('use_graph_decoding_at_test', True)
