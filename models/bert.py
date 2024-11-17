import torch.nn as nn
import torch

class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        super(BERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids))
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        return self.dense(context_layer.view(*new_shape))

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(hidden_size, num_attention_heads)
        self.attention_output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Sequential(
            nn.LayerNorm(intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(0.1)
        )

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        hidden_states = self.attention_output(attention_output + hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        
        # Project intermediate_output to match hidden_states' size
        hidden_states = self.output(intermediate_output) + hidden_states  # Ensure same dimensions for addition
        return hidden_states


class BERTModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads, num_encoder_layers, intermediate_size, max_position_embeddings):
        super(BERTModel, self).__init__()
        self.embeddings = BERTEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_encoder_layers)
        ])

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        hidden_states = self.embeddings(input_ids, token_type_ids, position_ids)
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class BERTForQA(nn.Module):
    def __init__(self, bert_model):
        super(BERTForQA, self).__init__()
        self.bert = bert_model
        self.start_logits = nn.Linear(bert_model.embeddings.word_embeddings.embedding_dim, 1)
        self.end_logits = nn.Linear(bert_model.embeddings.word_embeddings.embedding_dim, 1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        hidden_states = self.bert(input_ids, token_type_ids, position_ids)
        # print(f"Shape of hidden states: {hidden_states.shape}")  # Debugging line
        start_logits = self.start_logits(hidden_states).squeeze(-1)
        end_logits = self.end_logits(hidden_states).squeeze(-1)
        return start_logits, end_logits
