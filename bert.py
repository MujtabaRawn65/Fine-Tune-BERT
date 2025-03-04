import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    import math
    """
    Compute multi-head scaled dot-product attention.

    Args:
        key: Tensor of shape [bs, num_heads, seq_len, head_dim]
        query: Tensor of shape [bs, num_heads, seq_len, head_dim]
        value: Tensor of shape [bs, num_heads, seq_len, head_dim]
        attention_mask: Tensor broadcastable to [bs, num_heads, seq_len, seq_len],
                        with 0 for non-padding and large negative values for padding.

    Returns:
        Tensor of shape [bs, seq_len, hidden_size] where hidden_size = num_heads * head_dim.
    """ #  Written by Mujtaba

    # TODO: Mujtaba
    # Each attention is calculated following eq. (1) of https://arxiv.org/pdf/1706.03762.pdf.
    # Attention scores are calculated by multiplying the key and query to obtain
    # a score matrix S of size [bs, num_attention_heads, seq_len, seq_len].

    ## Retrieve head dimension for scaling
    d_k = query.size(-1)

    ## Compute raw attention scores using dot-product between query and key
    ## scores shape: [bs, num_heads, seq_len, seq_len]
    S = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # S[*, i, j, k] represents the (unnormalized) attention score between the j-th and k-th
    # token, given by i-th attention head.
    # Before normalizing the scores, use the attention mask to mask out the padding token scores.
    # Note that the attention mask distinguishes between non-padding tokens (with a value of 0)
    # and padding tokens (with a value of a large negative number).

    ## Apply the attention mask: Add large negative numbers to pad token positions
    S = S + attention_mask

    # Make sure to:
    # - Normalize the scores with softmax.
    # - Multiply the attention scores with the value to get back weighted values.
    # - Before returning, concatenate multi-heads to recover the original shape:
    #   [bs, seq_len, num_attention_heads * attention_head_size = hidden_size].

    ## Normalize the scores to probabilities using softmax on the last dimension
    attn_probs = torch.softmax(S, dim=-1)

    ## Use the attention probabilities to weight the values
    ## weighted_values shape: [bs, num_heads, seq_len, head_dim]
    weighted_values = torch.matmul(attn_probs, value)

    # Concatenate multi-head outputs:
    # First, transpose to bring the head dimension next to the sequence dimension.
    # Then, reshape to combine the heads with the head dimension.
    # Final shape: [bs, seq_len, num_heads * head_dim]
    bs, num_heads, seq_len, head_dim = weighted_values.size()
    ## Transpose to shape [bs, seq_len, num_heads, head_dim]
    weighted_values = weighted_values.transpose(1, 2).contiguous()
    ## Merge the last two dimensions to get [bs, seq_len, hidden_size]
    output = weighted_values.view(bs, seq_len, num_heads * head_dim)

    return output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = BertSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    This function is applied after the multi-head attention layer or the feed forward layer.
    
    Args:
        input: Tensor, the input of the previous layer
        output: Tensor, the output of the previous sub-layer (e.g., attention or feed-forward)
        dense_layer: nn.Module, used to transform the output (typically a linear layer)
        dropout: nn.Module, the dropout layer to be applied 
        ln_layer: nn.Module, the layer norm to be applied
        
    Returns:
        Tensor after applying dense transformation, dropout, residual connection, and layer normalization.
    """ # Modified by Mujtaba

    # TODO: Mujtaba
    # Hint: Remember that BERT applies dropout to the transformed output of each sub-layer,
    # before it is added to the sub-layer input and normalized with a layer norm.
    # Transform the output using the provided dense layer.
    transformed = dense_layer(output)
    
    ## Apply dropout to the transformed output.
    dropped = dropout(transformed)
    
    ## Add the original input (residual connection) to the dropped output.
    added = input + dropped
    
    ## Apply layer normalization to the result of the addition.
    normalized = ln_layer(added)
    
    return normalized


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # TODO: Talha
    ## attention_output: Tensor, shape [batch_size, seq_length, hidden_size]
    ## Pass the hidden states through the multi-head attention mechanism to get attention output.
    attention_output = self.self_attention(hidden_states, attention_mask)

    ## attention_output: Tensor, shape [batch_size, seq_length, hidden_size]
    ## Apply the add-norm operation after attention to stabilize and improve training.
    attention_output = self.add_norm(
            input=hidden_states,
            output=attention_output,
            dense_layer=self.attention_dense,
            dropout=self.attention_dropout,
            ln_layer=self.attention_layer_norm
          )

    ## feed_forward_output: Tensor, shape [batch_size, seq_length, hidden_size]
    ## Apply the feed-forward layer (dense layer followed by activation function).
    feed_forward_output = self.interm_dense(attention_output)
    feed_forward_output = self.interm_af(feed_forward_output)

    ## output: Tensor, shape [batch_size, seq_length, hidden_size]
    ## Apply another add-norm operation after the feed-forward layer to finalize the output.
    output = self.add_norm(
            input=attention_output,
            output=feed_forward_output,
            dense_layer=self.out_dense,
            dropout=self.out_dropout,
            ln_layer=self.out_layer_norm
        )

    return output



class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)
    
    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    # TODO: Talha
    ## input_shape: Tensor, shape [batch_size, seq_length]
    ## Get the shape of the input tensor to know the batch size and sequence length.
    input_shape = input_ids.size()  

    ## seq_length: Variable, represents the length of the input sequence
    ## Extract the sequence length from the input tensor's shape.
    seq_length = input_shape[1]  

    # Get word embedding from self.word_embedding into input_embeds.
    ## inputs_embeds: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Convert the input IDs into embeddings using the word embedding layer.
    inputs_embeds = self.word_embedding(input_ids)  

    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    ## pos_ids: Tensor, shape [batch_size, seq_length]
    ## Slice the position IDs tensor to match the sequence length.
    pos_ids = self.position_ids[:, :seq_length]  

    ## pos_embeds: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Retrieve position embeddings based on the position IDs.
    pos_embeds = self.pos_embedding(pos_ids)  

    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    ## tk_type_ids: Tensor, shape [batch_size, seq_length]
    ## Initialize token type IDs to zero, as we are not using token type embeddings.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)  

    ## tk_type_embeds: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Get token type embeddings (which will be zero in this case).
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)  

    # Add three embeddings together; then apply embed_layer_norm and dropout and return
    ## total: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Combine word, position, and token type embeddings.
    total = inputs_embeds + pos_embeds + tk_type_embeds  

    ## norm_layer: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Normalize the combined embeddings for stable training.
    norm_layer = self.embed_layer_norm(total)  

    ## embed_output: Tensor, shape [batch_size, seq_length, embedding_dim]
    ## Apply dropout to prevent overfitting after layer normalization.
    embed_output = self.embed_dropout(norm_layer)  

    return embed_output

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
