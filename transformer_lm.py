# models.py

import numpy as np
import torch.nn as nn
import torch
from torch import optim
import random
from utils import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20):
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)


    def forward(self, x, batched=False):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


class FeedFoward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)

class LanguageModelExample(object):
    def __init__(self, input: str, vocab_index: Indexer):
        new_input = " " + input[:-1]
        self.input_tensor = torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in new_input]))
        self.output_tensor = torch.LongTensor(np.array([vocab_index.index_of(ci) for ci in input]))

class Head(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.K = nn.Linear(d_model, d_internal)
        self.Q = nn.Linear(d_model, d_internal)
        self.V = nn.Linear(d_model, d_internal)
        self.w0 = nn.Linear(d_internal, d_model // num_heads)
        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))

    def forward(self, input_vecs):
        keys = self.K(input_vecs) # B, L, d_internal
        d_k = keys.shape[-1]
        queries = self.Q(input_vecs) # B, L, d_internal
        value = self.V(input_vecs) # B, L, d_internal
        weights = torch.matmul(queries, keys.transpose(-2, -1)) * d_k**-0.5# L, L
        weights = weights.masked_fill(self.tril == 0, float('-inf'))
        attention = torch.softmax(weights, dim=-1)

        logit = torch.matmul(attention , value) # B, L, d_internal
        logit = self.w0(logit)
        return logit

class MultiHeadAttention(nn.Module):

    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_length, d_model, num_heads, d_internal) for _ in range(num_heads)])
        self.linear1 = nn.Linear(d_model, d_model)

    def forward(self, input_vecs):
        out = torch.cat([head(input_vecs) for head in self.heads], dim=-1)
        out = self.linear1(out)
        return out

class MHATransformerLayer(nn.Module):
    def __init__(self, seq_length, d_model, num_heads, d_internal):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention( seq_length, d_model, num_heads, d_internal)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, input_vecs):
        x = self.multi_head_attention(self.ln1(input_vecs))
        x += input_vecs
        x = x + self.ffwd(self.ln2(x))

        return x

class MHATransformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, num_heads):
        super().__init__()
        self.num_positions = num_positions
        self.L = []
        for ly in range(num_layers):
            self.L.append(MHATransformerLayer(num_positions, d_model, num_heads, d_internal))
        self.transformer_layers = nn.Sequential(*self.L)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, num_positions=num_positions)
        self.layer_norm = nn.LayerNorm(d_model)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, indices, batched=False):
        logit = self.embedding(indices)
        logit = self.pos_embedding(logit, batched=batched)
        logit = self.transformer_layers(logit)
        logit = self.classifier(logit)
        logit = self.softmax(logit)
        if batched:
            return logit
        else:
            return logit.squeeze(0)

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index):
        self.model = model
        self.model.eval()
        self.vocab = vocab_index

    def get_next_char_log_probs(self, context):
        if len(context) < 20:
            context = " " * (20 - len(context)) + context
        context = context[len(context)-20:len(context)]
        input_tensor = torch.LongTensor(np.array([self.vocab.index_of(ci) for ci in context]))
        logit = self.model(input_tensor, batched=False)
        next_char_log_prob = logit[-1, :]
        return next_char_log_prob.detach().numpy()


    def get_log_prob_sequence(self, next_chars, context):
        total_prob = 0.0
        for n_c in next_chars:
            n_c_idx = self.vocab.index_of(n_c)
            log_probs = self.get_next_char_log_probs(context)
            total_prob += log_probs[n_c_idx]
            context += n_c
        return total_prob

def generate_dataset(source_text, vocab_index):
    examples = []
    for i in range(0, len(source_text)-20, 20):
        examples.append(LanguageModelExample(source_text[i:i+20], vocab_index))
    return examples
def train_lm(args, train_text, dev_text, vocab_index):
    train_exs = generate_dataset(train_text, vocab_index)
    dev_exs = generate_dataset(dev_text, vocab_index)

    model = MHATransformer(27, 20, 80, 40, 1, 8)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    batch_size = 64
    num_batches = len(train_exs) // batch_size
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        random.shuffle(train_exs)
        data_batches = np.array_split(train_exs[:num_batches * batch_size], num_batches)
        for train_batch in data_batches:
            batch_x = []
            batch_y = []
            for letter_ex in train_batch:
                batch_x.append(letter_ex.input_tensor)
                batch_y.append(letter_ex.output_tensor)
            batch_x = torch.stack(batch_x)
            batch_y = torch.stack(batch_y)
            logit = model(batch_x, batched=True)
            model.zero_grad()
            #print(logit.shape)
            loss = nn.NLLLoss()(logit.permute(0, 2, 1), batch_y)
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()
        print(f"finished epoch {t} with total loss {loss_this_epoch}")
    model.eval()
    return NeuralLanguageModel(model, vocab_index)
