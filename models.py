import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class Norm(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.norm(embeddings)


class PositionalEncoder(nn.Module):
    def __init__(
        self, embedding_dim: int, max_seq_len: int = 512, dropout: float = 0.1
    ) -> None:
        super(PositionalEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)

        # doing this because pe is not learned
        # should be transferred when changing devices
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * i + 1) / embedding_dim))
                )
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings * math.sqrt(self.embedding_dim)
        seq_length = embeddings.size(1)
        pe = Variable(self.pe[:, :seq_length], requires_grad=False).to(  # type: ignore
            embeddings.device
        )
        embeddings += pe

        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, dropout: float = 0.1) -> None:
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_dim = keys.size(-1)
        attention_scores = torch.matmul(
            queries / np.sqrt(key_dim), keys.transpose(2, 3)
        )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_probs = self.dropout(torch.softmax(attention_scores, dim=-1))

        output = torch.matmul(attention_probs, values)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = SelfAttention(dropout)
        self.num_heads = num_heads
        self.dim_per_head = embedding_dim // num_heads
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = queries.size(0)

        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        queries = queries.view(
            batch_size, -1, self.num_heads, self.dim_per_head
        ).transpose(1, 2)

        keys = keys.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(
            1, 2
        )
        values = values.view(
            batch_size, -1, self.num_heads, self.dim_per_head
        ).transpose(1, 2)

        scores = self.self_attention(queries, keys, values, attention_mask)

        # Reshape the output
        output = (
            scores.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        )

        # Apply the linear projection
        output = self.out(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)

    def forward(
        self, inputs: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x2 = self.norm1(inputs)
        # Add Muti-head attention
        x = inputs + self.dropout1(self.self_attention(x2, x2, x2, attention_mask))

        x2 = self.norm2(x)
        x = x + self.dropout2(self.feed_forward(x2))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        max_seq_len: int,
        encoder_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        depth: int = 5,
        fine_tune: bool = True,
    ) -> None:
        super(Encoder, self).__init__()
        self.eff = EfficientNet.from_pretrained(f"efficientnet-b{depth}")
        self.set_fine_tune(fine_tune)
        self.avg_pool = nn.AdaptiveAvgPool2d((max_seq_len - 1, 512))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(embedding_dim, num_heads, 2048, dropout)
                for _ in range(encoder_layers)
            ]
        )
        self.norm = Norm(embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.eff.extract_features(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        x = self.avg_pool(features)

        # Propagate through the layers
        for layer in self.layers:
            x = layer(x)

        # Normalize
        x = self.norm(x)

        return x

    def set_fine_tune(self, fine_tune=True):
        for p in self.eff.parameters():
            p.requires_grad = fine_tune


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = Norm(embedding_dim)
        self.norm2 = Norm(embedding_dim)
        self.norm3 = Norm(embedding_dim)

    def forward(
        self,
        target_embeddings: torch.Tensor,
        encoder_outputs: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x2 = self.norm1(target_embeddings)
        x = target_embeddings + self.dropout1(
            self.self_attention(x2, x2, x2, attention_mask)
        )

        x2 = self.norm2(x)
        x = x + self.dropout2(
            self.encoder_attention(x2, encoder_outputs, encoder_outputs)
        )

        x2 = self.norm3(x)
        x = x + self.dropout3(self.feed_forward(x2))

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        max_seq_len: int,
        decoder_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(embedding_dim, num_heads, 2048, dropout)
                for _ in range(decoder_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = Norm(embedding_dim)
        self.position_embedding = PositionalEncoder(embedding_dim, max_seq_len, dropout)

    def forward(
        self,
        target_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # memory is the encoder output
        x = self.embed(target_ids)

        # Add the position embeddings
        x = self.position_embedding(x)

        for layer in self.layers:
            x = layer(x, encoder_outputs, attention_mask)

        x = self.norm(x)

        return x


class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        max_seq_len: int,
        encoder_layers: int,
        decoder_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super(ImageCaptionModel, self).__init__()
        self.encoder = Encoder(
            embedding_dim, max_seq_len, encoder_layers, num_heads, dropout
        )
        self.decoder = Decoder(
            embedding_dim, vocab_size, max_seq_len, decoder_layers, num_heads, dropout
        )
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.norm = Norm(embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, images: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        # Encode the image
        encoder_output = self.encoder(images)

        # Create captions mask
        target_mask = self.make_mask(caption_ids)

        # Decode the image
        decoder_output = self.decoder(caption_ids, encoder_output, target_mask)

        # Apply the linear projection
        decoder_output = self.fc(decoder_output)

        return decoder_output

    def make_mask(self, caption_ids: torch.Tensor) -> torch.Tensor:
        _batch_size, len_target = caption_ids.size()

        subsequent_mask = (
            1
            - torch.triu(
                torch.ones((1, len_target, len_target), device=caption_ids.device),
                diagonal=1,
            )
        ).bool()
        return subsequent_mask
