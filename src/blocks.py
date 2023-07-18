import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchinfo import summary


IMAGE_SIZE = 299
VOCAB_SIZE = 250
EMBED_DIM = 512
SEQ_LENGTH = 25

# function cannot be work
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        base_model = efficientnet_b0(weights = weights)

        # freeze all trainable feature extraction layers parameters
        for param in base_model.features.parameters():
            param.requires_grad = False

        base_model_out = base_model.features
        
        self.cnn_model = nn.Sequential(
            base_model_out,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512)
        )

    def forward(self, inputs):

        return self.cnn_model(inputs)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention_1 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=0.0
        )

        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

        self.dense = nn.Linear(embed_dim, dense_dim)
        self.relu  = nn.ReLU()

    def forward(self, input_tensor):

        output = self.layernorm_1(input_tensor)
        output = self.relu(self.dense(output))

        attention_output = self.attention_1(
            query = output,
            value = output,
            key = output,
            # attention_mask = None,

        )

        output = self.layernorm_2(output + attention_output) # output type /Tensor attention_output /tuple

        return output
    

class PositionalEmbedding(nn.Module):
    def __init__(self, seq_length, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim)

        self.position_embeddings = nn.Embedding(
            num_embeddings=seq_length,
            embedding_dim=embed_dim
        )

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

    def compute_mask(self, inputs, mask=None):
        return torch.not_equal(inputs, 0)
    
    def forward(self, input_):
        length = input_.size(-1)
        positions = torch.arange(start=0, end=length)
        emebedded_tokens = self.token_embeddings(input_)
        emebedded_tokens = emebedded_tokens * self.embed_scale
        emebedded_positions = self.position_embeddings(positions)

        return emebedded_tokens + emebedded_positions
    
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super(TransformerDecoderBlock,self).__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads

        self.attention_1 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=0.1
        )

        self.attention_2 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=0.1
        )

        self.ffn_layer_1 = nn.Linear(in_features=embed_dim,out_features=ff_dim)
        self.ffn_layer_2 = nn.Linear(in_features=ff_dim,out_features=embed_dim)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.layernorm_1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.layernorm_3 = nn.LayerNorm(normalized_shape=embed_dim)

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )

        self.out = nn.Linear(in_features=EMBED_DIM, out_features=VOCAB_SIZE)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.5)

    def forward(self, input_, encoder_outputs, training, mask=None):
        inputs = self.embedding(input_)
        causal_mask = self.get_casual_attention_mask(inputs)

        if mask is not None:
            padding_mask = mask.unsqueeze(2).type(torch.int32)
            combined_mask = mask.unsqueeze(1).type(torch.int32)
            combined_mask = torch.minimum(combined_mask, causal_mask)

        attention_output_1, _= self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask
        )

        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2, _= self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask
        )

        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out =  self.relu(self.ffn_layer_1(out_2))
        ffn_out =  self.dropout_1(ffn_out)
        ffn_out =  self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out)

        preds = self.softmax(self.out(ffn_out))

        return preds
    
    def get_causal_attention_mask(self, inputs):
        inp_shape = inputs.shape
        batch_size, seq_length = inp_shape[0], inp_shape[2]
        i = torch.unsqueeze(torch.arange(seq_length), 1)



if __name__ == "__main__":
    pass



