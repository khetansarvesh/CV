import torch
import torch.nn as nn
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    r"""
    Attention Module for DiT.
    This is same as VIT code and does not have any changes
    from it.
    """
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['num_heads']
        self.hidden_size = config['hidden_size']
        self.head_dim = config['head_dim']

        self.att_dim = self.n_heads * self.head_dim

        # QKV projection for the input
        self.qkv_proj = nn.Linear(self.hidden_size, 3 * self.att_dim, bias=True)
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.hidden_size))

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)
        nn.init.xavier_uniform_(self.output_proj[0].weight)
        nn.init.constant_(self.output_proj[0].bias, 0)

    def forward(self, x):
        #  Converting to Attention Dimension
        ######################################################
        # Batch Size x Number of Patches x Dimension
        B, N = x.shape[:2]
        # Projecting to 3*att_dim and then splitting to get q, k v(each of att_dim)
        # qkv -> Batch Size x Number of Patches x (3* Attention Dimension)
        # q(as well as k and v) -> Batch Size x Number of Patches x Attention Dimension
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        # Batch Size x Number of Patches x Attention Dimension
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Heads x Number of Patches x Head Dimension
        # -> B x H x N x Head Dimension
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',n_h=self.n_heads, h_dim=self.head_dim)
        #########################################################

        # Compute Attention Weights
        #########################################################
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** (-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        #########################################################

        # Weighted Value Computation
        #########################################################
        #  B x H x N x N @ B x H x N x Head Dimension
        # -> B x H x N x Head Dimension
        out = torch.matmul(att, v)
        #########################################################

        # Converting to Transformer Dimension
        #########################################################
        # B x N x (Heads * Head Dimension) -> B x N x (Attention Dimension)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')
        #  B x N x Dimension
        out = self.output_proj(out)
        ##########################################################

        return out

def get_patch_position_embedding(pos_emb_dim, grid_size, device):
    assert pos_emb_dim % 4 == 0, 'Position embedding dimension must be divisible by 4'
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of patch tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([torch.sin(grid_h_emb), torch.cos(grid_h_emb)], dim=-1)
    # grid_h_emb -> (Number of patch tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([torch.sin(grid_w_emb), torch.cos(grid_w_emb)], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of patch tokens, pos_emb_dim)
    return pos_emb


class PatchEmbedding(nn.Module):
    r"""
    Layer to take in the input image and do the following:
        1.  Transform grid of image patches into a sequence of patches.
            Number of patches are decided based on image height,width and
            patch height, width.
        2. Add positional embedding to the above sequence
    """

    def __init__(self,
                 image_height,
                 image_width,
                 im_channels,
                 patch_height,
                 patch_width,
                 hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.im_channels = im_channels

        self.hidden_size = hidden_size

        self.patch_height = patch_height
        self.patch_width = patch_width

        # Input dimension for Patch Embedding FC Layer
        patch_dim = self.im_channels * self.patch_height * self.patch_width
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_size)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.patch_embed[0].weight)
        nn.init.constant_(self.patch_embed[0].bias, 0)

    def forward(self, x):
        grid_size_h = self.image_height // self.patch_height
        grid_size_w = self.image_width // self.patch_width

        # B, C, H, W -> B, (Patches along height * Patches along width), Patch Dimension
        # Number of tokens = Patches along height * Patches along width
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)

        # BxNumber of tokens x Patch Dimension -> B x Number of tokens x Transformer Dimension
        out = self.patch_embed(out)

        # Add 2d sinusoidal position embeddings
        pos_embed = get_patch_position_embedding(pos_emb_dim=self.hidden_size,
                                                 grid_size=(grid_size_h, grid_size_w),
                                                 device=x.device)
        out += pos_embed
        return out


class TransformerLayer(nn.Module):
    r"""
    Transformer block which is just doing the following based on VIT
        1. LayerNorm followed by Attention
        2. LayerNorm followed by Feed forward Block
        Both these also have residuals added to them

        For DiT we additionally have
        1. Layernorm mlp to predict layernorm affine parameters from
        2. Same Layernorm mlp to also predict scale parameters for outputs
            of both mlp/attention prior to residual connection.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']

        ff_hidden_dim = 4 * self.hidden_size

        # Layer norm for attention block
        self.att_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.attn_block = Attention(config)

        # Layer norm for mlp block
        self.ff_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        self.mlp_block = nn.Sequential(
            nn.Linear(self.hidden_size, ff_hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_dim, self.hidden_size),
        )

        # Scale Shift Parameter predictions for this layer
        # 1. Scale and shift parameters for layernorm of attention (2 * hidden_size)
        # 2. Scale and shift parameters for layernorm of mlp (2 * hidden_size)
        # 3. Scale for output of attention prior to residual connection (hidden_size)
        # 4. Scale for output of mlp prior to residual connection (hidden_size)
        # Total 6 * hidden_size
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 6 * self.hidden_size, bias=True)
        )

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.xavier_uniform_(self.mlp_block[0].weight)
        nn.init.constant_(self.mlp_block[0].bias, 0)
        nn.init.xavier_uniform_(self.mlp_block[-1].weight)
        nn.init.constant_(self.mlp_block[-1].bias, 0)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaptive_norm_layer(condition).chunk(6, dim=1)
        (pre_attn_shift, pre_attn_scale, post_attn_scale,
         pre_mlp_shift, pre_mlp_scale, post_mlp_scale) = scale_shift_params
        out = x
        attn_norm_output = (self.att_norm(out) * (1 + pre_attn_scale.unsqueeze(1))
                            + pre_attn_shift.unsqueeze(1))
        out = out + post_attn_scale.unsqueeze(1) * self.attn_block(attn_norm_output)
        mlp_norm_output = (self.ff_norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
                           pre_mlp_shift.unsqueeze(1))
        out = out + post_mlp_scale.unsqueeze(1) * self.mlp_block(mlp_norm_output)
        return out


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DIT(nn.Module):
    def __init__(self, 
                 im_size = 32,  #should be 128
                 im_channels = 4, #should be 3
                 config = {
                        'patch_size' : 2,
                        'hidden_size' : 768,
                        'num_heads' : 12,
                        'head_dim' : 64,
                        'timestep_emb_dim' : 768
                        }
                ):

        super().__init__()

        self.image_height = im_size
        self.image_width = im_size
        self.im_channels = im_channels
        self.hidden_size = 768
        self.patch_height = 2
        self.patch_width = 2

        self.timestep_emb_dim = 768

        # Number of patches along height and width
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height,
                                                image_width=self.image_width,
                                                im_channels=self.im_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # All Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
            TransformerLayer(config),
                    ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.hidden_size,
                                  self.patch_height * self.patch_width * self.im_channels)

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t):
        # Patchify
        out = self.patch_embed_layer(x)

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        # (Batch, timestep_emb_dim) -> (Batch, hidden_size)
        t_emb = self.t_proj(t_emb)

        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out, t_emb)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = self.adaptive_norm_layer(t_emb).chunk(2, dim=1)
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        return out
