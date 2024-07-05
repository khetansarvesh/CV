import torch
import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
                                        nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(16),nn.ReLU(),
                                        nn.Conv2d(16, 4, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(4),nn.ReLU(),
                                        nn.Conv2d(4, 2, kernel_size=1)
                                    )

        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=2)

        self.decoder = nn.Sequential(
                                        #nn.Conv2d(2, 4, kernel_size=1),
                                        nn.ConvTranspose2d(2, 16, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(16),nn.ReLU(),
                                        nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),nn.Tanh(), #no normalization in this layer
                                    )


    def forward(self, x):
        # encoder
        quant_input = self.encoder(x)

        # Reshaping
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))

        # Quantization Process
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1))) # Compute pairwise distances
        min_encoding_indices = torch.argmin(dist, dim=-1) # Find index of nearest embedding
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1)) # Select the embedding weights

        # Quantization Loss Calculation
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        quantize_losses = codebook_loss + 0.2*commitment_loss

        # Ensure straight through gradient
        quant_out = quant_input + (quant_out - quant_input).detach()

        # Reshaping back to original input shape
        quant_out = quant_out.reshape((B, H, W, C))
        quant_out = quant_out.permute(0, 3, 1, 2)

        # Decoder
        output = self.decoder(quant_out)

        return output, quantize_losses

# class VQGAN(nn.Module):
#     def __init__(self):
#         super(VQGAN, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(16),nn.ReLU(),
#             nn.Conv2d(16, 4, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(4),nn.ReLU(),
#             nn.Conv2d(4, 2, kernel_size=1)
#         )

#         self.embedding = nn.Embedding(num_embeddings = 1024, embedding_dim=2)
#         self.embedding.weight.data.uniform_(-1.0 / 1024, 1.0 / 1024)

#         self.decoder = nn.Sequential(
#             nn.Conv2d(2, 4, kernel_size=1),
#             nn.ConvTranspose2d(4, 16, kernel_size=4, stride=2, padding=1),nn.BatchNorm2d(16),nn.ReLU(),
#             nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),nn.Tanh(), #no normalization in this layer
#         )


#     def forward(self, imgs):

#         # Encoder
#         quant_conv_encoded_images = self.encoder(imgs)

#         # Reshaping
#         z = quant_conv_encoded_images.permute(0, 2, 3, 1).contiguous()
#         z_flattened = z.view(-1, 2)

#         # Quantization Process
#         d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2*(torch.matmul(z_flattened, self.embedding.weight.t())) # compute pairwise distance
#         min_encoding_indices = torch.argmin(d, dim=1) # find index of nearest embedding
#         z_q = self.embedding(min_encoding_indices).view(z.shape) # select the embedding weights

#         # Quantization Loss Calculation
#         commitment_loss = torch.mean((z_q.detach() - z)**2)
#         codebook_loss = torch.mean((z_q - z.detach())**2)
#         loss = codebook_loss + 0.2*commitment_loss

#         # Ensure straight through gradient
#         z_q = z + (z_q - z).detach()

#         # Reshaping back to original input shape
#         z_q = z_q.permute(0, 3, 1, 2)

#         # Decoder
#         output = self.decoder(z_q)

#         return output, loss