import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat

from net.vit import Transformer

class SSLSM(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        num_classes,
        masking_ratio = 0.75,
        decoder_depth = 1,
        bands,
        decoder_heads = 8,
        decoder_dim_head = 64,
        dim
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]
        # self.to_patch, self.patch_to_emb = encoder.to_patch_embedding[:2]
        self.to_spe_patch, self.spe_patch_to_emb= encoder.to_spe_patch_embedding[:2]
        # pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]

        # decoder parameters

        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(dim = decoder_dim, depth = decoder_depth, heads = decoder_heads, dim_head = decoder_dim_head, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.spe_decoder_pos_emb = nn.Embedding(bands, decoder_dim)
        # self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.to_spe_pixels = nn.Linear(decoder_dim, dim)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(bands*decoder_dim),
            nn.Dropout(0.5),
            nn.Linear(bands*decoder_dim,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, img):
        device = img.device

        # get patches

        # patches = self.to_patch(img)
        # batch, num_patches, *_ = patches.shape

        spe_img = self.to_spe_patch(img)
        batch, n, _ = spe_img.shape
        # patch to encoder tokens and add positions
        spe_num_masked = int(self.masking_ratio * n)
        # num_masked = int(self.masking_ratio * 81)
        spe_rand_indices = torch.rand(batch, n,device=device).argsort(dim=-1)
        # rand_indices = torch.rand(batch, 81, device=device).argsort(dim=-1)
        spe_img_pos = spe_img + self.encoder.pos_embedding[:, 1:(n + 1)]
        # patches_pos = patches + self.pos_embedding[:, 1:(81 + 1)]

        spe_masked_indices, spe_unmasked_indices = spe_rand_indices[:,:spe_num_masked], spe_rand_indices[:,spe_num_masked:]
        # masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        spe_batch_range = torch.arange(batch,device=device)[:, None]

        spe_tokens = spe_img_pos[spe_batch_range, spe_unmasked_indices]
        spe_masked_patches = spe_img[spe_batch_range, spe_masked_indices]

        # masked_patches = patches[spe_batch_range, masked_indices]
        # un_masked_patches = patches_pos[spe_batch_range, unmasked_indices]

        spe_encoded_tokens = self.encoder.transformer(spe_tokens)
        # encoded_tokens = self.encoder.transformer(un_masked_patches)
        decoder_tokens = self.enc_to_dec(spe_encoded_tokens)
        decoder_tokens = decoder_tokens + self.spe_decoder_pos_emb(spe_unmasked_indices)
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch, n=spe_num_masked)
        mask_tokens = mask_tokens + self.spe_decoder_pos_emb(spe_masked_indices)
        decoder_tokens = torch.cat((mask_tokens, decoder_tokens), dim=1)
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, :spe_num_masked]
        pred_pixel_values = self.to_spe_pixels(mask_tokens)
        recon_loss = F.mse_loss(pred_pixel_values, spe_masked_patches)
        x = self.fc(decoded_tokens)

        return recon_loss,x

