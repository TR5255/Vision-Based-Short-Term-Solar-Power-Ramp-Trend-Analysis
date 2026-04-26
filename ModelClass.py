import torch
import torch.nn as nn
import torchvision.models as models
class SequenceModel(nn.Module):

    def __init__(self, num_aux=2):
        super().__init__()

        # ------------------------
        # CNN BACKBONE
        # ------------------------
        backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
        self.cnn = backbone.features

        # Freeze most layers
        for p in self.cnn.parameters():
            p.requires_grad = False

        # Unfreeze last stage
        for p in self.cnn[-1].parameters():
            p.requires_grad = True

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ------------------------
        # TEMPORAL PROJECTION
        # ------------------------
        self.temporal_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU()
        )

        # ------------------------
        # FINAL HEAD
        # ------------------------
        self.head = nn.Sequential(
            nn.Linear(256 + num_aux, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, aux):

        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # flatten time into batch
        x = x.reshape(B*T ,C, H, W)

        # CNN feature extraction
        feat = self.cnn(x)
        feat = self.pool(feat).flatten(1)   # (B*T, 768)

        # restore time dimension
        feat = feat.view(B, T, 768)

        # ------------------------
        # TEMPORAL AGGREGATION (MEAN)
        # ------------------------
        feat = feat.mean(dim=1)

        # ------------------------
        # PROJECTION
        # ------------------------
        feat = self.temporal_fc(feat)

        # ------------------------
        # AUX + HEAD
        # ------------------------
        out = torch.cat([feat, aux], dim=1)

        return self.head(out).squeeze(1)