import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicGenreModel(nn.Module):
    """Multi-label genre classifier: feature attention + dedicated branches for hard genres."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.4)

        self.attention = nn.Linear(2048, 2048)

        # Genre branches (metal, punk, blues)
        def branch():
            return nn.Sequential(
                nn.Linear(2048, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
        self.metal_branch = branch()
        self.punk_branch = branch()
        self.blues_branch = branch()

        self.final_dense = nn.Linear(2048 + 256*3, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.silu(self.dense1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        att = torch.sigmoid(self.attention(x))
        x_att = x * att

        metal_feat = self.metal_branch(x_att)
        punk_feat = self.punk_branch(x_att)
        blues_feat = self.blues_branch(x_att)

        x_concat = torch.cat([x_att, metal_feat, punk_feat, blues_feat], dim=1)

        x = F.silu(self.final_dense(x_concat))
        x = self.bn2(x)
        x = self.dropout2(x)

        out = torch.sigmoid(self.output_layer(x))
        return out


class MoodClassifier(nn.Module):
    """Single-label mood classifier (outputs logits; apply softmax for probabilities)."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class WeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights  # tensor shape [num_classes]

    def forward(self, y_pred, y_true):
        loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        if self.weights is not None:
            loss = loss * self.weights
        return loss.mean()
