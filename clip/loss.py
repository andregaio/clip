import torch
import torch.nn.functional as F
import torch.nn as nn


class CrontrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(CrontrastiveLoss, self).__init__()
        self.temperature = temperature
        self.register_buffer('targets', torch.eye(batch_size, requires_grad = False))

    def forward(self, image_embeddings, text_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        texts_loss = F.cross_entropy(logits, self.targets, reduction='mean')
        images_loss = F.cross_entropy(logits.T, self.targets.T, reduction='mean')
        return (images_loss + texts_loss) / 2.0