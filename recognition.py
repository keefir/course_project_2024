from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50


class ResNet50EmbNorm(nn.Module):

    def __init__(self, emb_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, emb_size, bias=False)

    def forward(self, data):

        return F.normalize(self.model(data), p=2, dim=1)
