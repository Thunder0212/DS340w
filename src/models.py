import torch.nn as nn
import torchvision.models as tv
import timm

def make_model(backbone, num_classes=2, pretrained=True):
    if backbone.startswith("resnet"):
        fn = getattr(tv, backbone)
        m = fn(weights="DEFAULT" if pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    elif backbone.startswith("vit"):
        m = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        return m
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
