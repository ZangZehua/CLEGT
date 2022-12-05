import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class BackBoneNet(nn.Module):
    def __init__(self, base_model, out_dim, projector):
        super(BackBoneNet, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        self.in_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        # projector
        self.sizes = [self.in_feature] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(self.sizes) - 2):
            layers.append(nn.Linear(self.sizes[i], self.sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(self.sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(self.sizes[-2], self.sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(self.sizes[-1], affine=False)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        y = self.backbone(x)
        return self.projector(y)
