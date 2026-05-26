import torch
import torch.nn as nn
import torchvision.models as models
import torch.hub
import warnings
import timm
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 2  # You can make this dynamic


def load_weights(model, weight_path):
    weights = torch.load(weight_path, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in weights['state_dict'].items()}
    state_dict = {k.replace('model.', ''): v for k, v in weights['state_dict'].items()} # CHECK THIS LINE IF ERROR IN WEIGHT LOADING
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)

def _replace_classifier(model, n_classes):
    # generic helper to replace final classifier across different architectures
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
    elif hasattr(model, 'classifier'):
        cls = model.classifier
        if isinstance(cls, nn.Linear):
            in_features = cls.in_features
            model.classifier = nn.Linear(in_features, n_classes)
        elif isinstance(cls, nn.Sequential):
            # replace last Linear in the sequential classifier
            for i in reversed(range(len(cls))):
                if isinstance(cls[i], nn.Linear):
                    in_features = cls[i].in_features
                    cls[i] = nn.Linear(in_features, n_classes)
                    model.classifier = cls
                    break
        else:
            raise RuntimeError("Unknown classifier type, please adjust replacement logic.")
    else:
        raise RuntimeError("Model has no recognizable classifier ('fc' or 'classifier').")

def unfreeze_from_layer(model, layer_name="layer4"):
    """
    Freeze all parameters, then unfreeze everything from `layer_name` onward
    (inclusive), using named_parameters order.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze from target layer onward
    unfreeze = False
    for name, param in model.named_parameters():
        if layer_name in name:
            unfreeze = True
        if unfreeze:
            param.requires_grad = True


class ResNetModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class DenseNetModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "features.denseblock4.denselayer16")
    def forward(self, x):
        return self.model(x)

class ResNet34Model(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if hasattr(models, 'resnet34'):
            self.model = models.resnet34(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError("resnet34 not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class ResNet101Model(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if hasattr(models, 'resnet101'):
            self.model = models.resnet101(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError("resnet101 not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class EfficientNetV2SModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        # torchvision >= 0.13 provides efficientnet_v2_s
        if hasattr(models, 'efficientnet_v2_s'):
            self.model = models.efficientnet_v2_s(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError("efficientnet_v2_s not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "features.6.14")
    def forward(self, x):
        return self.model(x)

class MobileNetV2Model(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if hasattr(models, 'mobilenet_v2'):
            self.model = models.mobilenet_v2(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError("mobilenet_v2 not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "features.17")
    def forward(self, x):
        return self.model(x)

class MobileNetV3LargeModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        if hasattr(models, 'mobilenet_v3_large'):
            self.model = models.mobilenet_v3_large(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError("mobilenet_v3_large not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "features.15")
    def forward(self, x):
        return self.model(x)

class ResNeXtModel(nn.Module):
    def __init__(self, variant='resnext101_32x8d', weight_path=None):
        super().__init__()
        if hasattr(models, variant):
            self.model = getattr(models, variant)(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError(f"{variant} not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class WideResNetModel(nn.Module):
    def __init__(self, weight_path=None, variant='wide_resnet50_2'):
        super().__init__()
        if hasattr(models, variant):
            self.model = getattr(models, variant)(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            raise NotImplementedError(f"{variant} not available in torchvision.models in this environment.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class Res2NetModel(nn.Module):
    def __init__(self, weight_path=None, hub_repo=None, hub_model_name=None):
        super().__init__()
        # torchvision doesn't ship Res2Net; try torch.hub if repository/model provided
        if hasattr(models, 'res2net50'):
            self.model = models.res2net50(pretrained=True)
            _replace_classifier(self.model, n_classes)
        try:
            import timm
            # create_model will set the classifier when num_classes is provided
            self.model = timm.create_model('res2net50_26w_4s', pretrained=True, num_classes=n_classes)
            _replace_classifier(self.model, n_classes)
        except Exception:
            if hub_repo and hub_model_name:
                self.model = torch.hub.load(hub_repo, hub_model_name, pretrained=True)
                _replace_classifier(self.model, n_classes)
            else:
                raise NotImplementedError("Res2Net not found in torchvision.models. Provide hub_repo and hub_model_name to load via torch.hub.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)

class SENetModel(nn.Module):
    def __init__(self, weight_path=None, hub_repo=None, hub_model_name=None):
        super().__init__()
        # SENet not available in standard torchvision; allow torch.hub fallback
        if hasattr(models, 'squeezenet1_1'):
            # placeholder if a torchvision SENet variant appears in future
            self.model = models.squeezenet1_1(pretrained=True)
            _replace_classifier(self.model, n_classes)
        else:
            if hub_repo and hub_model_name:
                self.model = torch.hub.load(hub_repo, hub_model_name, pretrained=True)
                _replace_classifier(self.model, n_classes)
            else:
                raise NotImplementedError("SENet is not available in torchvision.models. Provide hub_repo and hub_model_name to load via torch.hub.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "features.12")
    def forward(self, x):
        return self.model(x)

class SKNetModel(nn.Module):
    def __init__(self, weight_path=None, hub_repo=None, hub_model_name=None):
        super().__init__()
        try:
            import timm
            self.model = timm.create_model('skresnet34.ra_in1k', pretrained=True, num_classes=n_classes)
            _replace_classifier(self.model, n_classes)
        except Exception:
            if hub_repo and hub_model_name:
                self.model = torch.hub.load(hub_repo, hub_model_name, pretrained=True)
                _replace_classifier(self.model, n_classes)
            else:
                raise NotImplementedError("SKNet is not available in torchvision.models. Provide hub_repo and hub_model_name to load via torch.hub.")
        if weight_path:
            self.model = load_weights(self.model, weight_path)
        unfreeze_from_layer(self.model, "layer4")
    def forward(self, x):
        return self.model(x)


class ViTModel(nn.Module):
    def __init__(self, variant='vit_base_patch16_224', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        #unfreeze_from_layer(self.model, "layer4")

    def forward(self, x):
        return self.model(x)

class DeiTModel(nn.Module):
    def __init__(self, variant='deit_base_patch16_224', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class PVTModel(nn.Module):
    def __init__(self, variant='pvt_v2_b2', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class CvTModel(nn.Module):
    def __init__(self, variant='cvt-13', weight_path=None):
        super().__init__()
        self.model = torch.hub.load("microsoft/CvT","cvt_13_224",pretrained=True)
        _replace_classifier(self.model, n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class TNTModel(nn.Module):
    def __init__(self, variant='tnt_s_patch16_224', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class CrossViTModel(nn.Module):
    def __init__(self, variant='crossvit_base_240', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class SwinV2Model(nn.Module):
    def __init__(self, variant='swinv2_base_window12_192_22k', weight_path=None):
        super().__init__()
        self.variant = variant
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        if self.variant =='swinv2_small_window8_256.ms_in1k':
            x = torch.nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            x = torch.nn.functional.interpolate(x, size=(192, 192), mode='bilinear', align_corners=False)
        return self.model(x)

class VisionMambaModel(nn.Module):
    def __init__(self, variant='vim_small_patch16_224', weight_path=None):
        super().__init__()
        self.model = torch.hub.load(
    "hustvl/Vim",
    "vim_small_patch16_224",
    pretrained=True,
)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class MAEModel(nn.Module):
    def __init__(self, variant='vit_base_patch16_224.mae', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            load_weights(self, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class BEiTModel(nn.Module):
    def __init__(self, variant='beit_base_patch16_224', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class SimMIMModel(nn.Module):
    def __init__(self, variant='swin_base_patch4_window7_224', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")
    def forward(self, x):
        return self.model(x)

class MobileViTModel(nn.Module):
    def __init__(self, variant='mobilevit_s', weight_path=None):
        super().__init__()
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)

class MaxViTModel(nn.Module):
    def __init__(self, variant='maxvit_tiny_tf_224.in1k', weight_path=None):
        super().__init__()
        # local_path = "/home/teaching/.cache/huggingface/hub/models--timm--maxvit_tiny_tf_224.in1k/snapshots/manual"
        # self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes,pretrained_cfg_overlay={'file': f"{local_path}/model.safetensors"})
        self.model = timm.create_model(variant, pretrained=True, num_classes=n_classes)

        if weight_path:
            self.model = load_weights(self.model, weight_path)

        # unfreeze_from_layer(model, layer_name="layer4")

    def forward(self, x):
        return self.model(x)


class DINOResNetModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        self.model.fc = nn.Linear(2048, n_classes)
        if weight_path:
            self.model = load_weights(self.model, weight_path)

    def forward(self, x):
        return self.model(x)

class DinoV2ViTModel(nn.Module):
    def __init__(self, weight_path=None):
        super().__init__()
        self.transformer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
        if weight_path:
            self.load_weights(weight_path)

    def load_weights(self, weight_path):
        load_weights(self, weight_path)

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        return self.classifier(x)


class EnsembleModel1(nn.Module):
    def __init__(self, base_models, thresholds):
        super().__init__()
        self.base_models = nn.ModuleDict(base_models)
        self.thresholds = thresholds
        self.num_models = len(base_models)

        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        self.attention = nn.Sequential(
            nn.Linear(self.num_models, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_models)
        )

        self.output_head = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def normalize_score(self, score, threshold):
        return torch.where(score >= threshold,
                           0.5 + 0.5 * (score - threshold) / (1 - threshold),
                           0.5 * (score / threshold))

    def forward(self, x):
        with torch.no_grad():
            outputs = []
            for name, model in self.base_models.items():
                raw = model(x)
                prob = torch.softmax(raw, dim=1)[:, 1]
                normed = self.normalize_score(prob, self.thresholds[name])
                outputs.append(normed.unsqueeze(1))

        stacked = torch.cat(outputs, dim=1)
        attn_weights = torch.softmax(self.attention(stacked), dim=1)
        fused_score = torch.sum(attn_weights * stacked, dim=1, keepdim=True)
        return self.output_head(fused_score)


class EnsembleModel2(nn.Module):
    def __init__(self, base_models, thresholds):
        super().__init__()
        self.base_models = nn.ModuleDict(base_models)
        self.thresholds = thresholds
        self.num_models = len(base_models)

        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        self.combiner = nn.Sequential(
            nn.Linear(self.num_models, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def normalize_score(self, score, threshold):
        return torch.where(score >= threshold,
                           0.5 + 0.5 * (score - threshold) / (1 - threshold),
                           0.5 * (score / threshold))

    def forward(self, x):
        with torch.no_grad():
            outputs = []
            for name, model in self.base_models.items():
                raw = model(x)
                prob = torch.softmax(raw, dim=1)[:, 1]
                normed = self.normalize_score(prob, self.thresholds[name])
                outputs.append(normed.unsqueeze(1))

        stacked = torch.cat(outputs, dim=1)
        return self.combiner(stacked)
