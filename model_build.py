import torch
import torch.nn as nn
import config

class CNNBlock(nn.Module):
    """
    CNN Block
    """
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    """
    Residual Block
    """
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    """
    The output branch for different scales prediction
    output = (samples, anchors, grid, grid, classes + 5)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # cells = 3 * (num_classes + 5), 3 is the number of anchor boxes, 5 is (objectness, x, y, w, h)
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, 3 * (num_classes + 5), bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        # split the bounding boxes into three dimensions
        # output = (samples, anchors, grid, grid, classes + 5)
        return (
            self.pred(x).reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )
    
class YOLOv3(nn.Module):
    """
    YOLOv3 model
    """
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
    
    def forward(self, x):
        outputs = []
        route_connections = []
        # loop through the layers for the scale prediction and the residual blocks for upsampling
        for layer in self.layers:
            # if the layer is scale prediction, append the output to the outputs list
            # and continue to the next iteration, since the scale prediction layer is the branch of the network
            # and we don't want to pass the output of the scale prediction layer to the next layer
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
            
        return outputs
            
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config.model_architecture:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1 if kernel_size == 3 else 0)
                )
                in_channels = out_channels
            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels *= 3
        
        return layers

def test():
    num_classes = 2
    IMAGE_SIZE = 480 # yolo v1 is 448, yolo v3 is 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    # three outputs, each output is the prediction of the scale
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")

if __name__ == "__main__":
    test()
