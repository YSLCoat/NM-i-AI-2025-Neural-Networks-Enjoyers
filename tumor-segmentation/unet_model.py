import segmentation_models_pytorch as smp

def get_unet_model(in_channels=1, out_classes=1):
    """
    Returns a UNet model with a ResNet34 encoder.

    Args:
        in_channels (int): Number of input channels. Default is 1 for grayscale images.
        out_classes (int): Number of output classes. Default is 1 for binary segmentation.

    Returns:
        model (torch.nn.Module): The UNet model instance.
    """
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use pretrained weights for encoder
        in_channels=in_channels,        # input channels (1 for grayscale images)
        classes=out_classes,            # output channels (1 for binary mask)
    )
    return model