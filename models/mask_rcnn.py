import torchvision

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2, 
    MaskRCNN_ResNet50_FPN_V2_Weights,
    faster_rcnn,
    mask_rcnn)


def get_model(num_classes):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.con5_mask.in_channels

    hidden_layer = 256
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model