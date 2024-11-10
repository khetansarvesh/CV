import torch
import torch.nn as nn
from CV.object_detection.yolo_v1.iou import intersection_over_union

def YoloLoss(S, B, C, predictions, target):

    mse = nn.MSELoss(reduction="sum")

    # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
    predictions = predictions.reshape(-1, S, S, C + B * 5)



    '''
    # ======================== #
    # Calculating which predicted bounding box has max IOU wrt the target bounding box    #
    # ======================== #

    '''
    # calculating IOU between predicted bounding box 1 and target bounding box
    iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])

    # calculating IOU between predicted bounding box 2 and target bounding box
    iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

    # concatenating the two IOUs
    ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)


    # Take the box with highest IoU out of the two prediction => Note that bestbox will be indices of 0, 1 for which bbox was best
    iou_maxes, bestbox = torch.max(ious, dim=0)
    exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i



    '''
    # ======================== #
    #    BOX COORDINATE LOSS    #
    # ======================== #
    '''
    # Set boxes with no object in them to 0. We only take out one of the two predictions, which is the one with highest Iou calculated previously.
    box_predictions = exists_box * ((bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]))
    box_targets = exists_box * target[..., 21:25]

    # Take sqrt of width, height in predicted bounding box
    box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))

    # Take sqrt of width, height in actual bounding box
    box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

    # loss value
    box_loss = mse(torch.flatten(box_predictions, end_dim=-2),torch.flatten(box_targets, end_dim=-2),)



    '''
    # ==================== #
    #    OBJECT LOSS    #
    # ==================== #
    '''
    pred_box = (bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21])
    object_loss = mse(torch.flatten(exists_box * pred_box),torch.flatten(exists_box * target[..., 20:21]),)



    '''
    # ======================= #
    #    NO OBJECT LOSS    #
    # ======================= #
    '''
    no_object_loss = mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1), torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),)
    no_object_loss += mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1))



    '''
    # ================== #
    #    CLASS LOSS   #
    # ================== #
    '''
    class_loss = mse(torch.flatten(exists_box * predictions[..., :20], end_dim=-2,), torch.flatten(exists_box * target[..., :20], end_dim=-2,),)


    return 5 * box_loss + object_loss + 0.5 * no_object_loss + class_loss