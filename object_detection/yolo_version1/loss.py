import torch
import torch.nn as nn


def get_iou(boxes1, boxes2):
    r"""
    IOU between two sets of boxes
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[..., 0], boxes2[..., 0])
    y_top = torch.max(boxes1[..., 1], boxes2[..., 1])

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[..., 2], boxes2[..., 2])
    y_bottom = torch.min(boxes1[..., 3], boxes2[..., 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1.clamp(min=0) + area2.clamp(min=0) - intersection_area
    iou = intersection_area / (union + 1E-6)
    return iou


class YOLOV1Loss(nn.Module):
    r"""
    Loss module for YoloV1 which caters to the following components:
    1. Localization Loss for responsible predictor boxes
    2. Objectness Loss for responsible predictor boxes
    2. Objectness Loss for non-responsible predictor boxes of cells assigned with objects
    2. Objectness Loss for ALL predictor boxes of cells not assigned with objects
    3. Classification Loss
    """
    def __init__(self, S=7, B=2, C=20):
        super(YOLOV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, preds, targets, use_sigmoid=False):
        r"""
        Main method of loss computation
        :param preds: (Batch, S*S*(5B+C)) tensor
        :param targets: (Batch, S, S, (5B+C)) tensor.
            Target element for each cell has been duplicated 5B times(done in VOCDataset)
        :param use_sigmoid: Whether to use sigmoid activation for box predicitons or not
        """
        batch_size = preds.size(0)

        # preds -> (Batch, S, S, 5B+C)
        preds = preds.reshape(batch_size, self.S, self.S, 5*self.B + self.C)

        # Generally sigmoid leads to quicker convergence
        if use_sigmoid:
            preds[..., :5 * self.B] = torch.nn.functional.sigmoid(preds[..., :5 * self.B])

        # Shifts for all grid cell locations.
        # Will use these for converting x_center_offset/y_center_offset
        # values to x1/y1/x2/y2(normalized 0-1)
        # S cells = 1 => each cell adds 1/S pixels of shift
        shifts_x = torch.arange(0, self.S,
                                dtype=torch.int32,
                                device=preds.device) * 1 / float(self.S)
        shifts_y = torch.arange(0, self.S,
                                dtype=torch.int32,
                                device=preds.device) * 1 / float(self.S)

        # Create a grid using these shifts
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # shifts -> (1, S, S, B)
        shifts_x = shifts_x.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        shifts_y = shifts_y.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # pred_boxes -> (Batch_size, S, S, B, 5)
        pred_boxes = preds[..., :5*self.B].reshape(batch_size,
                                                   self.S,
                                                   self.S,
                                                   self.B,
                                                   -1)

        # xc_offset yc_offset w h -> x1 y1 x2 y2 (normalized 0-1)
        # x_center = (xc_offset / S + shift_x)
        # x1 = x_center - 0.5 * w
        # x2 = x_center + 0.5 * w
        pred_boxes_x1 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         - 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x1 = pred_boxes_x1[..., None]
        pred_boxes_y1 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         - 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y1 = pred_boxes_y1[..., None]
        pred_boxes_x2 = ((pred_boxes[..., 0]/self.S + shifts_x)
                         + 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x2 = pred_boxes_x2[..., None]
        pred_boxes_y2 = ((pred_boxes[..., 1]/self.S + shifts_y)
                         + 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y2 = pred_boxes_y2[..., None]
        pred_boxes_x1y1x2y2 = torch.cat([
            pred_boxes_x1,
            pred_boxes_y1,
            pred_boxes_x2,
            pred_boxes_y2], dim=-1)

        # target_boxes -> (Batch_size, S, S, B, 5)
        target_boxes = targets[..., :5*self.B].reshape(batch_size,
                                                       self.S,
                                                       self.S,
                                                       self.B,
                                                       -1)
        target_boxes_x1 = ((target_boxes[..., 0] / self.S + shifts_x)
                           - 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x1 = target_boxes_x1[..., None]
        target_boxes_y1 = ((target_boxes[..., 1] / self.S + shifts_y)
                           - 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y1 = target_boxes_y1[..., None]
        target_boxes_x2 = ((target_boxes[..., 0] / self.S + shifts_x)
                           + 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x2 = target_boxes_x2[..., None]
        target_boxes_y2 = ((target_boxes[..., 1] / self.S + shifts_y)
                           + 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y2 = target_boxes_y2[..., None]
        target_boxes_x1y1x2y2 = torch.cat([
            target_boxes_x1,
            target_boxes_y1,
            target_boxes_x2,
            target_boxes_y2
        ], dim=-1)

        # iou -> (Batch_size, S, S, B)
        iou = get_iou(pred_boxes_x1y1x2y2, target_boxes_x1y1x2y2)

        # max_iou_val/max_iou_idx -> (Batch_size, S, S, 1)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True)

        #########################
        # Indicator Definitions #
        #########################
        # before max_iou_idx -> (Batch_size, S, S, 1) Eg [[0], [1], [0], [0]]
        # after repeating max_iou_idx -> (Batch_size, S, S, B)
        # Eg. [[0, 0], [1, 1], [0, 0], [0, 0]] assuming B = 2
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = (torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx)
                   .to(preds.device))
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()

        # obj_indicator -> (Batch_size, S, S, 1)
        obj_indicator = targets[..., 4:5]

        # Loss definitions start from here

        #######################
        # Classification Loss #
        #######################
        cls_target = targets[..., 5 * self.B:]
        cls_preds = preds[..., 5 * self.B:]
        cls_mse = (cls_preds - cls_target) ** 2
        # Only keep losses from cells with object assigned
        cls_mse = (obj_indicator * cls_mse).sum()

        ######################################################
        # Objectness Loss (For responsible predictor boxes ) #
        ######################################################
        # indicator is now object_cells * is_best_box
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        obj_mse = (pred_boxes[..., 4] - max_iou_val) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        #####################
        # Localization Loss #
        #####################
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        # Only keep losses from boxes of cells with object assigned
        # and that box which is the responsible predictor
        x_mse = (is_max_box_obj_indicator * x_mse).sum()

        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
        h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()

        #################################################
        # Objectness Loss
        # For boxes of cells assigned with object that
        # aren't responsible predictor boxes
        # and for boxes of cell not assigned with object
        #################################################
        no_object_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()

        ##############
        # Total Loss #
        ##############
        loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_mse + obj_mse
        loss += self.lambda_noobj*no_obj_mse
        loss = loss / batch_size
        return loss
