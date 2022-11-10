import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * (
                (1 - target) * torch.log((1 - output))
            )
        else:
            output = torch.clamp(output, min=1e-3, max=1 - 1e-3)
            bce = target * torch.log(output) + (1 - target) * torch.log((1 - output))
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, "predict & target shape do not match"
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class DSC_loss(nn.Module):
    def __init__(self):
        super(DSC_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / (
            (pred + target).sum(1) + self.epsilon
        )
        return 1 - DSC.sum() / float(batch_num)


class DICELossMultiClass(nn.Module):
    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):
        print("output.size, mask.size:", output.size(), mask.size())
        num_classes = output.size(1)
        dice_eso = 0
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            mask = torch.squeeze(mask[:, i, :, :], 1)

            num = probs * mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = mask * mask
            # print(den2.size())
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # dice_eso = dice[:, 1:]
            dice_eso += dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss(nn.Module):
    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):
        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        den1 = probs * probs
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        den2 = mask * mask
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class BCE_Dice_Loss(nn.Module):
    def __init__(self):
        super(BCE_Dice_Loss, self).__init__()
        self.ce = BCELoss()
        self.dc = DICELoss()

    def forward(self, input, target):
        dc_loss = self.dc(input, target)
        ce_loss = self.ce(input, target)
        result = ce_loss + dc_loss
        return result


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()
        # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = 1.1 * num_pos / num_total
        # target pixel = 1 -> weight beta
        # target pixel = 0 -> weight 1-beta
        weights = alpha * pos + beta * neg
        return F.binary_cross_entropy_with_logits(input, target, weights)


class DICELoss_LV(nn.Module):
    def __init__(self):
        super(DICELoss_LV, self).__init__()

    def forward(self, output, mask):
        intersection = output * mask
        intersection = torch.sum(intersection)

        den1 = output * output
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - dice_eso
        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedDiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, mask):
        output = self.sigmoid(outputs)
        intersection = output * mask
        intersection = torch.sum(intersection)

        den1 = output * output
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - dice_eso
        return loss


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, mask):
        dice = DICELoss_LV()
        Lsum = 0
        # print(output.size())
        # print(mask.size())
        for batch in range(output.size(0)):

            lv_loss = dice(output[batch], mask[batch])
            alpha = 0.5
            beta = 0.5
            L2 = 0

            for i in range(output.size(1)):
                L2 += dice(output[batch, i], mask[batch, i]) * dice(
                    output[batch, i], mask[batch, i]
                )

            Ltotal = alpha * lv_loss + beta * math.sqrt(L2)

            Lsum += Ltotal

        return Lsum / output.size(0)
