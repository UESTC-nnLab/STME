#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""训练时计算损失函数"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # 预测框和真实框一致
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        # 第一个2是预测框的中心点相较于该特征点的偏移情况，第二个2是预测框的宽高相较于对数指数的参数
        # 取左上点(偏移点减去宽高的一半)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        # 取右下点(偏移点加上宽高的一半)
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        # torch.prod(input), 返回input全部元素的乘积,dim代表第一维度
        # 预测框的面积
        area_p = torch.prod(pred[:, 2:], 1)
        # 真实框的面积
        area_g = torch.prod(target[:, 2:], 1)
        # 只有有交集才会tl<br
        en = (tl < br).type(tl.type()).prod(dim=1)
        # 交集的面积
        area_i = torch.prod(br - tl, 1) * en
        # 并集的面积
        area_u = area_p + area_g - area_i
        # iou的计算公式
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            # 框住真实框和预测框的最小框的左上角
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            # 框住真实框和预测框的最小框的右下角
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            # 最小框的面积
            area_c = torch.prod(c_br - c_tl, 1)
            # giou计算公式, clamp函数用于约束返回值到A和B之间，若value小于min，则返回min；若value大于max，则返回max
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class YOLOLoss(nn.Module):    
    def __init__(self, num_classes, strides=[8]):
        """
        :param num_classes: 类别数
        :param strides: 框在特征层恢复到原图的参数
        """
        super().__init__()
        self.num_classes        = num_classes
        self.strides            = strides

        self.bcewithlog_loss    = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss           = IOUloss(reduction="none")
        # 三个torch列表
        self.grids              = [torch.zeros(1)] * len(strides)

    def forward(self, inputs, labels=None):
        outputs             = []
        x_shifts            = [] # 特征点横坐标
        y_shifts            = [] # 特征点纵坐标
        expanded_strides    = [] # 存储网格中每个特征点的strides

        #-----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5] 每个特征点的4 + 1 + 20
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400] y_shifts是一样的
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        # expanded_strides [[batch_size, 400]
        #                   [batch_size, 1600]
        #                   [batch_size, 6400]]
        #-----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            output, grid = self.get_output_and_grid(output, k, stride)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))

    def get_output_and_grid(self, output, k, stride):
        """
        :param output: [batch_size, num_classes + 5, 20, 20]
        :param k: 0到2,表示哪层特征层
        :param stride: 框在特征层恢复到原图的参数
        :return:output,grid(特征层网格)
        """
        grid            = self.grids[k]
        # 特征层的宽高
        hsize, wsize    = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            # 生成xv为x, yv为y组成的网格, xv,yv为[20,20]
            yv, xv          = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            # 堆叠新增的维度在第二维度[xv[0][0],yx[0][0]],堆叠第二维为特征点坐标
            grid            = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k]   = grid
        # 将hsize和wsize的维度合并
        grid                = grid.view(1, -1, 2)

        # output变为[batch_size, 400, num_classes + 5](第零层)
        output              = output.flatten(start_dim=2).permute(0, 2, 1)
        # 特征点恢复到原图的坐标
        output[..., :2]     = (output[..., :2] + grid) * stride
        # 特征点预测框的宽高恢复到原图上预测框的宽高
        output[..., 2:4]    = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        """
        :param x_shifts: 特征点横坐标
        :param y_shifts: 特征点纵坐标
        :param expanded_strides: 存储网格中每个特征点的strides
        :param labels: 真实标签 [batch, ....]
        :param outputs:[batch, 400+1600+6400, num_classes + 5]
        :return: loss
        """
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 4] 预测框回归参数
        #-----------------------------------------------#
        bbox_preds  = outputs[:, :, :4]  
        #-----------------------------------------------#
        #   [batch, n_anchors_all, 1] 判断是否有检测物
        #-----------------------------------------------#
        obj_preds   = outputs[:, :, 4:5]
        #-----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]  种类概率
        #-----------------------------------------------#
        cls_preds   = outputs[:, :, 5:]  

         # 获取总的锚框个数
        total_num_anchors   = outputs.shape[1]
        #-----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        #-----------------------------------------------#
        x_shifts            = torch.cat(x_shifts, 1)
        y_shifts            = torch.cat(y_shifts, 1)
        expanded_strides    = torch.cat(expanded_strides, 1)

        # 分类的真实值
        cls_targets = []
        # 回归的真实值
        reg_targets = []
        # 存在的真实值
        obj_targets = []
        # fg代表foreground, 特征框是否对应真实框
        fg_masks    = []

        num_fg  = 0.0  # foreground个数
        for batch_idx in range(outputs.shape[0]):
            # gt表示grand truth, num_gt表示真实物体的个数
            num_gt          = len(labels[batch_idx])
            if num_gt == 0:  # 没有真实物体
                cls_target  = outputs.new_zeros((0, self.num_classes)) # [20]
                reg_target  = outputs.new_zeros((0, 4)) # [4]
                obj_target  = outputs.new_zeros((total_num_anchors, 1)) # [400+1600+6400, 1]
                fg_mask     = outputs.new_zeros(total_num_anchors).bool() # [400+1600+6400]
            else:
                #-----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, num_classes]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                #-----------------------------------------------#
                # 每张图片真实框
                gt_bboxes_per_image     = labels[batch_idx][..., :4]
                # 每张图片的真实类别
                gt_classes              = labels[batch_idx][..., 4]
                # 每张图片的回归预测值
                bboxes_preds_per_image  = bbox_preds[batch_idx]
                # 每张图片的分类预测值
                cls_preds_per_image     = cls_preds[batch_idx]
                # 每张图片的检测预测值
                obj_preds_per_image     = obj_preds[batch_idx]
                #  特征点对应的物品种类, 特征框是否对应真实框[n_anchors_all], 对应的ious[fg_mask], 返回得到种类的索引, 正样本个数
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments( 
                    num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts, 
                )
                torch.cuda.empty_cache() # 释放显存
                num_fg      += num_fg_img
                cls_target  = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target  = fg_mask.unsqueeze(-1)
                reg_target  = gt_bboxes_per_image[matched_gt_inds] # 可以筛选出是哪些特征框
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks    = torch.cat(fg_masks, 0)

        num_fg      = max(num_fg, 1)
        loss_iou    = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj    = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls    = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight  = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls

        return loss / num_fg

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        #-------------------------------------------------------#
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt)

        #-------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes] 这几个参数是过滤掉无效框的
        #   obj_preds_              [fg_mask, 1]
        #-------------------------------------------------------#
        bboxes_preds_per_image  = bboxes_preds_per_image[fg_mask]
        cls_preds_              = cls_preds_per_image[fg_mask]
        obj_preds_              = obj_preds_per_image[fg_mask]
        # 能对应上真实框的预测框个数
        num_in_boxes_anchor     = bboxes_preds_per_image.shape[0]

        #-------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask] 得到iou值
        #-------------------------------------------------------#
        pair_wise_ious      = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        
        #-------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        #-------------------------------------------------------#
        # 预测有对象的分类预测值
        cls_preds_          = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        # 真实的分类
        gt_cls_per_image    = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        # 计算分类loss
        pair_wise_cls_loss  = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        del cls_preds_

        # 计算cost代价函数,其中100000.0*(~is_in_boxes_and_center )指 正样本取反，
        # 剩下的都是负样本，一方面需要最小化正样本的损失，同时意味着需要最大化负样本的损失。
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        # 正样本个数, 特征点对应的物品种类, 对应的ious[fg_mask], 返回得到种类的索引
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg
    
    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        """这里的算法和Iouloss里面的算法一致,不多赘述"""
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt, center_radius = 2.5):
        """判断特征点（框）是否落在真实物体上"""
        #-------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #-------------------------------------------------------#
        expanded_strides_per_image  = expanded_strides[0]
        # 特征点的x,y是边缘的，移动0.5是右下角移动到中间。
        x_centers_per_image         = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image         = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        #-------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all] bbox:xywh
        #-------------------------------------------------------#
        # 得到每个图片真实框的左坐标
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        # 得到每个图片真实框的右坐标
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(1, total_num_anchors)
        # 得到每个图片真实框的上坐标（越高的地方坐标越小）
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)
        # 得到每个图片真实框的下坐标
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(1, total_num_anchors)

        #-------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4] 真实框的上下左右坐标和特征点中心坐标的上下左右的差
        #-------------------------------------------------------#
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        #-------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all]
        #   is_in_boxes_all [n_anchors_all]
        #-------------------------------------------------------#
        # 特征点是否对应一个真实物体
        is_in_boxes     = bbox_deltas.min(dim=-1).values > 0.0
        # 只要有一个true证明在num_gt中至少对应一个真实物体
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # 离真实框的中心距离 2.5*strides 的边界上下左右坐标, [num_gt, n_anchors_all] bbox:xywh
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors) + center_radius * expanded_strides_per_image.unsqueeze(0)

        #-------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        #-------------------------------------------------------#
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas       = torch.stack([c_l, c_t, c_r, c_b], 2)

        #-------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all]
        #   is_in_centers_all   [n_anchors_all]
        #-------------------------------------------------------#
        is_in_centers       = center_deltas.min(dim=-1).values > 0.0  # 特征点是否在真实物体中心所在的半径范围内
        is_in_centers_all   = is_in_centers.sum(dim=0) > 0 # 特征点至少在一个真实物体中心所在的半径范围内

        #-------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        #-------------------------------------------------------#
        is_in_boxes_anchor      = is_in_boxes_all | is_in_centers_all # 至少满足一个条件的特征框(至少对应一个)
        is_in_boxes_and_center  = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]  # 同时满足两个条件
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        """
        :param cost: 代价函数[num_gt, fg_mask]
        :param pair_wise_ious: 每对iou值[num_gt, fg_mask]
        :param gt_classes: 真实物体种类 [num_gt]
        :param num_gt: 真实框的个数
        :param fg_mask:[n_anchors_all] 特征框是否对应真实框
        :return:
        """
        #-------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        #   matching_matrix     [num_gt, fg_mask]
        #-------------------------------------------------------#
        matching_matrix         = torch.zeros_like(cost)

        #------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        #------------------------------------------------------------#
        n_candidate_k           = min(10, pair_wise_ious.size(1)) # 选取小的作为个数
        topk_ious, _            = torch.topk(pair_wise_ious, n_candidate_k, dim=1) # 获得ious中前n_candidate_k个最大值
        dynamic_ks              = torch.clamp(topk_ious.sum(1).int(), min=1) # 然后求和，判断应该有多少点用于该框预测(且不能小于一)
        
        for gt_idx in range(num_gt):
            #------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            #------------------------------------------------------------#
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            #  将这些点通过匹配矩阵保存下来
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        #------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        #------------------------------------------------------------#
        anchor_matching_gt = matching_matrix.sum(0) # 锚框匹配真实框
        if (anchor_matching_gt > 1).sum() > 0:
            #------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。所有行中匹配数大于1的列（行为真实框,列为锚框）,即一个锚框对应了多个真实框,要选出最适合那个。
            #------------------------------------------------------------#
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0 # 将其他的匹配归零
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0 # 只保留最合适的那个框
        #------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        #------------------------------------------------------------#
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0 # 对应真实框的预测框(一一对应的)
        num_fg          = fg_mask_inboxes.sum().item() 

        #------------------------------------------------------------#
        #   对fg_mask进行更新
        #------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        #------------------------------------------------------------#
        #   获得特征点对应的物品种类
        #------------------------------------------------------------#
        matched_gt_inds     = matching_matrix[:, fg_mask_inboxes].argmax(0) # 竖着比较返回位置
        gt_matched_classes  = gt_classes[matched_gt_inds]
        # pre_ious_this_matching [fg_mask] 对应的ious值
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
