import torch
from mmcv.ops.nms import batched_nms
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (DepthInstance3DBoxes,
                                          LiDARInstance3DBoxes,
                                          rotation_3d_in_axis)
from mmdet3d.models.builder import build_loss
from mmdet3d.models.losses import chamfer_distance
from mmdet.core import multi_apply
from mmdet.models import HEADS
from .vote_head import VoteHead


@HEADS.register_module()
class SSD3DHead(VoteHead):
    r"""Bbox head of `3DSSD <https://arxiv.org/abs/2002.10187>`_.

    Args:
        num_classes (int): The number of class.
        bbox_coder (:obj:`BaseBBoxCoder`): Bbox coder for encoding and
            decoding boxes.
        in_channels (int): The number of input feature channel.
        train_cfg (dict): Config for training.
        test_cfg (dict): Config for testing.
        vote_module_cfg (dict): Config of VoteModule for point-wise votes.
        vote_aggregation_cfg (dict): Config of vote aggregation layer.
        pred_layer_cfg (dict): Config of classfication and regression
            prediction layers.
        conv_cfg (dict): Config of convolution in prediction layer.
        norm_cfg (dict): Config of BN in prediction layer.
        act_cfg (dict): Config of activation in prediction layer.
        objectness_loss (dict): Config of objectness loss.
        center_loss (dict): Config of center loss.
        dir_class_loss (dict): Config of direction classification loss.
        dir_res_loss (dict): Config of direction residual regression loss.
        size_res_loss (dict): Config of size residual regression loss.
        corner_loss (dict): Config of bbox corners regression loss.
        vote_loss (dict): Config of candidate points regression loss.
    """

    def __init__(self,
                 num_classes,
                 bbox_coder,
                 in_channels=256,
                 use_anchor_free=True,
                 train_cfg=None,
                 test_cfg=None,
                 vote_module_cfg=None,
                 vote_aggregation_cfg=None,
                 pred_layer_cfg=None,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 use_orig_vote_loss=False,
                 assignment_strategy='pts_in_bbox',
                 objectness_loss=None,
                 center_loss=None,
                 dir_class_loss=None,
                 dir_res_loss=None,
                 size_class_loss=None,
                 size_res_loss=None,
                 corner_loss=None,
                 vote_loss=None,
                 velocity_loss=None):
        self.extra_reg_dim = 2 if velocity_loss is not None else 0
        super(SSD3DHead, self).__init__(
            num_classes,
            bbox_coder,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            vote_module_cfg=vote_module_cfg,
            vote_aggregation_cfg=vote_aggregation_cfg,
            pred_layer_cfg=pred_layer_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            objectness_loss=objectness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=size_class_loss,
            size_res_loss=size_res_loss,
            semantic_loss=None)

        self.use_anchor_free = use_anchor_free
        self.corner_loss = build_loss(corner_loss)
        self.vote_loss = build_loss(vote_loss)
        if velocity_loss is not None:
            self.velocity_loss = build_loss(velocity_loss)
        else:
            self.velocity_loss = None

        self.num_candidates = vote_module_cfg['num_points']
        self.use_orig_vote_loss = use_orig_vote_loss
        self.assignment_strategy = assignment_strategy

    def _get_cls_out_channels(self):
        """Return the channel number of classification outputs."""
        # Class numbers (k) + objectness (1)
        return self.num_classes

    def _get_reg_out_channels(self):
        """Return the channel number of regression outputs."""
        if self.num_sizes == 0:
            # Bbox classification and regression
            # (center residual (3), size regression (3)
            # heading class+residual (num_dir_bins*2)),
            return 3 + 3 + self.num_dir_bins * 2 + self.extra_reg_dim
        else:
            return 3 + self.num_dir_bins * 2 + self.num_sizes * 4 +\
                self.extra_reg_dim

    def _extract_input(self, feat_dict):
        """Extract inputs from features dictionary.

        Args:
            feat_dict (dict): Feature dict from backbone.

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
            torch.Tensor: Indices of input points.
        """
        seed_points = feat_dict['sa_xyz'][-1]
        seed_features = feat_dict['sa_features'][-1]
        seed_indices = feat_dict['sa_indices'][-1]

        return seed_points, seed_features, seed_indices

    def loss(self,
             bbox_preds,
             points,
             gt_bboxes_3d,
             gt_labels_3d,
             pts_semantic_mask=None,
             pts_instance_mask=None,
             img_metas=None,
             gt_bboxes_ignore=None):
        """Compute loss.

        Args:
            bbox_preds (dict): Predictions from forward of SSD3DHead.
            points (list[torch.Tensor]): Input points.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each sample.
            gt_labels_3d (list[torch.Tensor]): Labels of each sample.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise
                semantic mask.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise
                instance mask.
            img_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses of 3DSSD.
        """
        targets = self.get_targets(points, gt_bboxes_3d, gt_labels_3d,
                                   pts_semantic_mask, pts_instance_mask,
                                   bbox_preds)
        (vote_targets, center_targets, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, mask_targets, centerness_targets,
         corner3d_targets, extra_targets, vote_mask, positive_mask,
         negative_mask, centerness_weights, box_loss_weights,
         one_hot_size_targets, heading_res_loss_weight, proposal_recall,
         vote_recall) = targets

        # calculate centerness loss
        centerness_loss = self.objectness_loss(
            bbox_preds['obj_scores'].transpose(2, 1),
            centerness_targets,
            weight=centerness_weights)

        # calculate center loss
        center_loss = self.center_loss(
            bbox_preds['center_offset'],
            center_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate direction class loss
        dir_class_loss = self.dir_class_loss(
            bbox_preds['dir_class'].transpose(1, 2),
            dir_class_targets,
            weight=box_loss_weights)

        # calculate direction residual loss
        dir_res_loss = self.dir_res_loss(
            bbox_preds['dir_res_norm'],
            dir_res_targets.unsqueeze(-1).repeat(1, 1, self.num_dir_bins),
            weight=heading_res_loss_weight)

        # calculate size class loss
        if size_class_targets is not None:
            size_class_loss = self.size_class_loss(
                bbox_preds['size_class'].transpose(2, 1),
                size_class_targets,
                weight=box_loss_weights)
        else:
            size_class_loss = dir_res_loss * 0

        # calculate size residual loss
        if one_hot_size_targets is not None:
            size_res_pred = bbox_preds['size_res'] * \
                one_hot_size_targets.unsqueeze(-1)
            size_res_pred = size_res_pred.sum(dim=2)
        else:
            size_res_pred = bbox_preds['size_res']
        size_loss = self.size_res_loss(
            size_res_pred,
            size_res_targets,
            weight=box_loss_weights.unsqueeze(-1))

        # calculate corner loss
        one_hot_dir_class_targets = dir_class_targets.new_zeros(
            bbox_preds['dir_class'].shape)
        one_hot_dir_class_targets.scatter_(2, dir_class_targets.unsqueeze(-1),
                                           1)
        pred_bbox_dict = dict(
            center=bbox_preds['center'],
            dir_res=bbox_preds['dir_res'],
            dir_class=one_hot_dir_class_targets,
            size_res=bbox_preds['size_res'])
        if size_class_targets is not None:
            pred_bbox_dict['size_class'] = bbox_preds['size_class']

        pred_bbox3d = self.bbox_coder.decode(pred_bbox_dict)
        pred_bbox3d = pred_bbox3d.reshape(-1, pred_bbox3d.shape[-1])
        pred_bbox3d = img_metas[0]['box_type_3d'](
            pred_bbox3d.clone(),
            box_dim=pred_bbox3d.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=(0.5, 0.5, 0.5))
        pred_corners3d = pred_bbox3d.corners.reshape(-1, 8, 3)
        corner_loss = self.corner_loss(
            pred_corners3d,
            corner3d_targets.reshape(-1, 8, 3),
            weight=box_loss_weights.view(-1, 1, 1))

        # calculate vote loss
        if not self.use_orig_vote_loss:
            vote_loss = self.vote_loss(
                bbox_preds['vote_offset'].transpose(1, 2),
                vote_targets,
                weight=vote_mask.unsqueeze(-1))
        else:
            vote_loss = self.vote_module.get_loss(bbox_preds['seed_points'],
                                                  bbox_preds['vote_points'],
                                                  bbox_preds['seed_indices'],
                                                  vote_mask, vote_targets)

        losses = dict(
            centerness_loss=centerness_loss,
            center_loss=center_loss,
            dir_class_loss=dir_class_loss,
            dir_res_loss=dir_res_loss,
            size_class_loss=size_class_loss,
            size_res_loss=size_loss,
            corner_loss=corner_loss,
            vote_loss=vote_loss)

        if self.velocity_loss is not None:
            veloc_loss = self.velocity_loss(
                bbox_preds['extra_reg'],
                extra_targets,
                weight=box_loss_weights.unsqueeze(-1))
            losses['veloc_loss'] = veloc_loss

        losses['proposal_recall'] = vote_loss.new_tensor(
            proposal_recall).mean()
        losses['vote_recall'] = vote_loss.new_tensor(vote_recall).mean()
        losses['proposal_ratio'] = positive_mask.sum() / \
            float(positive_mask.shape[1] * positive_mask.shape[0])
        losses['vote_ratio'] = (vote_mask > 0).sum() / \
            float(vote_mask.shape[1] * vote_mask.shape[0])
        return losses

    def get_targets(self,
                    points,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,
                    bbox_preds=None):
        """Generate targets of ssd3d head.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.
            bbox_preds (torch.Tensor): Bounding box predictions of ssd3d head.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        # find empty example
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]

        aggregated_points = [
            bbox_preds['aggregated_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        seed_points = [
            bbox_preds['seed_points'][i, :self.num_candidates].detach()
            if self.num_candidates != -1 else bbox_preds['seed_points'][i]
            for i in range(len(gt_labels_3d))
        ]

        (vote_targets, center_targets, size_class_targets, size_res_targets,
         dir_class_targets, dir_res_targets, mask_targets, extra_targets,
         centerness_targets, corner3d_targets, vote_mask, positive_mask,
         negative_mask, proposal_recall, vote_recall) = multi_apply(
             self.get_targets_single, points, gt_bboxes_3d, gt_labels_3d,
             pts_semantic_mask, pts_instance_mask, aggregated_points,
             seed_points)

        center_targets = torch.stack(center_targets)
        positive_mask = torch.stack(positive_mask)
        negative_mask = torch.stack(negative_mask)
        dir_class_targets = torch.stack(dir_class_targets)
        dir_res_targets = torch.stack(dir_res_targets)
        size_res_targets = torch.stack(size_res_targets)
        mask_targets = torch.stack(mask_targets)
        centerness_targets = torch.stack(centerness_targets).detach()
        corner3d_targets = torch.stack(corner3d_targets)
        vote_targets = torch.stack(vote_targets)
        vote_mask = torch.stack(vote_mask)

        if extra_targets[0] is not None:
            extra_targets = torch.stack(extra_targets)

        if size_class_targets[0] is not None:
            size_class_targets = torch.stack(size_class_targets)
        else:
            size_class_targets = None

        center_targets -= bbox_preds['aggregated_points']

        centerness_weights = (positive_mask +
                              negative_mask).unsqueeze(-1).repeat(
                                  1, 1, self.num_classes).float()
        centerness_weights = centerness_weights / \
            (centerness_weights.sum() + 1e-6)
        if not self.use_orig_vote_loss:
            vote_mask = vote_mask / (vote_mask.sum() + 1e-6)

        box_loss_weights = positive_mask / (positive_mask.sum() + 1e-6)

        batch_size, proposal_num = dir_class_targets.shape[:2]
        heading_label_one_hot = dir_class_targets.new_zeros(
            (batch_size, proposal_num, self.num_dir_bins))
        heading_label_one_hot.scatter_(2, dir_class_targets.unsqueeze(-1), 1)
        heading_res_loss_weight = heading_label_one_hot * \
            box_loss_weights.unsqueeze(-1)

        if self.num_sizes != 0:
            one_hot_size_targets = size_class_targets.new_zeros(
                (batch_size, proposal_num, self.num_sizes))
            one_hot_size_targets.scatter_(2, size_class_targets.unsqueeze(-1),
                                          1)
        else:
            one_hot_size_targets = None

        return (vote_targets, center_targets, size_class_targets,
                size_res_targets, dir_class_targets, dir_res_targets,
                mask_targets, centerness_targets, corner3d_targets,
                extra_targets, vote_mask, positive_mask, negative_mask,
                centerness_weights, box_loss_weights, one_hot_size_targets,
                heading_res_loss_weight, proposal_recall, vote_recall)

    def get_targets_single(self,
                           points,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,
                           aggregated_points=None,
                           seed_points=None):
        """Generate targets of ssd3d head for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.
            aggregated_points (torch.Tensor): Aggregated points from
                candidate points layer.
            seed_points (torch.Tensor): Seed points of candidate points.

        Returns:
            tuple[torch.Tensor]: Targets of ssd3d head.
        """
        assert self.bbox_coder.with_rot or pts_semantic_mask is not None
        proposal_num = aggregated_points.shape[0]
        gt_bboxes_3d = gt_bboxes_3d.to(points.device)
        valid_gt = gt_labels_3d != -1
        gt_bboxes_3d = gt_bboxes_3d[valid_gt]
        gt_labels_3d = gt_labels_3d[valid_gt]
        gt_corner3d = gt_bboxes_3d.corners

        if self.use_anchor_free:
            (center_targets, size_res_targets, dir_class_targets,
             dir_res_targets,
             extra_targets) = self.bbox_coder.encode(gt_bboxes_3d,
                                                     gt_labels_3d)
        else:
            (center_targets, size_class_targets, size_res_targets,
             dir_class_targets, dir_res_targets, extra_targets) = \
                self.bbox_coder.encode(gt_bboxes_3d, gt_labels_3d)

        if self.assignment_strategy == 'pts_in_bbox':
            points_mask, assignment = self._assign_targets_by_points_inside(
                gt_bboxes_3d, aggregated_points)
        elif self.assignment_strategy == 'chamfer_distance':
            distance1, _, assignment, _ = chamfer_distance(
                aggregated_points.unsqueeze(0),
                center_targets.unsqueeze(0),
                reduction='none')
            assignment = assignment.squeeze(0)
            euclidean_distance1 = torch.sqrt(distance1.squeeze(0) + 1e-6)
            points_mask = points.new_zeros((proposal_num), dtype=torch.long)
            points_mask[
                euclidean_distance1 < self.train_cfg['pos_distance_thr']] = 1
            points_mask = points_mask.unsqueeze(-1)

        proposal_recall = assignment[
            points_mask.sum(-1) > 0].unique().shape[0] / gt_labels_3d.shape[0]

        center_targets = center_targets[assignment]
        size_res_targets = size_res_targets[assignment]
        mask_targets = gt_labels_3d[assignment]
        dir_class_targets = dir_class_targets[assignment]
        dir_res_targets = dir_res_targets[assignment]
        corner3d_targets = gt_corner3d[assignment]
        if self.extra_reg_dim != 0:
            extra_targets = extra_targets[assignment]

        half_size_target = gt_bboxes_3d.dims / 2
        half_size_target = half_size_target[assignment]

        top_center_targets = center_targets.clone()
        if isinstance(gt_bboxes_3d, LiDARInstance3DBoxes):
            top_center_targets[:, 2] += half_size_target[:, 2]
        dist = torch.norm(aggregated_points - top_center_targets, dim=1)
        dist_mask = dist < self.train_cfg.pos_distance_thr
        positive_mask = (points_mask.max(1)[0] > 0) * dist_mask
        dist_mask = dist > self.train_cfg.neg_distance_thr
        negative_mask = (points_mask.max(1)[0] == 0) * dist_mask

        # Centerness loss targets
        canonical_xyz = aggregated_points - center_targets
        if self.bbox_coder.with_rot:
            # TODO: Align points rotation implementation of
            # LiDARInstance3DBoxes and DepthInstance3DBoxes
            canonical_xyz = rotation_3d_in_axis(
                canonical_xyz.unsqueeze(0).transpose(0, 1),
                -gt_bboxes_3d.yaw[assignment], 2).squeeze(1)
        distance_front = torch.clamp(
            half_size_target[:, 0] - canonical_xyz[:, 0], min=0)
        distance_back = torch.clamp(
            half_size_target[:, 0] + canonical_xyz[:, 0], min=0)
        distance_left = torch.clamp(
            half_size_target[:, 1] - canonical_xyz[:, 1], min=0)
        distance_right = torch.clamp(
            half_size_target[:, 1] + canonical_xyz[:, 1], min=0)
        distance_top = torch.clamp(
            half_size_target[:, 2] - canonical_xyz[:, 2], min=0)
        distance_bottom = torch.clamp(
            half_size_target[:, 2] + canonical_xyz[:, 2], min=0)

        centerness_l = torch.min(distance_front, distance_back) / torch.max(
            distance_front, distance_back)
        centerness_w = torch.min(distance_left, distance_right) / torch.max(
            distance_left, distance_right)
        centerness_h = torch.min(distance_bottom, distance_top) / torch.max(
            distance_bottom, distance_top)
        centerness_targets = torch.clamp(
            centerness_l * centerness_w * centerness_h, min=0)
        centerness_targets = centerness_targets.pow(1 / 3.0)
        centerness_targets = torch.clamp(centerness_targets, min=0, max=1)

        proposal_num = centerness_targets.shape[0]
        one_hot_centerness_targets = centerness_targets.new_zeros(
            (proposal_num, self.num_classes))
        one_hot_centerness_targets.scatter_(1, mask_targets.unsqueeze(-1), 1)
        centerness_targets = centerness_targets.unsqueeze(
            1) * one_hot_centerness_targets

        if not self.use_anchor_free:
            size_class_targets = size_class_targets[assignment]
            mean_sizes = size_res_targets.new_tensor(
                self.bbox_coder.mean_sizes).unsqueeze(0)
            pos_mean_sizes = mean_sizes[0][size_class_targets]
            size_res_targets /= pos_mean_sizes
        else:
            size_class_targets = None
        # import pdb
        # pdb.set_trace()
        # Vote loss targets
        if not self.use_orig_vote_loss:
            enlarged_gt_bboxes_3d = gt_bboxes_3d.enlarged_box(
                self.train_cfg.expand_dims_length)
            enlarged_gt_bboxes_3d.tensor[:, 2] -= \
                self.train_cfg.expand_dims_length
            vote_mask, vote_assignment = self._assign_targets_by_points_inside(
                enlarged_gt_bboxes_3d, seed_points)

            vote_targets = gt_bboxes_3d.gravity_center
            vote_targets = vote_targets[vote_assignment] - seed_points
            vote_mask = vote_mask.max(1)[0] > 0

            vote_recall = vote_assignment[vote_mask.sum(-1) > 0].unique(
            ).shape[0] / gt_labels_3d.shape[0]
        else:
            # generate votes target
            vote_targets, vote_mask = self._generate_vote_targets(
                points, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask,
                pts_instance_mask)
            vote_recall = vote_mask.sum() / (vote_mask.shape[0] * 1.0)

        return (vote_targets, center_targets, size_class_targets,
                size_res_targets, dir_class_targets, dir_res_targets,
                mask_targets, extra_targets, centerness_targets,
                corner3d_targets, vote_mask, positive_mask, negative_mask,
                proposal_recall, vote_recall)

    def get_bboxes(self, points, bbox_preds, input_metas, rescale=False):
        """Generate bboxes from sdd3d head predictions.

        Args:
            points (torch.Tensor): Input points.
            bbox_preds (dict): Predictions from sdd3d head.
            input_metas (list[dict]): Point cloud and image's meta info.
            rescale (bool): Whether to rescale bboxes.

        Returns:
            list[tuple[torch.Tensor]]: Bounding boxes, scores and labels.
        """
        # decode boxes
        sem_scores = F.sigmoid(bbox_preds['obj_scores']).transpose(1, 2)
        obj_scores = sem_scores.max(-1)[0]
        bbox3d = self.bbox_coder.decode(bbox_preds)

        batch_size = bbox3d.shape[0]
        results = list()

        for b in range(batch_size):
            if input_metas[b]['box_type_3d'] == DepthInstance3DBoxes:
                bbox_selected, score_selected, labels = \
                    super(SSD3DHead, self).multiclass_nms_single(
                        obj_scores[b], sem_scores[b], bbox3d[b],
                        points[b, ..., :3], input_metas[b])
            elif input_metas[b]['box_type_3d'] == LiDARInstance3DBoxes:
                bbox_selected, score_selected, labels = \
                    self.multiclass_nms_single(obj_scores[b], sem_scores[b],
                                               bbox3d[b], points[b, ..., :3],
                                               input_metas[b])
            else:
                raise NotImplementedError
            bbox = input_metas[b]['box_type_3d'](
                bbox_selected.clone(),
                box_dim=bbox_selected.shape[-1],
                with_yaw=self.bbox_coder.with_rot)
            results.append((bbox, score_selected, labels))

        # import numpy as np
        # np.save('pts_vis', points[0].cpu().numpy())
        # sorted_idx = (-score_selected).argsort()
        # np.save('corners', bbox.corners[sorted_idx].cpu().numpy())
        # import pdb
        # pdb.set_trace()
        return results

    def multiclass_nms_single(self, obj_scores, sem_scores, bbox, points,
                              input_meta):
        """Multi-class nms in single batch.

        Args:
            obj_scores (torch.Tensor): Objectness score of bounding boxes.
            sem_scores (torch.Tensor): semantic class score of bounding boxes.
            bbox (torch.Tensor): Predicted bounding boxes.
            points (torch.Tensor): Input points.
            input_meta (dict): Point cloud and image's meta info.

        Returns:
            tuple[torch.Tensor]: Bounding boxes, scores and labels.
        """
        num_bbox = bbox.shape[0]
        if input_meta['box_type_3d'] == LiDARInstance3DBoxes:
            origin = (0.5, 0.5, 1.0)
        else:
            origin = (0.5, 0.5, 0.5)
        bbox = input_meta['box_type_3d'](
            bbox.clone(),
            box_dim=bbox.shape[-1],
            with_yaw=self.bbox_coder.with_rot,
            origin=origin)

        if isinstance(bbox, LiDARInstance3DBoxes):
            box_idx = bbox.points_in_boxes(points)
            box_indices = box_idx.new_zeros([num_bbox + 1])
            box_idx[box_idx == -1] = num_bbox
            box_indices.scatter_add_(0, box_idx.long(),
                                     box_idx.new_ones(box_idx.shape))
            box_indices = box_indices[:-1]
            nonempty_box_mask = box_indices > 0
        elif isinstance(bbox, DepthInstance3DBoxes):
            box_indices = bbox.points_in_boxes(points)
            nonempty_box_mask = box_indices.T.sum(1) > 0
        else:
            raise NotImplementedError('Unsupported bbox type!')

        corner3d = bbox.corners
        minmax_box3d = corner3d.new(torch.Size((corner3d.shape[0], 6)))
        minmax_box3d[:, :3] = torch.min(corner3d, dim=1)[0]
        minmax_box3d[:, 3:] = torch.max(corner3d, dim=1)[0]

        bbox_classes = torch.argmax(sem_scores, -1)
        nms_selected = batched_nms(
            minmax_box3d[nonempty_box_mask][:, [0, 1, 3, 4]],
            obj_scores[nonempty_box_mask], bbox_classes[nonempty_box_mask],
            self.test_cfg.nms_cfg)[1]

        if nms_selected.shape[0] > self.test_cfg.max_output_num:
            nms_selected = nms_selected[:self.test_cfg.max_output_num]

        # filter empty boxes and boxes with low score
        scores_mask = (obj_scores >= self.test_cfg.score_thr)
        nonempty_box_inds = torch.nonzero(nonempty_box_mask).flatten()
        nonempty_mask = torch.zeros_like(bbox_classes).scatter(
            0, nonempty_box_inds[nms_selected], 1)
        selected = (nonempty_mask.bool() & scores_mask.bool())

        if self.test_cfg.per_class_proposal:
            bbox_selected, score_selected, labels = [], [], []
            for k in range(sem_scores.shape[-1]):
                bbox_selected.append(bbox[selected].tensor)
                score_selected.append(sem_scores[selected][:, k])
                labels.append(
                    torch.zeros_like(bbox_classes[selected]).fill_(k))
            bbox_selected = torch.cat(bbox_selected, 0)
            score_selected = torch.cat(score_selected, 0)
            labels = torch.cat(labels, 0)
            scores_mask = (score_selected >= self.test_cfg.score_thr)
            bbox_selected = bbox_selected[scores_mask]
            score_selected = score_selected[scores_mask]
            labels = labels[scores_mask]
        else:
            bbox_selected = bbox[selected].tensor
            score_selected = obj_scores[selected]
            labels = bbox_classes[selected]

        return bbox_selected, score_selected, labels

    def _assign_targets_by_points_inside(self, bboxes_3d, points):
        """Compute assignment by checking whether point is inside bbox.

        Args:
            bboxes_3d (BaseInstance3DBoxes): Instance of bounding boxes.
            points (torch.Tensor): Points of a batch.

        Returns:
            tuple[torch.Tensor]: Flags indicating whether each point is
                inside bbox and the index of box where each point are in.
        """
        # TODO: align points_in_boxes function in each box_structures
        num_bbox = bboxes_3d.tensor.shape[0]
        if isinstance(bboxes_3d, LiDARInstance3DBoxes):
            assignment = bboxes_3d.points_in_boxes(points).long()
            points_mask = assignment.new_zeros(
                [assignment.shape[0], num_bbox + 1])
            assignment[assignment == -1] = num_bbox
            points_mask.scatter_(1, assignment.unsqueeze(1), 1)
            points_mask = points_mask[:, :-1]
            assignment[assignment == num_bbox] = num_bbox - 1
        elif isinstance(bboxes_3d, DepthInstance3DBoxes):
            points_mask = bboxes_3d.points_in_boxes(points)
            assignment = points_mask.argmax(dim=-1)
        else:
            raise NotImplementedError('Unsupported bbox type!')

        return points_mask, assignment
