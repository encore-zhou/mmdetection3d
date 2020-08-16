import torch

from mmdet3d.core import merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .two_stage import TwoStage3DDetector


@DETECTORS.register_module()
class H3DNet(TwoStage3DDetector):
    r"""H3DNet model.

    Please refer to the `paper <https://arxiv.org/abs/2006.05682>`_
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(H3DNet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_bboxes_ignore=None):
        """Forward of training.

        Args:
            points (list[torch.Tensor]): Points of each batch.
            img_metas (list): Image metas.
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): gt bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): gt class labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): point-wise instance
                label of each batch.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict: Losses.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        x['fp_xyz'] = [x['fp_xyz_net0'][-1]]
        x['fp_features'] = [x['hd_feature']]
        x['fp_indices'] = [x['fp_indices_net0'][-1]]

        losses = dict()
        if self.with_rpn:
            rpn_outs = self.rpn_head(x, self.train_cfg.rpn.sample_mod)
            rpn_loss_inputs = (points, gt_bboxes_3d, gt_labels_3d,
                               pts_semantic_mask, pts_instance_mask, img_metas)
            rpn_losses = self.rpn_head.loss(
                rpn_outs,
                *rpn_loss_inputs,
                gt_bboxes_ignore=gt_bboxes_ignore,
                ret_target=True)
            rpn_targets = rpn_losses.pop('targets')
            losses.update(rpn_losses)
            x.update(rpn_outs)
            x['targets'] = rpn_targets
        else:
            raise NotImplementedError

        roi_losses = self.roi_head.forward_train(
            x, self.train_cfg.rcnn.sample_mod, img_metas, points, gt_bboxes_3d,
            gt_labels_3d, pts_semantic_mask, pts_instance_mask,
            gt_bboxes_ignore)
        losses.update(roi_losses)

        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Forward of testing.

        Args:
            points (list[torch.Tensor]): Points of each sample.
            img_metas (list): Image metas.
            rescale (bool): Whether to rescale results.

        Returns:
            list: Predicted 3d boxes.
        """
        points_cat = torch.stack(points)

        x = self.extract_feat(points_cat)
        x['fp_xyz'] = [x['fp_xyz_net0'][-1]]
        x['fp_features'] = [x['hd_feature']]
        x['fp_indices'] = [x['fp_indices_net0'][-1]]

        if self.with_rpn:
            rpn_outs = self.rpn_head(x, self.test_cfg.rpn.sample_mod)
            x.update(rpn_outs)
        else:
            raise NotImplementedError

        return self.roi_head.simple_test(x, self.test_cfg.rcnn.sample_mod,
                                         img_metas, points_cat)

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test with augmentation."""
        points_cat = [torch.stack(pts) for pts in points]
        feats = self.extract_feats(points_cat, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, pts_cat, img_meta in zip(feats, points_cat, img_metas):
            if self.with_rpn:
                rpn_outs = self.rpn_head(x, self.train_cfg.rpn.sample_mod)
                x.update(rpn_outs)
            else:
                raise NotImplementedError

            bbox_results = self.roi_head.simple_test(
                x, self.test_cfg.rcnn.sample_mod, img_metas, points_cat)
            aug_bboxes.append(bbox_results)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return merged_bboxes
