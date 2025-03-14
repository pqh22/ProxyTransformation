import os

from mmengine.dist import master_only
from mmengine.visualization import Visualizer

from embodiedscan.registry import VISUALIZERS

try:
    import open3d as o3d

    from embodiedscan.visualization.utils import _9dof_to_box, nms_filter
except ImportError:
    o3d = None


@VISUALIZERS.register_module()
class EmbodiedScanBaseVisualizer(Visualizer):
    """EmbodiedScan Base Visualizer. Method to visualize 3D scenes and Euler
    boxes.

    Args:
        name (str): Name of the visualizer. Defaults to 'visualizer'.
        save_dir (str, optional): Directory to save visualizations.
            Defaults to None.
        vis_backends (list[ConfigType], optional):
            List of visualization backends to use. Defaluts to None.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 save_dir: str = None,
                 vis_backends=None) -> None:
        super().__init__(name=name,
                         vis_backends=vis_backends,
                         save_dir=save_dir)

        if o3d is None:
            raise ImportError('Please install open3d.')

    @staticmethod
    def get_root_dir(img_path):
        """Get the root directory of the dataset."""
        if 'posed_images' in img_path:
            return img_path.split('posed_images')[0]
        if 'sequence' in img_path:
            return img_path.split('sequence')[0]
        if 'matterport_color_images' in img_path:
            return img_path.split('matterport_color_images')[0]
        raise ValueError('Custom datasets are not supported.')

    @staticmethod
    def get_ply(root_dir, scene_name):
        """Get the path of the ply file."""
        s = scene_name.split('/')
        if len(s) == 2:
            dataset, region = s
        else:
            dataset, building, region = s
        if dataset == 'scannet':
            filepath = os.path.join('/cluster/nvme9a/embodied_data/scannet_raw_anno', 'scans', region,
                                    f'{region}_vh_clean_2.ply') # 统一格式
        elif dataset == '3rscan':
            filepath = os.path.join(root_dir, 'mesh.refined.v2.obj')
        elif dataset == 'matterport3d':
            filepath = os.path.join(root_dir, 'region_segmentations',
                                    f'{region}.ply')
        else:
            raise NotImplementedError
        return filepath

    @master_only
    def visualize_scene(self,
                        data_samples,
                        class_filter=None,
                        nms_args=dict(iou_thr=0.15,
                                      score_thr=0.075,
                                      topk_per_class=10),
                        pred_path='temp_dir'):
        """Visualize the 3D scene with 3D boxes.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]):
                The output of the model.
            class_filter (int, optional): Class filter for visualization.
                Default to None to show all classes.
            nms_args (dict): NMS arguments for filtering boxes.
                Defaults to dict(iou_thr = 0.15,
                                 score_thr = 0.075,
                                 topk_per_class = 10).
        """
        assert len(data_samples) == 1
        data_sample = data_samples[0]

        metainfo = data_sample.metainfo
        pred = data_sample.pred_instances_3d
        gt = data_sample.eval_ann_info

        if not hasattr(pred, 'labels_3d'):
            assert gt['gt_labels_3d'].shape[0] == 1
            gt_label = gt['gt_labels_3d'][0].item()
            _ = pred.bboxes_3d.tensor.shape[0]
            pseudo_label = pred.bboxes_3d.tensor.new_ones(_, ) * gt_label
            pred.labels_3d = pseudo_label
        pred_box, pred_label = nms_filter(pred, **nms_args)

        root_dir = self.get_root_dir(metainfo['img_path'][0])
        ply_file = self.get_ply(root_dir, metainfo['scan_id'])
        axis_align_matrix = metainfo['axis_align_matrix']
        mesh = o3d.io.read_triangle_mesh(ply_file, True)
        mesh.transform(axis_align_matrix) # <class 'open3d.cuda.pybind.geometry.TriangleMesh'>
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame() # <class 'open3d.cuda.pybind.geometry.TriangleMesh'>
        boxes = []
        # pred 3D box
        n = pred_box.shape[0]
        for i in range(n):
            box = pred_box[i]
            label = pred_label[i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(255, 0, 0))
            boxes.append(box_geo)
        # gt 3D box
        m = gt['gt_bboxes_3d'].tensor.shape[0]
        for i in range(m):
            box = gt['gt_bboxes_3d'].tensor[i]
            label = gt['gt_labels_3d'][i]
            if class_filter is not None and label != class_filter:
                continue
            box_geo = _9dof_to_box(box, color=(0, 255, 0)) # <class 'open3d.cuda.pybind.geometry.OrientedBoundingBox'>
            boxes.append(box_geo)
        # o3d.visualization.draw_geometries([mesh, frame] + boxes)
        import re
        import json
        if 'scannet' in ply_file:
            scene_name = os.path.basename(os.path.dirname(ply_file))
            path = os.path.join(pred_path, 'scannet', f'{scene_name}')
            if not os.path.exists(path):
                os.makedirs(path)
            # make a new target dir
            existing_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            target_dirs = [d for d in existing_dirs if re.match(r'target\d+', d)]
            max_i = 0
            for dir_name in target_dirs:
                num = int(re.search(r'\d+', dir_name).group())
                if num > max_i:
                    max_i = num
            new_target_dir = f'target{max_i + 1}'
            new_target_path = os.path.join(path, new_target_dir)
            os.makedirs(new_target_path)
            # save mesh, frame, boxes, annotations
            o3d.io.write_triangle_mesh(os.path.join(new_target_path, 'mesh.ply'), mesh)
            o3d.io.write_triangle_mesh(os.path.join(new_target_path, 'frame.ply'), frame)
            for i in range(len(boxes)):
                points = boxes[i].get_box_points()
                pcd = o3d.geometry.PointCloud()
                pcd.points = points
                if i == len(boxes) - 1:
                    o3d.io.write_point_cloud(os.path.join(new_target_path, f'box_gt.ply'), pcd)
                else:
                    o3d.io.write_point_cloud(os.path.join(new_target_path, f'box_pred{i}.ply'), pcd)
            annotation_dict = {
                "description": data_sample.text,
                "labels": data_sample.gt_instances_3d.labels_3d.item()
            }
            ann_path = os.path.join(new_target_path, 'annotation.json')
            with open(ann_path, 'w') as f:
                json.dump(annotation_dict, f)
        else:
            pass

        return 

# train gt visualization
@master_only
def visualize_scene_train_gt(data_samples,
                            class_filter=None,
                            pred_path='temp_dir'):
    """Visualize the 3D scene with 3D boxes.

    Args:
        data_samples (list[:obj:`Det3DDataSample`]):
            The output of the model.
        class_filter (int, optional): Class filter for visualization.
            Default to None to show all classes.
    """
    assert len(data_samples) == 1
    data_sample = data_samples[0]

    metainfo = data_sample.metainfo
    gt_bboxes_3d = data_sample.gt_instances_3d.bboxes_3d
    gt_labels_3d = data_sample.gt_instances_3d.labels_3d

    root_dir = get_root_dir(metainfo['img_path'][0])
    ply_file = get_ply(root_dir, metainfo['scan_id'])
    axis_align_matrix = metainfo['axis_align_matrix']
    mesh = o3d.io.read_triangle_mesh(ply_file, True)
    mesh.transform(axis_align_matrix) # <class 'open3d.cuda.pybind.geometry.TriangleMesh'>
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame() # <class 'open3d.cuda.pybind.geometry.TriangleMesh'>
    boxes = []

    # gt 3D box
    m = gt_bboxes_3d.tensor.shape[0]
    for i in range(m):
        box = gt_bboxes_3d.tensor[i]
        label = gt_labels_3d[i]
        if class_filter is not None and label != class_filter:
            continue
        box_geo = _9dof_to_box(box, color=(0, 255, 0)) # <class 'open3d.cuda.pybind.geometry.OrientedBoundingBox'>
        boxes.append(box_geo)
    # o3d.visualization.draw_geometries([mesh, frame] + boxes)
    import re
    import json
    if 'scannet' in ply_file:
        scene_name = os.path.basename(os.path.dirname(ply_file))
        path = os.path.join(pred_path, 'scannet', f'{scene_name}')
        if not os.path.exists(path):
            os.makedirs(path)
        # make a new target dir
        existing_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        target_dirs = [d for d in existing_dirs if re.match(r'target\d+', d)]
        max_i = 0
        for dir_name in target_dirs:
            num = int(re.search(r'\d+', dir_name).group())
            if num > max_i:
                max_i = num
        new_target_dir = f'target{max_i + 1}'
        new_target_path = os.path.join(path, new_target_dir)
        os.makedirs(new_target_path)
        # save mesh, frame, boxes, annotations
        o3d.io.write_triangle_mesh(os.path.join(new_target_path, 'mesh.ply'), mesh)
        o3d.io.write_triangle_mesh(os.path.join(new_target_path, 'frame.ply'), frame)
        # when training, only save the last gt box
        points = boxes[-1].get_box_points()
        pcd = o3d.geometry.PointCloud()
        pcd.points = points
        o3d.io.write_point_cloud(os.path.join(new_target_path, 'box_gt.ply'), pcd)
        annotation_dict = {
            "description": data_sample.text,
        }
        ann_path = os.path.join(new_target_path, 'annotation.json')
        with open(ann_path, 'w') as f:
            json.dump(annotation_dict, f)
    else:
        pass

    return 


def get_root_dir(img_path):
    """Get the root directory of the dataset."""
    if 'posed_images' in img_path:
        return img_path.split('posed_images')[0]
    if 'sequence' in img_path:
        return img_path.split('sequence')[0]
    if 'matterport_color_images' in img_path:
        return img_path.split('matterport_color_images')[0]
    raise ValueError('Custom datasets are not supported.')


def get_ply(root_dir, scene_name):
    """Get the path of the ply file."""
    s = scene_name.split('/')
    if len(s) == 2:
        dataset, region = s
    else:
        dataset, building, region = s
    if dataset == 'scannet':
        filepath = os.path.join('/cluster/nvme9a/embodied_data/scannet_raw_anno', 'scans', region,
                                f'{region}_vh_clean_2.ply') # 统一格式
    elif dataset == '3rscan':
        filepath = os.path.join(root_dir, 'mesh.refined.v2.obj')
    elif dataset == 'matterport3d':
        filepath = os.path.join(root_dir, 'region_segmentations',
                                f'{region}.ply')
    else:
        raise NotImplementedError
    return filepath