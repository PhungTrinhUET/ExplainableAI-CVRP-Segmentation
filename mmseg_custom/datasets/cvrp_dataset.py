# Dataset class cho CVRP — đăng ký với MMSegmentation
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class CVRPDataset(BaseSegDataset):
    """
    CVRP Rice Panicle Segmentation Dataset.
    Annotation: 0 = background, 1 = panicle (bông lúa)
    """
    METAINFO = dict(
        classes=('background', 'panicle'),
        palette=[[0, 0, 0], [128, 0, 0]]
    )

    def __init__(self, img_suffix='.jpg', seg_map_suffix='.png', **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )