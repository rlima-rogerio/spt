import logging
import os
import os.path as osp
from src.datasets.itv_config import *
from src.datasets.itv import read_itv_room, ITV

DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['ITVRoom', 'MiniITVRoom']


########################################################################
#                              ITV Room                              #
########################################################################

class ITVRoom(ITV):
    """ITV dataset, for aligned room-wise prediction.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    align : `bool`
        Whether the rooms should be canonically aligned, as described in
        section 3.2 of the ITV paper
    stage : {'train', 'val', 'test', 'trainval'}
    transform : `callable`
        transform function operating on data.
    pre_transform : `callable`
        pre_transform function operating on data.
    pre_filter : `callable`
        pre_filter function operating on data.
    on_device_transform: `callable`
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    def __init__(self, *args, align=True, **kwargs):
        self.align = align
        super().__init__(*args, **kwargs)

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': [
                f'Area_{i}/{r}' for i in range(1, 9) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'val': [
                f'Area_{i}/{r}' for i in range(1, 9) if i != self.fold
                for r in ROOMS[f'Area_{i}']],
            'test': [
                f'Area_{self.fold}/{r}' for r in ROOMS[f'Area_{self.fold}']]}

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        return read_itv_room(
            raw_cloud_path, xyz=True, rgb=False, semantic=True, instance=True,
            xyz_room=True, align=self.align, is_val=True, verbose=False)

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        room_folders = self.raw_file_names_3d
        area_folders = [f'Area_{i + 1}' for i in range(6)]
        alignment_files = [
            osp.join(a, f"{a}_alignmentAngle.txt") for a in area_folders]
        return room_folders + alignment_files

    def processed_to_raw_path(self, processed_path):
        """Given a processed cloud path from `self.processed_paths`,
        return the absolute path to the corresponding raw cloud.

        Overwrite this method if your raw data does not follow the
        default structure.
        """
        # Extract useful information from <path>
        stage, hash_dir, area_id, room_id = \
            osp.splitext(processed_path)[0].split(os.sep)[-4:]
        cloud_id = osp.join(area_id, room_id)

        # Remove the tiling in the cloud_id, if any
        base_cloud_id = self.id_to_base_id(cloud_id)

        # Read the raw cloud data
        raw_ext = osp.splitext(self.raw_file_names_3d[0])[1]
        raw_path = osp.join(self.raw_dir, base_cloud_id + raw_ext)

        return raw_path


########################################################################
#                              MiniITV                               #
########################################################################

class MiniITVRoom(ITVRoom):
    """A mini version of ITV with only 2 areas per stage for
    experimentation.
    """
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
