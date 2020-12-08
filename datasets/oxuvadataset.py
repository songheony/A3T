import os
import numpy as np
import cv2
import csv
import path_config
from datasets.data import Sequence, BaseDataset, SequenceList


TASK_FIELDS = [
    'video_id', 'object_id',
    'init_frame', 'last_frame', 'xmin', 'xmax', 'ymin', 'ymax',
]


class Task(object):
    '''Describes a tracking task with optional ground-truth annotations.'''

    def __init__(self, init_time, init_rect, labels=None, last_time=None, attributes=None):
        '''Create a trasking task.
        Args:
            init_time -- Time of supervision (in frames).
            init_rect -- Rectangle dict.
            labels -- SparseTimeSeries of frame annotation dicts.
                Does not include first frame.
            last_time -- Time of last frame of interest, inclusive (optional).
                Consider frames init_time <= t <= last_time.
            attributes -- Dictionary with extra attributes.
        If last_time is None, then the last frame of labels will be used.
        '''
        self.init_time = init_time
        self.init_rect = init_rect
        if labels:
            if init_time in labels:
                raise ValueError('labels should not contain init time')
        self.labels = labels
        if last_time is None and labels is not None:
            self.last_time = labels.sorted_keys()[-1]
        else:
            self.last_time = last_time
        self.attributes = attributes or {}

    def len(self):
        return self.last_time - self.init_time + 1


class VideoObjectDict(object):
    '''Represents map video -> object -> element.
    Behaves as a dictionary with keys of (video, object) tuples.
    Example:
        for key in tracks.keys():
            print(tracks[key])
        tracks = VideoObjectDict()
        ...
        for vid in tracks.videos():
            for obj in tracks.objects(vid):
                print(tracks[(vid, obj)])
    '''

    def __init__(self, elems=None):
        if elems is None:
            self._elems = dict()
        elif isinstance(elems, VideoObjectDict):
            self._elems = dict(elems._elems)
        else:
            self._elems = dict(elems)

    def videos(self):
        return set([vid for vid, obj in self._elems.keys()])

    def objects(self, vid):
        # TODO: This is somewhat inefficient if called for all videos.
        return [obj_i for vid_i, obj_i in self._elems.keys() if vid_i == vid]

    def __len__(self):
        return len(self._elems)

    def __getitem__(self, key):
        return self._elems[key]

    def __setitem__(self, key, value):
        self._elems[key] = value

    def __delitem__(self, key):
        del self._elems[key]

    def keys(self):
        return self._elems.keys()

    def values(self):
        return self._elems.values()

    def items(self):
        return self._elems.items()

    def __iter__(self):
        for k in self._elems.keys():
            yield k

    def to_nested_dict(self):
        elems = {}
        for (vid, obj), elem in self._elems.items():
            elems.setdefault(vid, {})[obj] = elem
        return elems

    def update_from_nested_dict(self, elems):
        for vid, vid_elems in elems.items():
            for obj, elem in vid_elems.items():
                self._elems[(vid, obj)] = elem


def load_dataset_tasks_csv(fp):
    '''Loads the problem definitions for an entire dataset from one CSV file.'''
    reader = csv.DictReader(fp, fieldnames=TASK_FIELDS)
    rows = [row for row in reader]

    tasks = VideoObjectDict()
    for row in rows:
        key = (row['video_id'], row['object_id'])
        tasks[key] = Task(
            init_time=int(row['init_frame']),
            last_time=int(row['last_frame']),
            init_rect={
                'xmin': float(row['xmin']),
                'xmax': float(row['xmax']),
                'ymin': float(row['ymin']),
                'ymax': float(row['ymax'])})
    return tasks


def rect_to_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs = rect['xmin'] * imwidth
    ymin_abs = rect['ymin'] * imheight
    xmax_abs = rect['xmax'] * imwidth
    ymax_abs = rect['ymax'] * imheight
    return (xmin_abs, ymin_abs, xmax_abs - xmin_abs, ymax_abs - ymin_abs)


def rect_from_opencv(rect, imsize_hw):
    imheight, imwidth = imsize_hw
    xmin_abs, ymin_abs, width_abs, height_abs = rect
    xmax_abs = xmin_abs + width_abs
    ymax_abs = ymin_abs + height_abs
    return {
        'xmin': xmin_abs / imwidth,
        'ymin': ymin_abs / imheight,
        'xmax': xmax_abs / imwidth,
        'ymax': ymax_abs / imheight,
    }


def OxUvADataset():
    return OxUvADatasetClass().get_sequence_list()


class OxUvADatasetClass(BaseDataset):
    """ OxUvA test set.

    Publication:
        Long-term Tracking in the Wild: A Benchmark
        Jack Valmadre, Luca Bertinetto, João F. Henriques, Ran Tao, Andrea Vedaldi, Arnold Smeulders, Philip Torr and Efstratios Gavves
        ECCV, 2018
        https://arxiv.org/abs/1803.09502

    Download the dataset using the toolkit https://oxuva.github.io/long-term-tracking-benchmark/.
    """
    def __init__(self):
        super().__init__()
        self.base_path = path_config.OXUVA_PATH
        self.tasks = self._get_tasks(self.base_path)
        self.imfile = lambda vid, t: os.path.join(self.base_path, 'images', "test", vid, '{:06d}.jpeg'.format(t))

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(key, task) for key, task in self.tasks.items()])

    def _construct_sequence(self, key, task):
        vid, obj = key

        init_image = self.imfile(vid, task.init_time)
        im = cv2.imread(init_image, cv2.IMREAD_COLOR)
        imheight, imwidth, _ = im.shape

        frames_list = [self.imfile(vid, t) for t in range(task.init_time, task.last_time + 1)]

        ground_truth_rect = np.zeros((len(frames_list), 4))
        init_box = rect_to_opencv(task.init_rect, imsize_hw=(imheight, imwidth))
        ground_truth_rect[0, :] = init_box

        return Sequence(vid, frames_list, 'oxuva', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_tasks(self, root):
        tasks_file = os.path.join(root, 'test.csv')
        with open(tasks_file, 'r') as fp:
            tasks = load_dataset_tasks_csv(fp)

        return tasks
