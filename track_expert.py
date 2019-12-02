import numpy as np
import multiprocessing
import os
from itertools import product
from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from datasets.got10kdataset import GOT10KDatasetVal


def run_sequence(seq, tracker, debug=False):
    """Runs a tracker on a sequence."""

    base_results_path = "{}/{}".format(tracker.results_dir, seq.name)
    results_path = "{}.txt".format(base_results_path)
    times_path = "{}_time.txt".format(base_results_path)

    if os.path.isfile(results_path):
        return

    print("Tracker: {},  Sequence: {}".format(tracker.name, seq.name))

    if debug:
        tracked_bb, exec_times = tracker.run(seq)
    else:
        try:
            tracked_bb, exec_times = tracker.run(seq)
        except Exception as e:
            print(e)
            return

    tracked_bb = np.array(tracked_bb).astype(float)
    exec_times = np.array(exec_times).astype(float)

    print("FPS: {}".format(len(exec_times) / exec_times.sum()))
    if not debug:
        np.savetxt(results_path, tracked_bb, delimiter="\t", fmt="%f")
        np.savetxt(times_path, exec_times, delimiter="\t", fmt="%f")


def run_dataset(dataset, trackers, debug=False, threads=0):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    if threads == 0:
        mode = "sequential"
    else:
        mode = "parallel"

    if mode == "sequential":
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == "parallel":
        param_list = [
            (seq, tracker_info, debug)
            for seq, tracker_info in product(dataset, trackers)
        ]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print("Done")


def run_tracker(tracker, dataset, sequence=None, debug=0, threads=0):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset: Dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [tracker]

    run_dataset(dataset, trackers, debug, threads)


def main(tracker_name, dataset_name):
    if tracker_name == "ATOM":
        from experts.atom import ATOM

        tracker = ATOM()
    elif tracker_name == "BACF":
        from experts.bacf import BACF

        tracker = BACF()
    elif tracker_name == "CSRDCF":
        from experts.csrdcf import CSRDCF

        tracker = CSRDCF()
    elif tracker_name == "DaSiamRPN":
        from experts.dasiamrpn import DaSiamRPN

        tracker = DaSiamRPN()
    elif tracker_name == "DiMP":
        from experts.dimp import DiMP

        tracker = DiMP()
    elif tracker_name == "ECO":
        from experts.eco import ECO

        tracker = ECO()
    elif tracker_name == "ECO-HC":
        from experts.eco_hc import ECO_HC

        tracker = ECO_HC()
    elif tracker_name == "GradNet":
        from experts.gradnet import GradNet

        tracker = GradNet()
    elif tracker_name == "LDES":
        from experts.ldes import LDES

        tracker = LDES()
    elif tracker_name == "MemTrack":
        from experts.memtrack import MemTrack

        tracker = MemTrack()
    elif tracker_name == "RT-MDNet":
        from experts.rt_mdnet import RTMDNet

        tracker = RTMDNet()
    elif tracker_name == "SiamDW":
        from experts.siamdw import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCRes22":
        from experts.siamdw_siamfcres import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCIncep22":
        from experts.siamdw_siamfcincep import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCNext22":
        from experts.siamdw_siamfcnext import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamRPNRes22":
        from experts.siamdw_siamrpnres import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCRes22_VOT":
        from experts.siamdw_siamfcres_vot import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCIncep22_VOT":
        from experts.siamdw_siamfcincep_vot import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCNext22_VOT":
        from experts.siamdw_siamfcnext_vot import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamRPNRes22_VOT":
        from experts.siamdw_siamrpnres_vot import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCRes22_G":
        from experts.siamdw_siamfcres_g import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamDW_SiamFCRes22W_G":
        from experts.siamdw_siamfcresw_g import SiamDW

        tracker = SiamDW()
    elif tracker_name == "SiamFC":
        from experts.siamfc import SiamFC

        tracker = SiamFC()
    elif tracker_name == "SiamMCF":
        from experts.siammcf import SiamMCF

        tracker = SiamMCF()
    elif tracker_name == "SiamRPN":
        from experts.siamrpn import SiamRPN

        tracker = SiamRPN()
    elif tracker_name == "SiamRPN++":
        from experts.siamrpnpp import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_AlexNet":
        from experts.siamrpnpp_alexnet import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_AlexNet_OTB":
        from experts.siamrpnpp_alexnet_otb import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_ResNet-50":
        from experts.siamrpnpp_resnet import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_ResNet-50_OTB":
        from experts.siamrpnpp_resnet_otb import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_ResNet-50_LT":
        from experts.siamrpnpp_resnet_lt import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_MobileNetV2":
        from experts.siamrpnpp_mobilenetv2 import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SiamRPN++_SiamMask":
        from experts.siamrpnpp_siammask import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name == "SPM":
        from experts.spm import SPM

        tracker = SPM()
    elif tracker_name == "Staple":
        from experts.staple import Staple

        tracker = Staple()
    elif tracker_name == "STRCF":
        from experts.strcf import STRCF

        tracker = STRCF()
    elif tracker_name == "THOR":
        from experts.thor import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamFC_Dynamic_OTB":
        from experts.thor_siamfc_dynamic_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamFC_Dynamic_VOT":
        from experts.thor_siamfc_dynamic_vot import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamFC_Ensemble_OTB":
        from experts.thor_siamfc_ensemble_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamFC_Ensemble_VOT":
        from experts.thor_siamfc_ensemble_vot import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamMask_Dynamic_OTB":
        from experts.thor_siammask_dynamic_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamMask_Dynamic_VOT":
        from experts.thor_siammask_dynamic_vot import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamMask_Ensemble_OTB":
        from experts.thor_siammask_ensemble_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamMask_Ensemble_VOT":
        from experts.thor_siammask_ensemble_vot import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamRPN_Dynamic_OTB":
        from experts.thor_siamrpn_dynamic_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamRPN_Dynamic_VOT":
        from experts.thor_siamrpn_dynamic_vot import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamRPN_Ensemble_OTB":
        from experts.thor_siamrpn_ensemble_otb import THOR

        tracker = THOR()
    elif tracker_name == "THOR_SiamRPN_Ensemble_VOT":
        from experts.thor_siamrpn_ensemble_vot import THOR

        tracker = THOR()
    else:
        raise ValueError("Unknown expert name")

    if dataset_name == "OTB":
        dataset = OTBDataset()
    elif dataset_name == "NFS":
        dataset = NFSDataset()
    elif dataset_name == "UAV":
        dataset = UAVDataset()
    elif dataset_name == "TPL":
        dataset = TPLDataset()
    elif dataset_name == "VOT":
        dataset = VOTDataset()
    elif dataset_name == "LaSOT":
        dataset = LaSOTDataset()
    elif dataset_name == "Got10K":
        dataset = GOT10KDatasetVal()
    else:
        raise ValueError("Unknown dataset name")

    run_tracker(tracker, dataset, debug=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expert", default="ECO-HC", type=str, help="expert")
    parser.add_argument("-d", "--dataset", default="LaSOT", type=str, help="dataset")
    args = parser.parse_args()

    main(args.expert, args.dataset)
