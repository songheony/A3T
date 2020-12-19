from datasets.otbdataset import OTBDataset
from datasets.otbnoisydataset import OTBNoisyDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.trackingnetdataset import TrackingNetDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset
from datasets.got10kdataset import GOT10KDatasetVal
from print_manager import do_not_print


@do_not_print
def select_expert(tracker_name):
    if tracker_name == "ATOM":
        from experts.atom import ATOM

        tracker = ATOM()
    elif tracker_name == "DaSiamRPN":
        from experts.dasiamrpn import DaSiamRPN

        tracker = DaSiamRPN()
    elif tracker_name == "DiMP":
        from experts.dimp import DiMP50

        tracker = DiMP50()
    elif tracker_name == "DROL":
        from experts.drol import DROL

        tracker = DROL()
    elif tracker_name == "GradNet":
        from experts.gradnet import GradNet

        tracker = GradNet()
    elif tracker_name == "KYS":
        from experts.kys import KYS

        tracker = KYS()
    elif tracker_name == "MemDTC":
        from experts.memdtc import MemDTC

        tracker = MemDTC()
    elif tracker_name == "MemTrack":
        from experts.memtrack import MemTrack

        tracker = MemTrack()
    elif tracker_name == "Ocean":
        from experts.ocean import Ocean

        tracker = Ocean()
    elif tracker_name == "PrDiMP":
        from experts.prdimp import PrDiMP50

        tracker = PrDiMP50()
    elif tracker_name == "RLS-RTMDNet":
        from experts.rls_rtmdnet import RLS_RTMDNet

        tracker = RLS_RTMDNet()
    elif tracker_name == "SiamBAN":
        from experts.siamban import SiamBAN

        tracker = SiamBAN()
    elif tracker_name == "SiamCAR":
        from experts.siamcar import SiamCAR

        tracker = SiamCAR()
    elif tracker_name == "SiamDW":
        from experts.siamdw import SiamDW

        tracker = SiamDW()
    elif tracker_name.startswith("SiamDW"):
        from experts.siamdw_group import SiamDWGroup

        parameter = tracker_name.split("/")

        tracker = SiamDWGroup(parameter[1], parameter[2])
    elif tracker_name == "SiamFC":
        from experts.siamfc import SiamFC

        tracker = SiamFC()
    elif tracker_name == "SiamFC++":
        from experts.siamfcpp import SiamFCPP

        tracker = SiamFCPP()
    elif tracker_name == "SiamMCF":
        from experts.siammcf import SiamMCF

        tracker = SiamMCF()
    elif tracker_name == "SiamR-CNN":
        from experts.siamrcnn import SiamRCNN

        tracker = SiamRCNN()
    elif tracker_name == "SiamRPN":
        from experts.siamrpn import SiamRPN

        tracker = SiamRPN()
    elif tracker_name == "SiamRPN++":
        from experts.siamrpnpp import SiamRPNPP

        tracker = SiamRPNPP()
    elif tracker_name.startswith("SiamRPN++"):
        from experts.siamrpnpp_group import SiamRPNPPGroup

        parameter = tracker_name.split("/")

        tracker = SiamRPNPPGroup(parameter[1], parameter[2])
    elif tracker_name == "SPM":
        from experts.spm import SPM

        tracker = SPM()
    elif tracker_name == "Staple":
        from experts.staple import Staple

        tracker = Staple()
    elif tracker_name == "THOR":
        from experts.thor import THOR

        tracker = THOR()
    elif tracker_name == "TRAS":
        from experts.tras import ETRAS

        tracker = ETRAS()
    elif tracker_name == "TRAST":
        from experts.tras import ETRAST

        tracker = ETRAST()
    elif tracker_name == "TRASFUST":
        from experts.tras import ETRASFUST

        tracker = ETRASFUST()
    else:
        raise ValueError("Unknown expert name")

    return tracker


def select_algorithms(algorithm_name, experts, **kwargs):
    n_experts = len(experts)
    mode = kwargs["mode"]
    if algorithm_name == "AAA":
        from algorithms.aaa import AAA

        algorithm = AAA(n_experts, **kwargs)
    elif algorithm_name == "Average":
        from algorithms.average import Average

        algorithm = Average(n_experts, mode)
    elif algorithm_name == "MCCT":
        from algorithms.mcct import MCCT

        algorithm = MCCT(n_experts, mode, mu=kwargs["threshold"])
    elif algorithm_name == "Max":
        from algorithms.baseline import Baseline

        algorithm = Baseline(
            n_experts, name=f"Max_{mode}", use_iou=False, use_feature=True
        )
    elif algorithm_name == "Random":
        from algorithms.random import Random

        algorithm = Random(n_experts, mode)
    elif algorithm_name == "HDT":
        from algorithms.hdt import HDT

        algorithm = HDT(n_experts, mode, beta=kwargs["threshold"])
    else:
        raise ValueError("Unknown algorithm name")

    return algorithm


def select_datasets(dataset_name):
    if dataset_name == "OTB2015":
        dataset = OTBDataset()
    elif dataset_name == "OTB2015-80%":
        dataset = OTBNoisyDataset(0.8)
    elif dataset_name == "OTB2015-60%":
        dataset = OTBNoisyDataset(0.6)
    elif dataset_name == "OTB2015-40%":
        dataset = OTBNoisyDataset(0.4)
    elif dataset_name == "OTB2015-20%":
        dataset = OTBNoisyDataset(0.2)
    elif dataset_name == "NFS":
        dataset = NFSDataset()
    elif dataset_name == "UAV123":
        dataset = UAVDataset()
    elif dataset_name == "TColor128":
        dataset = TPLDataset()
    elif dataset_name == "TrackingNet":
        dataset = TrackingNetDataset()
    elif dataset_name == "VOT2018":
        dataset = VOTDataset()
    elif dataset_name == "LaSOT":
        dataset = LaSOTDataset()
    elif dataset_name == "Got10K":
        dataset = GOT10KDatasetVal()
    else:
        raise ValueError("Unknown dataset name")

    return dataset
