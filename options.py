from datasets.otbdataset import OTBDataset
from datasets.votdataset import VOTDataset
from datasets.tpldataset import TPLDataset
from datasets.uavdataset import UAVDataset
from datasets.nfsdataset import NFSDataset
from datasets.lasotdataset import LaSOTDataset


def select_expert(tracker_name):
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

        algorithm = MCCT(n_experts, mode, mu=kwargs["feature_threshold"])
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

        algorithm = HDT(n_experts, mode, beta=kwargs["feature_threshold"])
    elif algorithm_name == "HDTC":
        from algorithms.hdtc import HDTC

        algorithm = HDTC(n_experts, mode, beta=kwargs["feature_threshold"])
    else:
        raise ValueError("Unknown algorithm name")

    return algorithm


def select_datasets(dataset_name):
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
    else:
        raise ValueError("Unknown dataset name")

    return dataset
