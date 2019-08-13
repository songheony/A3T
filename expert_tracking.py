from experiments.otb import ExperimentOTB
from experiments.vot import ExperimentVOT
from experiments.tcolor128 import ExperimentTColor128
from experiments.nfs import ExperimentNfS
from experiments.uav123 import ExperimentUAV123
from experiments.lasot import ExperimentLaSOT


otb = ExperimentOTB("/home/heonsong/Disk2/Dataset/OTB")
vot = ExperimentVOT("/home/heonsong/Disk2/Dataset/VOT2018", version=2018)
tcolor128 = ExperimentTColor128("/home/heonsong/Disk2/Dataset/TColor128")
nfs = ExperimentNfS("/home/heonsong/Disk2/Dataset/NFS")
uav123 = ExperimentUAV123("/home/heonsong/Disk2/Dataset/UAV123")
lasot = ExperimentLaSOT("/home/heonsong/Disk2/Dataset/LaSOT")
datasets = [otb, vot, tcolor128, nfs, uav123, lasot]

# from experts.atom import ATOM
# tracker = ATOM()
# from experts.dasiamrpn import DaSiamRPN
# tracker = DaSiamRPN()
from experts.eco import ECO
tracker = ECO()
# from experts.mdnet import MDnet
# tracker = MDnet()
# from experts.siamdw import SiamDW
# tracker = SiamDW()
# from experts.siamfc import SiamFC
# tracker = SiamFC()
# from experts.siamrpn import SiamRPN
# tracker = SiamRPN()
# from experts.tadt import TADT
# tracker = TADT()
# from experts.vital import Vital
# tracker = Vital()
# from experts.bacf import BACF
# tracker = BACF()
# from experts.csrdcf import CSRDCF
# tracker = CSRDCF()
# from experts.staple import Staple
# tracker = Staple()
# from experts.strcf import STRCF
# tracker = STRCF()

for dataset in datasets:
    dataset.run(tracker)
    # dataset.report(["ATOM", "DaSiamRPN", "ECO", "MDNet", "SiamDW", "SiamFC", "SiamRPN", "TADT", "Vital", "BACF", "CSRDCF", "Staple", "STRCF"])
    break
