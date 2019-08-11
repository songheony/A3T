from experiments.otb import ExperimentOTB
from experts.dasiamrpn import DaSiamRPN


tracker = DaSiamRPN()
otb = ExperimentOTB("/home/heonsong/Disk2/Dataset/OTB")
otb.run(tracker)
otb.report(["ATOM", "DaSiamRPN", "ECO", "MDNet", "SiamDW", "TADT", "Vital"])
