from experiments.otb import ExperimentOTB
from experts.atom import ATOM


tracker = ATOM()
otb = ExperimentOTB("/home/heonsong/Disk2/Dataset/OTB")
otb.run(tracker)
