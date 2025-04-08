import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/ptycho_gui/')
from nsls2ptycho2.core.ptycho.utils import parse_config
from nsls2ptycho2.core.ptycho.recon_ptycho_gui import recon_gui

#param = parse_config('/nsls2/data2/hxn/legacy/users/2025Q1/Boyu_2025Q1/ptycho/recon_result/S334806/testlive/recon_data/334806_testlive.ptycho_root_8691_57.txt')
param = parse_config('./ptycho_config.txt')

recon,rank = recon_gui(param)

recon.setup()

