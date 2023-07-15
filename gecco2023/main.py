import math, pdb
import numpy as np
import pandas as pd
import _pickle as cPickle
from past.builtins import execfile
from argparse import ArgumentParser
parser = ArgumentParser(description='Input parameters for Deep Generative Neural Network')
parser.add_argument('--noise', default=16, type=int, help='Number of Noise Variables for GNN')
parser.add_argument('--cnndim', default=2, type=int, help='Size of Latent Dimensions for GNN')
parser.add_argument('--iter', default=50, type=int, help='Number of Total Iterations for Solver')
parser.add_argument('--batch', default=1024, type=int, help='Number of Evaluations in an Iteration')
parser.add_argument('--rseed', default=0, type=int, help='Random Seed for Network Initialization')
args = parser.parse_args()

import torch
import torch.nn as nn
from entmax import entmax15, sparsemax

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.rseed)
torch.manual_seed(args.rseed)
torch.cuda.manual_seed(args.rseed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open('Dataset.pkl', 'rb') as f: Dataset = cPickle.load(f)

ban_stocks = input('Enter tickers to exclude (comma-separate if multiple). Press enter to skip this step.\n').split()
ban_stocks = [etf for etf in ban_stocks if etf in Dataset.columns]
Dataset = Dataset.drop(columns = ban_stocks)

selected = False
while not selected:
    selected = input('Enter ETFs to track (comma-separate if multiple equal weighted ETFs) e.g. RYH, VGT\n').split()
    selected = [etf for etf in selected if etf in Dataset.columns]

#RTH, RYH, VGT, FXG, XLP, DEF
sel_ret = torch.Tensor(Dataset[selected].values).to(device)
mweights = torch.ones(len(selected)).to(device)
mweights /= mweights.sum()
index = (sel_ret * mweights).sum(dim=1)

ETFs = ['ASA','TY','ADX','BCV','INSI','GAM','JHI','MCI','DNP','MMT','CMU','MGF','NUV','PCF','NCA','NNY','ASG','CIK','CRF','EEA','GAB','KF','MFM','USA','MXF','PAI','PEO','RVT','SOR','HQH','JHS','SWZ','TSI','FAX','HYB','PPT','IAF','MIN','GIM','PIM','BKT','CIF','OIA','NMI','CXE','CXH','FT','JMM','KSM','KTF','MHF','MPV',
'PMM','TWN','VLT','ZTR','PDT','CEF','DSM','EMF','MCR','MFV','GF','ECF','MVF','CEE','JOF','IRL','MXE','PFD','NQP','NUO','VKQ','DTF','DMF','MYD','PFO','VGM','FCO','MYN','CET','SBI','NXP','MYI','HQL','VTN','VMO','NXC','NXN','CHN','MMU','JEQ','MNP','SPY','BKN','AWF','PCM','RMT','IFN','CUBA','BTO','DDF','HIO','IIM','MIY',
'MPA','MQT','MQY','MSD','MUA','MVT','NAZ','NIM','NMT','NPV','PMO','RCS','RFI','SPE','TEI','VBF','VCV','VPV','GGT','IIF','TDF','GCV','MDY','NOM','VFL','VKI','EWA','EWC','EWD','EWG','EWH','EWI','EWJ','EWK','EWL','EWM','EWN','EWO','EWP','EWQ','EWS','EWU','EWW','RCG','MHD','MHN','DIA','MUC','MUJ','DSU','DHF','HIX','VVR',
'DHY','EVF','XLB','XLE','XLF','XLI','XLK','XLP','XLU','XLV','XLY','CEV','EVN','MUE','QQQ','NAC','NAD','NAN','GUT','NSL','BBH','PPH','EWY','IVV','IWB','IYW','IJH','IJR','IVE','IVW','IWD','IWF','IWM','IWV','IYF','SMH','IYE','IYH','IYR','IDU','IYM','IYG','EWT','IYC','EWZ','IEV','IJK','IJS','IJT','IUSG','IWN','IWO','IUSV',
'DGT','SLYG','SLYV','SPYG','SPYV','SPTM','OEF','IOO','AEF','LEO','IBB','OIH','IGM','RTH','VTI','PCQ','PMF','PNF','IGN','SOXX','IWR','IWS','BFK','BFZ','BNY','IWP','RECS','EFA','RWR','NZF','EPP','ILF','JPXN','IXC','JRS','IXG','IXJ','IXN','IXP','BHK','PCN','VXF','AFB','RQI','WEA','PHT','PML','PNI','CHI','IEF','LQD','SHY',
'TLT','BLE','CLM','EIM','ENX','EVM','NVG','NXJ','JPS','NBO','NKG','FEZ','SPEU','NBH','NBW','BYM','PMX','PYN','PZC','ADRE','NEA','HPF','FMN','PTY','NKX','NRK','FFC','EZA','EAD','JPC','NCV','EEM','PHK','AVK','RSP','PWC','CHY','HYT','EVV','HPS','ERC','RNP','JQC','HPI','MHI','BLW','EHI','NCZ','NHS','MUI','FTF','AGG','EVT',
'WIA','ONEQ','FLC','FVD','PFL','MAV','FRA','NRO','DVY','NMZ','EFR','GDV','EMD','TIP','GRF','ITOT','ETG','VAW','VB','VBK','VBR','VCR','VDC','VFH','VGT','VHT','VO','VPU','VTV','VUG','VV','HTD','IGR','SCD','TYG','UTG','WIW','JFR','CSQ','ERH','FTDS','LGI','MFD','UTF','CII','ETO','FCT','GLU','FEN','ILCG','ISCG','IMCG','IMCV',
'ISCV','EFT','GLV','JRO','MCN','FFA','BGT','KYN','VDE','VIS','VNQ','VOX','FXI','BXMX','EOI','PFN','GLD','FAM','PEY','PGJ','BGR','PHD','EOS','IAU','EVG','NFJ','PBW','PWB','PWV','XMVM','XSMO','VGK','VPL','VWO','BME','GGN','IGD','ETB','GLQ','XLG','BOE','FMY','PGP','PBE','PBJ','PBS','PEJ','PSI','PSJ','PXQ','ETV','IWC','MGU',
'BDJ','PFM','PID','FDM','PKB','PPA','PUI','PXE','PXJ','CGO','EGF','IGA','KBE','KCE','KIE','MDYG','MDYV','SDY','SLY','SPLG','SPMD','ETW','SPXX','PHO','FXE','PRF','DBC','XBI','XHB','XSD','BTA','RPG','RPV','RZG','RZV','FDL','USO','VIG','SLV','GLO','QQEW','QTEC','IAI','IAT','IEZ','IHE','IHF','IHI','RYJ','GDX','FPX','GSP',
'AIVI','AIVL','DES','DEW','DFE','DFJ','DHS','DIM','DLN','DLS','DNL','DOL','DON','DTD','DTH','DXJ','DDM','DOG','MVV','MYY','PSQ','QLD','SH','SSO','KRE','XES','XME','XOP','XPH','XRT','FBT','FXA','FXB','FXC','FXF','FTCS','DXD','MZZ','QID','SDS','GSG','AGD','VOE','VOT','DBV','PRFZ','CVY','JXI','MXI','EXI','CAF','PFI','PRN',
'PSL','PTF','PTH','PXI','PYZ','EVX','SLX','ERTH','PSP','DJP','RCD','RYH','RYT','RYU','VYM','DSI','ETY','FOF','EQWL','PGF','XMHQ','CSD','DEF','RWX','PKW','BTZ','DBA','DBB','DBE','DBO','DBP','DBS','DGL','GBF','IEI','IGIB','IGSB','SHV','TLH','USIG','CWI','RWM','SAA','SBB','SDD','TWM','UWM','AOD','GII','DIG','DUG','QQQX',
'ROM','RXD','RXL','SIJ','SKF','SRS','SSG','SZK','UPW','USD','UXI','UYG','UYM','REW','SCC','SMN','UCC','URE','FXY','QCLN','EES','EPS','EZM','NIE','RESP','PDP','UDN','UUP','VEU','SDP','MBB','GXC','SPEM','EOD','IAE','JCE','PFF','CZA','BIV','BLV','BND','BSV','HYG','UNG','EXG','GDL','EDD','AWP','GWX','SPDW','RSX','REZ','USRT',
'QQXT','SMOG','FEX','FNX','FRI','FXD','FXG','FXH','FXL','FXN','FXO','FXR','FXU','FXZ','FYX','FAD','FCG','FIW','FNI','FTC','CGW','FAB','FTA','BGY','FGB','BIL','HNW','SPAB','SPIP','SPTI','SPTL','WTRE','IDHQ','PBD','PIO','GRX','PXF','DEM','CHW','DEX','VEA','VTA','ETJ','GOF','WPS','NLR','SRV','DTRE','FDD','MOO','MUB','TFI',
'HTY','PDN','PXH','CMF','BWX','NYF','PCY','PLW','PZT','PWZ','PZA','SHM','EFU','EFZ','DGS','EEV','EUM','EWV','FXP','CUT','PHB','PVI','AIA','BKF','FGD','JNK','USL','IGF','SCZ','TOK','EMB','IEUS','IFGL','MGC','MGK','MGV','PIE','PIZ','BJK','GCC','PGX','GSY','FUE','GRU','DWX','EPI','DGP','DZZ','UGA','DGZ','PIN','RWK','WIP',
'ACWI','EIS','TUR','ACWX','THD','TAN','CYB','PST','RDOG','RWO','TBT','SEF','PNQI','DDG','ICLN','NIB','WOOD','VT','FAN','AFK','AAXJ','HAP','RBLD','SPXL','AGZ','SUB','AOA','AOM','AOR','AOK','ERX','ERY','FAS','FAZ','SPXS','TNA','TZA','PSR','EUO','SCO','UCO','ULE','YCS','UGL','ZSL','AGQ','YCL','EDC','EDZ','TECL','TECS']
ETFs = [etf for etf in ETFs if etf in Dataset.columns]

Dataset = Dataset.drop(columns=ETFs)
assets = list(Dataset.columns.values)
valid_data = np.array(Dataset)
valid_data = torch.from_numpy(valid_data).float()
test_size = len(valid_data)//5

q3 = valid_data.abs().quantile(0.75)
q3 =  q3 + 1.5 * (q3 - valid_data.abs().quantile(0.25))
valid_data[valid_data > q3] = q3
valid_data[valid_data < -q3] = -q3

if torch.cuda.is_available():
    valid_data = valid_data.pin_memory().to(device, non_blocking=True)

def calculate_reward(weights, valid_data, index_data, train = False):
    diff = weights.matmul(valid_data.T) - index_data
    if not train: return diff.clamp(max=0.0).pow(2).mean(dim=1)

    #weight the training returns by recency in performance calculation
    ww = torch.arange(1, diff.shape[1]+1).pow(0.5).to(device)

    return (diff.clamp(max=0.0).pow(2) * (ww/ww.sum())).sum(dim=1)

method = input('Do you want to use "GNN" or "CMA" for Portfolio Optimization for Selected Index(s)?\n')
if method == 'GNN':
    print('Starting GNN optimizer. You can check best_weights.csv for the best portfolio found.')
    execfile('gnn_es_index.py')
elif method == 'CMA':
    print('Starting CMA optimizer. You can check best_weights.csv for the best portfolio found.')
    execfile('cma_es_index.py')    
else:
    print('Invalid entry. Continuing with PyTorch optimizers. Check best_weights.csv for results.')
    execfile('torch_opt_index.py')