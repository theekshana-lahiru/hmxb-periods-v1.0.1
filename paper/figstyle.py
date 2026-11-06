import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical slots (dataviz reference palette, light mode, all-pairs safe for 3)
C_BE, C_SG, C_SFXT = "#2a78d6", "#eb6834", "#1baf7a"
C_OTHER = "#7a7974"   # neutral grey for other / unclassified
C_INK, C_INK2, C_GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MARK = {"Be": "o", "sg": "s", "SFXT": "^", "other": "D"}
COL = {"Be": C_BE, "sg": C_SG, "SFXT": C_SFXT, "other": C_OTHER}

COL_W = 3.4     # inches, one AN column (~86 mm)
FULL_W = 7.0    # inches, full text width (~178 mm)

def setup():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.edgecolor": C_INK2, "axes.linewidth": 0.6, "axes.labelcolor": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0, "legend.frameon": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def cls_group(c):
    c = str(c)
    if c.startswith("Be"):
        return "Be"
    if c.startswith("sg"):
        return "sg"
    if c.startswith("SFXT"):
        return "SFXT"
    return "other"
