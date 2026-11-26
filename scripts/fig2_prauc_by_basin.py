"""Figure 2 of the manuscript: M>=3 PR-AUC by basin and model, 12-month lag (values from phase17_tables.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

R = [  # basin, model, value, lo, hi  (12-month lag; from the manuscript's Figure 2 data)
("Oklahoma","Climatology",.076,.057,.101),("Oklahoma","Seismogenic index",.443,.380,.496),("Oklahoma","ETAS",.463,.402,.517),
("Oklahoma","Classifier (LOBO)*",.197,.120,.244),("Oklahoma","Count hybrid (local)",.420,.370,.470),("Oklahoma","Count hybrid (pooled)",.420,.370,.470),
("Oklahoma","ETAS, 2017–2022*",.279,.189,.335),("Oklahoma","Count hybrid (LOBO)*",.251,.172,.304),
("Kansas","Climatology",.042,.022,.076),("Kansas","Seismogenic index",.073,.038,.158),("Kansas","ETAS",.114,.052,.225),
("Kansas","Classifier (LOBO)",.167,.078,.297),("Kansas","Count hybrid (local)",.088,.039,.181),("Kansas","Count hybrid (pooled)",.132,.058,.264),
("Kansas","Count hybrid (LOBO)",.142,.055,.278),
("Delaware Basin","Climatology",.359,.310,.407),("Delaware Basin","Seismogenic index",.581,.521,.633),("Delaware Basin","ETAS",.590,.533,.637),
("Delaware Basin","Classifier (LOBO)",.616,.557,.666),("Delaware Basin","Count hybrid (local)",.538,.480,.593),("Delaware Basin","Count hybrid (pooled)",.581,.529,.628),
("Delaware Basin","Count hybrid (LOBO)",.594,.540,.640),
("Midland Basin (held out, v1)","Climatology",.039,.029,.054),("Midland Basin (held out, v1)","Seismogenic index",.150,.113,.201),
("Midland Basin (held out, v1)","ETAS",.151,.112,.205),("Midland Basin (held out, v1)","Classifier (LOBO)",.159,.114,.210),
("Midland Basin (held out, v1)","Count hybrid (local)",.122,.080,.173),("Midland Basin (held out, v1)","Count hybrid (pooled)",.149,.115,.202),
("Midland Basin (held out, v1)","Count hybrid (LOBO)",.151,.114,.204),
("Fort Worth Basin (held out, v2)","Climatology",.006,.001,.020),("Fort Worth Basin (held out, v2)","Seismogenic index",.095,.007,.252),
("Fort Worth Basin (held out, v2)","ETAS",.240,.020,.466),("Fort Worth Basin (held out, v2)","Classifier (LOBO)",.192,.027,.323),
("Fort Worth Basin (held out, v2)","Count hybrid (local)",.138,.009,.392),("Fort Worth Basin (held out, v2)","Count hybrid (pooled)",.139,.011,.262),
("Fort Worth Basin (held out, v2)","Count hybrid (LOBO)",.165,.013,.293)]
BASE = {"Oklahoma":.043,"Kansas":.004,"Delaware Basin":.042,"Midland Basin (held out, v1)":.010,"Fort Worth Basin (held out, v2)":.001}
BLUE, GREY, DARK = "#2a6fdb", "#b9b8ad", "#3a3a3a"

plt.rcParams.update({"font.family": "Arial", "font.size": 8})
basins = list(BASE)
n = [sum(r[0] == b for r in R) for b in basins]
fig, axes = plt.subplots(len(basins), 1, figsize=(6.5, 10.5), gridspec_kw={"height_ratios": n, "hspace": 0.75})
for i, (b, ax) in enumerate(zip(basins, axes)):
    rs = [r for r in R if r[0] == b]
    xmax = max(r[4] for r in rs) * 1.18
    for j, (_, m, v, lo, hi) in enumerate(rs):
        y = len(rs) - 1 - j
        hl, et = m.startswith("Count hybrid (LOBO)"), m.startswith("ETAS")
        col = BLUE if hl else (DARK if et else GREY)
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
        ax.plot(v, y, "o", ms=5.5, mfc="white" if et else col, mec=DARK if et else col, mew=1.2, zorder=3)
        ax.text(hi + xmax * 0.012, y, f"{v:.3f}", va="center", fontsize=7.5, color="#555")
    base = BASE[b]
    ax.axvline(base, ls=(0, (3, 3)), color="#888", lw=0.9, zorder=1)
    ax.text(base + xmax * 0.008, len(rs) - 0.05, f"base rate {base:.3f}", fontsize=7, color="#666", va="bottom")
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([r[1] for r in rs][::-1])
    for lab in ax.get_yticklabels():
        t = lab.get_text()
        if t.startswith("Count hybrid (LOBO)"):
            lab.set_fontweight("bold")
        if not (t.startswith("Count hybrid (LOBO)") or t.startswith("ETAS")):
            lab.set_color("#555")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, len(rs) + 0.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="x", labelsize=7.5, length=3)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"({chr(97 + i)}) {b}", loc="left", fontsize=9, fontweight="bold", x=-0.36)
axes[-1].set_xlabel("M≥3 PR-AUC (each panel has its own scale)", fontsize=8)
out = "figure2_prauc_by_basin"   
fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
fig.savefig(out + ".tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(out + ".pdf", bbox_inches="tight")
print("saved")
