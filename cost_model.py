"""
MUSE AWS cost model: one day, a month at the end of year 1, egress, storage, and L3 on
GPUs.

Everything is priced in Oregon (us-west-2); N. California (us-west-1) is kept only for
the "how much more" comparison. Prices were checked 2026-09-23 against regional catalogs
published September 11-22. On-Demand throughout: no Spot, reservations or archive tiers
beyond Glacier are assumed.

`python cost_model.py` prints a report. Quarto runs the script as a pre-render step to
write _variables.yml, which the slides read as {{< var cost.KEY >}}.
"""

import json
import os

# ---- Calendar ------------------------------------------------------------------------------
DAYS = 30  # one month of daily batches; monthly storage prices are prorated by this
HOURS = 24 * DAYS  # 720 h; AWS bills actual hours, 730 h is only an annual-average convention

# ---- Data volumes, GiB per input day -------------------------------------------------------
GIB = 2**30
L0_GB_DAY = 389e9
L0 = L0_GB_DAY / GIB  # 362.28
L1 = L0 * 75_052_800 / 73_987_200  # measured L1/L0 on the 20-file run
L2 = L0 * 74_836_800 / 73_987_200  # measured L2/L0
# L3: one merged file per SG observation, float32 moments + errors + flags, lossless GZIP_2.
# The 124 GiB synthetic-flare estimate still needs a linked size report. Alternatives run from
# 40-45 (low-signal pixels blanked) to 370-570 (L3A/L3B/L2.5 all published); a published VDEM
# cube would add 530-680 GiB/day.
L3 = 124.0

# Object counts from the full-size synthetic corpus: 16,340 L0 -> 16,340 L1 + 56 L2
L0_FILES = L0_GB_DAY / (48.64509504e9 / 16_340)  # ~130.7k/day, ~3 MB each
L2_FILES = L0_FILES * 56 / 16_340  # ~448/day
L3_FILES = L2_FILES / 4  # one L3 per 3 SG L2 files; CI (1 of 4 cameras) has none

# ---- AWS list prices, (Oregon, N. California) ----------------------------------------------
# Hourly unless noted; storage per GiB-month; requests and transitions per object.
REGIONS = ("us-west-2", "us-west-1")
REGION = REGIONS[0]
_RATES = {
    "ec2": (0.4032, 0.4704),  # m7i.2xlarge On-Demand
    "gp3": (0.080, 0.096),  # EBS
    "snap": (0.050, 0.055),  # EBS snapshots
    "nat": (0.045, 0.048),  # NAT gateway hourly; its per-GB processing charge is not modeled
    "ipv4": (0.005, 0.005),  # public IPv4 address
    "rds_saz": (0.168, 0.192),  # db.m7g.large Single-AZ
    "rds_gp3_saz": (0.115, 0.138),  # RDS storage
    "rds_backup": (0.095, 0.095),  # backup storage beyond the free allowance
    "s3": (0.023, 0.026),  # S3 Standard, first 50 TiB
    "glacier": (0.0036, 0.0045),  # Glacier Flexible Retrieval
    "git": (0.004, 0.005),  # Glacier Instant Retrieval
    "put": (0.005e-3, 0.0055e-3),
    "get": (0.0004e-3, 0.00044e-3),
    "glacier_transition": (0.03e-3, 0.033e-3),
    "git_transition": (0.02e-3, 0.02e-3),
}
PRICES = {region: {key: rates[i] for key, rates in _RATES.items()} for i, region in enumerate(REGIONS)}
EGRESS_TIERS = ((10_240, 0.09), (40_960, 0.085), (102_400, 0.07), (float("inf"), 0.05))  # GiB in tier, $/GiB
FREE_EGRESS = 100  # GiB/month, assumed all available to MUSE
INTER_REGION = 0.02  # $/GiB between Regions

# ---- Topology and allowances ---------------------------------------------------------------
EBS_GIB = 1_024  # proposed working disk; needs eviction/restore and a working-set proof
EBS_STATE_GIB = 132  # root + Docker/log/artifact state; the rest is science data
GPU_EBS_GIB = 64  # two 32 GiB GPU boot disks, kept all month; extra scratch is a separate input
SCIENTIST_SHARES = (0.0, 0.25)  # user downloads as a share of new monthly L2/L3 bytes; 25% is a ceiling

# ---- GPUs for L2 -> L3, Oregon: key -> (label, $/hour for the pair) ------------------------
GPUS = {
    "t4": ("2x T4 16 GB (2x g4dn.xlarge)", 2 * 0.526),
    "l4": ("2x L4 24 GB (2x g6.xlarge)", 2 * 0.8048),
    "a10g": ("2x A10G 24 GB (2x g5.xlarge)", 2 * 1.006),
    "g54": ("2x A10G 24 GB, 16 vCPU / 64 GiB each (2x g5.4xlarge)", 2 * 1.624),
    "l40s": ("2x L40S 48 GB (2x g6e.xlarge)", 2 * 1.861),
    "h100": ("2x H100 80 GB (2x p5.4xlarge)", 2 * 6.88),
}
L3_GPU = "g54"  # chosen for now; whether L2 -> L3 needs this much CPU and RAM is not qualified
L3_PAIR = GPUS[L3_GPU][1]


# ---- Model ---------------------------------------------------------------------------------


def egress(gib: float, free_gib: float = FREE_EGRESS) -> float:
    """
    Internet transfer out for one month, through the volume tiers.
    """
    left, cost = max(0.0, gib - free_gib), 0.0
    for size, rate in EGRESS_TIERS:
        step = min(left, size)
        cost += step * rate
        left -= step
    return cost


def always_on_month(region: str = REGION) -> dict:
    """
    Charges that accrue whether or not data moves: host, database, network, requests,
    allowances.
    """
    p = PRICES[region]
    puts, gets = L0_FILES + L2_FILES + L3_FILES, L0_FILES + L2_FILES  # L2 is read once, by whichever L3 worker
    return {
        "EC2 m7i.2xlarge": HOURS * p["ec2"],
        "RDS Single-AZ + 100 GiB": HOURS * p["rds_saz"] + 100 * p["rds_gp3_saz"],
        "NAT + IPv4": HOURS * (p["nat"] + p["ipv4"]),
        "S3 requests": DAYS * (puts * p["put"] + gets * p["get"]),
        "Backups + snapshots": 100 * (p["rds_backup"] + p["snap"]),
        "Logs, secrets": 30.0,
    }


def one_day(region: str = REGION) -> dict:
    """
    The daily slide: the always-on month over 30 days, plus the disk, one day's data in
    S3, the L2 download.
    """
    p = PRICES[region]
    day = {k: x / DAYS for k, x in always_on_month(region).items()}
    day["EBS working disk"] = EBS_GIB * p["gp3"] / DAYS
    day["S3 storage, one day"] = (L0 + L2 + L3) * p["s3"] / DAYS
    day["L2 download"] = egress(L2, free_gib=0)  # the free 100 GB assumed used up
    return day


def ebs_month(days_retained: float, region: str = REGION) -> float:
    """
    Today's code never deletes: the working disk holds every L0, L1 and L2.
    """
    return (EBS_STATE_GIB + days_retained * (L0 + L1 + L2)) * PRICES[region]["gp3"]


def s3_month(year: float, l23_mission: bool, region: str = REGION) -> dict:
    """
    S3 run rate after `year` years.

    L0: 30 days in Standard, then Glacier Flexible Retrieval for the mission.

    L2/L3: 90 days in Standard, then Glacier Instant Retrieval for the mission (l23_mission) or deleted.

    Glacier bills 8 KiB of Standard and 32 KiB of Glacier metadata per archived object.
    """
    p, days = PRICES[region], 365 * year
    old_l0 = max(0, days - 30)
    old_l23 = max(0, days - 90) if l23_mission else 0
    recent_l23 = min(days, 90) * (L2 + L3)
    metadata_gib = old_l0 * L0_FILES * 8192 / GIB
    standard_gib = min(days, 30) * L0 + metadata_gib + recent_l23
    return {
        "L0 recent (Standard)": min(days, 30) * L0 * p["s3"],
        "L0 archive (bytes)": old_l0 * L0 * p["glacier"],
        "L0 archive metadata": metadata_gib * p["s3"] + old_l0 * L0_FILES * 32768 / GIB * p["glacier"],
        "L0 archive moves": DAYS * L0_FILES * p["glacier_transition"] if old_l0 else 0,
        "L2+L3 recent (Standard)": recent_l23 * p["s3"],
        "L2+L3 archive (Glacier IR)": old_l23 * (L2 + L3) * p["git"],
        "L2+L3 archive moves": DAYS * (L2_FILES + L3_FILES) * p["git_transition"] if old_l23 else 0,
        "Standard volume discount": -max(0, standard_gib - 51_200) * 0.001,  # beyond 50 TiB, $0.001 less
    }


def download_gib() -> tuple[float, float]:
    """
    Low/high internet bytes billed to MUSE per month.

    Low: L2 once, to the local L3 machine. High adds an L0 archive copy, a NASA copy of L2/L3
    and user downloads at the 25% ceiling of new L2/L3 bytes. No routine L1 export; no cache credit.
    """
    l23 = DAYS * (L2 + L3)
    low = DAYS * L2 + SCIENTIST_SHARES[0] * l23
    high = DAYS * (L2 + L0) + l23 + SCIENTIST_SHARES[1] * l23
    return low, high


def month_range(region: str = REGION) -> tuple[float, float]:
    """
    Low/high month at the end of year 1: always-on + working disk + S3 + egress.

    Low keeps L2/L3 for 90 days and downloads L2 once; high keeps them for the mission
    and adds the optional exports. Two policies, not confidence bounds; conditional on
    the bounded disk.
    """
    fixed = sum(always_on_month(region).values()) + EBS_GIB * PRICES[region]["gp3"]
    return tuple(
        fixed + sum(s3_month(1, mission, region).values()) + egress(gib)
        for mission, gib in zip((False, True), download_gib(), strict=True)
    )


def processing_month(region: str = REGION) -> dict:
    """
    Compute only, if NASA carries S3 and user egress: the L0 -> L2 host with its disk,
    database and network.
    """
    a = always_on_month(region)
    return {
        "EC2 m7i.2xlarge, 24/7": a["EC2 m7i.2xlarge"],
        "EBS working disk": EBS_GIB * PRICES[region]["gp3"],
        "RDS Single-AZ + 100 GiB": a["RDS Single-AZ + 100 GiB"],
        "NAT + IPv4": a["NAT + IPv4"],
        "Backups, snapshots, logs, secrets": a["Backups + snapshots"] + a["Logs, secrets"],
    }


# ---- Output --------------------------------------------------------------------------------


def usd(x: float, decimals: int = 0, step: float = 0) -> str:
    return f"${round(x / step) * step if step else x:,.{decimals}f}"


def usd_range(a: float, b: float, step: float = 1) -> str:
    return f"${round(a / step) * step:,.0f}-{round(b / step) * step:,.0f}"


def slide_variables() -> dict:
    """
    Every number the AWS slides show, formatted; read by Quarto as {{< var cost.KEY >}}.
    """
    p = PRICES[REGION]
    m = {k: DAYS * x for k, x in {"L0": L0, "L1": L1, "L2": L2, "L3": L3}.items()}  # GiB per month
    v = {}

    # Paper estimates: one day
    day, day_nc = one_day(), one_day("us-west-1")
    total, l2 = sum(day.values()), day["L2 download"]
    big = {"ec2": "EC2 m7i.2xlarge", "rds": "RDS Single-AZ + 100 GiB", "ebs": "EBS working disk"}
    rest = total - l2 - sum(day[k] for k in big.values())
    v["day_l2"], v["day_total"], v["day_nc"] = usd(l2, 2), usd(total, 2), usd(sum(day_nc.values()), 2)
    for key, name in big.items():
        v[f"day_{key}"], v[f"w_{key}"] = usd(day[name], 2), f"{100 * day[name] / l2:.0f}%"  # bar widths vs L2
    v["day_rest"], v["w_rest"] = usd(rest, 2), f"{100 * rest / l2:.0f}%"
    small = {"s3": "S3 storage, one day", "req": "S3 requests", "nat": "NAT + IPv4"}
    small |= {"backups": "Backups + snapshots", "logs": "Logs, secrets"}
    for key, name in small.items():
        v[f"day_{key}"] = usd(day[name], 2)
    v.update(
        l0_gib=f"{L0:.0f}",
        l1_gib=f"{L1:.0f}",
        l2_gib=f"{L2:.0f}",
        l0_files=f"{L0_FILES / 1000:.0f}k",
        day_puts=f"{(L0_FILES + L2_FILES + L3_FILES) / 1000:.0f}k",
        day_gets=f"{(L0_FILES + L2_FILES) / 1000:.0f}k",
        ec2_rate=f"${p['ec2']:g}",
        rds_rate=f"${p['rds_saz']:g}",
    )

    # Monthly cost at year 1
    lo, hi = month_range()
    low_gib, high_gib = download_gib()
    s3_lo, s3_hi = (sum(s3_month(1, mission).values()) for mission in (False, True))
    s3_lo_y2, s3_hi_y2 = (sum(s3_month(2, mission).values()) for mission in (False, True))
    v.update(
        m_always=usd(sum(always_on_month().values())),
        m_ebs=usd(EBS_GIB * p["gp3"]),
        m_s3_lo=usd(s3_lo),
        m_s3_hi=usd(s3_hi),
        m_eg_lo=usd(egress(low_gib)),
        m_eg_hi=usd(egress(high_gib)),
        m_lo="~" + usd(lo, step=100),
        m_hi="~" + usd(hi, step=100),
        m_l0_moves=usd(DAYS * L0_FILES * p["glacier_transition"]),
        share_hi=f"{SCIENTIST_SHARES[1]:.0%}",
        shares=f"{SCIENTIST_SHARES[0] * 100:.0f}-{SCIENTIST_SHARES[1]:.0%}",
    )

    # What leaves AWS each month, cumulative rows
    rows = {"eg_l2": m["L2"], "eg_l1": m["L2"] + m["L1"], "eg_l0": m["L2"] + m["L1"] + m["L0"]}
    rows["eg_nasa"] = rows["eg_l0"] + m["L2"] + m["L3"]
    for key, gib in rows.items():
        v[key], v[key + "_tib"] = usd(egress(gib)), f"{gib / 1024:.1f} TiB"
    sci = [rows["eg_nasa"] + share * (m["L2"] + m["L3"]) for share in SCIENTIST_SHARES]
    v["eg_sci"] = usd_range(egress(sci[0]), egress(sci[1]))
    v["eg_sci_tib"] = f"{sci[0] / 1024:.1f}-{sci[1] / 1024:.1f} TiB"

    # Storage over the mission
    v.update(
        glacier_rate=f"${p['glacier']:g}",
        l0_growth=usd(s3_lo_y2 - s3_lo),
        l1_90d=usd(90 * L1 * p["s3"], step=10),
        l23_90d=usd(90 * (L2 + L3) * p["s3"]),
        l23_growth=usd(365 * (L2 + L3) * p["git"]),
        s3_y1="~" + usd_range(s3_lo, s3_hi, 100),
        s3_growth=usd_range(s3_lo_y2 - s3_lo, s3_hi_y2 - s3_hi, 100),
        ebs_min=f"{EBS_STATE_GIB + L0 + L1 + L2:,.0f}",
        ebs_30=usd(ebs_month(30)),
        ebs_365=usd(ebs_month(365)),
    )

    # L2 -> L3 on two GPUs
    for key, (_, rate) in GPUS.items():
        v[f"gpu_{key}"], v[f"gpu_{key}_8"], v[f"gpu_{key}_24"] = (
            usd(rate, 2),
            usd(rate * 8 * DAYS),
            usd(rate * 24 * DAYS),
        )
    avoided = [egress(gib) - egress(gib - m["L2"]) for gib in (low_gib, high_gib)]  # marginal tier in each case
    v["avoided_egress"] = usd_range(min(avoided), max(avoided))
    for key, gpu in (("l4_hours", "l4"), ("l3_gpu_hours", L3_GPU)):
        hours = [(a - GPU_EBS_GIB * p["gp3"]) / (DAYS * GPUS[gpu][1]) for a in avoided]  # break-even, h/day each
        v[key] = f"{min(hours):.1f}-{max(hours):.1f}"
    v["sci_l3"] = usd_range(*(share * m["L3"] * 0.09 for share in SCIENTIST_SHARES), 10)
    v["xr_l2"] = usd(m["L2"] * INTER_REGION, step=10)

    # Processing only: NASA carries S3 storage and user egress
    proc = processing_month()
    host, disk = sum(proc.values()), GPU_EBS_GIB * p["gp3"]
    v.update({f"proc_{k}": usd(x, 2) for k, x in zip(("ec2", "ebs", "rds", "nat", "ops"), proc.values(), strict=True)})
    v["proc_host"] = usd(host, 2)
    for hours in (4, 8, 12, 24):
        pair = L3_PAIR * hours * DAYS + disk
        v[f"proc_one_{hours}"], v[f"proc_pair_{hours}"] = usd(pair / 2), usd(pair)
        v[f"proc_host_one_{hours}"], v[f"proc_host_pair_{hours}"] = usd(host + pair / 2), usd(host + pair)

    # N. California vs Oregon
    lo_nc, hi_nc = month_range("us-west-1")
    more = [sum(day_nc.values()) / total - 1, lo_nc / lo - 1, hi_nc / hi - 1]
    v["nc_pct"] = f"{min(more) * 100:.0f}-{max(more):.0%}"
    return v


def main() -> None:
    p = PRICES[REGION]
    print(f"GiB/day  L0 {L0:.2f}  L1 {L1:.2f}  L2 {L2:.2f}  L3 {L3:.2f}")
    print(f"files/day  L0 {L0_FILES:,.0f}  L2 {L2_FILES:,.0f}  L3 {L3_FILES:,.0f}")
    for region in REGIONS:
        day = one_day(region)
        print(f"\n== one day, {region}: ${sum(day.values()):.2f}")
        for k, x in day.items():
            print(f"  {k:28s} {x:8.2f}  {100 * x / sum(day.values()):4.1f}%")
    print(f"\n== always-on month, {REGION}: ${sum(always_on_month().values()):,.2f}")
    for k, x in always_on_month().items():
        print(f"  {k:28s} {x:9.2f}")
    print(f"\n== EBS /month: proposed {EBS_GIB:,} GiB ${EBS_GIB * p['gp3']:,.2f}; if nothing is ever deleted:")
    for d in (30, 90, 365):
        print(f"  day {d:3d}  ${ebs_month(d):10,.2f}  ({(EBS_STATE_GIB + d * (L0 + L1 + L2)) / 1024:,.0f} TiB)")
    for label, mission in (("L2/L3 kept 90 days", False), ("L2/L3 kept for the mission", True)):
        s = s3_month(1, mission)
        print(f"\n== S3 run rate after year 1, {label}: ${sum(s.values()):,.2f}/month")
        for k, x in s.items():
            if x:
                print(f"  {k:32s} {x:9.2f}")
    low, high = download_gib()
    print(f"\n== egress /month: low {low / 1024:.1f} TiB ${egress(low):,.2f}", end=";")
    print(f"  high {high / 1024:.1f} TiB ${egress(high):,.2f}")
    print(f"\n== month at year 1, {REGION}, low-high: {usd_range(*month_range())}")
    host = sum(processing_month().values())
    pair8 = L3_PAIR * 8 * DAYS + GPU_EBS_GIB * p["gp3"]
    print(f"== processing only (no S3, no egress): host ${host:,.2f}; + GPU pair 8 h/day ${host + pair8:,.2f}")
    print(f"\n  N. California vs Oregon: {slide_variables()['nc_pct']} more")


def write_variables(path: str = "_variables.yml") -> None:
    lines = ["# Generated by cost_model.py (Quarto pre-render). Do not edit.", "cost:"]
    lines += [f"  {k}: {json.dumps(val, ensure_ascii=False)}" for k, val in slide_variables().items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "QUARTO_PROJECT_DIR" not in os.environ:  # quiet when run as Quarto's pre-render step
        main()
    write_variables()
