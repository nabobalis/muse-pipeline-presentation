"""MUSE AWS cost model: one day, one month, egress, storage, and L3 on two GPUs.

Prices are AWS On-Demand list prices from the regional price-list files published
2026-09-16..22 (EC2 2026-09-21, RDS 2026-09-22, S3 2026-09-18, data transfer 2026-09-16).

Storage policy: the EC2 disk only holds files in flight. L0 is fetched from S3, L1 is
temporary (never stored in S3, rebuilt from L0 when needed), L2 is uploaded to S3, then
local copies are deleted. Today's code never deletes; ebs_month() prices that case.
"""

import json
import os

GIB = 2**30
REGION = "us-west-2"  # everything runs in Oregon; N. California (us-west-1) kept for comparison
DAYS = 30          # the daily doc's month for storage proration and daily batches
HOURS = 730        # AWS billing month for hourly resources

# ---- data volumes (GiB per input day) -------------------------------------------
L0_GB_DAY = 389e9
L0 = L0_GB_DAY / GIB                                  # 362.28
L1 = L0 * 75_052_800 / 73_987_200                    # measured L1/L0 on the 20-file run
L2 = L0 * 74_836_800 / 73_987_200                    # measured L2/L0
# L3: one merged file per SG observation, float32 moments + errors + uint8/uint16 flags, lossless
# GZIP_2. Measured on the baseline_gpu run (synthetic M-flare): 0.34x L2 per SG exposure,
# compressed to 0.79x. Range 40-45 (low-signal pixels blanked) .. 370-570 (L3A/L3B/L2.5 all published);
# the VDEM cube, if published, adds 530-680 GiB/day.
L3 = 124.0

# object counts from the full-size synthetic corpus: 16,340 L0 -> 16,340 L1 + 56 L2
FILE_BYTES = 48.64509504e9 / 16_340
L0_FILES = L0_GB_DAY / FILE_BYTES                    # ~130.7k/day
L1_FILES = L0_FILES
L2_FILES = L0_FILES * 56 / 16_340                    # ~448/day
L3_FILES = L2_FILES / 4                              # one L3 per 3 SG L2 files; CI (1 of 4 cameras) has none

PRICES = {
    "us-west-1": dict(ec2=0.4704, gp3=0.096, snap=0.055, nat=0.048, ipv4=0.005,
                      rds_maz=0.384, rds_saz=0.192, rds_gp3_maz=0.276, rds_gp3_saz=0.138, rds_backup=0.095,
                      s3=0.026, glacier=0.0045, gir=0.005, deep=0.002,
                      put=0.0055e-3, get=0.00044e-3, glacier_transition=0.033e-3, gir_transition=0.02e-3,
                      m7g_2xl=0.3808, m7g_spot_saving=0.54, t4g_medium=0.04, ec2_1yr=0.31025,
                      rds_t4g_medium_saz=0.085, s3_int_ia=0.0144, deep_restore=0.0035),
    "us-west-2": dict(ec2=0.4032, gp3=0.080, snap=0.050, nat=0.045, ipv4=0.005,
                      rds_maz=0.337, rds_saz=0.168, rds_gp3_maz=0.230, rds_gp3_saz=0.115, rds_backup=0.095,
                      s3=0.023, glacier=0.0036, gir=0.004, deep=0.00099,
                      put=0.005e-3, get=0.0004e-3, glacier_transition=0.03e-3, gir_transition=0.02e-3,
                      m7g_2xl=0.3264, m7g_spot_saving=0.63, t4g_medium=0.0336, ec2_1yr=0.26672,
                      rds_t4g_medium_saz=0.065, s3_int_ia=0.0125, deep_restore=0.0025),
}
# m7g_spot_saving: AWS Spot Advisor typical saving on 2026-09-23 (live Spot prices move daily);
# ec2_1yr: m7i.2xlarge 1-year no-upfront reservation; deep_restore: Deep Archive bulk restore per GB.
EGRESS_TIERS = ((10_240, 0.09), (40_960, 0.085), (102_400, 0.07), (float("inf"), 0.05))
FREE_EGRESS = 100
INTER_REGION = 0.02
EBS_GIB = 1_024              # root + state + files in flight; nothing kept after upload
SCIENTIST_SHARES = (0.10, 0.25)  # share of L2 and L3 users download; depends on what is observed
EBS_STATE_GIB = 132         # root + Docker/log/artifact state; the rest is science data

GPUS = {  # two GPUs, Oregon; $/hour for the pair
    "2x T4 16 GB (2x g4dn.xlarge)": 2 * 0.526,
    "2x L4 24 GB (2x g6.xlarge)": 2 * 0.8048,
    "2x A10G 24 GB (2x g5.xlarge)": 2 * 1.006,
    "2x A10G 24 GB, 16 vCPU / 64 GiB each (2x g5.4xlarge)": 2 * 1.624,
    "2x L40S 48 GB (2x g6e.xlarge)": 2 * 1.861,
    "2x 96 GB (g7e.12xlarge)": 8.28608,
    "2x H100 80 GB (2x p5.4xlarge)": 2 * 6.88,
}
# Chosen for now; whether L2 -> L3 needs this much CPU and RAM is not yet qualified.
L3_GPU = "2x A10G 24 GB, 16 vCPU / 64 GiB each (2x g5.4xlarge)"

# ---- savings options (region-specific prices are in PRICES)
CORE_S_PER_GB = 1.5433 * 5212.20 / 48.64509504   # benchmark: avg worker cores x seconds / GB of L0
AWS_SLOWDOWN = 1.5                               # assumption: AWS vCPU vs the benchmark desktop core
BUNDLE_GB = 1.0                                  # L0 bundled into ~1 GB files before upload
SAVE = dict(lambda_arm=0.0000133334, lambda_x86=0.0000166667, lambda_gb_per_vcpu=1.769,  # same both Regions
            science_share=0.25)


def egress(gib: float) -> float:
    left, cost = max(0.0, gib - FREE_EGRESS), 0.0
    for size, rate in EGRESS_TIERS:
        step = min(left, size)
        cost += step * rate
        left -= step
    return cost


def requests_day(p: dict) -> float:
    puts = L0_FILES + L2_FILES + L3_FILES                     # L0 lands in our bucket too; L1 is not uploaded
    gets = L0_FILES + L2_FILES                                # EC2 reads L0; external L3 reads L2
    return puts * p["put"] + gets * p["get"]


def one_day(region: str = REGION) -> dict:
    p = PRICES[region]
    return {
        "EC2 m7i.2xlarge incl. Dagster": 24 * p["ec2"],
        "EBS gp3 1,024 GiB": EBS_GIB * p["gp3"] / DAYS,
        "RDS Multi-AZ + 100 GiB": 24 * p["rds_maz"] + 100 * p["rds_gp3_maz"] / DAYS,
        "L2 internet download": L2 * 0.09,
        "S3 L0+L2+L3, one day": (L0 + L2 + L3) * p["s3"] / DAYS,
        "S3 requests": requests_day(p),
        "NAT gateway + public IPv4": 24 * (p["nat"] + p["ipv4"]),
        "RDS backup allowance": 100 * p["rds_backup"] / DAYS,
        "EBS snapshot allowance": 100 * p["snap"] / DAYS,
        "Logs, secrets, control traffic": 1.00,
    }


def always_on_month(region: str = REGION) -> dict:
    p = PRICES[region]
    return {
        "EC2 m7i.2xlarge": HOURS * p["ec2"],
        "RDS Multi-AZ + 100 GiB": HOURS * p["rds_maz"] + 100 * p["rds_gp3_maz"],
        "NAT + IPv4": HOURS * (p["nat"] + p["ipv4"]),
        "S3 requests": DAYS * requests_day(p),
        "Backups + snapshots": 100 * p["rds_backup"] + 100 * p["snap"],
        "Logs, secrets": 30.0,
    }


def ebs_month(days_retained: float, region: str = REGION) -> float:
    """Today's code never deletes: working disk holds every L0, L1 and L2."""
    return (EBS_STATE_GIB + days_retained * (L0 + L1 + L2)) * PRICES[region]["gp3"]


def s3_month(year: float, l23_mission: bool, region: str = REGION) -> dict:
    """Run rate after `year` years. L0: 30 d Standard then Glacier Flexible, kept.
    L1: not stored in S3. L2+L3: 90 d Standard, then Glacier Instant
    Retrieval (mission) or deleted. Flat first-tier Standard price (tiers cut ~4 % above 50 TB)."""
    p = PRICES[region]
    days = 365 * year
    old_l0 = max(0.0, days - 30)
    old_l23 = max(0.0, days - 90) if l23_mission else 0.0
    return {
        "L0 recent (Standard)": 30 * L0 * p["s3"],
        "L0 archive (Glacier Flexible)": old_l0 * L0 * p["glacier"],
        "L0 archive moves (per file)": DAYS * L0_FILES * p["glacier_transition"],
        "L2+L3 90 days (Standard)": 90 * (L2 + L3) * p["s3"],
        "L2+L3 archive (Glacier IR)": old_l23 * (L2 + L3) * p["gir"],
    }


def month_range(region: str = REGION) -> tuple[float, float]:
    """Steady month at the end of year 1: low = L2 download only, L2/L3 kept 90 days;
    high = L0 archive copy, NASA copy, scientists pull SAVE['science_share'], L2/L3 kept."""
    p, m = PRICES[region], {k: DAYS * v for k, v in dict(L0=L0, L2=L2, L3=L3).items()}
    fixed = sum(always_on_month(region).values()) + EBS_GIB * p["gp3"]
    high_gib = 2 * m["L2"] + m["L0"] + m["L3"] + SAVE["science_share"] * (m["L2"] + m["L3"])
    return (fixed + sum(s3_month(1, False, region).values()) + egress(m["L2"]),
            fixed + sum(s3_month(1, True, region).values()) + egress(high_gib))


NO_REGRET = ("EC2: + Graviton Spot", "RDS: single-zone db.t4g.medium", "NAT: public IPv4 only",
             "S3: L0 archive in Deep Archive", "S3: L0 bundled in ~1 GB files", "S3: L2+L3 Intelligent-Tiering")


def savings(region: str = REGION) -> dict:
    """Monthly (now, after) for each option, steady month at the end of year 1."""
    p, s = PRICES[region], {**SAVE, **PRICES[region]}
    vcpu_h_day = 389 * CORE_S_PER_GB * AWS_SLOWDOWN / 3600
    # worker hours: 8 vCPUs scale perfectly .. stuck at the benchmark's 2 workers (12 h/day)
    worker_h = (vcpu_h_day / 8 * 365 / 12, 12 * 365 / 12)
    control = HOURS * s["t4g_medium"]                      # Dagster stays on
    spot = s["m7g_2xl"] * (1 - s["m7g_spot_saving"])
    m = {k: DAYS * v for k, v in dict(L0=L0, L2=L2, L3=L3).items()}
    l23 = m["L2"] + m["L3"]
    high = 2 * m["L2"] + m["L0"] + m["L3"] + s["science_share"] * l23   # steady month, no L1 delivery
    ec2 = HOURS * p["ec2"]
    return {
        "vCPU-hours per day": (vcpu_h_day, vcpu_h_day),
        "EC2: workers per pass, off after": (ec2, tuple(h * p["ec2"] + control for h in worker_h)),
        "EC2: + Graviton Spot": (ec2, tuple(h * spot + control for h in worker_h)),
        "EC2: Lambda (ARM .. x86)": (ec2, tuple(vcpu_h_day * 3600 * s["lambda_gb_per_vcpu"] * DAYS * s[k]
                                                for k in ("lambda_arm", "lambda_x86"))),
        "EC2: 1-year Savings Plan, 24/7": (ec2, HOURS * s["ec2_1yr"]),
        "RDS: single-zone db.t4g.medium": (HOURS * p["rds_maz"] + 100 * p["rds_gp3_maz"],
                                           HOURS * s["rds_t4g_medium_saz"] + 100 * p["rds_gp3_saz"]),
        "NAT: public IPv4 only": (HOURS * (p["nat"] + p["ipv4"]), HOURS * p["ipv4"]),
        "S3: L0 archive in Deep Archive": (335 * L0 * p["glacier"], 335 * L0 * p["deep"]),
        "S3: L0 bundled in ~1 GB files": (DAYS * L0_FILES * (p["glacier_transition"] + p["put"]),
                                          DAYS * L0_GB_DAY / (BUNDLE_GB * 1e9) * (p["glacier_transition"] + p["put"])),
        "S3: L2+L3 Intelligent-Tiering": (90 * l23 / DAYS * p["s3"],
                                          (30 * p["s3"] + 60 * s["s3_int_ia"]) * l23 / DAYS),
        "Egress: NASA copy inside AWS": (egress(high), egress(high - l23)),
        "Egress: scientists pay (Requester Pays)": (egress(high - l23),
                                                    egress(high - l23 - s["science_share"] * l23)),
        "Egress: L3 in AWS, no L2 download": (egress(m["L2"]), 0.0),
        "Deep Archive bulk restore, per day of L0": (0.0, L0 * s["deep_restore"]),
    }


def main() -> None:
    print(f"GiB/day  L0 {L0:.2f}  L1 {L1:.2f}  L2 {L2:.2f}  L3 {L3:.2f}  total {L0+L1+L2+L3:.2f}")
    print(f"files/day L0 {L0_FILES:,.0f}  L2 {L2_FILES:,.0f}")
    for region in PRICES:
        day = one_day(region)
        print(f"\n== one day, {region}: ${sum(day.values()):.2f}")
        for k, v in day.items():
            print(f"  {k:34s} {v:8.2f}  {100*v/sum(day.values()):4.1f}%")
    print(f"\n== always-on month, {REGION}")
    base = always_on_month()
    for k, v in base.items():
        print(f"  {k:24s} {v:9.2f}")
    print(f"  {'total':24s} {sum(base.values()):9.2f}")
    print(f"  N. California total      {sum(always_on_month('us-west-1').values()):9.2f}")
    print(f"\n== EC2 disk (EBS), {REGION} /month")
    print(f"  bounded {EBS_GIB:,} GiB        {EBS_GIB*PRICES[REGION]['gp3']:9.2f}")
    for d in (30, 90, 365):
        print(f"  never deletes, day {d:3d}  {ebs_month(d):9.2f}   ({(EBS_STATE_GIB + d*(L0+L1+L2))/1024:,.0f} TiB)")
    print(f"\n== egress /month, {REGION}")
    month = {k: DAYS * v for k, v in dict(L2=L2, L1=L1, L0=L0, L3=L3).items()}
    cases = {
        "L2 once, for external L3": month["L2"],
        "+ L1 delivery (start of mission)": month["L2"] + month["L1"],
        "+ L0 archive copy": month["L2"] + month["L1"] + month["L0"],
        "+ NASA copy of L2 and L3": month["L2"] * 2 + month["L1"] + month["L0"] + month["L3"],
    }
    nasa = cases["+ NASA copy of L2 and L3"]
    for share in SCIENTIST_SHARES:
        cases[f"+ users pull {share:.0%} of L2 and L3"] = nasa + share * (month["L2"] + month["L3"])
    cases["steady month, no L1, users 25%"] = nasa - month["L1"] + SAVE["science_share"] * (month["L2"] + month["L3"])
    for k, v in cases.items():
        print(f"  {k:34s} {v/1024:6.1f} TiB  ${egress(v):9.2f}")
    print(f"  one extra full stream at the 2nd tier: ${month['L2']*0.085:.0f}")
    for label, mission in (("L2/L3 90 days", False), ("L2/L3 mission", True)):
        for year in (1, 2):
            s = s3_month(year, mission)
            print(f"\n== S3 run rate after year {year}, {label}: ${sum(s.values()):,.0f}/month")
            for k, v in s.items():
                print(f"  {k:32s} {v:9.2f}")
    p1 = PRICES[REGION]
    print(f"\n  L0 archive growth per year: ${365*L0*p1['glacier']:.0f}/month (Deep Archive ${365*L0*p1['deep']:.0f})")
    print(f"  L2+L3 archive growth per year: ${365*(L2+L3)*p1['gir']:.0f}/month")
    print("\n== L3 on two GPUs, /day and /month")
    for name, rate in GPUS.items():
        cells = "  ".join(f"{h:2d} h ${rate*h:7.2f}/d ${rate*h*DAYS:8.0f}/mo" for h in (8, 12, 24))
        print(f"  {name:44s} ${rate:6.3f}/h  {cells}")
    print(f"\n  saved if L3 runs in the same Region: L2 download ${L2*0.09:.2f}/day (${egress(month['L2']):.0f}/month)")
    print(f"  cross-Region L2: ${L2*INTER_REGION:.2f}/day; L3 back: ${L3*INTER_REGION:.2f}/day")
    print(f"  users still pull L3 once: ${L3*0.09:.2f}/day first tier")
    print(f"  break-even: 2x L4 hours/day covered by the L2 download: {L2*0.09/GPUS['2x L4 24 GB (2x g6.xlarge)']:.1f} h")
    print(f"\n== savings options, {REGION} /month (now -> after)")
    fmt = lambda v: "-".join(f"{x:,.0f}" for x in v) if isinstance(v, tuple) else f"{v:,.2f}"
    for k, (now, after) in savings().items():
        print(f"  {k:42s} {fmt(now):>8s} -> {fmt(after)}")
    sv = savings()
    cut = [sum(sv[k][0] - (sv[k][1][i] if isinstance(sv[k][1], tuple) else sv[k][1]) for k in NO_REGRET)
           for i in (1, 0)]                                  # smaller saving first
    lo, hi = month_range()
    data = sum(sv[k][0] - sv[k][1] for k in ("Egress: NASA copy inside AWS", "Egress: scientists pay (Requester Pays)"))
    print(f"\n  month now {lo:,.0f} .. {hi:,.0f}; no-regret saves {cut[0]:,.0f}-{cut[1]:,.0f}"
          f" -> {lo-cut[1]:,.0f}-{lo-cut[0]:,.0f} .. {hi-cut[1]:,.0f}-{hi-cut[0]:,.0f};"
          f" + data inside AWS saves {data:,.0f} -> high {hi-cut[1]-data:,.0f}-{hi-cut[0]-data:,.0f}")


def usd(x: float, nd: int = 0, step: float = 0) -> str:
    return f"${round(x / step) * step if step else x:,.{nd}f}"


def usd_range(a: float, b: float, step: float = 1) -> str:
    return f"${round(a / step) * step:,.0f}-{round(b / step) * step:,.0f}"


def slide_variables() -> dict:
    """Every number the AWS slides show, formatted; read by Quarto as {{< var cost.KEY >}}."""
    p = PRICES[REGION]
    m = {k: DAYS * v for k, v in dict(L0=L0, L1=L1, L2=L2, L3=L3).items()}
    tib = lambda g: f"{g / 1024:.1f} TiB"
    v = {}
    # one day
    day, day_nc = one_day(), one_day("us-west-1")
    l2 = day["L2 internet download"]
    big = {"ec2": "EC2 m7i.2xlarge incl. Dagster", "rds": "RDS Multi-AZ + 100 GiB", "ebs": "EBS gp3 1,024 GiB"}
    rest = sum(day.values()) - l2 - sum(day[k] for k in big.values())
    v["day_l2"] = usd(l2, 2)
    for key, name in big.items():
        v[f"day_{key}"], v[f"w_{key}"] = usd(day[name], 2), f"{100 * day[name] / l2:.0f}%"
    v["day_rest"], v["w_rest"] = usd(rest, 2), f"{100 * rest / l2:.0f}%"
    v["day_total"], v["day_nc"] = usd(sum(day.values()), 2), usd(sum(day_nc.values()), 2)
    v["day_saz"] = usd(24 * (p["rds_maz"] - p["rds_saz"]) + 100 * (p["rds_gp3_maz"] - p["rds_gp3_saz"]) / DAYS)
    v.update(l0_gib=f"{L0:.0f}", l1_gib=f"{L1:.0f}", l2_gib=f"{L2:.0f}", l3_gib=f"{L3:.0f}",
             l0_files=f"{L0_FILES / 1000:.0f}k", ec2_rate=f"${p['ec2']:g}", rds_rate=f"${p['rds_maz']:g}")
    v["day_s3"] = usd(day["S3 L0+L2+L3, one day"], 2)
    v["day_puts"] = f"{(L0_FILES + L2_FILES + L3_FILES) / 1000:.0f}k"
    v["day_gets"] = f"{(L0_FILES + L2_FILES) / 1000:.0f}k"
    v["day_req"], v["day_nat"] = usd(day["S3 requests"], 2), usd(day["NAT gateway + public IPv4"], 2)
    v["day_backups"] = usd(day["RDS backup allowance"] + day["EBS snapshot allowance"], 2)
    v["day_logs"] = usd(day["Logs, secrets, control traffic"], 2)
    # a month at the end of year 1
    lo, hi = month_range()
    high_gib = 2 * m["L2"] + m["L0"] + m["L3"] + SAVE["science_share"] * (m["L2"] + m["L3"])
    s3_lo, s3_hi = sum(s3_month(1, False).values()), sum(s3_month(1, True).values())
    v.update(m_always=usd(sum(always_on_month().values())), m_ebs=usd(EBS_GIB * p["gp3"]),
             m_s3_lo=usd(s3_lo), m_s3_hi=usd(s3_hi), m_eg_lo=usd(egress(m["L2"])), m_eg_hi=usd(egress(high_gib)),
             m_lo="~" + usd(lo, step=100), m_hi="~" + usd(hi, step=100),
             m_l0_moves=usd(DAYS * L0_FILES * p["glacier_transition"]),
             share_hi=f"{SAVE['science_share']:.0%}",
             shares=f"{SCIENTIST_SHARES[0] * 100:.0f}-{SCIENTIST_SHARES[1]:.0%}")
    # egress table (cumulative)
    rows = {"eg_l2": m["L2"], "eg_l1": m["L2"] + m["L1"], "eg_l0": m["L2"] + m["L1"] + m["L0"]}
    rows["eg_nasa"] = rows["eg_l0"] + m["L2"] + m["L3"]
    for key, gib in rows.items():
        v[key], v[key + "_tib"] = usd(egress(gib)), tib(gib)
    sci = [rows["eg_nasa"] + sh * (m["L2"] + m["L3"]) for sh in SCIENTIST_SHARES]
    v["eg_sci"] = usd_range(egress(sci[0]), egress(sci[1]))
    v["eg_sci_tib"] = f"{sci[0] / 1024:.1f}-{sci[1] / 1024:.1f} TiB"
    v["eg_extra"] = usd(m["L2"] * 0.085, step=10)
    # storage over the mission
    v.update(glacier_rate=f"${p['glacier']:g}", l0_growth=usd(365 * L0 * p["glacier"]),
             l0_growth_deep=usd(365 * L0 * p["deep"]), l1_90d=usd(90 * L1 * p["s3"], step=10),
             l23_90d=usd(90 * (L2 + L3) * p["s3"]), l23_growth=usd(365 * (L2 + L3) * p["gir"]),
             ebs_day_tib=f"{(L0 + L1 + L2) / 1024:.1f}",
             ebs_30_tib=f"{(EBS_STATE_GIB + 30 * (L0 + L1 + L2)) / 1024:.0f}",
             ebs_365_tib=f"{(EBS_STATE_GIB + 365 * (L0 + L1 + L2)) / 1024:.0f}",
             s3_y1="~" + usd_range(s3_lo, s3_hi, 100),
             s3_growth=usd_range(sum(s3_month(2, False).values()) - s3_lo,
                                 sum(s3_month(2, True).values()) - s3_hi, 100))
    # two GPUs
    for key, name in (("t4", "2x T4 16 GB (2x g4dn.xlarge)"), ("l4", "2x L4 24 GB (2x g6.xlarge)"),
                      ("a10g", "2x A10G 24 GB (2x g5.xlarge)"), ("g54", L3_GPU), ("l40s", "2x L40S 48 GB (2x g6e.xlarge)"),
                      ("h100", "2x H100 80 GB (2x p5.4xlarge)")):
        rate = GPUS[name]
        v[f"gpu_{key}"], v[f"gpu_{key}_8"], v[f"gpu_{key}_24"] = usd(rate, 2), usd(rate * 8 * DAYS), usd(rate * 24 * DAYS)
    v["l4_hours"] = f"{L2 * 0.09 / GPUS['2x L4 24 GB (2x g6.xlarge)']:.0f}"
    v["l3_gpu_hours"] = f"{L2 * 0.09 / GPUS[L3_GPU]:.0f}"
    v["sci_l3"] = usd_range(*(sh * m["L3"] * 0.09 for sh in SCIENTIST_SHARES), 10)
    v["xr_l2"], v["xr_l3"] = usd(m["L2"] * INTER_REGION, step=10), usd(m["L3"] * INTER_REGION, step=10)
    # savings
    sv = savings()
    fmt = lambda x: usd_range(*x) if isinstance(x, tuple) else usd(x)
    for key, name in (("scale", "EC2: workers per pass, off after"), ("spot", "EC2: + Graviton Spot"),
                      ("lambda", "EC2: Lambda (ARM .. x86)"), ("sp", "EC2: 1-year Savings Plan, 24/7"),
                      ("rds", "RDS: single-zone db.t4g.medium"), ("nat", "NAT: public IPv4 only")):
        v[f"sv_{key}_now"], v[f"sv_{key}"] = usd(sv[name][0]), fmt(sv[name][1])
    v.update(vcpu_h=f"{sv['vCPU-hours per day'][0]:.0f}", spot_pct=f"{p['m7g_spot_saving']:.0%}",
             slowdown=f"{AWS_SLOWDOWN:g}×")
    saved = {k: sv[k][0] - sv[k][1] for k in ("S3: L0 archive in Deep Archive", "S3: L0 bundled in ~1 GB files",
                                              "S3: L2+L3 Intelligent-Tiering", "Egress: NASA copy inside AWS",
                                              "Egress: scientists pay (Requester Pays)")}
    v.update(sv_deep=usd(saved["S3: L0 archive in Deep Archive"]), sv_bundle=usd(saved["S3: L0 bundled in ~1 GB files"]),
             sv_it=usd(saved["S3: L2+L3 Intelligent-Tiering"]),
             sv_store3=usd(sum(list(saved.values())[:3]), step=10),
             bundles_day=f"{L0_GB_DAY / (BUNDLE_GB * 1e9):.0f}",
             restore_day=usd(sv["Deep Archive bulk restore, per day of L0"][1], 2),
             sv_nasa="~" + usd(saved["Egress: NASA copy inside AWS"], step=10),
             sv_nasa_xr=usd((m["L2"] + m["L3"]) * INTER_REGION, step=10),
             sv_sci="~" + usd(saved["Egress: scientists pay (Requester Pays)"], step=10))
    cut = sum(sv[k][0] - (sum(sv[k][1]) / 2 if isinstance(sv[k][1], tuple) else sv[k][1]) for k in NO_REGRET)
    data = saved["Egress: NASA copy inside AWS"] + saved["Egress: scientists pay (Requester Pays)"]
    v.update(after_lo="~" + usd(lo - cut, step=100), after_hi=usd(hi - cut, step=100).lstrip("$"),
             after_data_hi="~" + usd(hi - cut - data, step=100),
             now_range="~" + usd_range(lo, hi, 100))
    # N. California vs Oregon
    lo_nc, hi_nc = month_range("us-west-1")
    more = [sum(day_nc.values()) / sum(day.values()) - 1, lo_nc / lo - 1, hi_nc / hi - 1]
    v["nc_pct"] = f"{min(more) * 100:.0f}-{max(more):.0%}"
    return v


def write_variables(path: str = "_variables.yml") -> None:
    lines = ["# Generated by cost_model.py (Quarto pre-render). Do not edit.", "cost:"]
    lines += [f"  {k}: {json.dumps(val, ensure_ascii=False)}" for k, val in slide_variables().items()]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if "QUARTO_PROJECT_DIR" not in os.environ:          # quiet when run as Quarto's pre-render step
        main()
    write_variables()
