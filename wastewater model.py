#!/usr/bin/env python
# coding: utf-8

# In[1]:


# eb_tea_app.py — Interactive TEA dashboard (Streamlit, USD edition v2.5)
"""Electron Beam Techno Economic Assessment
────────────────────────────────────────────
* All currency units in USD (consistent throughout)
* Deterministic + Monte Carlo L_COT
* Triangular uncertainty for electricity price & utilisation
* Log normal uncertainty for dose and efficiency

v2.5 changes
────────────
* Unified all currency units to USD (was mixed CNY/USD)
* Updated labels and metrics to reflect USD consistency
* Fixed distribution graph currency unit

Run
───
streamlit run eb_tea_app.py
"""
from __future__ import annotations

from typing import Final
import numpy as np
import pandas as pd
import streamlit as st

# ---------- plotting backend ---------------------------------------------
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL: Final[bool] = True
except ModuleNotFoundError:
    HAS_MPL = False
    import plotly.express as px  # type: ignore

# ---------- defaults -----------------------------------------------------
DEFAULT = {
    "PlantCapacity": 30_000,   # m³ d⁻¹
    "OperatingDays": 330,      # d a⁻¹
    "Utilisation":    0.90,
    "DoseMode":      2.0,     # kGy
    "EffMode":       0.48,    # –
    "ElecMu":        0.063,   # $ kWh⁻¹ (most likely) - converted from 0.453 CNY
    "CAPEX":         21.0,    # M$ (total – one off) - converted from 150 MCNY
    "Labour":        0.154,   # M$ a⁻¹ - converted from 1.1 MCNY
    "MaintFrac":     0.03,    # – (annual % of CAPEX)
}
CRF: Final[float] = 0.10          # annuity factor ≈ 20 yr @ 8 %
MC_N_DEFAULT: Final[int] = 5_000
FX_RATE: Final[float] = 7.15      # CNY to USD conversion rate

# ---------- core maths ---------------------------------------------------

def _unit_energy(dose_kgy: float, eff: float) -> float:
    """kWh m⁻³ for given kGy & efficiency (MJ → kWh / η)."""
    return dose_kgy * 1e3 / 3600.0 / eff  # (kGy ≡ kJ kg⁻¹)

def deterministic(inp: dict[str, float]) -> pd.Series:
    """Single point L_COT & cost breakdown in USD."""
    vol = inp["PlantCapacity"] * inp["OperatingDays"] * inp["Utilisation"]
    unit_e = _unit_energy(inp["DoseMode"], inp["EffMode"])
    elec_kwh = vol * unit_e

    opex_e = elec_kwh * inp["ElecMu"] / 1e6  # M$
    opex_m = inp["CAPEX"] * inp["MaintFrac"]  # M$
    opex_tot = opex_e + opex_m + inp["Labour"]  # M$
    capex_a = inp["CAPEX"] * CRF  # M$

    l_usd = (opex_tot + capex_a) * 1e6 / vol  # $ m⁻³
    return pd.Series({
        "Annual volume (Mm³)"      : vol / 1e6,
        "Unit energy (kWh/m³)"     : unit_e,
        "Electric OPEX (M$)"       : opex_e,
        "Maintenance OPEX (M$)"    : opex_m,
        "Labour OPEX (M$)"         : inp["Labour"],
        "Total OPEX (M$)"          : opex_tot,
        "Total CAPEX (M$)"         : inp["CAPEX"],
        "Annual CAPEX (M$)"        : capex_a,
        "L_COT (USD/m³)"           : l_usd,
    })

def monte_carlo(
    inp: dict[str, float],
    elec_tri: tuple[float, float, float],
    util_tri: tuple[float, float, float],
    ln_sigma: float,
    n: int,
) -> np.ndarray:
    """Return L_COT USD vector — triangular for price/util, log normal for others."""
    rng = np.random.default_rng(42)

    price = rng.triangular(*elec_tri, n)
    util  = rng.triangular(*util_tri, n)

    dose = inp["DoseMode"] * np.exp(rng.normal(0, ln_sigma, n))
    eff  = inp["EffMode"]  * np.exp(rng.normal(0, ln_sigma, n))

    vol = inp["PlantCapacity"] * inp["OperatingDays"] * util
    capex_a = inp["CAPEX"] * CRF
    opex_m  = inp["CAPEX"] * inp["MaintFrac"]

    unit_e = _unit_energy(dose, eff)
    l_usd = ((vol * unit_e * price / 1e6) + opex_m + inp["Labour"] + capex_a) * 1e6 / vol
    return l_usd  # USD/m³

# ---------- Streamlit UI -------------------------------------------------

st.set_page_config(page_title="EB TEA USD Dashboard", layout="wide")
st.title("Electron Beam Techno Economic Assessment (USD)")

with st.sidebar:
    st.header("Input Parameters - Base Values")
    p: dict[str, float] = {}
    p["PlantCapacity"] = st.number_input("Plant Capacity (m³/day)", 1_000, 100_000, int(DEFAULT["PlantCapacity"]), 1_000)
    p["OperatingDays"] = st.slider("Operating Days per Year", 200, 365, int(DEFAULT["OperatingDays"]))
    p["Utilisation"]   = st.slider("Utilization (Mode)", 0.5, 1.0, float(DEFAULT["Utilisation"]), 0.01)
    p["DoseMode"]      = st.slider("Dose (kGy)", 0.5, 5.0, float(DEFAULT["DoseMode"]), 0.1)
    p["EffMode"]       = st.slider("Efficiency (η)", 0.3, 0.8, float(DEFAULT["EffMode"]), 0.01)
    p["ElecMu"]        = st.number_input("Electricity Price ($/kWh)", 0.02, 0.15, float(DEFAULT["ElecMu"]), 0.001)
    p["CAPEX"]         = st.number_input("Total CAPEX (Million $)", 1.0, 70.0, float(DEFAULT["CAPEX"]), 0.1)
    p["Labour"]        = st.number_input("Labour Cost (Million $/year)", 0.01, 1.5, float(DEFAULT["Labour"]), 0.01)
    p["MaintFrac"]     = st.slider("Maintenance % of CAPEX", 0.01, 0.10, float(DEFAULT["MaintFrac"]), 0.005)

    st.header("Uncertainty Ranges - Triangular Distribution")
    st.caption("Electricity price: $0.052-$0.077/kWh, Utilization: 0.70-0.90")
    e_min  = st.number_input("Elec Min ($/kWh)", 0.01, 0.20, 0.052, 0.001)
    e_mode = st.number_input("Elec Mode ($/kWh)", 0.01, 0.20, p["ElecMu"], 0.001)
    e_max  = st.number_input("Elec Max ($/kWh)", 0.01, 0.20, 0.077, 0.001)

    u_min  = st.slider("Util Min", 0.3, 1.0, 0.70, 0.01)
    u_mode = p["Utilisation"]
    u_max  = st.slider("Util Max", 0.3, 1.0, 0.90, 0.01)

    st.header("Other Settings")
    ln_sigma = st.slider("Log Standard Deviation (Dose/Efficiency)", 0.05, 0.40, 0.20, 0.01)
    samples  = st.slider("Monte Carlo Samples", 1_000, 20_000, MC_N_DEFAULT, 1_000)

# ---------- results ------------------------------------------------------

res = deterministic(p)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deterministic L_COT", f"${res['L_COT (USD/m³)']:.3f}/m³")
c2.metric("Electric OPEX", f"${res['Electric OPEX (M$)']:.3f}M")
c3.metric("Maintenance OPEX", f"${res['Maintenance OPEX (M$)']:.3f}M")
c4.metric("Annual CAPEX", f"${res['Annual CAPEX (M$)']:.3f}M")

left, right = st.columns(2)
with left:
    st.subheader("Deterministic Cost Breakdown")
    st.dataframe(res.to_frame("Value").style.format("{:.3f}"))
with right:
    st.subheader("Cost Composition (Million $)")
    pie_vals = [res["Electric OPEX (M$)"],
                res["Maintenance OPEX (M$)"],
                p["Labour"],
                res["Annual CAPEX (M$)"]]
    pie_lbls = ["Electricity", "Maintenance", "Labour", "Annual CAPEX"]
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(pie_vals, labels=pie_lbls, autopct="%.1f%%")
        ax.axis("equal")
        ax.set_title("Cost Composition (USD)")
        st.pyplot(fig)
    else:
        fig = px.pie(values=pie_vals, names=pie_lbls, hole=0.4, title="Cost Composition (USD)")
        st.plotly_chart(fig)

# ---------- Monte Carlo --------------------------------------------------
mc = monte_carlo(p, (e_min, e_mode, e_max), (u_min, u_mode, u_max), ln_sigma, samples)

st.subheader("Monte Carlo Simulation - L_COT Distribution ($/m³)")
mc_l, mc_r = st.columns([3, 1])
with mc_l:
    if HAS_MPL:
        fig2, ax2 = plt.subplots()
        ax2.hist(mc, bins=40, color="teal", alpha=0.7)
        ax2.set_xlabel("L_COT ($/m³)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("L_COT Probability Distribution")
        st.pyplot(fig2)
    else:
        fig = px.histogram(x=mc, nbins=40, opacity=0.75,
                          labels={"x": "L_COT ($/m³)", "y": "Frequency"},
                          title="L_COT Probability Distribution")
        st.plotly_chart(fig)
with mc_r:
    st.metric("P10 (10th Percentile)", f"${np.percentile(mc, 10):.3f}/m³")
    st.metric("P50 (Median)", f"${np.percentile(mc, 50):.3f}/m³")
    st.metric("P90 (90th Percentile)", f"${np.percentile(mc, 90):.3f}/m³")

# ---------- CLI test -----------------------------------------------------
if __name__ == "__main__":
    print("Deterministic L_COT:", deterministic(DEFAULT)["L_COT (USD/m³)"], "$/m³")


# In[2]:



from typing import Final
import numpy as np
import pandas as pd
import streamlit as st

# ---------- plotting backend ---------------------------------------------
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MPL: Final[bool] = True
except ModuleNotFoundError:
    HAS_MPL = False
    import plotly.express as px  # type: ignore

# ---------- defaults -----------------------------------------------------
FX_DEFAULT: Final[float] = 7.15  # CNY → USD (CNY per 1 USD)
DEFAULT = {
    "PlantCapacity": 30_000,   # m³ d⁻¹
    "OperatingDays": 330,      # d a⁻¹
    "Utilisation":    0.90,
    "DoseMode":      2.0,      # kGy
    "EffMode":       0.48,     # –
    "ElecMu":        0.453,    # ¥ kWh⁻¹ (mode)
    "CAPEX":         150.0,    # M¥ (total – one off)
    "Labour":        1.1,      # M¥ a⁻¹
    "MaintFrac":     0.03,     # – (annual % of CAPEX)
}
CRF: Final[float] = 0.10          # annuity factor ≈ 20 yr @ 8 %
MC_N_DEFAULT: Final[int] = 5_000

# ---------- core maths ---------------------------------------------------

def _unit_energy(dose_kgy: float, eff: float) -> float:
    """kWh m⁻³ for given kGy & efficiency (MJ → kWh / η)."""
    return dose_kgy * 1e3 / 3600.0 / eff  # (kGy ≡ kJ kg⁻¹ ~ kJ L⁻¹)

def deterministic(inp: dict[str, float], fx: float) -> pd.Series:
    """Single point L_COT & cost breakdown. All monetary outputs in **USD**."""
    vol = inp["PlantCapacity"] * inp["OperatingDays"] * inp["Utilisation"]
    unit_e = _unit_energy(inp["DoseMode"], inp["EffMode"])
    elec_kwh = vol * unit_e

    # ---- compute in CNY first (millions), then convert to USD (millions) ----
    opex_e_m_cny = elec_kwh * inp["ElecMu"] / 1e6         # M¥
    opex_m_m_cny = inp["CAPEX"] * inp["MaintFrac"]         # M¥
    labour_m_cny = inp["Labour"]                           # M¥
    capex_a_m_cny = inp["CAPEX"] * CRF                     # M¥
    opex_tot_m_cny = opex_e_m_cny + opex_m_m_cny + labour_m_cny

    # conversions
    def cnyM_to_usdM(x_m_cny: float) -> float:
        return x_m_cny / fx

    opex_e_m_usd  = cnyM_to_usdM(opex_e_m_cny)
    opex_m_m_usd  = cnyM_to_usdM(opex_m_m_cny)
    labour_m_usd  = cnyM_to_usdM(labour_m_cny)
    capex_a_m_usd = cnyM_to_usdM(capex_a_m_cny)
    opex_tot_m_usd= cnyM_to_usdM(opex_tot_m_cny)
    capex_tot_m_usd = cnyM_to_usdM(inp["CAPEX"])           # M$ (total CAPEX)

    l_cny_per_m3 = (opex_tot_m_cny + capex_a_m_cny) * 1e6 / vol  # ¥/m³
    l_usd_per_m3 = l_cny_per_m3 / fx

    return pd.Series({
        "Annual vol (Mm³)"      : vol / 1e6,
        "Unit energy (kWh/m³)"  : unit_e,
        # ---- USD outputs (millions where noted) ----
        "Electric OPEX (M$)"    : opex_e_m_usd,
        "Maintenance (M$)"      : opex_m_m_usd,
        "Labour (M$)"           : labour_m_usd,
        "Total OPEX (M$)"       : opex_tot_m_usd,
        "Total CAPEX (M$)"      : capex_tot_m_usd,
        "Ann. CAPEX (M$)"       : capex_a_m_usd,
        "L_COT (USD/m³)"        : l_usd_per_m3,
    })

def monte_carlo(
    inp: dict[str, float],
    elec_tri: tuple[float, float, float],
    util_tri: tuple[float, float, float],
    ln_sigma: float,
    n: int,
    fx: float,
) -> np.ndarray:
    """Return L_COT USD vector — triangular for price/util, log normal for dose/eff."""
    rng = np.random.default_rng(42)

    price = rng.triangular(*elec_tri, n)     # ¥/kWh
    util  = rng.triangular(*util_tri, n)

    dose = inp["DoseMode"] * np.exp(rng.normal(0, ln_sigma, n))
    eff  = inp["EffMode"]  * np.exp(rng.normal(0, ln_sigma, n))

    vol = inp["PlantCapacity"] * inp["OperatingDays"] * util
    capex_a_m_cny = inp["CAPEX"] * CRF
    opex_m_m_cny  = inp["CAPEX"] * inp["MaintFrac"]
    labour_m_cny  = inp["Labour"]

    unit_e = _unit_energy(dose, eff)

    # L_COT in CNY first, then convert to USD
    l_cny = ((vol * unit_e * price / 1e6) + opex_m_m_cny + labour_m_cny + capex_a_m_cny) * 1e6 / vol
    return l_cny / fx

# ---------- Streamlit UI -------------------------------------------------

st.set_page_config(page_title="EB TEA — USD outputs", layout="wide")
st.title("Electron Beam Techno-Economic Assessment (All outputs in USD)")

with st.sidebar:
    st.header("Inputs – base values (CNY)")
    p: dict[str, float] = {}
    p["PlantCapacity"] = st.number_input("Plant capacity (m³ d⁻¹)", 1_000, 100_000, int(DEFAULT["PlantCapacity"]), 1_000)
    p["OperatingDays"] = st.slider("Operating days a⁻¹", 200, 365, int(DEFAULT["OperatingDays"]))
    p["Utilisation"]   = st.slider("Utilisation (mode)", 0.5, 1.0, float(DEFAULT["Utilisation"]), 0.01)
    p["DoseMode"]      = st.slider("Dose mode (kGy)", 0.5, 5.0, float(DEFAULT["DoseMode"]), 0.1)
    p["EffMode"]       = st.slider("Efficiency mode (η)", 0.3, 0.8, float(DEFAULT["EffMode"]), 0.01)
    p["ElecMu"]        = st.number_input("Elec price mode (¥ kWh⁻¹)", 0.20, 1.00, float(DEFAULT["ElecMu"]), 0.001)
    p["CAPEX"]         = st.number_input("Total CAPEX (M¥)", 10.0, 500.0, float(DEFAULT["CAPEX"]), 1.0)
    p["Labour"]        = st.number_input("Labour (M¥ a⁻¹)", 0.1, 10.0, float(DEFAULT["Labour"]), 0.1)
    p["MaintFrac"]     = st.slider("Maintenance % of CAPEX", 0.01, 0.10, float(DEFAULT["MaintFrac"]), 0.005)

    st.header("Uncertainty ranges – triangular (CNY for prices)")
    st.caption("Electricity price 0.372–0.554 ¥ kWh⁻¹ & utilisation 0.70–0.90 default")
    e_min  = st.number_input("Elec min (¥/kWh)", 0.1, 2.0, 0.372, 0.001)
    e_mode = st.number_input("Elec mode (¥/kWh)", 0.1, 2.0, p["ElecMu"], 0.001)
    e_max  = st.number_input("Elec max (¥/kWh)", 0.1, 2.0, 0.554, 0.001)

    u_min  = st.slider("Util min", 0.3, 1.0, 0.70, 0.01)
    u_mode = p["Utilisation"]
    u_max  = st.slider("Util max", 0.3, 1.0, 0.90, 0.01)

    st.header("Other settings")
    ln_sigma = st.slider("ln sigma (dose/eff)", 0.05, 0.40, 0.20, 0.01)
    samples  = st.slider("MC samples", 1_000, 20_000, MC_N_DEFAULT, 1_000)
    fx       = st.number_input("CNY per USD (汇率)", 5.0, 10.0, float(FX_DEFAULT), 0.01)

# ---------- results ------------------------------------------------------

res = deterministic(p, fx)

c1, c2, c3, c4 = st.columns(4)
c1.metric("L_COT det (USD/m³)", f"{res['L_COT (USD/m³)']:.3f}")
c2.metric("Electric OPEX (M$)", f"{res['Electric OPEX (M$)']:.3f}")
c3.metric("Maintenance (M$)",   f"{res['Maintenance (M$)']:.3f}")
c4.metric("Ann. CAPEX (M$)",    f"{res['Ann. CAPEX (M$)']:.3f}")

left, right = st.columns(2)
with left:
    st.subheader("Deterministic breakdown (USD)")
    # 统一格式：非货币列也一起三位小数即可
    st.dataframe(res.to_frame("Value").style.format("{:.3f}"))
with right:
    st.subheader("Cost split (USD, millions)")
    pie_vals = [res["Electric OPEX (M$)"],
                res["Maintenance (M$)"],
                res["Labour (M$)"],
                res["Ann. CAPEX (M$)"]]
    pie_lbls = ["Electric", "Maintenance", "Labour", "Ann. CAPEX"]
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(pie_vals, labels=pie_lbls, autopct="%.1f%%")
        ax.axis("equal")
        st.pyplot(fig)
    else:
        st.plotly_chart(px.pie(values=pie_vals, names=pie_lbls, hole=0.4))

# ---------- Monte Carlo --------------------------------------------------
mc = monte_carlo(p, (e_min, e_mode, e_max), (u_min, u_mode, u_max),
                 ln_sigma, samples, fx)

st.subheader("Monte Carlo – L_COT distribution (USD/m³)")
mc_l, mc_r = st.columns([3, 1])
with mc_l:
    if HAS_MPL:
        fig2, ax2 = plt.subplots()
        ax2.hist(mc, bins=40, color="teal", alpha=0.7)
        ax2.set_xlabel("L_COT ($/m³)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.plotly_chart(px.histogram(x=mc, nbins=40, opacity=0.75))
with mc_r:
    st.metric("P10", f"{np.percentile(mc, 10):.3f}")
    st.metric("P50", f"{np.percentile(mc, 50):.3f}")
    st.metric("P90", f"{np.percentile(mc, 90):.3f}")

# ---------- CLI test -----------------------------------------------------
if __name__ == "__main__":
    print("Deterministic L_COT (USD/m³):",
          deterministic(DEFAULT, FX_DEFAULT)["L_COT (USD/m³)"])


# In[ ]:




