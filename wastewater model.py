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
