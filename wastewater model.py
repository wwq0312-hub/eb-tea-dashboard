#!/usr/bin/env python
# coding: utf-8

# In[12]:


# eb_tea_app.py — Interactive TEA dashboard (Streamlit, USD edition v2.4)
"""Electron Beam Techno Economic Assessment
────────────────────────────────────────────
* Currency inputs **CNY**, KPIs **USD** (sidebar FX)
* Deterministic + Monte Carlo L_COT
* **Triangular** uncertainty for electricity price & utilisation (user set min / most likely / max)
* Log normal uncertainty (ln sigma) retained for dose and efficiency if desired.
* Works with Matplotlib **or** Plotly (fallback).

v2.4 changes
────────────
* **Total CAPEX** (input, M¥) is now listed explicitly in the deterministic table.
* Pie chart label changed from *CAPEX* → **Ann. CAPEX** to avoid ambiguity.

Run
───
```bash
pip install streamlit pandas numpy matplotlib plotly
streamlit run eb_tea_app.py            # → http://localhost:8501
```
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
FX_DEFAULT: Final[float] = 7.15  # CNY → USD
DEFAULT = {
    "PlantCapacity": 30_000,   # m³ d⁻¹
    "OperatingDays": 330,      # d a⁻¹
    "Utilisation":    0.90,
    "DoseMode":      2.0,     # kGy
    "EffMode":       0.48,    # –
    "ElecMu":        0.453,   # ¥ kWh⁻¹ (most likely)
    "CAPEX":         150.0,   # M¥ (total – one off)
    "Labour":        1.1,     # M¥ a⁻¹
    "MaintFrac":     0.03,    # – (annual % of CAPEX)
}
CRF: Final[float] = 0.10          # annuity factor ≈ 20 yr @ 8 %
MC_N_DEFAULT: Final[int] = 5_000

# ---------- core maths ---------------------------------------------------

def _unit_energy(dose_kgy: float, eff: float) -> float:
    """kWh m⁻³ for given kGy & efficiency (MJ → kWh / η)."""
    return dose_kgy * 1e3 / 3600.0 / eff  # (kGy ≡ kJ kg⁻¹)

def deterministic(inp: dict[str, float], fx: float) -> pd.Series:
    """Single point L_COT & cost breakdown."""
    vol = inp["PlantCapacity"] * inp["OperatingDays"] * inp["Utilisation"]
    unit_e = _unit_energy(inp["DoseMode"], inp["EffMode"])
    elec_kwh = vol * unit_e

    opex_e = elec_kwh * inp["ElecMu"] / 1e6
    opex_m = inp["CAPEX"] * inp["MaintFrac"]
    opex_tot = opex_e + opex_m + inp["Labour"]
    capex_a = inp["CAPEX"] * CRF

    l_cny = (opex_tot + capex_a) * 1e6 / vol  # ¥ m⁻³
    return pd.Series({
        "Annual vol (Mm³)"      : vol / 1e6,
        "Unit energy (kWh/m³)"  : unit_e,
        "Electric OPEX (M¥)"    : opex_e,
        "Maintenance (M¥)"      : opex_m,
        "Labour (M¥)"           : inp["Labour"],
        "Total OPEX (M¥)"       : opex_tot,
        "Total CAPEX (M¥)"      : inp["CAPEX"],            # NEW
        "Ann. CAPEX (M¥)"       : capex_a,
        "L_COT (USD/m³)"        : l_cny / fx,
    })

def monte_carlo(
    inp: dict[str, float],
    elec_tri: tuple[float, float, float],
    util_tri: tuple[float, float, float],
    ln_sigma: float,
    n: int,
    fx: float,
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
    l_cny = ((vol * unit_e * price / 1e6) + opex_m + inp["Labour"] + capex_a) * 1e6 / vol
    return l_cny / fx

# ---------- Streamlit UI -------------------------------------------------

st.set_page_config(page_title="EB TEA USD Dashboard", layout="wide")
st.title("Electron Beam Techno Economic Assessment (USD)")

with st.sidebar:
    st.header("Inputs – base values")
    p: dict[str, float] = {}
    p["PlantCapacity"] = st.number_input("Plant capacity (m³ d⁻¹)", 1_000, 100_000, int(DEFAULT["PlantCapacity"]), 1_000)
    p["OperatingDays"] = st.slider("Operating days a⁻¹", 200, 365, int(DEFAULT["OperatingDays"]))
    p["Utilisation"]   = st.slider("Utilisation (mode)", 0.5, 1.0, float(DEFAULT["Utilisation"]), 0.01)
    p["DoseMode"]      = st.slider("Dose mode (kGy)", 0.5, 5.0, float(DEFAULT["DoseMode"]), 0.1)
    p["EffMode"]       = st.slider("Efficiency mode (η)", 0.3, 0.8, float(DEFAULT["EffMode"]), 0.01)
    p["ElecMu"]        = st.number_input("Elec price mode (¥ kWh⁻¹)", 0.20, 1.00, float(DEFAULT["ElecMu"]), 0.001)
    p["CAPEX"]         = st.number_input("Total CAPEX (M¥)", 10.0, 500.0, float(DEFAULT["CAPEX"]), 1.0)
    p["Labour"]        = st.number_input("Labour (M¥ a⁻¹)", 0.1, 10.0, float(DEFAULT["Labour"]), 0.1)
    p["MaintFrac"]     = st.slider("Maintenance % of CAPEX", 0.01, 0.10, float(DEFAULT["MaintFrac"]), 0.005)

    st.header("Uncertainty ranges – triangular")
    st.caption("Electricity price 0.372–0.554 ¥ kWh⁻¹ & utilisation 0.70–0.90 default")
    e_min  = st.number_input("Elec min", 0.1, 2.0, 0.372, 0.001)
    e_mode = st.number_input("Elec mode", 0.1, 2.0, p["ElecMu"], 0.001)
    e_max  = st.number_input("Elec max", 0.1, 2.0, 0.554, 0.001)

    u_min  = st.slider("Util min", 0.3, 1.0, 0.70, 0.01)
    u_mode = p["Utilisation"]
    u_max  = st.slider("Util max", 0.3, 1.0, 0.90, 0.01)

    st.header("Other settings")
    ln_sigma = st.slider("ln sigma (dose/eff)", 0.05, 0.40, 0.20, 0.01)
    samples  = st.slider("MC samples", 1_000, 20_000, MC_N_DEFAULT, 1_000)
    fx       = st.number_input("CNY per USD", 5.0, 10.0, float(FX_DEFAULT), 0.01)

# ---------- results ------------------------------------------------------

res = deterministic(p, fx)

c1, c2, c3, c4 = st.columns(4)
c1.metric("L_COT det (USD)", f"{res['L_COT (USD/m³)']:.3f}")
c2.metric("Electric OPEX M¥", f"{res['Electric OPEX (M¥)']:.3f}")
c3.metric("Maint OPEX M¥",   f"{res['Maintenance (M¥)']:.3f}")
c4.metric("Ann. CAPEX M¥",    f"{res['Ann. CAPEX (M¥)']:.3f}")

left, right = st.columns(2)
with left:
    st.subheader("Deterministic breakdown")
    st.dataframe(res.to_frame("Value").style.format("{:.3f}"))
with right:
    st.subheader("Cost split (CNY)")
    pie_vals = [res["Electric OPEX (M¥)"],
                res["Maintenance (M¥)"],
                p["Labour"],
                res["Ann. CAPEX (M¥)"]]
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

st.subheader("Monte Carlo – L_COT USD distribution")
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
    print("Deterministic L_COT USD",
          deterministic(DEFAULT, FX_DEFAULT)["L_COT (USD/m³)"])

