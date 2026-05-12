"""
Sistema de Estimación de Reservas Mineras — Versión Web (Streamlit)
Autor: adaptado para Streamlit desde versión PySide6
"""

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from math import sqrt
from datetime import datetime

import streamlit as st
from scipy.spatial import cKDTree

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Estimación de Reservas Mineras",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS PERSONALIZADO
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ECEFF1; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #BBDEFB;
        color: #0D47A1;
        border-radius: 6px 6px 0 0;
        font-weight: bold;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1976D2 !important;
        color: white !important;
    }
    .metric-box {
        background: white;
        border: 2px solid #1976D2;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .report-box {
        background: #263238;
        color: #00E676;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        padding: 16px;
        border-radius: 8px;
        white-space: pre;
        overflow-x: auto;
    }
    .info-box {
        background: #E1F5FE;
        border: 2px solid #0277BD;
        border-radius: 8px;
        padding: 14px;
    }
    .success-box {
        background: #C8E6C9;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
# LÓGICA DE NEGOCIO  (sin cambios respecto al original)
# ═══════════════════════════════════════════

def generate_dummy_data(seed: int = 42) -> pd.DataFrame:
    """Genera datos de prueba simulando sondajes diamantinos."""
    np.random.seed(seed)
    n_drillholes = 15
    rows = []
    for dh_id in range(1, n_drillholes + 1):
        cx = np.random.uniform(50, 450)
        cy = np.random.uniform(50, 450)
        depth = np.random.uniform(40, 80)
        n_samples = int(depth / 0.5)
        for i in range(n_samples):
            z = -i * 0.5
            from_d = i * 0.5
            to_d = (i + 1) * 0.5
            dist = sqrt((cx - 250) ** 2 + (cy - 250) ** 2)
            base = 0.5 + 4.0 * np.exp(-(dist ** 2 / (2 * 100 ** 2)))
            depth_factor = 1.0 + 0.5 * np.exp(-abs(z + 30) ** 2 / 200)
            grade = max(0.1, base * depth_factor + np.random.normal(0, 0.4))
            rows.append({
                "BHID": f"DDH-{dh_id:03d}", "X": cx, "Y": cy, "Z": z,
                "FROM": from_d, "TO": to_d, "LENGTH": 0.5,
                "grade": round(grade, 3),
                "density": round(np.random.uniform(2.5, 2.8), 2),
            })
    return pd.DataFrame(rows)


def composite_data(df: pd.DataFrame, comp_length: float = 2.0) -> pd.DataFrame:
    """Composita datos por intervalos regulares ponderados por longitud."""
    if df.empty:
        return pd.DataFrame()
    composites = []
    for bhid, group in df.groupby("BHID"):
        group = group.sort_values("FROM").reset_index(drop=True)
        collar_x = group["X"].iloc[0]
        collar_y = group["Y"].iloc[0]
        depth_min = group["FROM"].min()
        depth_max = group["TO"].max()
        for start in np.arange(depth_min, depth_max, comp_length):
            end = start + comp_length
            mask = (group["TO"] > start) & (group["FROM"] < end)
            samp = group[mask]
            if samp.empty:
                continue
            total_len = 0.0
            w_grade = 0.0
            w_density = 0.0
            for _, row in samp.iterrows():
                i_start = max(row["FROM"], start)
                i_end = min(row["TO"], end)
                i_len = i_end - i_start
                if i_len > 0:
                    w_grade += row["grade"] * i_len
                    w_density += row["density"] * i_len
                    total_len += i_len
            if total_len > 0:
                composites.append({
                    "BHID": bhid, "X": collar_x, "Y": collar_y,
                    "Z": -(start + end) / 2,
                    "FROM": start, "TO": end, "LENGTH": total_len,
                    "grade": round(w_grade / total_len, 3),
                    "density": round(w_density / total_len, 2),
                })
    return pd.DataFrame(composites)


def calculate_cutoff_grade(
    price: float, recovery: float,
    cost_mine: float, cost_plant: float, cost_refine: float,
    factor: float = 31.1035,
) -> float | None:
    """Fórmula: Cutoff = [(Cm + Cp)*100] / [(P - Cr) * R * F]"""
    try:
        denom = (price - cost_refine) * recovery * factor
        if denom <= 0:
            return None
        return (cost_mine + cost_plant) * 100 / denom
    except Exception:
        return None


def generate_block_model_3d(
    xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz
) -> pd.DataFrame:
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    zs = np.linspace(zmin, zmax, nz + 1)
    blocks = []
    bid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x0, x1 = xs[i], xs[i + 1]
                y0, y1 = ys[j], ys[j + 1]
                z0, z1 = zs[k], zs[k + 1]
                blocks.append({
                    "BLOCK_ID": bid,
                    "CX": (x0 + x1) / 2, "CY": (y0 + y1) / 2, "CZ": (z0 + z1) / 2,
                    "X0": x0, "X1": x1, "Y0": y0, "Y1": y1, "Z0": z0, "Z1": z1,
                    "DX": x1 - x0, "DY": y1 - y0, "DZ": z1 - z0,
                    "VOLUME": (x1 - x0) * (y1 - y0) * (z1 - z0),
                })
                bid += 1
    return pd.DataFrame(blocks)


def estimate_blocks_idw(
    blocks_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    k: int = 12,
    power: float = 2.0,
) -> pd.DataFrame:
    """IDW – Inverso de la Distancia al Cuadrado."""
    if samples_df.empty:
        return blocks_df
    sc = samples_df[["X", "Y", "Z"]].values
    bc = blocks_df[["CX", "CY", "CZ"]].values
    tree = cKDTree(sc)
    grades, densities, n_used = [], [], []
    for center in bc:
        dists, idx = tree.query(center, k=min(k, len(sc)))
        if not isinstance(idx, np.ndarray):
            idx, dists = np.array([idx]), np.array([dists])
        dists = np.where(dists < 1e-10, 1e-10, dists)
        weights = 1.0 / dists ** power
        ws = weights.sum()
        grades.append((weights * samples_df.iloc[idx]["grade"].values).sum() / ws)
        densities.append((weights * samples_df.iloc[idx]["density"].values).sum() / ws)
        n_used.append(len(idx))
    bdf = blocks_df.copy()
    bdf["GRADE"] = grades
    bdf["DENSITY"] = densities
    bdf["N_SAMPLES"] = n_used
    bdf["TONNAGE"] = bdf["VOLUME"] * bdf["DENSITY"]
    bdf["METAL"] = bdf["TONNAGE"] * bdf["GRADE"]
    return bdf


# ═══════════════════════════════════════════
# ESTADO DE SESIÓN
# ═══════════════════════════════════════════

def init_state():
    defaults = {
        "raw_data": pd.DataFrame(),
        "composited_data": pd.DataFrame(),
        "block_model": pd.DataFrame(),
        "cutoff_grade": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/pick-emoji.png", width=60)
    st.title("⛏️ Reservas Mineras")
    st.markdown("**Sistema Profesional v2.0**")
    st.divider()

    # ── Datos ──────────────────────────────
    st.subheader("📁 Datos de Sondajes")
    uploaded = st.file_uploader(
        "Cargar Excel / CSV",
        type=["xlsx", "xls", "csv"],
        help="Columnas requeridas: BHID, X, Y, Z, FROM, TO, grade",
    )
    if uploaded:
        try:
            df_raw = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
            st.success(f"Archivo cargado: {len(df_raw)} filas")

            st.markdown("**Mapeo de columnas:**")
            cols = ["-- No mapear --"] + list(df_raw.columns)
            mapping = {}
            fields = [
                ("BHID", "ID Taladro"), ("X", "Coord. X"),
                ("Y", "Coord. Y"), ("Z", "Elevación/Prof."),
                ("FROM", "Prof. inicial"), ("TO", "Prof. final"),
                ("grade", "Ley (g/t)"), ("density", "Densidad (t/m³) – opcional"),
            ]
            for field, label in fields:
                default_idx = 0
                for ci, c in enumerate(df_raw.columns):
                    if field.lower() in str(c).lower():
                        default_idx = ci + 1
                        break
                mapping[field] = st.selectbox(
                    label, cols,
                    index=default_idx,
                    key=f"map_{field}",
                )

            if st.button("✅ Aplicar mapeo y cargar"):
                required = ["BHID", "X", "Y", "Z", "FROM", "TO", "grade"]
                missing = [f for f in required if mapping[f] == "-- No mapear --"]
                if missing:
                    st.error(f"Faltan campos: {', '.join(missing)}")
                else:
                    mapped = {}
                    for field, col in mapping.items():
                        if col != "-- No mapear --":
                            mapped[field] = df_raw[col] if field == "BHID" else df_raw[col].astype(float)
                    df_mapped = pd.DataFrame(mapped)
                    if "density" not in df_mapped.columns:
                        df_mapped["density"] = 2.65
                    if "LENGTH" not in df_mapped.columns:
                        df_mapped["LENGTH"] = df_mapped["TO"] - df_mapped["FROM"]
                    st.session_state["raw_data"] = df_mapped
                    st.session_state["composited_data"] = pd.DataFrame()
                    st.session_state["block_model"] = pd.DataFrame()
                    st.success("Datos cargados correctamente.")
        except Exception as e:
            st.error(f"Error al leer archivo: {e}")

    if st.button("🎲 Generar datos de prueba"):
        st.session_state["raw_data"] = generate_dummy_data()
        st.session_state["composited_data"] = pd.DataFrame()
        st.session_state["block_model"] = pd.DataFrame()
        st.success("Datos de prueba generados.")

    if not st.session_state["raw_data"].empty:
        df = st.session_state["raw_data"]
        st.markdown(
            f"""<div class='success-box'>
            ✅ <b>{len(df)}</b> muestras cargadas<br>
            🔩 <b>{df['BHID'].nunique()}</b> taladros<br>
            📊 Ley prom.: <b>{df['grade'].mean():.3f} g/t</b>
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Compositación ──────────────────────
    st.subheader("🔧 Compositación")
    comp_length = st.number_input("Longitud composito (m)", 0.5, 10.0, 2.0, 0.5)
    if st.button("▶️ Compositar"):
        if st.session_state["raw_data"].empty:
            st.warning("Cargue datos primero.")
        else:
            with st.spinner("Compositando..."):
                comp = composite_data(st.session_state["raw_data"], comp_length)
            st.session_state["composited_data"] = comp
            st.session_state["block_model"] = pd.DataFrame()
            st.success(f"{len(comp)} compositos generados.")

    st.divider()

    # ── Ley de corte ───────────────────────
    st.subheader("💰 Ley de Corte")
    price = st.number_input("Precio Au ($/oz)", 100.0, 5000.0, 1900.0, 50.0)
    recovery = st.slider("Recuperación (%)", 50, 99, 90) / 100
    cost_mine = st.number_input("Costo Mina ($/t)", 0.0, 200.0, 25.0, 1.0)
    cost_plant = st.number_input("Costo Planta ($/t)", 0.0, 200.0, 15.0, 1.0)
    cost_refine = st.number_input("Costo Refino ($/oz)", 0.0, 100.0, 5.0, 1.0)

    if st.button("🎯 Calcular Cutoff"):
        co = calculate_cutoff_grade(price, recovery, cost_mine, cost_plant, cost_refine)
        if co:
            st.session_state["cutoff_grade"] = co
            st.success(f"Cutoff = **{co:.4f} g/t**")
        else:
            st.error("Parámetros inválidos.")

    if st.session_state["cutoff_grade"]:
        st.info(f"🎯 Cutoff activo: **{st.session_state['cutoff_grade']:.4f} g/t**")

    st.divider()

    # ── Modelo de bloques ──────────────────
    st.subheader("📦 Modelo de Bloques")
    nx = st.slider("Bloques X", 5, 60, 20)
    ny = st.slider("Bloques Y", 5, 60, 20)
    nz = st.slider("Bloques Z", 1, 30, 10)
    k_neigh = st.slider("Vecinos IDW", 4, 24, 12)
    power_idw = st.slider("Potencia IDW", 1.0, 5.0, 2.0, 0.5)

    if st.button("▶️ Generar Bloques"):
        src = (
            st.session_state["composited_data"]
            if not st.session_state["composited_data"].empty
            else st.session_state["raw_data"]
        )
        if src.empty:
            st.warning("Cargue datos primero.")
        else:
            with st.spinner("Generando modelo de bloques..."):
                blocks = generate_block_model_3d(
                    src["X"].min(), src["X"].max(),
                    src["Y"].min(), src["Y"].max(),
                    src["Z"].min(), src["Z"].max(),
                    nx, ny, nz,
                )
                blocks = estimate_blocks_idw(blocks, src, k=k_neigh, power=power_idw)
            st.session_state["block_model"] = blocks
            st.success(f"{len(blocks):,} bloques estimados.")


# ═══════════════════════════════════════════
# PANEL PRINCIPAL — TABS
# ═══════════════════════════════════════════

st.title("⛏️ Sistema de Estimación de Reservas Mineras")
st.caption("Flujo: Cargar datos → Compositar → Calcular Cutoff → Generar Bloques → Analizar")

tab_data, tab_drill, tab_comp, tab_blocks, tab_stats, tab_report = st.tabs([
    "📋 Datos", "🎯 Taladros 3D", "📊 Compositos", "🗺️ Bloques", "📈 Estadísticas", "📄 Reporte",
])


# ─── TAB: DATOS ────────────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Vista de Datos Cargados")
    df = st.session_state["raw_data"]
    if df.empty:
        st.markdown(
            """<div class='info-box'>
            <b>📄 Formato requerido</b><br><br>
            Columnas obligatorias: <b>BHID, X, Y, Z, FROM, TO, grade</b><br>
            Columna opcional: <b>density</b> (se asume 2.65 si falta)<br><br>
            Use el panel lateral para cargar un archivo Excel/CSV o generar datos de prueba.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total muestras", f"{len(df):,}")
        c2.metric("Taladros", df["BHID"].nunique())
        c3.metric("Ley prom. (g/t)", f"{df['grade'].mean():.3f}")
        c4.metric("Densidad prom.", f"{df['density'].mean():.2f}")
        st.dataframe(df, use_container_width=True, height=400)

    comp = st.session_state["composited_data"]
    if not comp.empty:
        st.subheader("Vista de Compositos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total compositos", f"{len(comp):,}")
        c2.metric("Taladros", comp["BHID"].nunique())
        c3.metric("Ley prom. (g/t)", f"{comp['grade'].mean():.3f}")
        c4.metric("Densidad prom.", f"{comp['density'].mean():.2f}")
        st.dataframe(comp, use_container_width=True, height=300)


# ─── TAB: TALADROS 3D ─────────────────────────────────────────────────────────
with tab_drill:
    st.subheader("Visualización 3D de Taladros Diamantinos")
    df = st.session_state["raw_data"]
    if df.empty:
        st.info("Cargue datos para ver la visualización.")
    else:
        fig = go.Figure()
        gmin = df["grade"].min()
        gmax = df["grade"].max()

        for bhid, group in df.groupby("BHID"):
            group = group.sort_values("Z")
            norm = (group["grade"] - gmin) / (gmax - gmin + 1e-9)

            fig.add_trace(go.Scatter3d(
                x=group["X"], y=group["Y"], z=group["Z"],
                mode="lines+markers",
                line=dict(
                    color=group["grade"],
                    colorscale="Jet",
                    cmin=gmin, cmax=gmax,
                    width=6,
                ),
                marker=dict(
                    size=3,
                    color=group["grade"],
                    colorscale="Jet",
                    cmin=gmin, cmax=gmax,
                ),
                name=bhid,
                showlegend=False,
                hovertemplate=(
                    f"<b>{bhid}</b><br>"
                    "X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{z:.1f}<br>"
                    "Ley: %{marker.color:.3f} g/t<extra></extra>"
                ),
            ))

        # Collares
        collars = df.groupby("BHID").first().reset_index()
        fig.add_trace(go.Scatter3d(
            x=collars["X"], y=collars["Y"], z=collars["Z"],
            mode="markers",
            marker=dict(size=7, color="red", symbol="diamond"),
            name="Collar",
            hovertemplate="<b>Collar: %{text}</b><extra></extra>",
            text=collars["BHID"],
        ))

        fig.update_layout(
            title="Taladros Diamantinos 3D — Coloreados por Ley",
            scene=dict(
                xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                bgcolor="#ECEFF1",
            ),
            height=650,
            paper_bgcolor="#FAFAFA",
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── TAB: COMPOSITOS ─────────────────────────────────────────────────────────
with tab_comp:
    st.subheader("Compositos — Vista en Planta")
    comp = st.session_state["composited_data"]
    if comp.empty:
        st.info("Ejecute la compositación desde el panel lateral.")
    else:
        co = st.session_state["cutoff_grade"]
        if co:
            comp_plot = comp.copy()
            comp_plot["Clasificación"] = comp_plot["grade"].apply(
                lambda g: f"≥ Cutoff ({co:.3f})" if g >= co else f"< Cutoff ({co:.3f})"
            )
            fig = px.scatter(
                comp_plot, x="X", y="Y", color="grade",
                color_continuous_scale="Jet",
                hover_data={"BHID": True, "grade": ":.3f", "Z": ":.1f"},
                title="Compositos por Ley (g/t)",
            )
        else:
            fig = px.scatter(
                comp, x="X", y="Y", color="grade",
                color_continuous_scale="Jet",
                hover_data={"BHID": True, "grade": ":.3f", "Z": ":.1f"},
                title="Compositos por Ley (g/t)",
            )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
        fig.update_layout(height=550, paper_bgcolor="#FAFAFA")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig, use_container_width=True)

        # Estadísticas descriptivas
        with st.expander("Estadísticas descriptivas"):
            st.dataframe(comp[["grade", "density", "LENGTH"]].describe().round(4))


# ─── TAB: BLOQUES ─────────────────────────────────────────────────────────────
with tab_blocks:
    st.subheader("Modelo de Bloques")
    bm = st.session_state["block_model"]
    if bm.empty:
        st.info("Genere el modelo de bloques desde el panel lateral.")
    else:
        co = st.session_state["cutoff_grade"]

        # Vista en planta — nivel superior
        z_max = bm["CZ"].max()
        top = bm[bm["CZ"] >= z_max - bm["DZ"].mean() * 1.5].copy()

        if co:
            top["Cat"] = top["GRADE"].apply(
                lambda g: "Mineral (≥Cutoff)" if g >= co else "Estéril (<Cutoff)"
            )
            fig = px.scatter(
                top, x="CX", y="CY",
                color="Cat",
                color_discrete_map={
                    "Mineral (≥Cutoff)": "#1976D2",
                    "Estéril (<Cutoff)": "#D32F2F",
                },
                hover_data={"GRADE": ":.4f", "TONNAGE": ":,.0f"},
                title=f"Vista en Planta — Cutoff {co:.4f} g/t",
            )
        else:
            fig = px.scatter(
                top, x="CX", y="CY",
                color="GRADE",
                color_continuous_scale="Jet",
                hover_data={"GRADE": ":.4f", "TONNAGE": ":,.0f"},
                title="Vista en Planta — Ley de bloques (g/t)",
            )

        fig.update_traces(
            marker=dict(size=10, symbol="square", line=dict(width=0.3, color="black"))
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(height=550, paper_bgcolor="#FAFAFA")
        st.plotly_chart(fig, use_container_width=True)

        # Métricas rápidas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total bloques", f"{len(bm):,}")
        c2.metric("Tonelaje total", f"{bm['TONNAGE'].sum():,.0f} t")
        c3.metric("Ley promedio", f"{bm['GRADE'].mean():.4f} g/t")
        c4.metric("Metal total", f"{bm['METAL'].sum() / 31.1035:,.0f} oz")

        if co:
            eco = bm[bm["GRADE"] >= co]
            waste = bm[bm["GRADE"] < co]
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🟦 Mineral Económico")
                st.metric("Bloques", f"{len(eco):,} ({len(eco)/len(bm)*100:.1f}%)")
                st.metric("Tonelaje", f"{eco['TONNAGE'].sum():,.0f} t")
                st.metric("Ley promedio", f"{eco['GRADE'].mean():.4f} g/t")
                st.metric("Metal", f"{eco['METAL'].sum() / 31.1035:,.2f} oz")
            with col2:
                st.markdown("### 🟥 Estéril")
                st.metric("Bloques", f"{len(waste):,} ({len(waste)/len(bm)*100:.1f}%)")
                st.metric("Tonelaje", f"{waste['TONNAGE'].sum():,.0f} t")
                st.metric("Ley promedio", f"{waste['GRADE'].mean():.4f} g/t")
                strip = len(waste) / max(len(eco), 1)
                st.metric("Ratio estéril/mineral", f"{strip:.2f} : 1")


# ─── TAB: ESTADÍSTICAS ────────────────────────────────────────────────────────
with tab_stats:
    st.subheader("Análisis Estadístico")
    df = st.session_state["raw_data"]
    bm = st.session_state["block_model"]
    co = st.session_state["cutoff_grade"]

    if df.empty:
        st.info("Cargue datos primero.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Histograma leyes raw
            fig = px.histogram(
                df, x="grade", nbins=35,
                title="Distribución de Leyes — Datos Raw",
                color_discrete_sequence=["#64B5F6"],
                labels={"grade": "Ley (g/t)"},
            )
            if co:
                fig.add_vline(x=co, line_color="green", line_dash="dash",
                              annotation_text=f"Cutoff {co:.3f}")
            fig.add_vline(x=df["grade"].mean(), line_color="red", line_dash="dash",
                          annotation_text=f"Media {df['grade'].mean():.3f}")
            fig.update_layout(paper_bgcolor="#FAFAFA", height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Histograma densidad
            fig = px.histogram(
                df, x="density", nbins=20,
                title="Distribución de Densidad",
                color_discrete_sequence=["#EF5350"],
                labels={"density": "Densidad (t/m³)"},
            )
            fig.add_vline(x=df["density"].mean(), line_color="darkred", line_dash="dash",
                          annotation_text=f"Media {df['density'].mean():.2f}")
            fig.update_layout(paper_bgcolor="#FAFAFA", height=320)
            st.plotly_chart(fig, use_container_width=True)

        if not bm.empty:
            col3, col4 = st.columns(2)

            with col3:
                # Histograma bloques
                fig = px.histogram(
                    bm, x="GRADE", nbins=35,
                    title="Distribución de Leyes — Bloques",
                    color_discrete_sequence=["#81C784"],
                    labels={"GRADE": "Ley estimada (g/t)"},
                )
                if co:
                    fig.add_vline(x=co, line_color="red", line_dash="dash",
                                  annotation_text=f"Cutoff {co:.3f}")
                fig.add_vline(x=bm["GRADE"].mean(), line_color="darkgreen", line_dash="dash",
                              annotation_text=f"Media {bm['GRADE'].mean():.3f}")
                fig.update_layout(paper_bgcolor="#FAFAFA", height=320)
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                # Curva Tonelaje-Ley
                sorted_bm = bm.sort_values("GRADE", ascending=False).reset_index(drop=True)
                sorted_bm["cum_tonnage_kt"] = sorted_bm["TONNAGE"].cumsum() / 1000
                fig = px.line(
                    sorted_bm, x="GRADE", y="cum_tonnage_kt",
                    title="Curva Tonelaje-Ley",
                    labels={"GRADE": "Ley de Corte (g/t)", "cum_tonnage_kt": "Tonelaje Acumulado (kt)"},
                    color_discrete_sequence=["#7E57C2"],
                )
                if co:
                    cutoff_ton = sorted_bm[sorted_bm["GRADE"] >= co]["cum_tonnage_kt"]
                    if not cutoff_ton.empty:
                        fig.add_vline(x=co, line_color="red", line_dash="dash")
                        fig.add_hline(y=cutoff_ton.iloc[-1], line_color="red", line_dash="dash")
                fig.update_layout(paper_bgcolor="#FAFAFA", height=320)
                st.plotly_chart(fig, use_container_width=True)


# ─── TAB: REPORTE ─────────────────────────────────────────────────────────────
with tab_report:
    st.subheader("📄 Reporte de Reservas")
    bm = st.session_state["block_model"]
    co = st.session_state["cutoff_grade"]

    if bm.empty:
        st.info("Genere el modelo de bloques para ver el reporte.")
    else:
        total_blocks = len(bm)
        total_tonnage = bm["TONNAGE"].sum()
        avg_grade = bm["GRADE"].mean()
        total_metal_g = bm["METAL"].sum()
        total_metal_oz = total_metal_g / 31.1035

        lines = [
            "╔═════════════════════════════════════════════════════════╗",
            "║       REPORTE DE ESTIMACIÓN DE RESERVAS MINERAS        ║",
            "╚═════════════════════════════════════════════════════════╝",
            "",
            f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "─────────────────────────────────────────────────────────",
            "  RESUMEN GENERAL DEL MODELO",
            "─────────────────────────────────────────────────────────",
            f"  Total de Bloques:       {total_blocks:>15,}",
            f"  Tonelaje Total:         {total_tonnage:>15,.1f} t",
            f"  Ley Promedio:           {avg_grade:>15.4f} g/t",
            f"  Metal Total (g):        {total_metal_g:>15,.2f} g",
            f"  Metal Total (oz):       {total_metal_oz:>15,.2f} oz",
        ]

        if co:
            eco = bm[bm["GRADE"] >= co]
            wst = bm[bm["GRADE"] < co]
            strip = len(wst) / max(len(eco), 1)
            lines += [
                "",
                "═════════════════════════════════════════════════════════",
                f"  LEY DE CORTE: {co:.4f} g/t",
                "═════════════════════════════════════════════════════════",
                "",
                "─────────────────────────────────────────────────────────",
                "  MINERAL ECONÓMICO (>= Cutoff)",
                "─────────────────────────────────────────────────────────",
                f"  Bloques:              {len(eco):>12,} ({len(eco)/total_blocks*100:.1f}%)",
                f"  Tonelaje:             {eco['TONNAGE'].sum():>12,.1f} t",
                f"  Ley promedio:         {eco['GRADE'].mean():>12.4f} g/t",
                f"  Metal contenido:      {eco['METAL'].sum():>12,.2f} g",
                f"                        {eco['METAL'].sum()/31.1035:>12,.2f} oz",
                "",
                "─────────────────────────────────────────────────────────",
                "  ESTÉRIL (< Cutoff)",
                "─────────────────────────────────────────────────────────",
                f"  Bloques:              {len(wst):>12,} ({len(wst)/total_blocks*100:.1f}%)",
                f"  Tonelaje:             {wst['TONNAGE'].sum():>12,.1f} t",
                f"  Ley promedio:         {wst['GRADE'].mean():>12.4f} g/t",
                "",
                "═════════════════════════════════════════════════════════",
                f"  RATIO ESTÉRIL/MINERAL:  {strip:>8.2f} : 1",
                "═════════════════════════════════════════════════════════",
            ]

        report_text = "\n".join(lines)
        st.markdown(f"<div class='report-box'>{report_text}</div>", unsafe_allow_html=True)

        # Botón de descarga
        st.download_button(
            label="💾 Descargar Reporte (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"reporte_reservas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

        # Descargar modelo de bloques como CSV
        csv_buf = io.StringIO()
        bm.to_csv(csv_buf, index=False)
        st.download_button(
            label="📦 Descargar Modelo de Bloques (.csv)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"bloque_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
