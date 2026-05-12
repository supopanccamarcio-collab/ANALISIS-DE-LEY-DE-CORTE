# ⛏️ Sistema de Estimación de Reservas Mineras

Aplicación web profesional para la estimación de reservas mineras mediante modelos de bloques 3D e Inverso de la Distancia (IDW).

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🚀 Demo en vivo

> Una vez desplegado en Streamlit Community Cloud, reemplaza este enlace con tu URL pública.

---

## ✨ Características

| Módulo | Descripción |
|---|---|
| 📁 **Carga de datos** | Excel (.xlsx/.xls) o CSV con mapeo automático de columnas |
| 🔧 **Compositación** | Composición ponderada por longitud en intervalos regulares |
| 💰 **Ley de Corte** | Cálculo económico: Cutoff = [(Cm + Cp)×100] / [(P − Cr) × R × F] |
| 📦 **Modelo de Bloques** | Grilla 3D con estimación IDW (Inverso de la Distancia) |
| 🎯 **Taladros 3D** | Visualización interactiva 3D con Plotly |
| 📈 **Estadísticas** | Histogramas, curva tonelaje-ley y análisis descriptivo |
| 📄 **Reporte** | Reporte de reservas descargable en .txt y modelo en .csv |

---

## 📋 Formato de datos de entrada

Tu archivo Excel o CSV debe contener estas columnas (el nombre exacto no importa, el sistema las detecta automáticamente):

| Campo | Descripción | Obligatorio |
|---|---|---|
| `BHID` | ID del taladro / sondaje | ✅ |
| `X` | Coordenada Este (m) | ✅ |
| `Y` | Coordenada Norte (m) | ✅ |
| `Z` | Elevación / Profundidad (m) | ✅ |
| `FROM` | Profundidad inicial del intervalo (m) | ✅ |
| `TO` | Profundidad final del intervalo (m) | ✅ |
| `grade` | Ley del mineral (g/t) | ✅ |
| `density` | Densidad (t/m³) | ⬜ se asume 2.65 |

---

## 🛠️ Instalación local

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`.

---

## ☁️ Despliegue en Streamlit Community Cloud (gratis)

1. Sube este repositorio a **GitHub** (debe ser público).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"**.
4. Selecciona tu repositorio, rama (`main`) y archivo principal (`app.py`).
5. Haz clic en **"Deploy!"** — listo en ~2 minutos.

> **Nota:** Streamlit Community Cloud es gratuito para repositorios públicos.

---

## 📐 Fórmula de Ley de Corte

```
Cutoff (g/t) = [(Costo_Mina + Costo_Planta) × 100]
               ─────────────────────────────────────────
               [(Precio - Costo_Refino) × Recuperación × 31.1035]
```

Donde `31.1035` es el factor de conversión gramos → onzas troy.

---

## 📦 Tecnologías utilizadas

- **[Streamlit](https://streamlit.io/)** — Framework web
- **[Plotly](https://plotly.com/python/)** — Visualizaciones interactivas 3D/2D
- **[Pandas](https://pandas.pydata.org/)** & **[NumPy](https://numpy.org/)** — Procesamiento de datos
- **[SciPy](https://scipy.org/)** — KDTree para búsqueda espacial eficiente

---

## 📂 Estructura del proyecto

```
📁 tu-repositorio/
├── app.py              ← Aplicación principal Streamlit
├── requirements.txt    ← Dependencias Python
└── README.md           ← Este archivo
```

---

## 📄 Licencia

MIT — libre uso, modificación y distribución.
