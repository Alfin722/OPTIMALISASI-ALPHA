import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import gamma
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Judul aplikasi
st.title("Optimasi Parameter α dan c")

st.markdown("Aplikasi ini mencari nilai α (0 < α < 1) dan c yang meminimalkan RMSE antara data nyata dan model:")
st.latex(r"""
x(t) = \frac{1}{c\,\Gamma(\alpha)} \int_{0}^{t} (t - \tau)^{\alpha - 1} m \,d\tau
""")

# Upload data
uploaded_file = st.sidebar.file_uploader(
    "Unggah file (CSV atau Excel) dengan kolom: t, x",
    type=["csv", "xlsx", "xls"]
)

# Input massa m
st.sidebar.header("2. Parameter Massa")
m = st.sidebar.number_input("Nilai massa m", min_value=0.0, value=1.0, step=0.1)

# Input rentang α dan c (α antara 0 dan 1)
st.sidebar.header("3. Rentang Pencarian α dan c")
alpha_min = st.sidebar.number_input(
    "α minimal (lebih besar dari 0)",
    value=0.01, min_value=1e-6, max_value=0.99999,
    step=0.01, format="%.5f"
)
alpha_max = st.sidebar.number_input(
    "α maksimal (kurang dari 1)",
    value=0.99, min_value=0.01, max_value=0.99999,
    step=0.01, format="%.5f"
)
alpha_step = st.sidebar.number_input(
    "Delta α", value=0.01, min_value=1e-6, max_value=0.5,
    step=0.01, format="%.5f"
)

c_min = st.sidebar.number_input(
    "c minimal", value=0.1, min_value=1e-6,
    step=0.1, format="%.5f"
)
c_max = st.sidebar.number_input(
    "c maksimal", value=10.0, min_value=1e-6,
    step=0.1, format="%.5f"
)
c_step = st.sidebar.number_input(
    "Delta c", value=0.1, min_value=1e-6,
    step=0.1, format="%.5f"
)

# Tombol mulai optimasi
action = st.sidebar.button("Mulai Optimasi")

if uploaded_file is not None and action:
    # Baca data
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # Pastikan kolom t dan x ada
    if 't' not in df.columns or 'x' not in df.columns:
        st.error("File harus memiliki kolom 't' dan 'x'.")
        st.stop()

    t_data = df['t'].values
    x_data = df['x'].values

    # Optimasi parameter α dan c
    best_rmse = np.inf
    best_alpha = None
    best_c = None
    best_A = None

    for alpha in np.arange(alpha_min, alpha_max + alpha_step, alpha_step):
        A = m * (t_data ** alpha) / alpha
        for c in np.arange(c_min, c_max + c_step, c_step):
            x_model = A / c
            rmse = np.sqrt(mean_squared_error(x_data, x_model))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_c = c
                best_A = A

    # Tampilkan hasil di badan utama menggunakan Markdown
    st.markdown("### Hasil Optimasi")
    st.markdown(f"- **α terbaik:** {best_alpha:.5f}")
    st.markdown(f"- **c terbaik:** {best_c:.5f}")
    st.markdown(f"- **RMSE:** {best_rmse:.5f}")

    # Hitung x_model dengan parameter terbaik
    x_best = best_A / best_c

    # Buat DataFrame untuk plotting
    df_plot = pd.DataFrame({
        't': t_data,
        'x_nyata': x_data,
        'x_model': x_best
    })

    # Bangun Figure secara manual menggunakan graph_objects:
    fig = go.Figure()

    # Tambahkan trace scatter untuk data nyata (titik)
    fig.add_trace(
        go.Scatter(
            x=df_plot['t'],
            y=df_plot['x_nyata'],
            mode='markers',
            name='Data Nyata',
            marker=dict(size=6, symbol='circle', color='blue')
        )
    )

    # Tambahkan trace line untuk model (garis)
    fig.add_trace(
        go.Scatter(
            x=df_plot['t'],
            y=df_plot['x_model'],
            mode='lines',
            name='Model',
            line=dict(width=2, dash='solid', color='red')
        )
    )

    # Atur layout
    fig.update_layout(
        title='Perbandingan Data Nyata (titik) dan Model (garis)',
        xaxis_title='Waktu (t)',
        yaxis_title='Panjang (x)',
        legend_title='Serie'
    )

    # Tampilkan di Streamlit
    st.plotly_chart(fig)
    
st.markdown("---")
st.markdown("© 2025 - ALFIN | Math Wizard")