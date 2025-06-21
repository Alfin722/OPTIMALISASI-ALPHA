import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import gamma
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Judul aplikasi
st.title("Optimasi Parameter α dan c")

st.markdown("Aplikasi ini mencari nilai α (0 < α < 1) dan c yang meminimalkan RMSE antara data nyata dan model:")
st.latex(r"""
x(t) = \frac{1}{c\,\Gamma(\alpha)} \int_{0}^{t} (t - \tau)^{\alpha - 1} m \,d\tau
""" )

# Upload data
uploaded_file = st.sidebar.file_uploader(
    "Unggah file (CSV atau Excel) dengan kolom: t, x",
    type=["csv", "xlsx", "xls"]
)

# Input gaya m
st.sidebar.header("2. Parameter Massa")
m = st.sidebar.number_input("Nilai gaya F (m)", min_value=0.0, value=1.0, step=0.1)

# Input rentang α dan c (α antara 0 dan 1)
st.sidebar.header("3. Rentang Pencarian α dan c")
alpha_min = st.sidebar.number_input(
    "α minimal (lebih besar dari 0)",
    value=0.01, min_value=1e-6, max_value=0.99999,
    step=0.01, format="%.5f"
)
alpha_max = st.sidebar.number_input(
    "α maksimal (kurang dari 1)",
    value=0.99, min_value=1e-6, max_value=0.99999,
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

    # Buat array nilai alpha dan c
    alpha_values = np.arange(alpha_min, alpha_max + alpha_step/2, alpha_step)
    c_values = np.arange(c_min, c_max + c_step/2, c_step)
    if len(alpha_values) == 0 or len(c_values) == 0:
        st.error("Rentang α atau c tidak valid. Pastikan nilai min, max, dan step menghasilkan setidaknya satu nilai.")
        st.stop()

    # Inisialisasi grid RMSE
    rmse_grid = np.zeros((len(alpha_values), len(c_values)))
    best_rmse = np.inf
    best_alpha = None
    best_c = None
    best_A = None

    # Loop dengan indeks untuk mengisi rmse_grid
    for i, alpha in enumerate(alpha_values):
        # Hindari alpha nol
        if alpha <= 0:
            continue
        # Hitung A sekali per alpha: menggunakan model A = m * t^alpha / gamma(alpha)
        
        A = m * (t_data ** alpha) / gamma(alpha + 1)
        for j, c in enumerate(c_values):
            if c == 0:
                rmse = np.inf
            else:
                x_model = A / c
                rmse = np.sqrt(mean_squared_error(x_data, x_model))
            rmse_grid[i, j] = rmse
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
                best_c = c
                best_A = A

    # Tampilkan hasil di badan utama menggunakan Markdown
    st.markdown("### Hasil Optimasi")
    if best_alpha is None or best_c is None:
        st.warning("Tidak ditemukan parameter yang valid. Periksa rentang dan data input.")
    else:
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

        # Plot data dan model menggunakan Plotly 2D
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=df_plot['t'],
                y=df_plot['x_nyata'],
                mode='markers',
                name='Data Nyata',
                marker=dict(size=6, symbol='circle', color='red')
            )
        )
        fig1.add_trace(
            go.Scatter(
                x=df_plot['t'],
                y=df_plot['x_model'],
                mode='lines',
                name='Model',
                line=dict(width=2, dash='solid', color='blue')
            )
        )
        fig1.update_layout(
            title='Model vs Data Nyata',
            xaxis_title='t',
            yaxis_title='x',
            legend_title='Seri'
        )
        st.plotly_chart(fig1)

        # Plot 3D interaktif menggunakan Plotly
        C_mesh, Alpha_mesh = np.meshgrid(c_values, alpha_values)
        fig3d = go.Figure(data=[
            go.Surface(
                x=C_mesh,
                y=Alpha_mesh,
                z=rmse_grid,
                colorscale='Viridis',
                showscale=True,
                opacity=0.9,
                contours={
                    'x': {'show': False},
                    'y': {'show': False},
                    'z': {'show': True, 'start': np.nanmin(rmse_grid), 'end': np.nanmax(rmse_grid), 'size': (np.nanmax(rmse_grid)-np.nanmin(rmse_grid))/10}
                }
            )
        ])
        # Tambahkan titik terbaik
        fig3d.add_trace(
            go.Scatter3d(
                x=[best_c],
                y=[best_alpha],
                z=[best_rmse],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Titik Terbaik'
            )
        )
        fig3d.update_layout(
            title='3D Surface: RMSE terhadap α dan c (interaktif)',
            scene=dict(
                xaxis_title='c',
                yaxis_title='α',
                zaxis_title='RMSE',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
            ),
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.markdown("© 2025 - ALFIN | Math Wizard")
