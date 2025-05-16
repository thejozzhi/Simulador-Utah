import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import graphviz

# Configuración de página
st.set_page_config(
    page_title="Simulador Utah", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stSlider>div>div>div>div {
        background: #4f8bf9;
    }
    .st-b7 {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f8bf9;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .footer {
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Datos constantes
P_real_2024 = 3503613  # Población real en 2024

# Modelos exponenciales
modelos = {
    "Modelo 1 (1995-2015)": {"C": 2014177, "k": 0.0199276, "t": 29, "año_inicio": 1995},
    "Modelo 2 (2000-2007)": {"C": 2244502, "k": 0.020880, "t": 24, "año_inicio": 2000},
    "Modelo 3 (1997-2004)": {"C": 2119784, "k": 0.017830, "t": 27, "año_inicio": 1997},
    "Modelo 4 (2009-2013)": {"C": 2723421, "k": 0.016496, "t": 15, "año_inicio": 2009},
    "Modelo 5 (1998-2002)": {"C": 2165960, "k": 0.017694, "t": 26, "año_inicio": 1998},
}

# Datos históricos completos
datos_historicos = {
    1995: 2014177, 1996: 2067976, 1997: 2119784, 1998: 2165960,
    1999: 2203482, 2000: 2244502, 2001: 2283715, 2002: 2324815,
    2003: 2360137, 2004: 2401580, 2005: 2457719, 2006: 2525507,
    2007: 2597746, 2008: 2663029, 2009: 2723421, 2010: 2776212,
    2011: 2818798, 2012: 2861360, 2013: 2909190, 2014: 2951948,
    2015: 3000449, 2016: 3064277, 2017: 3126779, 2018: 3181595,
    2019: 3233028, 2020: 3284077, 2021: 3339738, 2022: 3391011,
    2023: 3443222, 2024: 3503613
}

# --- BARRA LATERAL ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Flag_of_Utah.svg/1200px-Flag_of_Utah.svg.png", 
             use_container_width=True)
    st.title("Configuración")
    
    año_prediccion = st.slider("Año para predicción", 2024, 2050, 2024)
    mostrar_datos = st.checkbox("Mostrar datos históricos")
    
    st.markdown("---")
    st.markdown("**Proyecto para Feria Científica**")
    st.markdown("Desarrollado por:")
    st.markdown("- Joseph Stephen Ramirez Arias")
    st.markdown("- Samuel Jurado Perez")
    st.markdown("- Tomas Ruiz Mendez")

# --- CONTENIDO PRINCIPAL ---
st.title("🌄 Simulador Interactivo de Crecimiento Poblacional")
st.markdown("""
Explora modelos matemáticos que predicen el crecimiento demográfico del estado de Utah (1995-2024).
""")

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📈 Modelos Exponenciales", "📊 Modelo Logístico", "📌 Comparativa", "🧮 Laplace + Extras"])

# --- PESTAÑA 1: MODELOS EXPONENCIALES ---
with tab1:
    st.header("Modelos Exponenciales")
    st.markdown("""
    Los modelos exponenciales describen un crecimiento sin restricciones:
    $$
    \\frac{dP}{dt} = kP \\quad \\Rightarrow \\quad P(t) = P_0 e^{kt}
    $$
    """)
    
    modelo_seleccionado = st.selectbox(
        "Selecciona un modelo exponencial:", 
        list(modelos.keys()),
        key="exp_model_selector"
    )
    
    datos = modelos[modelo_seleccionado]
    
    # Cálculos
    t_pred = año_prediccion - datos["año_inicio"]
    P_estimado = datos["C"] * np.exp(datos["k"] * t_pred)
    error = abs(P_estimado - P_real_2024) / P_real_2024 * 100 if año_prediccion == 2024 else None
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Población estimada", f"{P_estimado:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if error:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Error porcentual (2024)", f"{error:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.latex(f"P(t) = {datos['C']} \\cdot e^{{{datos['k']:.5f}t}}")
    
    # Gráfico
    t_vals = np.linspace(0, datos["t"] + (año_prediccion - 2024), 100)
    años = datos["año_inicio"] + t_vals
    P_vals = datos["C"] * np.exp(datos["k"] * t_vals)
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(años, P_vals, label="Modelo", color="#4f8bf9", linewidth=2.5)
    ax1.scatter([2024], [P_real_2024], color='red', s=80, label='Dato Real 2024')
    ax1.scatter([año_prediccion], [P_estimado], color='green', s=100, label=f'Predicción {año_prediccion}')
    ax1.set_xlabel('Año', fontweight='bold')
    ax1.set_ylabel('Población', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)
    
    if mostrar_datos:
        st.subheader("Datos Históricos")
        df_historico = pd.DataFrame(list(datos_historicos.items()), columns=["Año", "Población"])
        st.dataframe(df_historico)

# --- PESTAÑA 2: MODELO LOGÍSTICO ---
with tab2:
    st.header("Modelo Logístico")
    st.markdown("""
    Considera una capacidad máxima de carga (K):
    $$
    \\frac{dP}{dt} = rP\\left(1 - \\frac{P}{K}\\right) \\quad \\Rightarrow \\quad P(t) = \\frac{K}{1 + Ce^{-rt}}
    $$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        r = st.slider("Tasa de crecimiento (r)", 0.01, 0.1, 0.076, 0.001, key="logistic_r")
    with col2:
        k = st.slider("Capacidad de carga (K)", 3.5e6, 5.0e6, 3.6e6, 0.1e6, format="%.0f", key="logistic_k")
    
    # Parámetros del modelo logístico
    P0_log = 2723421  # Población inicial (2009)
    C_log = (k - P0_log)/P0_log
    t_pred_log = año_prediccion - 2009
    
    # Cálculo de la curva
    t_vals_log = np.linspace(0, max(25, t_pred_log), 100)
    P_logistic = k / (1 + C_log * np.exp(-r * t_vals_log))
    P_pred_log = k / (1 + C_log * np.exp(-r * t_pred_log))
    
    # Gráfico
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(2009 + t_vals_log, P_logistic, 'b-', linewidth=2.5, label='Modelo Logístico')
    ax2.scatter([2024], [P_real_2024], color='red', s=80, label='Dato Real 2024')
    ax2.scatter([año_prediccion], [P_pred_log], color='green', s=100, label=f'Predicción {año_prediccion}')
    ax2.set_xlabel('Año', fontweight='bold')
    ax2.set_ylabel('Población', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)
    
    # Resultados
    st.subheader("Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Población estimada", f"{P_pred_log:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        if año_prediccion == 2024:
            error_log = abs(P_pred_log - P_real_2024)/P_real_2024 * 100
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Error porcentual", f"{error_log:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

# --- PESTAÑA 3: COMPARATIVA ---
with tab3:
    st.header("Comparativa de Modelos")
    st.markdown("Comparación de las predicciones de todos los modelos para el año seleccionado.")
    
    # Calcular predicciones
    comparacion = []
    for name, params in modelos.items():
        t_pred_comp = año_prediccion - params["año_inicio"]
        P_est = params["C"] * np.exp(params["k"] * t_pred_comp)
        error_comp = abs(P_est - P_real_2024)/P_real_2024 * 100 if año_prediccion == 2024 else None
        comparacion.append({
            "Modelo": name,
            "Ecuación": f"P(t) = {params['C']:,.0f}·e^{params['k']:.5f}t",
            f"Población {año_prediccion}": f"{P_est:,.0f}",
            "Error % (2024)": f"{error_comp:.2f}%" if error_comp else "N/A",
            "Período": name.split("(")[1][:-1]
        })
    
    # Añadir modelo logístico
    t_pred_log_comp = año_prediccion - 2009
    P_log_comp = k / (1 + C_log * np.exp(-r * t_pred_log_comp))
    error_log_comp = abs(P_log_comp - P_real_2024)/P_real_2024 * 100 if año_prediccion == 2024 else None
    comparacion.append({
        "Modelo": "Logístico",
        "Ecuación": f"P(t) = {k:,.0f}/(1 + {C_log:.2f}e^{{-{r:.3f}t}}",
        f"Población {año_prediccion}": f"{P_log_comp:,.0f}",
        "Error % (2024)": f"{error_log_comp:.2f}%" if error_log_comp else "N/A",
        "Período": "2009-actual"
    })
    
    # Mostrar tabla
    st.dataframe(
        pd.DataFrame(comparacion).set_index("Modelo"),
        use_container_width=True
    )
    
    # Gráfico comparativo
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(modelos)))
    
    # Modelos exponenciales
    for i, (name, params) in enumerate(modelos.items()):
        t_vals = np.linspace(0, params["t"] + (año_prediccion - 2024), 100)
        años = params["año_inicio"] + t_vals
        P_vals = params["C"] * np.exp(params["k"] * t_vals)
        ax3.plot(años, P_vals, label=name, color=colors[i], linewidth=2)
    
    # Modelo logístico
    ax3.plot(2009 + t_vals_log, P_logistic, 'k--', linewidth=3, label='Modelo Logístico')
    
    # Líneas de referencia
    ax3.axvline(2024, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(P_real_2024, color='red', linestyle=':', label='Población Real 2024')
    
    ax3.set_xlabel('Año', fontweight='bold')
    ax3.set_ylabel('Población', fontweight='bold')
    ax3.set_title('Comparación de Modelos Poblacionales', fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    plt.close(fig3)

# --- PESTAÑA 4: LAPLACE Y EXTRAS ---
with tab4:
    st.header("Transformada de Laplace y Métodos Numéricos")
    
    # Dividir en subpestañas
    subtab1, subtab2, subtab3 = st.tabs(["Transformada de Laplace", "Métodos Numéricos", "Diagramas y Exportación"])
    
    with subtab1:
        st.markdown("### Solución de la Ecuación Logística usando Laplace")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### Pasos de Solución:
            1. Reescribir como ecuación de Bernoulli
            2. Aplicar sustitución \( u = P^{-1} \)
            3. Resolver la ecuación lineal resultante
            4. Aplicar transformada de Laplace
            5. Resolver en el dominio \( s \)
            6. Aplicar transformada inversa
            """)
            
            st.latex(r"""
            P(t) = \frac{K}{1 + \left(\frac{K}{P_0} - 1\right)e^{-rt}}
            """)
        
        with col2:
            r_laplace = st.slider("Tasa r", 0.01, 0.1, 0.07, 0.001, key="laplace_r")
            K_laplace = st.slider("Capacidad K", 3.0e6, 5.0e6, 3.6e6, 0.1e6, key="laplace_k")
            P0_laplace = st.slider("Población inicial", 1.0e6, 3.0e6, 2.7e6, 0.1e6, key="laplace_p0")
            
            t_vals = np.linspace(0, 50, 100)
            C = (K_laplace/P0_laplace) - 1
            P_vals = K_laplace / (1 + C * np.exp(-r_laplace * t_vals))
            
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            ax4.plot(t_vals, P_vals, 'b-', linewidth=2.5)
            ax4.set_xlabel('Tiempo (años)')
            ax4.set_ylabel('Población')
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            plt.close(fig4)
    
    with subtab2:
        st.markdown("### Comparación con Métodos Numéricos")
        
        method = st.radio("Método numérico", ["Euler", "Runge-Kutta 4"], key="metodo_numerico")
        h = st.slider("Paso de tiempo (h)", 0.1, 2.0, 1.0, 0.1, key="paso_tiempo")
        
        # Implementación básica de Euler
        if method == "Euler":
            t_euler = np.arange(0, 50 + h, h)
            P_euler = np.zeros_like(t_euler)
            P_euler[0] = P0_laplace
            
            for i in range(1, len(t_euler)):
                dP = r_laplace * P_euler[i-1] * (1 - P_euler[i-1]/K_laplace)
                P_euler[i] = P_euler[i-1] + h * dP
            
            fig_num, ax_num = plt.subplots(figsize=(10, 5))
            ax_num.plot(t_euler, P_euler, 'g-', linewidth=2, label='Método Euler')
            
            # Solución analítica para comparación
            t_vals = np.linspace(0, 50, 100)
            P_analitica = K_laplace / (1 + C * np.exp(-r_laplace * t_vals))
            ax_num.plot(t_vals, P_analitica, 'b--', linewidth=2, label='Solución Analítica')
            
            ax_num.set_xlabel('Tiempo (años)')
            ax_num.set_ylabel('Población')
            ax_num.legend()
            ax_num.grid(True, alpha=0.3)
            st.pyplot(fig_num)
            plt.close(fig_num)
    
    with subtab3:
        st.markdown("### Diagrama del Proceso")
        
        try:
            diagram = graphviz.Digraph()
            diagram.edge("Ecuación Diferencial", "Transformada de Laplace")
            diagram.edge("Transformada de Laplace", "Solución en s")
            diagram.edge("Solución en s", "Transformada Inversa")
            diagram.edge("Transformada Inversa", "Solución Temporal")
            st.graphviz_chart(diagram)
        except Exception as e:
            st.error(f"Error al renderizar el diagrama. Asegúrate de tener Graphviz instalado: {str(e)}")
        
        st.markdown("### Exportar Resultados")
        
        # Exportar gráfico actual
        if 'fig4' in locals():
            buf = BytesIO()
            fig4.savefig(buf, format="png")
            st.download_button(
                label="Descargar gráfico actual",
                data=buf,
                file_name="modelo_poblacional.png",
                mime="image/png"
            )
        else:
            st.warning("Genera un gráfico en las pestañas anteriores para exportar")

# --- FOOTER ---
st.markdown("---")
st.markdown('<div class="footer">Proyecto de Ecuaciones Diferenciales | Feria Científica 2023</div>', unsafe_allow_html=True)
