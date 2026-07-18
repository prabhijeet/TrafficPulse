# visualization/dashboard.py

'''
Day 9 — Streamlit Dashboard.
Interactive web dashboard with 5 tabs covering all project layers.
Embeds the LLM chatbot directly in the dashboard.

Tabs:
  1. Network Overview    — live intersection status and queue lengths
  2. Process Mining      — flow distribution and bottleneck analysis
  3. ML Model            — feature importance and prediction accuracy
  4. Agent Performance   — before/after comparison charts
  5. LLM Assistant       — embedded chatbot interface

Run with: streamlit run visualization/dashboard.py
'''

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import BASE_DIR

# paths
LOG_FILE      = os.path.join(BASE_DIR, 'data', 'logs', 'traffic_log.csv')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR     = os.path.join(BASE_DIR, 'results', 'plots')
ANIMATION_DIR = os.path.join(BASE_DIR, 'results', 'animation')

# page config
st.set_page_config(
    page_title = 'TrafficPulse Dashboard',
    page_icon  = '🚦',
    layout     = 'wide'
)

# custom CSS for dark theme
st.markdown('''
<style>
    .main { background-color: #0d0d1a; }
    .metric-card {
        background-color: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: bold;
    }
</style>
''', unsafe_allow_html=True)


@st.cache_data
def load_log():
    '''Loads and caches traffic log CSV.'''
    return pd.read_csv(LOG_FILE)


@st.cache_data
def load_baseline():
    '''Loads fixed timer baseline metrics.'''
    try:
        return pd.read_csv(
            os.path.join(PROCESSED_DIR, 'baseline_intersection.csv'),
            index_col=0
        )
    except FileNotFoundError:
        return None


@st.cache_data
def load_adaptive():
    '''Loads adaptive agent metrics.'''
    try:
        return pd.read_csv(
            os.path.join(PROCESSED_DIR, 'adaptive_intersection.csv'),
            index_col=0
        )
    except FileNotFoundError:
        return None


@st.cache_data
def load_features():
    '''Loads engineered feature CSV.'''
    try:
        return pd.read_csv(os.path.join(PROCESSED_DIR, 'features.csv'))
    except FileNotFoundError:
        return None


def tab_network_overview(df):
    '''Tab 1 — Network Overview'''
    st.header('Network Overview')
    st.markdown('Live intersection status from the 24-hour simulation.')

    cars       = df[df['event_type'] != 'signal_change']
    departures = cars[cars['event_type'] == 'car_departed']
    arrivals   = cars[cars['event_type'] == 'car_arrived']

    # headline metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Total Cars', f"{len(arrivals):,}")
    with col2:
        st.metric('Cars Served', f"{len(departures):,}")
    with col3:
        avg_wait = round(departures['wait_time'].mean(), 2)
        st.metric('Avg Wait Time', f'{avg_wait}s')
    with col4:
        max_wait = round(departures['wait_time'].max(), 2)
        st.metric('Max Wait Time', f'{max_wait}s')

    st.divider()

    # per intersection metrics
    st.subheader('Per Intersection Performance')
    col1, col2, col3 = st.columns(3)

    for col, int_id in zip([col1, col2, col3],
                           ['INT_1', 'INT_2', 'INT_3']):
        with col:
            int_dep = departures[departures['intersection'] == int_id]
            int_arr = arrivals[arrivals['intersection'] == int_id]
            avg     = round(int_dep['wait_time'].mean(), 2)
            mx      = round(int_dep['wait_time'].max(), 2)
            served  = len(int_dep)
            avg_q   = round(int_arr['queue_length'].mean(), 2)

            is_bottleneck = int_id == 'INT_2'
            color = '🔴' if is_bottleneck else '🟢'

            st.markdown(f'### {color} {int_id}')
            st.metric('Avg Wait', f'{avg}s',
                      delta='bottleneck' if is_bottleneck else None,
                      delta_color='inverse')
            st.metric('Max Wait',    f'{mx}s')
            st.metric('Cars Served', f'{served:,}')
            st.metric('Avg Queue',   f'{avg_q} cars')

    st.divider()

    # hourly arrivals chart
    st.subheader('Cars Arriving Per Hour')
    hourly = arrivals.groupby('hour').size().reset_index(name='count')

    fig = px.bar(
        hourly, x='hour', y='count',
        color='count',
        color_continuous_scale='RdYlGn',
        labels={'hour': 'Hour of Day', 'count': 'Cars Arriving'},
        title='Traffic Volume Throughout the Day'
    )
    fig.update_layout(
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font_color='white',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # queue length over time per intersection
    st.subheader('Average Queue Length Per Hour')
    hourly_q = arrivals.groupby(
        ['hour', 'intersection']
    )['queue_length'].mean().reset_index()

    fig2 = px.line(
        hourly_q, x='hour', y='queue_length',
        color='intersection',
        color_discrete_map={
            'INT_1': '#4488ff',
            'INT_2': '#ff4444',
            'INT_3': '#44cc44'
        },
        labels={'hour': 'Hour', 'queue_length': 'Avg Queue (cars)'},
        title='Queue Length Throughout the Day'
    )
    fig2.update_layout(
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font_color='white'
    )
    st.plotly_chart(fig2, use_container_width=True)


def tab_process_mining(df):
    '''Tab 2 — Process Mining'''
    st.header('Process Mining')
    st.markdown('Flow patterns and bottlenecks discovered automatically from event logs.')

    cars     = df[df['event_type'] != 'signal_change']
    arrivals = cars[cars['event_type'] == 'car_arrived']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Traffic Flow Distribution')
        flow_counts = arrivals[arrivals['flow_id'] != 'N/A']['flow_id'].value_counts()
        fig = px.bar(
            x=flow_counts.index,
            y=flow_counts.values,
            color=flow_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Flow ID', 'y': 'Number of Events'},
            title='Events per Traffic Flow'
        )
        fig.update_layout(
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font_color='white',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Events Per Intersection')
        int_counts = df.groupby('intersection').size()
        int_counts = int_counts[int_counts.index.isin(['INT_1','INT_2','INT_3'])]
        fig2 = px.pie(
            values=int_counts.values,
            names=int_counts.index,
            color_discrete_map={
                'INT_1': '#4488ff',
                'INT_2': '#ff4444',
                'INT_3': '#44cc44'
            },
            title='Event Distribution'
        )
        fig2.update_layout(
            paper_bgcolor='#1a1a2e',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # bottleneck heatmap
    st.subheader('Bottleneck Heatmap — Wait Time by Intersection × Hour')
    departures = cars[cars['event_type'] == 'car_departed']
    pivot = departures.pivot_table(
        values='wait_time',
        index='intersection',
        columns='hour',
        aggfunc='mean'
    ).round(2)

    fig3 = px.imshow(
        pivot,
        color_continuous_scale='YlOrRd',
        labels={'color': 'Avg Wait (s)'},
        title='Wait Time Heatmap'
    )
    fig3.update_layout(
        paper_bgcolor='#1a1a2e',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)

    # process map image
    process_map = os.path.join(PLOTS_DIR, '6_process_map.png')
    if os.path.exists(process_map):
        st.subheader('Process Map (pm4py)')
        st.image(process_map, caption='Directly-Follows Graph — discovered from event log')


def tab_ml_model():
    '''Tab 3 — ML Model'''
    st.header('ML Congestion Forecasting')
    st.markdown('Random Forest model trained to predict queue length 5 minutes ahead.')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Model Type', 'Random Forest')
    with col2:
        st.metric('R² Score', '0.41')
    with col3:
        st.metric('MAE', '0.38 cars')

    col4, col5 = st.columns(2)
    with col4:
        st.metric('Training Rows', '20,693')
    with col5:
        st.metric('Test Rows', '5,174')

    st.divider()

    # feature importance plot
    fi_plot = os.path.join(PLOTS_DIR, '9_feature_importance.png')
    pred_plot = os.path.join(PLOTS_DIR, '10_predicted_vs_actual.png')

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(fi_plot):
            st.image(fi_plot, caption='Feature Importance — queue_length dominates at 0.516')
    with col2:
        if os.path.exists(pred_plot):
            st.image(pred_plot, caption='Predicted vs Actual Queue Length')

    st.divider()
    st.subheader('Key Insights')
    st.markdown('''
    - **queue_length** is the dominant feature (0.516) — current queue strongly predicts next queue
    - **minute_of_day** (0.113) is more useful than hour (0.005) — congestion patterns are fine-grained
    - **Rolling averages** improve with longer windows — 15-event window beats 5-event window
    - **R² = 0.41** — sufficient for the agent to rank approaches correctly
    ''')


def tab_agent_performance():
    '''Tab 4 — Agent Performance'''
    st.header('Adaptive Agent Performance')
    st.markdown('Fixed timer vs adaptive ML agent — measured over 24-hour simulation.')

    baseline = load_baseline()
    adaptive = load_adaptive()

    if baseline is None or adaptive is None:
        st.warning('Agent comparison data not found. Run agent/evaluate.py first.')
        return

    # headline metrics
    b_avg  = round(baseline['avg_wait_time'].mean(), 3)
    a_avg  = round(adaptive['avg_wait_time'].mean(), 3)
    b_max  = round(baseline['max_wait_time'].max(), 3)
    a_max  = round(adaptive['max_wait_time'].max(), 3)
    b_cars = int(baseline['total_served'].sum())
    a_cars = int(adaptive['total_served'].sum())

    pct_avg  = round(((a_avg - b_avg) / b_avg) * 100, 1)
    pct_max  = round(((a_max - b_max) / b_max) * 100, 1)
    pct_cars = round(((a_cars - b_cars) / b_cars) * 100, 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Avg Wait Time',
                  f'{a_avg}s',
                  delta=f'{pct_avg}%',
                  delta_color='inverse')
    with col2:
        st.metric('Max Wait Time',
                  f'{a_max}s',
                  delta=f'{pct_max}%',
                  delta_color='inverse')
    with col3:
        st.metric('Cars Served',
                  f'{a_cars:,}',
                  delta=f'+{a_cars - b_cars}',
                  delta_color='normal')

    st.divider()

    # before/after comparison charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Average Wait Time Comparison')
        comparison_data = []
        for i in ['INT_1', 'INT_2', 'INT_3']:
            comparison_data.append({
                'Intersection': i,
                'Fixed Timer' : round(baseline.loc[i, 'avg_wait_time'], 2),
                'Adaptive'    : round(adaptive.loc[i, 'avg_wait_time'], 2)
            })
        comp_df = pd.DataFrame(comparison_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Fixed Timer',
            x=comp_df['Intersection'],
            y=comp_df['Fixed Timer'],
            marker_color='#ff8844'
        ))
        fig.add_trace(go.Bar(
            name='Adaptive Agent',
            x=comp_df['Intersection'],
            y=comp_df['Adaptive'],
            marker_color='#00cc66'
        ))
        fig.update_layout(
            barmode='group',
            plot_bgcolor='#1a1a2e',
            paper_bgcolor='#1a1a2e',
            font_color='white',
            yaxis_title='Avg Wait Time (seconds)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Before/After Animation')
        gif_path = os.path.join(ANIMATION_DIR, 'before_after.gif')
        if os.path.exists(gif_path):
            st.image(gif_path, caption='24-hour traffic simulation — Fixed Timer vs Adaptive Agent')
        else:
            static_path = os.path.join(PLOTS_DIR, '14_before_after_static.png')
            if os.path.exists(static_path):
                st.image(static_path, caption='Rush Hour 8am — Fixed Timer vs Adaptive Agent')

    st.divider()
    st.subheader('Why INT_1 Got Slightly Worse (+6.9%)')
    st.markdown('''
    The agent correctly prioritized **INT_2** (the central bottleneck) at the slight expense of INT_1.
    This is a deliberate network-level trade-off:
    - Clearing INT_2 faster reduces spillback to INT_1 and INT_3
    - Net saving: **25,735 car-seconds** — 6.7x more than the cost at INT_1
    - This mirrors how real adaptive signal systems like SCOOT optimize for network throughput
    ''')


def tab_llm_assistant():
    '''Tab 5 — LLM Assistant'''
    st.header('LLM Traffic Assistant')
    st.markdown('Ask questions about the traffic network in plain English.')
    st.markdown('*Powered by Groq + LLaMA 3.3 70B with tool-calling and RAG*')

    # initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # example questions
    with st.expander('Example questions'):
        st.markdown('''
        - What is the status of INT_2?
        - Why was there congestion at INT_2 at 8am?
        - Which intersection had the worst performance today?
        - Was the morning commute bad today?
        - Is the adaptive agent actually helping?
        - Which traffic light causes the most delays?
        ''')

    # chat input
    if prompt := st.chat_input('Ask about the traffic network...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

        with st.chat_message('assistant'):
            with st.spinner('Thinking...'):
                try:
                    from llm.assistant_api import ask_once
                    response = ask_once(prompt)
                except Exception as e:
                    error_msg = str(e)
                    if 'rate_limit' in error_msg or '429' in error_msg:
                        response = '⏳ Daily token limit reached. The Groq free tier allows 100,000 tokens per day. Please try again in a few minutes or tomorrow.'
                    elif 'GROQ_API_KEY' in error_msg or 'api_key' in error_msg:
                        response = '🔑 API key not found. Make sure GROQ_API_KEY is set in your environment.'
                    else:
                        response = f'❌ Something went wrong: {error_msg}'

            st.write(response)
            st.session_state.messages.append(
                {'role': 'assistant', 'content': response}
            )


def main():
    '''Main dashboard entry point.'''
    st.title('🚦 TrafficPulse Dashboard')
    st.markdown('*AI-Powered Smart Traffic Signal Control System*')
    st.divider()

    # load data
    df = load_log()

    # tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        '🌐 Network Overview',
        '⛏️ Process Mining',
        '🤖 ML Model',
        '📊 Agent Performance',
        '💬 LLM Assistant'
    ])

    with tab1:
        tab_network_overview(df)
    with tab2:
        tab_process_mining(df)
    with tab3:
        tab_ml_model()
    with tab4:
        tab_agent_performance()
    with tab5:
        tab_llm_assistant()


if __name__ == '__main__':
    main()