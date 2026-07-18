# main.py

'''
TrafficPulse — Main Entry Point
Ties all project layers together in one runnable script.

Usage:
    python main.py                    # runs full pipeline
    python main.py --simulate         # Day 2: run simulation only
    python main.py --visualize        # Day 3: generate plots only
    python main.py --mine             # Day 4: process mining only
    python main.py --train            # Day 5: train ML model only
    python main.py --evaluate         # Day 6: run agent evaluation only
    python main.py --chat             # Day 7: start LLM assistant
    python main.py --dashboard        # Day 9: launch Streamlit dashboard

Project layers:
    Layer 1: Digital Twin (SimPy simulation)
    Layer 2: Visualization & Validation
    Layer 3A: Process Mining (pm4py)
    Layer 3B: ML Forecasting (Random Forest)
    Layer 4: Adaptive Signal Agent
    Layer 5A: LLM Tool-Calling Assistant (Groq)
    Layer 5B: RAG Knowledge Base (ChromaDB)
    Layer 6: Dashboard & Animation (Streamlit)
'''

import argparse
import os
import sys


def run_simulation():
    print('=' * 55)
    print('Layer 1: Running Traffic Digital Twin Simulation')
    print('=' * 55)
    from simulation.run_simulation import main
    main()


def run_visualization():
    print('=' * 55)
    print('Layer 2: Generating Visualization Plots')
    print('=' * 55)
    from visualization.plot_queues import main
    main()


def run_mining():
    print('=' * 55)
    print('Layer 3A: Running Process Mining')
    print('=' * 55)
    from mining.process_miner import main
    main()
    from mining.bottleneck import main as bottleneck_main
    bottleneck_main()


def run_training():
    print('=' * 55)
    print('Layer 3B: Training ML Forecasting Model')
    print('=' * 55)
    from forecasting.features import main as features_main
    features_main()
    from forecasting.train_model import main as train_main
    train_main()


def run_evaluation():
    print('=' * 55)
    print('Layer 4: Running Agent Evaluation')
    print('=' * 55)
    from agent.fixed_timer import main as baseline_main
    baseline_main()
    from agent.adaptive_agent import main as agent_main
    agent_main()
    from agent.evaluate import main as eval_main
    eval_main()


def run_chat():
    print('=' * 55)
    print('Layer 5: Starting LLM Assistant')
    print('=' * 55)
    from llm.assistant_api import run_conversation
    run_conversation()


def run_dashboard():
    print('=' * 55)
    print('Layer 6: Launching Streamlit Dashboard')
    print('=' * 55)
    os.system(
        'streamlit run visualization/dashboard.py'
    )


def run_full_pipeline():
    '''
    Runs the complete TrafficPulse pipeline from scratch.
    Use this to regenerate everything after changing config.py.
    Takes 5-10 minutes total.
    '''
    print('=' * 55)
    print('TrafficPulse — Full Pipeline')
    print('Estimated time: 5-10 minutes')
    print('=' * 55)
    print()

    run_simulation()
    print()
    run_visualization()
    print()
    run_mining()
    print()
    run_training()
    print()
    run_evaluation()
    print()

    print('=' * 55)
    print('Full pipeline complete.')
    print('Run: python main.py --dashboard')
    print('     python main.py --chat')
    print('=' * 55)


def main():
    parser = argparse.ArgumentParser(
        description='TrafficPulse — AI-Powered Smart Traffic Signal Control'
    )
    parser.add_argument('--simulate',  action='store_true',
                        help='Run traffic simulation (Day 2)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots (Day 3)')
    parser.add_argument('--mine',      action='store_true',
                        help='Run process mining (Day 4)')
    parser.add_argument('--train',     action='store_true',
                        help='Train ML model (Day 5)')
    parser.add_argument('--evaluate',  action='store_true',
                        help='Run agent evaluation (Day 6)')
    parser.add_argument('--chat',      action='store_true',
                        help='Start LLM assistant (Day 7)')
    parser.add_argument('--dashboard', action='store_true',
                        help='Launch Streamlit dashboard (Day 9)')

    args = parser.parse_args()

    if args.simulate:
        run_simulation()
    elif args.visualize:
        run_visualization()
    elif args.mine:
        run_mining()
    elif args.train:
        run_training()
    elif args.evaluate:
        run_evaluation()
    elif args.chat:
        run_chat()
    elif args.dashboard:
        run_dashboard()
    else:
        run_full_pipeline()


if __name__ == '__main__':
    main()