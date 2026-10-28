TrafficPulse 🚦

Adaptive traffic signal control: simulation, forecasting, and an LLM assistant over the results

An end-to-end project that simulates a 3-intersection urban road network, learns its congestion patterns from the resulting event logs, controls signal timing adaptively, and answers questions about its own decisions in natural language.

The adaptive controller cuts average wait time by 9.7% and peak wait time by 32.4% against a fixed-timer baseline.

Results

Measured over a full simulated day, adaptive agent vs. fixed-timer baseline:

Metric	Fixed Timer	Adaptive Agent	Change
Average wait time	8.483 s	7.658 s	−9.7%
Maximum wait time	49.96 s	33.79 s	−32.4%
Cars served	25,877	26,251	+374
INT_2 (bottleneck) wait	10.226 s	7.300 s	−28.6%

Both controllers run against the same simulated traffic with the same random seed, so the comparison isolates the control policy.

Why this project exists

Most traffic-control demos either simulate without learning, or train a model without anything to act on. This one closes the loop: the simulation produces real event logs, the logs train a forecaster, the forecaster informs a controller, and the controller's behaviour is measured against a baseline and then explained in plain language.

The simulation exists because the data had to come from somewhere. The interesting work is layers 3 through 5.

Architecture
Layer 1: Digital twin (SimPy)           → generates traffic event logs
Layer 2: Visualization (matplotlib)     → validates simulation realism
Layer 3A: Process mining (pm4py)        → discovers flow patterns from logs
Layer 3B: Forecasting (scikit-learn)    → predicts congestion 5 min ahead
Layer 4: Adaptive agent                 → ML-informed signal timing control
Layer 5A: LLM assistant (Groq/Ollama)   → natural-language Q&A over real data
Layer 5B: RAG knowledge base (ChromaDB) → grounds answers in a policy document
Layer 6: Dashboard (Streamlit)          → interactive web interface
Network design
         North                      North
           ↓                          ↓
West → [INT_1] ←──────────────→ [INT_2] → East
                                    ↕
                                 [INT_3]
                                    ↕
                                  South
INT_1 — western entry/exit, 2-phase signal
INT_2 — central junction, main bottleneck, 3-phase signal
INT_3 — southern branch, 2-phase signal
6 bidirectional traffic flows across the network
Technology stack
Technology	Role
Python 3.12	Core language
SimPy	Discrete-event simulation
pandas / numpy	Data engineering
pm4py	Process mining
scikit-learn	Random Forest forecasting
matplotlib / plotly	Visualization
Groq + LLaMA 3.3 70B	LLM assistant (API)
Ollama + Mistral 7B	LLM assistant (local)
ChromaDB	RAG vector database
Streamlit	Interactive dashboard
Key design decisions

Why discrete-event simulation for the digital twin? Events — car arrivals, signal changes — drive the simulation clock rather than fixed continuous time steps. This matches how queueing systems actually behave and keeps the generated event log in the same shape as a real sensor feed, which is what makes the process-mining layer possible.

Why process mining on the logs? pm4py reconstructed all six traffic flows automatically from raw event logs without being told they existed. This validates that the pipeline learns structure from data rather than having it hard-coded, which is the property that would matter if the logs came from real intersections.

Why Random Forest rather than a neural network? It handles mixed feature types without scaling, gives interpretable feature-importance scores, trains in seconds at this dataset size, and resists overfitting. R² = 0.41 — modest in absolute terms, but the agent only needs a relative ranking of which intersection is about to congest, not a precise wait-time estimate. A more complex model would have bought accuracy the controller cannot use.

Why does the agent prioritize INT_2? INT_2 is the central junction serving all six flows. The net saving from clearing it faster is 25,735 car-seconds — 6.7× the cost incurred at INT_1. This mirrors how production adaptive systems such as SCOOT optimize network throughput rather than per-intersection fairness.

Why Groq rather than local Ollama for the final assistant? Mistral 7B struggled with multi-step reasoning and tool selection. LLaMA 3.3 70B via Groq handles multi-tool queries correctly. Both paths are implemented — local first, then API — so the deployment tradeoff is visible rather than assumed.

Project structure
TrafficPulse/
├── simulation/         # SimPy digital twin
│   ├── config.py       # Network settings, signal phases, arrival rates
│   ├── traffic_sim.py  # SimPy processes: signals, cars, event logger
│   └── run_simulation.py
├── mining/             # Process mining
│   ├── process_miner.py
│   └── bottleneck.py
├── forecasting/        # ML model
│   ├── features.py
│   ├── train_model.py
│   └── predict.py
├── agent/              # Adaptive signal agent
│   ├── fixed_timer.py
│   ├── adaptive_agent.py
│   └── evaluate.py
├── llm/                # LLM assistant + RAG
│   ├── tools.py        # 8 tool definitions
│   ├── assistant.py    # Ollama + Mistral 7B (local)
│   ├── assistant_api.py# Groq + LLaMA 3.3 70B (API)
│   └── rag.py          # ChromaDB RAG pipeline
├── visualization/      # Plots + dashboard
│   ├── plot_queues.py
│   ├── animate.py
│   └── dashboard.py
├── docs/
│   └── traffic_policy.md  # RAG knowledge base
├── data/
│   ├── logs/           # traffic_log.csv (simulation output)
│   └── processed/      # features.csv, model.pkl, metrics CSVs
├── results/
│   ├── plots/          # Generated charts
│   └── animation/      # before_after.gif
├── main.py             # Entry point — runs any layer
└── requirements.txt
Setup
Prerequisites
Python 3.12
Graphviz (for process-mining diagrams)
Developed on macOS; runs on Linux with the same steps
Installation
bash
git clone https://github.com/prabhijeet/TrafficPulse.git
cd TrafficPulse
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Graphviz:

bash
brew install graphviz        # macOS
sudo apt install graphviz    # Debian/Ubuntu
Environment variables
bash
export GROQ_API_KEY='your-groq-key'      # free tier at console.groq.com

For the local LLM path (optional):

bash
brew install ollama
ollama pull mistral
Usage

Run the full pipeline from scratch:

bash
python main.py

Run individual layers:

bash
python main.py --simulate    # generate traffic data
python main.py --visualize   # generate plots
python main.py --mine        # process mining
python main.py --train       # train ML model
python main.py --evaluate    # run agent comparison
python main.py --chat        # start LLM assistant
python main.py --dashboard   # launch web dashboard

Launch the dashboard directly:

bash
streamlit run visualization/dashboard.py

Start the assistant directly:

bash
python -m llm.assistant_api   # Groq API version (recommended)
python -m llm.assistant       # local Ollama version
LLM assistant — example questions

The assistant has eight tools over the simulation results and a RAG index over the signal-policy document. Questions it handles:

What is the status of INT_2?
Why was there congestion at INT_2 at 8am?
Compare the performance of INT_1 and INT_2
Give me a full summary of today's traffic
Is the adaptive agent actually helping?
Which traffic light causes the most delays?
Why does INT_2 always get priority when there's a tie?

The last question is answered from the policy document rather than the data — it is the RAG path rather than the tool-calling path.

License

MIT — see LICENSE.
