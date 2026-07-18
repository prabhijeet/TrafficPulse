# TrafficPulse 🚦

**AI-Powered Smart Traffic Signal Control System with Digital Twin, Process Mining, and LLM Assistant**

A complete end-to-end data science project that learns traffic patterns from a simulated 3-intersection urban network, predicts congestion, autonomously controls signal timing, and explains decisions in natural language.

---

## Results

| Metric | Fixed Timer | Adaptive Agent | Improvement |
|---|---|---|---|
| Average wait time | 8.483s | 7.658s | **-9.7%** |
| Maximum wait time | 49.96s | 33.79s | **-32.4%** |
| Cars served | 25,877 | 26,251 | **+374** |
| INT_2 bottleneck | 10.226s | 7.300s | **-28.6%** |

---

## Architecture
```
Layer 1: Digital Twin (SimPy)          → generates traffic event logs
Layer 2: Visualization (matplotlib)    → validates simulation realism
Layer 3A: Process Mining (pm4py)       → discovers flow patterns automatically
Layer 3B: ML Forecasting (scikit-learn)→ predicts congestion 5 min ahead
Layer 4: Adaptive Agent                → ML-informed signal timing control
Layer 5A: LLM Assistant (Groq/Ollama) → natural language Q&A over real data
Layer 5B: RAG Knowledge Base (ChromaDB)→ grounds answers in policy document
Layer 6: Dashboard (Streamlit)         → interactive web interface
```

## Network Design
```
         North                      North
           ↓                          ↓
West → [INT_1] ←──────────────→ [INT_2] → East
                                    ↕
                                 [INT_3]
                                    ↕
                                  South
```

- **INT_1** — Western entry/exit, 2-phase signal
- **INT_2** — Central junction, main bottleneck, 3-phase signal
- **INT_3** — Southern branch, 2-phase signal
- 6 bidirectional traffic flows across the network

---

## Project Structure
```
TrafficPulse/
├── simulation/         # Day 2: SimPy digital twin
│   ├── config.py       # Network settings, signal phases, arrival rates
│   ├── traffic_sim.py  # SimPy processes: signals, cars, event logger
│   └── run_simulation.py
├── mining/             # Day 4: Process mining
│   ├── process_miner.py
│   └── bottleneck.py
├── forecasting/        # Day 5: ML model
│   ├── features.py
│   ├── train_model.py
│   └── predict.py
├── agent/              # Day 6: Adaptive signal agent
│   ├── fixed_timer.py
│   ├── adaptive_agent.py
│   └── evaluate.py
├── llm/                # Days 7-8: LLM assistant + RAG
│   ├── tools.py        # 8 tool definitions
│   ├── assistant.py    # Ollama + Mistral 7B (local)
│   ├── assistant_api.py# Groq + LLaMA 3.3 70B (API)
│   └── rag.py          # ChromaDB RAG pipeline
├── visualization/      # Days 3, 9: Plots + dashboard
│   ├── plot_queues.py
│   ├── animate.py
│   └── dashboard.py
├── docs/
│   └── traffic_policy.md  # RAG knowledge base
├── data/
│   ├── logs/           # traffic_log.csv (simulation output)
│   └── processed/      # features.csv, model.pkl, metrics CSVs
├── results/
│   ├── plots/          # All generated charts (plots 1-14)
│   └── animation/      # before_after.gif
├── main.py             # Entry point — runs any layer
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.12
- macOS (tested on MacBook Pro 2019, Intel i9, 16GB RAM)

### Installation

```bash
git clone https://github.com/yourusername/TrafficPulse.git
cd TrafficPulse
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install graphviz ollama
```

### Environment Variables

```bash
export GROQ_API_KEY='your-groq-key'      # free at console.groq.com
```

For local LLM (optional):
```bash
ollama pull mistral
```

---

## Usage

### Run everything from scratch
```bash
python main.py
```

### Run individual layers
```bash
python main.py --simulate    # generate traffic data
python main.py --visualize   # generate plots
python main.py --mine        # process mining
python main.py --train       # train ML model
python main.py --evaluate    # run agent comparison
python main.py --chat        # start LLM chatbot
python main.py --dashboard   # launch web dashboard
```

### Launch dashboard directly
```bash
streamlit run visualization/dashboard.py
```

### Start LLM assistant
```bash
python -m llm.assistant_api   # Groq API version (recommended)
python -m llm.assistant       # Local Ollama version
```

---

## Technology Stack

| Technology | Role | Maps to |
|---|---|---|
| Python 3.12 | Core language | — |
| SimPy | Discrete-event simulation | P2: OFacT paradigm |
| pandas / numpy | Data engineering | P2, P13 |
| pm4py | Process mining | **P2: explicitly stated** |
| scikit-learn | Random Forest forecasting | **P2: regression/classification** |
| matplotlib / plotly | Visualization | P2, P12 |
| Groq + LLaMA 3.3 70B | LLM assistant (API) | **P12: tool-calling** |
| Ollama + Mistral 7B | LLM assistant (local) | P12: local deployment |
| ChromaDB | RAG vector database | **P12: RAG pipeline** |
| Streamlit | Interactive dashboard | P12: dialogue interface |

---

## Key Design Decisions

**Why SimPy for the digital twin?**
SimPy uses discrete-event simulation — the same paradigm as OFacT (HSBI + Fraunhofer ISST). Events (car arrivals, signal changes) drive the simulation clock rather than continuous time steps, which is how real production systems are modeled.

**Why process mining with pm4py?**
Project 2 explicitly describes process mining as a core method. pm4py reconstructed all 6 traffic flows automatically from raw event logs without being told they existed — demonstrating the "learn from data" principle central to the research.

**Why Random Forest over deep learning?**
Handles mixed feature types without scaling, gives interpretable feature importance scores, trains fast on our dataset size, and is resistant to overfitting. R²=0.41 is sufficient for the agent's relative ranking task.

**Why the agent prioritizes INT_2?**
INT_2 is the central junction serving all 6 flows. The net saving from clearing it faster is 25,735 car-seconds — 6.7x more than the cost at INT_1. This mirrors how real adaptive systems like SCOOT optimize for network throughput rather than individual intersection performance.

**Why Groq over local Ollama for the final LLM?**
Mistral 7B (7B parameters) showed limitations on multi-step reasoning and tool selection. LLaMA 3.3 70B via Groq handles complex multi-tool queries correctly. Both are demonstrated — local first, then API migration — showing understanding of deployment tradeoffs.

---

## LLM Assistant — Example Questions
```
What is the status of INT_2?
Why was there congestion at INT_2 at 8am?
Compare the performance of INT_1 and INT_2
Give me a full summary of today's traffic
Was the morning commute bad today?
Is the adaptive agent actually helping?
Which traffic light causes the most delays?
Why does INT_2 always get priority when there's a tie?
```

---

## License

MIT License — free to use and modify.