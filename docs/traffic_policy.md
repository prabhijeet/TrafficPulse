# TrafficPulse — Traffic Signal Policy Document

This document defines the operational rules, constraints, and guidelines
for the TrafficPulse adaptive signal control system. The LLM assistant
uses this document to justify signal decisions and answer policy questions.

---

## 1. Network Overview

The TrafficPulse network consists of 3 intersections in a T-shaped layout:

- **INT_1** — Western entry/exit point. 2-phase signal (NS and EW).
- **INT_2** — Central junction and primary bottleneck. 3-phase signal (NS, EW, BRANCH).
- **INT_3** — Southern branch entry/exit. 2-phase signal (NS and EW).

All roads between intersections are bidirectional. INT_2 is the only
junction where cars choose a direction — 60% continue east, 40% turn south
when arriving from the west, and traffic splits 50/50 when arriving from the south.

---

## 2. Signal Phase Definitions

### INT_1 and INT_3 — 2-Phase Cycle
- **Phase NS**: North and South approaches get green. Duration: 30 seconds.
- **Phase EW**: East and West approaches get green. Duration: 30 seconds.
- Total cycle time: 60 seconds.

### INT_2 — 3-Phase Cycle
- **Phase NS**: North and South approaches get green. Duration: 30 seconds.
- **Phase EW**: East and West approaches get green. Duration: 30 seconds.
- **Phase BRANCH**: South-North branch road gets green. Duration: 20 seconds.
- Total cycle time: 80 seconds.

INT_2 has a longer cycle than INT_1 and INT_3 because it must serve
3 competing traffic directions instead of 2. This is why INT_2 has
structurally higher average wait times even when traffic volume is equal.

---

## 3. Green Time Constraints

These constraints apply to the adaptive agent and must never be violated:

| Constraint | Value | Reason |
|---|---|---|
| Minimum green time (all phases) | 10 seconds | Pedestrian safety — minimum crossing time |
| Maximum green time (all phases) | 90 seconds | Fairness — no approach starved indefinitely |
| INT_2 BRANCH phase minimum | 15 seconds | Branch road needs more time due to merge complexity |
| Maximum bonus green time | 10 seconds | Agent can add up to 10s over base duration based on load |

Violating minimum green time risks pedestrian safety.
Violating maximum green time causes fairness violations — other approaches
can wait up to 3 cycles without service, which is unacceptable.

---

## 4. Rush Hour Definitions

| Period | Hours | Description |
|---|---|---|
| Morning rush | 07:00 — 09:00 | Commute to work, dominant east-west flow |
| Off-peak morning | 09:00 — 16:00 | Normal flow, lower volume |
| Evening rush | 16:00 — 19:00 | Commute home, increased south branch flow |
| Night quiet | 19:00 — 07:00 | Minimal traffic, signal cycles at minimum |

During rush hours the adaptive agent applies directional bias:
- Morning rush (7-9am): East-West corridor gets priority
- Evening rush (4-7pm): North-South and branch flows get slight priority

---

## 5. The 6 Traffic Flows

| Flow | Route | Description |
|---|---|---|
| FLOW_1 | West → INT_1 → INT_2 → East | Main corridor, westbound to eastbound |
| FLOW_2 | East → INT_2 → INT_1 → West | Main corridor, eastbound to westbound |
| FLOW_3 | South → INT_3 → INT_2 → East | South branch joining, going east |
| FLOW_4 | South → INT_3 → INT_2 → INT_1 → West | South branch joining, going west |
| FLOW_5 | West → INT_1 → INT_2 → INT_3 → South | Main corridor turning south |
| FLOW_6 | East → INT_2 → INT_3 → South | East turning south |

FLOW_4 consistently experiences the highest wait times because it
enters at INT_2 directly from the east boundary — hitting the busiest
junction first without passing through any prior intersection.

---

## 6. Adaptive Agent Policy

### How the Agent Makes Decisions

The adaptive agent replaces the fixed 30-second timer with ML-informed
signal timing. At each signal cycle it:

1. Reads current queue lengths at all approaches
2. Uses a trained Random Forest model to predict future queue lengths
3. Allocates green time proportionally to predicted congestion
4. Applies the 4-level tie-breaking policy when predictions are equal

### Green Time Calculation

base_duration = original fixed phase duration (30s for NS/EW, 20s for BRANCH)
load_ratio    = predicted queue score / average queue score across all phases
bonus         = min(10s, (load_ratio - 1) * 5)
final_duration = base_duration + bonus

The maximum bonus is capped at 10 seconds to prevent over-correction
and avoid starving other approaches.

### 4-Level Tie-Breaking Policy

When predicted queue lengths are equal across phases, the agent uses
this priority hierarchy:

**Level 1 — INT_2 Structural Priority**
INT_2 always wins a tie because it is the central junction serving all
6 flows. Clearing INT_2 faster reduces spillback to INT_1 and INT_3,
improving the entire network simultaneously.

**Level 2 — Longest Waiting Approach**
If INT_2 is not involved in the tie, the approach that has waited
the longest since its last green phase gets priority. This prevents
starvation — no approach should wait more than 3 cycles.

**Level 3 — Time-of-Day Rush Hour Bias**
During morning rush (7-9am), East-West phases get priority.
During evening rush (4-7pm), North-South phases get slight priority.

**Level 4 — Round-Robin Rotation**
If all above levels yield a tie (only in the first few seconds of
simulation before wait times accumulate), the agent cycles through
phases in order. This is a safe fallback that never crashes.

---

## 7. Performance Benchmarks

These are the measured results from the 24-hour simulation:

### Fixed Timer Baseline
| Metric | Value |
|---|---|
| Overall average wait time | 8.483 seconds |
| Overall maximum wait time | 49.96 seconds |
| Total cars served | 25,877 |
| INT_2 average wait time | 10.226 seconds |
| INT_2 East worst approach | 15.78 seconds average |

### Adaptive Agent Results
| Metric | Value | vs Baseline |
|---|---|---|
| Overall average wait time | 7.658 seconds | -9.7% |
| Overall maximum wait time | 33.79 seconds | -32.4% |
| Total cars served | 26,251 | +374 cars |
| INT_2 average wait time | 7.300 seconds | -28.6% |
| INT_1 average wait time | 7.960 seconds | +6.9% (trade-off) |

The INT_1 regression (+6.9%) is an intentional network-level trade-off.
The agent correctly prioritizes INT_2 because clearing the bottleneck
benefits the entire network. The net saving is 25,735 car-seconds —
6.7x more than the cost at INT_1.

---

## 8. Emergency and Special Conditions

### Emergency Vehicle Priority
When an emergency vehicle is detected (not currently implemented in
simulation), all signals should immediately give green to the emergency
vehicle's approach and hold all other phases.

### Maximum Wait Constraint
No approach should be held at red for more than 3 consecutive signal
cycles. If this occurs, the approach gets mandatory green regardless
of predicted queue lengths. This is a fairness override that takes
priority over all agent decisions.

### Starvation Prevention
The adaptive agent tracks time_since_last_served for every approach.
If any approach has waited more than 2 full signal cycles, it is
flagged for Level 2 priority in the next tie-breaking decision.

---

## 9. Glossary

| Term | Definition |
|---|---|
| Approach | A direction of traffic entering an intersection (N/S/E/W) |
| Phase | A signal state where specific approaches get green |
| Queue length | Number of cars waiting at a red light at one approach |
| Wait time | Seconds a car spends waiting at a red light |
| Bottleneck | An intersection where all flows converge, causing highest delays |
| Spillback | Congestion at one intersection backing up into adjacent ones |
| Flow | A specific route cars take through the network (6 flows total) |
| Tie-breaking | Policy used when predicted queues are equal across phases |
| Starvation | When one approach never gets a green light — prevented by policy |