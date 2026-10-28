# Traffic Management System: Simulation Metrics & Performance Analysis

This document contains empirical performance comparison data between **Fixed-Time Baseline** signal control and the **AI-Based Adaptive Signal Control** model implemented in this project.
Both simulations were executed over a simulated duration of **20 minutes** under identical, randomized Poisson vehicle arrival distributions with peak-hour load factors.

---

## 1. Key Performance Comparison Table

| Performance Metric | Fixed-Time Baseline | AI Adaptive Signal | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Control Philosophy** | Predetermined 30s Cycles | Dynamic Queue/Density Scaling | Real-time AI Driven |
| **Total Vehicles Cleared** | 998 | 1083 | **+85 vehicles** |
| **Unserved Queue Remaining** | 1897 | 1812 | **-85 backlog** |
| **Average Waiting Time** | 369.57 seconds | 258.69 seconds | **30.0% Reduction** |
| **Maximum Waiting Time** | 871.0 seconds | 546.0 seconds | **-325.0s Peak Delay** |
| **Average Queue Length** | 238.03 vehicles | 227.34 vehicles | **4.5% Shorter Queues** |
| **Throughput Rate** | 49.9 veh/min | 54.15 veh/min | **+8.5% Higher Capacity** |

---

## 2. Directional Queue Status at Simulation Close

| Direction | Fixed-Time Queue | AI Adaptive Queue | Difference |
| :--- | :--- | :--- | :--- |
| **NORTH (Heavy Flow)** | 765 | 474 | -291 |
| **SOUTH (Heavy Flow)** | 627 | 353 | -274 |
| **EAST (Moderate Flow)** | 318 | 558 | --240 |
| **WEST (Light Flow)** | 187 | 427 | --240 |

---

## 3. Analysis & Viva Talking Points

1. **Why Adaptive Beats Fixed-Time**:
   - In Fixed-Time mode, green time is squandered on empty lanes while congested approaches face spillover delays.
   - The AI-based algorithm monitors vehicle count, queue size, and road occupancy (ROI bounding box intersection), scaling green duration between 10s and 60s.
2. **Heavy Vehicle Compensation**:
   - The system adds an acceleration allowance for buses and trucks, preventing premature phase termination.
3. **Prevention of Starvation**:
   - Minimum green limits (10s) and competing queue urgency checks ensure secondary roads never experience indefinite red lock.
