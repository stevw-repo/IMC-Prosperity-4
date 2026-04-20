# IMC-Prosperity-4
***

### I made it to the finals (phase 2) as a solo participant. The most significant earning challenge for me is the round 2 manual challenge, earning +212,174 XIRECs and ranked 30th worldwide. Here's how I did it:

***

The "Invest & Expand" manual challenge requires participants to allocate a 50,000 XIRECs budget across Research, Scale, and Speed where:
> ### Final PnL = Research(r) × Scale(s) × Speed_Multiplier(sp) – 50,000

As I somehow got -29,661 PnL in round 1, I set myself to get at least 160,000 in the manual challenge to safely qualify into phase 2. Therefore, rather than simply maximising EV, my general approach is first setting a (list of) target PnLs, and utilise Monte Carlo Simulation to find the allocation that gives the **highest confidence level** for a certain PnL "k" in an exhaustive search. The main challenge in this approach is the rank‑based Speed Multiplier: the highest speed investment among all participants gets 0.9, the lowest 0.1, and others linearly scaled between.


To model the expected distribution of other people's allocation on Speed:
1. Assume the Speed distribution is uniform and observe the rough "optimal" allocation
2. Get a rough idea of the expected distribution by observing (1) the "optimal" Speed allocation on uniform distribution, (2) the results of the many different polls made by the good people on the IMC Prosperity discord and (3) what people revealed about their intended allocation.
3. Use a trimodal distribution to model the cluster at ~35% (the "optimal" solution and what most people chooses) and at 100% (people simply trolling) while also adding jittering to take account of uncertainty.


The rough expected distribution is given as follows:


<img width="2978" height="740" alt="imc_population_dist" src="https://github.com/user-attachments/assets/afdd1cdf-ff1f-4c7a-97fd-9116b69155ae" />


Now for all 5151 allocations, calculate Pr(PnL>k) (or EV), then find the best allocation that maximises Pr(PnL>k) (or EV).


<img width="2984" height="2357" alt="imc_main" src="https://github.com/user-attachments/assets/04035d12-6bcc-407f-889e-0b135f51e881" />
<img width="1784" height="881" alt="imc_deterministic" src="https://github.com/user-attachments/assets/1b8d387f-f392-4a9f-bc86-a76a1624c9d9" />
<img width="2232" height="740" alt="imc_pnl_dist" src="https://github.com/user-attachments/assets/70d32523-c499-47a7-9b19-631a527875a7" />




