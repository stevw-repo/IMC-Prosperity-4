# IMC-Prosperity-4
***

### I made it to the finals (phase 2) with a huge comeback. The most significant earning challenge for me is the round 2 manual challenge, earning +212,174 XIRECs and ranked 30th worldwide. Here is how I did it:

***

The "Invest & Expand" manual challenge requires participants to allocate a 50,000 XIRECs budget across Research, Scale, and Speed where:
> ### Final PnL = Research(r) × Scale(s) × Speed_Multiplier(sp) – 50,000

The general approach to find the optimal approach is to utilise Monte Carlo Simulation to do an exhaustive search. The main challenge in this approach is the rank‑based Speed Multiplier: the highest speed investment among all participants gets 0.9, the lowest 0.1, and others linearly scaled between.

To find the expected distribution of other people's allocation on Speed, I first assumed the speed distribution is uniform, then find a 
I roughly modeled the expected distribution of other people's allocation on Speed by simply observing the results of the many different polls made by the good people on the IMC Prosperity discord and people sharing their intended allocation:
<img width="2978" height="740" alt="imc_population_dist" src="https://github.com/user-attachments/assets/afdd1cdf-ff1f-4c7a-97fd-9116b69155ae" />


Now for all 5151 allocations, compute the expected PnL and 


