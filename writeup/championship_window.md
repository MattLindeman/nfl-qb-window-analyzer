# The QB Tax: Does Paying a Quarterback Max Money Close the Championship Window?

*A data-driven analysis of 71 qualifying NFL QB contracts from 2000–2024*

---

## The Snap Before the Spiral

In March 2022, the Cleveland Browns handed Deshaun Watson the richest contract in NFL history: $230 million, fully guaranteed, before he'd taken a single snap in a Cleveland uniform. The move was audacious. It was also a masterclass in what happens when a franchise bets the entire cap on one position.

Two seasons later, the Browns had missed the playoffs, Watson had played a combined 12 games due to suspension and injury, and the organization had almost no room to address the glaring holes around him. The window — never truly open — had effectively shut before it started.

Watson's situation is extreme, but the underlying question it raises is not: **when a team allocates 15, 20, or even 25 percent of its entire salary cap to a single quarterback, what actually happens to championship probability over the next five years?**

We tried to find out.

---

## The Question

The NFL salary cap creates a zero-sum roster construction problem. Every dollar allocated to the quarterback is a dollar unavailable for pass rushers, wide receivers, offensive linemen, and the depth that separates playoff teams from pretenders. Common wisdom holds that paying a QB "max money" — a term that has inflated dramatically as cap percentages have climbed — effectively trades near-term Super Bowl odds for long-term stability.

But common wisdom is often wrong. Brady won three Super Bowls while taking well below market value. Mahomes won two while on his second contract. The question isn't just whether the cap hit matters — it's *how* it interacts with the age and context of the quarterback receiving it.

---

## The Data

We collected 71 qualifying QB contracts signed between 2000 and 2024, defined as any new extension or free-agent signing with an Average Annual Value (AAV) representing at least 10% of that year's salary cap. Rookie contracts and franchise tag seasons were excluded — they do not represent market-rate allocation decisions.

All contract values were normalized to cap percentage rather than raw dollars. This is essential: a $25 million AAV in 2018 (14.1% of cap) is fundamentally different from the same dollar figure in 2013 (20.3% of cap). Raw numbers mislead; percentages reveal the actual resource tradeoff.

For each contract, we tracked team performance for the five seasons following the signing year, or until the QB departed the team, whichever came first. Team outcomes were sourced from Pro Football Reference. The result: 263 contract-season observations across 66 unique contracts, organized into four cap tiers: 10–15%, 15–18%, 18–21%, and 21%+.

---

## Key Findings

**The 18–21% tier outperforms everyone — including the highest-paid QBs.**

This is the most counterintuitive result in the dataset. Teams with QBs in the 18–21% cap range posted a **64.7% playoff rate** and a **23.5% Super Bowl appearance rate** across their five-year windows. Both figures exceed those of the 21%+ tier (50% playoff rate, 12.5% SB rate) by a significant margin.

The pattern suggests there may be an inflection point somewhere around the 20–21% threshold. Below it, teams are paying market rate for a franchise QB while retaining enough cap flexibility to surround him with talent. Above it, the resource constraint begins to bite — and it bites hard by years three through five.

**Championship windows decay fastest at the top of the pay scale.**

When we track playoff probability by season post-signing, the 21%+ tier shows the steepest decline. In year one, these teams remain competitive. By year four and five, their playoff rate approaches league average (37.5%). The 18–21% tier degrades more slowly. The pattern is consistent with what we'd expect if roster quality is the binding constraint: max contracts erode the surrounding cast over time, and the compounding effect becomes visible only after a few seasons.

**Age at signing interacts with cap percentage more powerfully than either variable alone.**

The single most important feature in our Random Forest model was not cap_pct — it was the interaction term between cap percentage and age at signing, which captured 51% of the model's predictive power. This is the finding that changes how to think about the question.

Paying a 26-year-old 20% of your cap is not the same decision as paying a 34-year-old 20% of your cap. The young QB has five or six peak seasons ahead; the veterans' window is compressing even as the contract begins. The Mahomes deal looks radically different from the Rodgers-to-New York deal not primarily because of the cap number — both are in the 21%+ tier — but because of who is receiving it and when.

**Linear regression alone undersells the complexity.**

A simple OLS model shows that each additional percentage point of cap allocated to the QB is associated with approximately +0.19 wins per season — a positive relationship, but one that barely clears statistical significance (p = 0.051) and explains only 7% of variance in outcomes. This is genuinely informative: cap percentage is a blunt predictor because it ignores the quarterback's age, the team's existing roster quality, and the nonlinear threshold effects visible in the tier analysis. The Random Forest model, which captures these interactions, reduces test RMSE from 2.76 (baseline) to 2.23 — a 19% improvement.

---

## The Outliers

Any good analysis has to grapple with the cases that don't fit. Three stand out.

**Tom Brady (NE, multiple contracts):** Brady repeatedly signed below market value, enabling the Patriots to build one of the deepest rosters in NFL history around him. His contracts ranged from 10.0% to 22.5% of the cap, and his teams reached the Super Bowl in four of his post-extension windows. He is the most powerful outlier in this dataset — and his story is precisely about the cap flexibility his self-sacrifice created.

**Patrick Mahomes (KC, 2020):** The 10-year, $450M deal represented 24.2% of the 2020 cap on an AAV basis, placing him firmly in the 21%+ tier. Yet the Chiefs have appeared in four consecutive Super Bowls since signing. The explanation involves aggressive draft efficiency (Kelce, Mahomes himself, multiple late-round contributors), defensive coordinator investment, and Andy Reid's ability to scheme around cap constraints. Mahomes' age (25 at signing) also means the age-cap interaction works in Kansas City's favor.

**Joe Flacco (BAL, 2013):** Fresh off a Super Bowl MVP, Flacco signed for $20.1M AAV (16.3% of the 2013 cap). Baltimore reached the playoffs in two of the next five seasons but never returned to the Super Bowl. Flacco's contract is the cautionary tale about paying a QB for a peak that may already have passed — the age-cap interaction working in reverse.

---

## Implications

The data doesn't say don't pay your quarterback. Franchise QBs are worth a premium; the alternative — cycling through marginal starters — is demonstrably worse. What the data says is more nuanced.

**First, when you pay matters as much as how much you pay.** Signing a 25-year-old to 20% of the cap is a different bet from signing a 33-year-old to the same percentage. Teams would be better served thinking about the age-adjusted cap commitment rather than the headline number.

**Second, the 21% threshold appears meaningful.** Whether this reflects a genuine tipping point in roster construction mathematics or is an artifact of sample size is worth monitoring as more contracts accumulate. But the consistent underperformance of the highest-paid tier relative to the tier just below it is striking enough to warrant caution when front offices push contracts into that territory.

**Third, flexibility compounds.** The window decay curves show that cap-constrained teams don't just perform worse — they perform worse *over time*, as early-round picks turn into cap extensions and the roster gradually hollows out. Teams that keep enough room to re-sign draft hits and add free-agent depth maintain their windows longer.

---

## Methodology Notes

Several limitations deserve acknowledgment.

This analysis treats team wins as the primary outcome, which obscures variation in defensive quality, coaching, and schedule strength. A team that wins 10 games with a mediocre QB and an elite defense will appear identical to one winning 10 games with a great QB and a poor defense, yet the two franchises have very different championship trajectories.

The sample size (57 contracts in the modeling dataset) is modest for machine learning. The Random Forest's 5-fold cross-validated RMSE of 2.37 reflects genuine uncertainty in individual predictions; the model is better understood as capturing distributional tendencies than as a precise forecasting tool for any one contract.

Causality runs in both directions. Teams that expect to compete tend to pay QBs more; teams already on the decline often don't. This selection effect means the observed relationship between cap percentage and team performance cannot be cleanly interpreted as causal — a higher cap spend may partially *reflect* a competitive team rather than *cause* one.

Finally, the contract data for older signings (pre-2010) is less complete, particularly for fully guaranteed amounts. Missing guarantees were filled with zero, which likely understates true contract richness for early-era deals.

---

*Data sources: OverTheCap (contract history, salary cap totals), Pro Football Reference (team records, playoff results). All cap percentages computed as AAV ÷ league salary cap in signing year. Analysis performed in Python using scikit-learn, statsmodels, and SQLite.*
