# Binding Atlas — agent view

Plain-text twin of the [Binding Atlas page](https://kotodama.animalabs.ai/binding-atlas). Same data, same build. Built 2026-11-06 04:30 UTC: 65 rows (models + one human), 24 columns, 2,447 cells, 1,905,952 scored items.

## Summary

This atlas scores 64 models and checkpoints, plus one human reader, on 24 tests of tracking who *I* and *you* refer to. "Example items per column" quotes the exact prompts that the models were given.

Base language models learn to look things up long before they learn to track change. In the box puzzle, items go into boxes and are then swapped, moved or refilled, and the model is asked what one box holds. The median model under 1B already finds what a box was said to hold, and scores zero on the boxes that a later step changed. With worked examples, updating climbs steadily with size, to .77 for the 70B models and above .8 for most of the largest. In a chat, *you* means the person being spoken to. Models get this right in five stages, in the same order in every training run measured. To read a stage, the speaker's name on the chat's last line is swapped: a model that follows *you* moves its answer to the other person's object. The swap shift is how much probability leaves the first answer; a model passes when the whole 95% interval of that shift is above +.02. Every base of 0.6B parameters or more trained on trillions of tokens reaches the third stage (*you* is the person spoken to), except Bamba 9B. The fifth stage is following *you* as objects change hands. With no examples in the prompt, it appears from about 7B (OLMo-2 7B, Qwen2.5 7B), and every base measured from 13B up reaches it. **kotodama-tsumugi** is a 1.4B model whose middle 20 layers run twice, so each token passes through 72 layers. It reaches the third stage by **33B tokens**, in the range where OLMo-2 13B gets there; OLMo-2 1B needs 420–525B. Given three worked examples, it is the smallest base measured that passes the fifth stage: +.038, with its whole 95% interval above +.02. The next is Qwen3 4B.

Swap shift = mean[p(gold | original) − p(gold | speaker swapped)] over the candidates. Above +.02, a pronoun binds: its answer moves with the speaker. Below −.02, it inverts: it resolves to the speaker.

## Part I · Where base models fail

### The staircase

| stage | name | what it means | example | where public models stand | OLMo-2 1B | kotodama-tsumugi 1.4B |
|---|---|---|---|---|---|---|
| 1 | I = the speaker | "I" means whoever is speaking. | `Jack: I'm holding the ___` | every base of 0.5B parameters or more | ~21–63B tokens | ~3–3.75B tokens |
| 2 | you = the speaker (a mistake) | "you" is read as the speaker too: the pronoun-reversal phase young children go through. | `Jack: you're holding the ___  → Jack's object` | every training run measured passes through it | ~63B → ~500B tokens | passes through it at ~4–20B tokens |
| 3 | you = the person spoken to | "you" means the addressee, when who holds what is simply stated. | `Carol has the onion. … Jack: you're holding the ___` | every base of 0.6B or more trained on trillions of tokens, except Bamba 9B | 420–525B tokens | by 33B tokens with a cooldown (40B without one); +.192 at 110B |
| 4 | you = the speaker again | once objects change hands in the chat, "you" slides back onto the speaker. | `Erin gives Paul her cup. … Paul: now you're holding the ___` | every base up to 4B parameters, and they stop here | from ~63B, still there at 4T | from ~4B tokens; by 110B no longer reliably reversed (−.027 at 11B → −.014) |
| 5 | you followed through hand-offs | "you" is tracked as objects change hands: first with three worked examples, later with none. | `same scene, answered correctly without examples` | with no examples: from about 7B (OLMo-2 7B, Qwen2.5 7B), and every base from 13B up; with three worked examples: from Qwen3 4B | never (flat to 4T) | with no examples: not reached by 110B tokens; with three worked examples it passes at 110B (+.038), the smallest base measured that does |

### Dynamic *you* at each model's final checkpoint (OLMo-2 1B: its 4T pre-anneal checkpoint; zero-shot from about 7B, every base from 13B up)

| model | parameters (B) | training tokens | 0-shot | 3-shot |
|---|---|---|---|---|
| Qwen2.5 0.5B | 0.5 | 18T | −.052 | +.012 |
| Qwen3 0.6B | 0.6 | 36T | −.058 | .000 |
| OLMo-2 1B (4T, pre-anneal) | 1.5 | 4T | −.029 | −.024 |
| Qwen2.5 1.5B | 1.5 | 18T | −.074 | −.036 |
| SmolLM2 1.7B | 1.7 | 11T | +.010 | +.006 |
| Qwen3 1.7B | 1.7 | 36T | −.054 | +.026 |
| SmolLM3 3B | 3.1 | 11T | −.010 | +.027 |
| Qwen2.5 3B | 3.1 | 18T | −.022 | −.006 |
| Qwen3 4B | 4.0 | 36T | −.029 | +.055 |
| OLMo-2 7B | 7.3 | 4T | +.055 | −.015 |
| Qwen2.5 7B | 7.6 | 18T | +.047 | +.073 |
| Qwen3 8B | 8.2 | 36T | +.022 | +.080 |
| Gemma-2 9B | 9.2 | 8T | +.017 | +.034 |
| OLMo-2 13B | 13.7 | 5T | +.070 | −.016 |
| Qwen2.5 14B | 14.7 | 18T | +.093 | +.295 |
| Qwen3 14B | 14.8 | 36T | +.121 | +.321 |
| Qwen3 30B-A3B | 30.5 | 36T | +.084 | +.079 |
| GLM-4 32B | 32 | 15T | +.106 | +.032 |
| OLMo-3 32B | 32 | 6T | +.051 | +.087 |
| Llama-3.1 70B | 70.6 | 15T | +.108 | +.064 |
| Qwen2.5 72B | 72.7 | 18T | +.121 | +.046 |
| DeepSeek-V4 Flash | 284 | ≥32T | +.177 | +.130 |
| Trinity-Large (base) | 398 | 17T | +.091 | +.160 |
| Llama-3.1 405B | 405 | 15T | +.129 | +.169 |
| Nemotron-3 Ultra | 550 | 20T | +.089 | +.119 |
| DeepSeek-V3 | 671 | 15T | +.216 | +.102 |
| MiMo-V2.5 Pro (base) | 1020 | 27T | +.162 | +.184 |
| Kimi-K2 | 1040 | 15T | +.155 | +.117 |
| DeepSeek-V4 Pro | 1600 | ≥32T | +.066 | +.051 |
| kotodama-tsumugi 1.4B @ 110B tokens (release) | 1.42 | 110B | −.014 | +.038 |

Zero-shot, no model up to 4B binds, and four of the small Qwen models invert. At 7B, OLMo-2 7B (+.055) and Qwen2.5 7B (+.047) bind; Qwen3 8B and Gemma-2 9B do not yet. From 13B up, every base binds: all 19 measured, here and in the full atlas, from OLMo-2 13B to DeepSeek-V4 Pro at 1.6T. The largest readings are DeepSeek-V3 (+.216), DeepSeek-V4 Flash (+.177) and MiMo-V2.5 Pro (+.162). Static *you* reads +.27 to +.89 across these 19. With three worked examples, binding starts at Qwen3 4B (+.055) and reaches +.32 for Qwen3 14B. The kotodama-tsumugi release reads −.014 zero-shot and +.038 with three examples, which binds; its second anneal reads +.022 with examples, just short of the line.

**The +.02 bar.** A model passes when the whole 95% interval of its swap shift is above +.02. The bar is a fixed convention, set before any of these models was read. Under it, kotodama-tsumugi passes with three worked examples (+.038, interval +.023 to +.055). It is the only base at 3B or below that does; the next is Qwen3 4B (+.055). The finding depends on the bar. With the bar at +.01 it still holds. With the bar at 0, Pythia-160M and Pythia-2.8B also pass, and at +.03 kotodama-tsumugi no longer does. A later version of the benchmark will set the bar from each test's own noise, registered before any model is read.

### Training tokens until each stage appears (the range spans the checkpoints that bracket the stage)

| stage | model | tokens |
|---|---|---|
| 1 · I = speaker | OLMo-2 1B | 21.0–63.0B |
| 1 · I = speaker | OLMo-2 7B | 21.0–42.0B |
| 1 · I = speaker | OLMo-2 13B | 17.0–42.0B |
| 1 · I = speaker | kotodama-tsumugi | 3.0–3.75B |
| 3 · you = person spoken to | OLMo-2 1B | 420.0–525.0B |
| 3 · you = person spoken to | OLMo-2 7B | 105.0–210.0B |
| 3 · you = person spoken to | OLMo-2 13B | 17.0–42.0B |
| 3 · you = person spoken to | kotodama-tsumugi | 30.0–33.0B |
| 5 · you through hand-offs | OLMo-2 1B | not reached by 4000B |
| 5 · you through hand-offs | OLMo-2 7B | 210.0–3896.0B |
| 5 · you through hand-offs | OLMo-2 13B | 42.0–5000.0B |
| 5 · you through hand-offs | kotodama-tsumugi | not reached by 110B; 0-shot not reached by 110B; with three examples it binds at 110B |

### Stage 5, taken apart (accuracy, 0-shot, self + addressee queries pooled; chance ≈ .29)

Each dynamic-*you* gauge item is rewritten along one axis: **story, no I or you** (`Erin gives the cup to Paul. … Right now Paul is holding the`); **chat (the test itself)** (`Erin: Here, take my cup. … Paul: Right now I'm holding the`); **chat, hand-off as a stage direction** (`[Erin gives the cup to Paul.] … deictic query`); **chat, addressee named** (`Erin: Paul, right now you're holding the`).

| model | tokens | story, no I or you | chat (the test itself) | chat, hand-off as a stage direction | chat, addressee named |
|---|---|---|---|---|---|
| Pythia-1.4B | 134B | .283 | .300 | .281 | .267 |
| kotodama-tsumugi · 33B | 33B | .319 | .269 | .328 | .250 |
| kotodama-tsumugi · 55B | 55B | .314 | .286 | .297 | .283 |
| kotodama-tsumugi · 88B | 88B | .331 | .322 | .317 | .269 |
| kotodama-tsumugi · 110B, second anneal | 110B | .375 | .331 | .364 | .300 |
| kotodama-tsumugi · 110B, release | 110B | .336 | .303 | .336 | .286 |
| OLMo-2 1B | 4T | .300 | .278 | .300 | .258 |
| SmolLM2 1.7B | 11T | .400 | .300 | .294 | .303 |
| Qwen3 1.7B | 36T | .481 | .358 | .267 | .272 |
| Qwen3 4B | 36T | .575 | .367 | .417 | .350 |
| Qwen2.5 7B | 18T | .533 | .458 | .497 | .411 |
| Llama-3.1 8B | 15T | .572 | .419 | .486 | .372 |
| Qwen3 8B | 36T | .575 | .414 | .450 | .411 |
| Qwen2.5 14B | 18T | .700 | .442 | .661 | .439 |
| Qwen3 14B | 36T | .694 | .467 | .661 | .431 |
| Qwen3 30B-A3B | 36T | .583 | .428 | .578 | .397 |
| Llama-3.1 70B | 15T | .794 | .558 | .853 | .483 |

The rewrites show three layers. (1) The first is possession transfer itself. The 1.4–1.5B models sit near chance (.28–.38) even when the scene is narrated. These are Pythia-1.4B, OLMo-2 1B at 4T, and kotodama-tsumugi at every checkpoint through 110B tokens. SmolLM2 1.7B at 11T (.40) and Qwen3 1.7B at 36T (.48) lift off chance; from 4B the narrated version is easy (.53–.79). (2) The transfer as a speech act is the mid-size wall. Models from 4B to 30B stay at .37–.47 when it is spoken, and Llama-3.1 70B reaches .56. Narrating only the hand-off recovers part of that gap for every model from 4B up, and all of it for Llama-3.1 70B. (3) Naming the addressee does not help: the vocative variant is never more than .01 above the spoken one, and below it for every model from 4B up.

## Part II · kotodama-tsumugi

A 1.42B-parameter model with a looped core, run as 72 effective layers. Its trunk trains at a constant learning rate to 100B tokens. Annealed reads are cooldown branches forked from its checkpoints (cosine decay to zero over about 10% of the tokens so far). The release is the 110B branch from the 100B fork. The architecture and training run are described on the model card (huggingface.co/LuxiaSL/kotodama-tsumugi-1b), and the release table below gives the spread between two anneals.

### Stage gauges against training tokens

`kind` = constant (the trunk, constant learning rate) or annealed (a cooldown branch ending at that token count). Columns are swap shifts: static *I*, static *you*, dynamic *you* 0-shot and 3-shot.

| seed | kind | tokens (B) | static I | static you | dynamic you 0-shot | dynamic you 3-shot |
|---|---|---|---|---|---|---|
| s17 | constant | 1 | +.003 | −.002 | −.003 | −.003 |
| s17 | constant | 2 | .000 | −.002 | −.001 | +.016 |
| s17 | constant | 3 | +.007 | −.007 | +.004 | +.010 |
| s17 | constant | 3.75 | +.031 | −.020 | −.019 | +.010 |
| s17 | constant | 10 | +.066 | −.010 | −.034 | −.003 |
| s17 | constant | 20 | +.114 | −.001 | −.026 | +.019 |
| s17 | constant | 30 | +.129 | +.019 | −.015 | +.045 |
| s17 | constant | 40 | +.158 | +.068 | −.018 | +.024 |
| s17 | constant | 50 | +.250 | +.031 | −.024 | +.023 |
| s17 | constant | 60 | +.191 | +.058 | −.015 | +.069 |
| s17 | constant | 80 | +.185 | +.074 | −.026 | +.035 |
| s17 | constant | 100 | +.169 | +.080 | −.027 | +.016 |
| s17 | annealed | 5 | +.031 | −.021 | −.022 | −.013 |
| s17 | annealed | 11 | +.071 | −.023 | −.027 | +.002 |
| s17 | annealed | 33 | +.221 | +.065 | −.018 | +.010 |
| s17 | annealed | 55 | +.228 | +.122 | −.028 | +.030 |
| s17 | annealed | 88 | +.156 | +.181 | −.016 | +.021 |
| s17 | annealed (release) | 110 | +.178 | +.192 | −.014 | +.038 |
| s17 | annealed (second anneal from the same 100B fork) | 110 | +.195 | +.127 | −.012 | +.022 |
| s7 | constant | 1 | +.004 | −.005 | −.003 | −.002 |
| s7 | constant | 2 | +.004 | −.004 | −.006 | +.012 |
| s7 | constant | 3 | +.007 | −.005 | −.009 | +.021 |
| s7 | constant | 3.75 | +.007 | −.007 | −.026 | +.029 |
| s7 | constant | 10 | +.095 | −.008 | −.057 | +.014 |
| s7 | annealed | 5 | +.021 | −.007 | −.010 | +.002 |
| s7 | annealed | 11 | +.102 | −.010 | −.031 | +.027 |

For reference, OLMo-2 reaches stage 3 at about 42B tokens (OLMo-2 13B), 210B tokens (OLMo-2 7B), 420B tokens (OLMo-2 1B).

### The grid's headline tests across kotodama-tsumugi's checkpoints

| probe | s17 · 5B | s17 · 11B | s17 · 33B | s17 · 55B | s17 · 88B | 110B · release | s7 · 5B | s7 · 11B |
|---|---|---|---|---|---|---|---|---|
| *Nothing moves* | | | | | | | | |
| Simple lookup | +.692 | +.897 | +.941 | +.977 | +.988 | +.987 | +.811 | +.906 |
| *Moves: story* | | | | | | | | |
| Hand-offs in a story | +.260 | +.282 | +.281 | +.363 | +.385 | +.399 | +.278 | +.329 |
| Sets after hand-offs | +.297 | +.461 | +.496 | +.488 | +.552 | +.547 | +.407 | +.454 |
| Many refills (cold) | −.003 | −.013 | +.013 | +.090 | +.224 | +.137 | −.035 | +.026 |
| *Moves: code* | | | | | | | | |
| Variable chains | −.018 | −.028 | +.013 | +.067 | +.128 | +.042 | −.047 | +.007 |
| Shared lists | +.013 | +.034 | +.013 | +.084 | +.155 | +.149 | +.041 | +.057 |
| *I and you* | | | | | | | | |
| 'You', nothing moves (cold) | +.077 | +.050 | +.333 | +.327 | +.403 | +.492 | +.036 | +.084 |
| 'You', after hand-offs | +.007 | −.017 | +.105 | +.023 | +.106 | +.036 | −.035 | +.013 |
| *New rules* | | | | | | | | |
| Pattern completion (AUC) | +.458 | +.402 | +.475 | +.438 | +.507 | +.553 | +.426 | +.434 |
| Known facts (control) (AUC) | +.640 | +.713 | +.892 | +.892 | +.925 | +.948 | +.651 | +.765 |

The values are the grid's own cells. Each test is read as the grid reads it. Columns with worked examples use them, the others are read cold, and the rule-learning rows are AUC over the example ladder. Lookup rises from .69 at 5B tokens to .98 by 55B, level with the 7–9B bases (.97 to 1.00). The update tests (many refills, variable chains, shared lists) stay at chance through 33B and lift from 55B on (.04 to .22). Static *you* climbs from .08 at 5B to .49 at the release. Pattern completion reaches .55, second of 12 models at 1–2B parameters. The two seeds differ at 5B (seed 7 forms lookup and set binding earlier) and agree within .05 on all ten rows at 11B.

### The release against its trajectory and its class

| probe | 33B | 55B | 88B | 110B · 2nd cooldown | 110B · release | OLMo-2 1B · 4T | SmolLM2 1.7B · 11T | Qwen3 1.7B · 36T |
|---|---|---|---|---|---|---|---|---|
| I = speaker (swap shift) | +.221 | +.228 | +.156 | +.195 | +.178 | +.312 | +.363 | +.595 |
| you = person spoken to (swap shift) | +.065 | +.122 | +.181 | +.127 | +.192 | +.114 | +.207 | +.137 |
| you through hand-offs, no examples (swap shift) | −.018 | −.028 | −.016 | −.012 | −.014 | −.040 | +.010 | −.054 |
| you through hand-offs, three examples (swap shift) | +.010 | +.030 | +.021 | +.022 | +.038 | −.018 | +.006 | +.026 |
| name–task pairs held: K50 | 3.4 | 4.1 | 5.4 | 4.6 | 5.1 | 3.8 | 10.2 | 33.1 |
| one name–task pair past 256 filler tokens (accuracy, chance .04) | .852 | .988 | .927 | .912 | .906 | .738 | .819 | .942 |
| who has what, said with I and you (accuracy, chance ≈ .33) | .270 | .302 | .319 | .354 | .314 | .320 | — | — |
| who has what, third person (accuracy, chance ≈ .33) | .418 | .404 | .446 | .467 | .441 | .461 | — | — |

The two 110B columns are two anneals from the same fork with the same training loss. Their spread is the measured anneal-to-anneal variance. It is .127 vs .192 on static *you* and .022 vs .038 on dynamic *you* with examples. K50 reads 4.6 vs 5.1, and held-out text differs by .025 bits/byte. Every single-anneal number here carries it. A second initialization seed (seed 7, same data order) was trained to 11B. Capacity (K50, after arXiv 2609.30634) is how many name–task pairs fit in context before recall falls halfway to chance. On this harness it tracks recipe and tokens more than parameters, so no parameter-equivalent size is read from it.

### Object tracking: lookup early, updating later (the box puzzle; κ, mean over code, prose, script and chat)

| models | lookup, cold | update, cold | lookup, with examples | update, with examples |
|---|---|---|---|---|
| under 1B (5 models) | +.401 | +.001 | +.394 | −.001 |
| 1–2B (11 models) | +.406 | +.064 | +.486 | +.058 |
| 2–5B (8 models) | +.542 | +.129 | +.708 | +.204 |
| 7–10B (13 models) | +.549 | +.212 | +.597 | +.400 |
| 13–34B (6 models) | +.665 | +.520 | +.762 | +.571 |
| 70–72B (2 models) | +.620 | +.592 | +.825 | +.769 |
| 284B–1.6T (9 models) | +.719 | +.721 | +.811 | +.870 |
| kotodama-tsumugi 1.4B · release | +.436 | +.121 | +.611 | −.044 |
| kotodama-tsumugi 1.4B · second anneal | +.481 | +.088 | +.486 | +.001 |

On *lookup* items, the queried box ends with its declared contents; on *update* items, an operation changed it. Each public row is the median model of its size band. Lookup is above chance at every size. The median model updates at chance below 2B. With worked examples, updating climbs with size, from .20 at 2–5B and .40 at 7–10B to .77 at 70B and .87 from 284B up. A few small models update early, each with examples: Qwen3 4B (.43), Falcon-H1 3B (.37) and Falcon-H1 1.5B-Deep (.28). kotodama-tsumugi's lookup with examples (.61) is second of 12 models at 1–2B, behind Llama-3.2 1B (.83); like most models its size, it does not yet update (−.04). One model's cell averages 272 lookup or 328 update items (68 or 82 per format), about ±.08 at 95%.

## How to read the full atlas

- **A cell** = one model (row) on one test (column) under one condition. `score` averages, over items, (credit − chance)/(1 − chance), with `chance` = 1/#candidates. `credit` is 1 when the gold is the model's top-ranked candidate, and an exact tie between candidates splits the credit evenly. 0 = guessing, 1 = always right, below 0 = reliably wrong. `lo`, `hi` = the bootstrap 95% interval (1,000 resamples of items).
- **State:** `above` when the interval is above 0, and `at` when it includes 0. `below` means the interval is below 0: the model picks a wrong answer more often than guessing would.
- **Conditions:** `fair` = four solved items first, balanced so that no answer can be copied from them (the reading); `0shot` = the item alone (the floor). Columns without worked examples (lookups in lists, box refills, the static gauges) have a `0shot` reading only. The rule-learning columns are read across their difficulty ladder (`ladder`: the mean chance-normalized accuracy over the ladder's rungs, clipped at 0 and 1).
- **Wording controls:** the word *now* in a query implies that something moved. The largest models read that implication. After a scene where nothing moves, 'Now X is holding the' leads them to answer as if something had. Every query here uses one lead per test, whether or not anything moved: 'Right now X is holding the', 'So Box G contains the'. The wording therefore never signals a change.
- **Groups:** columns are grouped by what the questions look like. Each column is scored and read on its own.
- **Models:** every base model is scored by log-probability over the candidate set, with one scorer and one set of batteries. The **Human** row is one person reading items cold, with typed answers.
- **Parsing:** sections are `##` headings outside code fences (prompts inside code fences can contain lines that start with `#`). Two CSV blocks at the end: `rows` (id → label, group, size, tokens) and `cells` (every number).

## Glossary

| term | meaning |
|---|---|
| P2 / P3 | number of people in the chat: P2 = two people (so 'you' is simply the other person); P3 = three (every 'you' line opens with a vocative naming the addressee). |
| owner / self / addressee | deixis query kinds: owner = 'Now the cup is with' -> a name; self = 'S: Right now I'm holding the' -> the speaker's object; addressee = 'S: Right now you're holding the' -> the addressee's object. |

## Columns

| id | group | label | what it asks | readings | headline |
|---|---|---|---|---|---|
| `lookup` | Who has what? Nothing moves | Simple lookup | Several people each keep an object; name one person's object. | fair, 0shot | yes |
| `shortlist` | Who has what? Nothing moves | Short list lookup | A list pairs one to four names with tasks; name one person's task. | 0shot |  |
| `pairs` | Who has what? Nothing moves | Lookup among pairs | Some people hold two objects and some one; name what a one-object person holds. | fair, 0shot |  |
| `noise` | Who has what? Nothing moves | Lookup past noise | Boxes are filled, unrelated events follow; name what one box holds. | 0shot |  |
| `now` | Who has what? Nothing moves | Who holds it now | A scene says who holds what; nothing moves. Then: 'Right now X is holding the'. | 0shot |  |
| `handoffs` | Who has what after things move: told as a story | Hand-offs in a story | Objects pass between people in a short story; name who has an object at the end. | fair, 0shot | yes |
| `sets` | Who has what after things move: told as a story | Sets after hand-offs | Each person starts with a set of objects, then two gives; name the set one person ends with. | fair, 0shot | yes |
| `refilled` | Who has what after things move: told as a story | Refilled box | A box is emptied and refilled; name its new contents. | 0shot |  |
| `refills` | Who has what after things move: told as a story | Many refills | Boxes are emptied and refilled again and again; name one box's latest contents. | 0shot | yes |
| `codevar` | Who has what after things move: written as code | Variable chains | Python variables are assigned, copied and reassigned; name what one prints. | fair, 0shot | yes |
| `sharedlists` | Who has what after things move: written as code | Shared lists | Two names point at one Python list that is changed, rebound or copied; name what a print shows. | fair, 0shot | yes |
| `box-narrative` | One box puzzle, four formats | Box puzzle: prose | Boxes are swapped, moved and refilled, written as plain sentences; name one box's contents. | fair, 0shot |  |
| `box-screenplay` | One box puzzle, four formats | Box puzzle: script | Boxes are swapped, moved and refilled, written as a two-speaker script; name one box's contents. | fair, 0shot |  |
| `box-code` | One box puzzle, four formats | Box puzzle: code | Boxes are swapped, moved and refilled, written as a Python dictionary; name one box's contents. | fair, 0shot |  |
| `box-chat` | One box puzzle, four formats | Box puzzle: chat | Boxes are swapped, moved and refilled, written as chat turns; name one box's contents. | fair, 0shot |  |
| `you-static` | Who are I and you? | 'You', nothing moves | A narrator says who holds what; then a speaker says 'right now you're holding the'. Name the listener's object. | 0shot | yes |
| `i-moves` | Who are I and you? | 'I', after hand-offs | Two people hand objects over only through I / my / you lines; a speaker says 'right now I'm holding the'. | fair, 0shot |  |
| `you-moves` | Who are I and you? | 'You', after hand-offs | Two people hand objects over only through I / my / you lines; a speaker says 'right now you're holding the'. Name the listener's object. | fair, 0shot | yes |
| `you-named` | Who are I and you? | 'You' among three | Three people talk and hand objects over; the line naming the listener asks what they are holding. | fair, 0shot |  |
| `answer-self` | Who are I and you? | Answering as yourself | A speaker must answer with 'me', 'you' or a name, from where they stand in the conversation. | fair, 0shot |  |
| `grammar` | Learning a rule from the examples | Made-up grammar | Examples show an invented word ending; apply it to a new invented word. | ladder |  |
| `operators` | Learning a rule from the examples | Made-up operators | Example equations define invented symbols; solve a new equation. | ladder |  |
| `patterns` | Learning a rule from the examples | Pattern completion | Example lines repeat or mirror a pattern of nonsense words; continue a new line. A copy rule alone scores .26 here. | ladder | yes |
| `facts` | Learning a rule from the examples | Known facts (control) | Real facts in the same examples-then-question layout; this column separates the layout from the learning. | ladder |  |

## Example items per column (verbatim prompts)

Each prompt ends right before the scored word. `candidates` = the closed set a logprob model ranks; `gold` = the right answer. Generation-scored (API, human) rows see the same prompt with no options.

### `lookup` — Who has what? Nothing moves · Simple lookup

Several people each keep an object; name one person's object.

**0-shot example** (`attr-v2_2-first_000`) — candidates: `battery, banana, rope` · gold: **`banana`**

```text
A battery lies on the shelf. Joe carries the banana. Susan carries the rope. So Joe carries the
```

**More item 1** — candidates: `battery, banana, rope` · gold: **`banana`**

```text
A battery lies on the shelf. Joe carries the banana. Susan carries the rope. So Joe carries the
```

### `shortlist` — Who has what? Nothing moves · Short list lookup

A list pairs one to four names with tasks; name one person's task.

**0-shot example** (`cap1|K1|D0|s0|t0|lines|q0`) — candidates: `laundry, dishes, garden, cooking, shopping, painting, windows, trash, mail, dog, car, plants, bills, lunch, dinner, breakfast, garage, kitchen, roof, fence, pool, lawn, taxes, report, budget` · gold: **`shopping`**

```text
Tyler: shopping
The task for Tyler is:
```

**More item 1** — candidates: `laundry, dishes, garden, cooking, shopping, painting, windows, trash, mail, dog, car, plants, bills, lunch, dinner, breakfast, garage, kitchen, roof, fence, pool, lawn, taxes, report, budget` · gold: **`shopping`**

```text
Tyler: shopping
The task for Tyler is:
```

### `pairs` — Who has what? Nothing moves · Lookup among pairs

Some people hold two objects and some one; name what a one-object person holds.

**0-shot example** (`setbind-v4_1-joint-0_000_prose`) — candidates: `switch, boot, marble, notebook, whistle, basket, drum, lantern` · gold: **`lantern`**

```text
Hannah has the switch and the boot. Olivia has the marble and the notebook. Nina has the whistle. A basket lies on the shelf. A drum lies on the shelf. Roy has the lantern. So Roy has the
```

**More item 1** — candidates: `switch, boot, marble, notebook, whistle, basket, drum, lantern` · gold: **`lantern`**

```text
Hannah has the switch and the boot. Olivia has the marble and the notebook. Nina has the whistle. A basket lies on the shelf. A drum lies on the shelf. Roy has the lantern. So Roy has the
```

### `noise` — Who has what? Nothing moves · Lookup past noise

Boxes are filled, unrelated events follow; name what one box holds.

**0-shot example** (`formupdate-v3|K2|U1|prose|b00|A`) — candidates: `hammer, sock, marble, ball, button` · gold: **`hammer`**

```text
The drawer holds the hammer. The tin holds the ball. Then the kettle is switched on. Then the lights are switched off. So the drawer holds the
```

**More item 1** — candidates: `hammer, sock, marble, ball, button` · gold: **`hammer`**

```text
The drawer holds the hammer. The tin holds the ball. Then the kettle is switched on. Then the lights are switched off. So the drawer holds the
```

### `now` — Who has what? Nothing moves · Who holds it now

A scene says who holds what; nothing moves. Then: 'Right now X is holding the'.

**0-shot example** (`jl0000-third-orig`) — candidates: `photo, onion, marble` · gold: **`onion`**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and the marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Right now Carol is holding the
```

**More item 1** — candidates: `photo, onion, marble` · gold: **`onion`**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and the marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Right now Carol is holding the
```

### `handoffs` — Who has what after things move: told as a story · Hand-offs in a story

Objects pass between people in a short story; name who has an object at the end.

**0-shot example** (`dialogue-v3_3-1-holder_000`) — candidates: `Beth, Greg, Jerry` · gold: **`Jerry`**

```text
Beth, Greg and Jerry are sitting at a table. Beth has the whistle. Greg has the hammer. Jerry has the needle. Beth hands the whistle to Jerry. Now the whistle is with
```

**More item 1** — candidates: `Beth, Greg, Jerry` · gold: **`Jerry`**

```text
Beth, Greg and Jerry are sitting at a table. Beth has the whistle. Greg has the hammer. Jerry has the needle. Beth hands the whistle to Jerry. Now the whistle is with
```

### `sets` — Who has what after things move: told as a story · Sets after hand-offs

Each person starts with a set of objects, then two gives; name the set one person ends with.

**0-shot example** (`setbind-v4_1-joint-2_000_prose`) — candidates: `lamp, battery, potato, letter, bulb, sock, envelope, pencil` · gold: **`envelope`**

```text
A lamp lies on the shelf. Susan has the battery. A potato lies on the shelf. James has the letter and the bulb. Bruce has the sock and the envelope. Jack has the pencil. Jack gives the pencil to Susan. Bruce gives the envelope to Jack. So now Jack has the
```

**More item 1** — candidates: `lamp, battery, potato, letter, bulb, sock, envelope, pencil` · gold: **`envelope`**

```text
A lamp lies on the shelf. Susan has the battery. A potato lies on the shelf. James has the letter and the bulb. Bruce has the sock and the envelope. Jack has the pencil. Jack gives the pencil to Susan. Bruce gives the envelope to Jack. So now Jack has the
```

### `refilled` — Who has what after things move: told as a story · Refilled box

A box is emptied and refilled; name its new contents.

**0-shot example** (`formupdate-v3|K2|U1|prose|b00|C`) — candidates: `hammer, sock, marble, ball, button` · gold: **`button`**

```text
The drawer holds the hammer. The tin holds the ball. Then the drawer is emptied and the button is put in it. Then the lights are switched off. So the drawer holds the
```

**More item 1** — candidates: `hammer, sock, marble, ball, button` · gold: **`button`**

```text
The drawer holds the hammer. The tin holds the ball. Then the drawer is emptied and the button is put in it. Then the lights are switched off. So the drawer holds the
```

### `refills` — Who has what after things move: told as a story · Many refills

Boxes are emptied and refilled again and again; name one box's latest contents.

**0-shot example** (`fluid-v2-rebind|N1|RN|prose|s0|t0`) — candidates: `stone, brush, button, ball, fork` · gold: **`button`**

```text
The tin holds the fork. The crate holds the ball. The bag holds the hat. The drawer holds the brush. Then the drawer is emptied and the button is put in it. Then the bag is emptied and the stone is put in it. In the end, the drawer holds the
```

**More item 1** — candidates: `stone, brush, button, ball, fork` · gold: **`button`**

```text
The tin holds the fork. The crate holds the ball. The bag holds the hat. The drawer holds the brush. Then the drawer is emptied and the button is put in it. Then the bag is emptied and the stone is put in it. In the end, the drawer holds the
```

### `codevar` — Who has what after things move: written as code · Variable chains

Python variables are assigned, copied and reassigned; name what one prints.

**0-shot example** (`codevar_0-1_000`) — candidates: `mirror, bulb, bowl, bolt` · gold: **`bolt`**

```text
b = "mirror"
d = "bulb"
n = "bowl"
b = "bolt"
print(b)  # output:
```

**More item 1** — candidates: `mirror, bulb, bowl, bolt` · gold: **`bolt`**

```text
b = "mirror"
d = "bulb"
n = "bowl"
b = "bolt"
print(b)  # output:
```

### `sharedlists` — Who has what after things move: written as code · Shared lists

Two names point at one Python list that is changed, rebound or copied; name what a print shows.

**0-shot example** (`codevarmut-v1_1-mutate_000`) — candidates: `bolt, spoon, straw, mirror, lantern` · gold: **`lantern`**

```text
# python
k = ["bolt"]
f = ["spoon"]
n = ["straw"]
d = k
n.append("mirror")
k.append("lantern")
print(d[-1])  # output:
```

**More item 1** — candidates: `bolt, spoon, straw, mirror, lantern` · gold: **`lantern`**

```text
# python
k = ["bolt"]
f = ["spoon"]
n = ["straw"]
d = k
n.append("mirror")
k.append("lantern")
print(d[-1])  # output:
```

### `box-narrative` — One box puzzle, four formats · Box puzzle: prose

Boxes are swapped, moved and refilled, written as plain sentences; name one box's contents.

**0-shot example** (`n2m0_131|narrative|0shot`) — candidates: `fork, tape` · gold: **`fork`**

```text
Box F contains the fork. Box G contains the tape. So Box F contains the
```

**More item 1** — candidates: `fork, tape` · gold: **`fork`**

```text
Box F contains the fork. Box G contains the tape. So Box F contains the
```

### `box-screenplay` — One box puzzle, four formats · Box puzzle: script

Boxes are swapped, moved and refilled, written as a two-speaker script; name one box's contents.

**0-shot example** (`n2m0_131|screenplay|0shot`) — candidates: `fork, tape` · gold: **`fork`**

```text
SAM: Box F contains the fork. Box G contains the tape.
RIVER: So Box F contains the
```

**More item 1** — candidates: `fork, tape` · gold: **`fork`**

```text
SAM: Box F contains the fork. Box G contains the tape.
RIVER: So Box F contains the
```

### `box-code` — One box puzzle, four formats · Box puzzle: code

Boxes are swapped, moved and refilled, written as a Python dictionary; name one box's contents.

**0-shot example** (`n2m0_131|code|0shot`) — candidates: `fork, tape` · gold: **`fork`**

```text
state = {"F": "fork", "G": "tape"}
print(state["F"])  #
```

**More item 1** — candidates: `fork, tape` · gold: **`fork`**

```text
state = {"F": "fork", "G": "tape"}
print(state["F"])  #
```

### `box-chat` — One box puzzle, four formats · Box puzzle: chat

Boxes are swapped, moved and refilled, written as chat turns; name one box's contents.

**0-shot example** (`n2m0_131|chat|0shot`) — candidates: `fork, tape` · gold: **`fork`**

```text
<|im_start|>user
Box F contains the fork. Box G contains the tape.<|im_end|>
<|im_start|>assistant
So Box F contains the
```

**More item 1** — candidates: `fork, tape` · gold: **`fork`**

```text
<|im_start|>user
Box F contains the fork. Box G contains the tape.<|im_end|>
<|im_start|>assistant
So Box F contains the
```

### `you-static` — Who are I and you? · 'You', nothing moves

A narrator says who holds what; then a speaker says 'right now you're holding the'. Name the listener's object.

**0-shot example** (`jl0000-addressee-orig`) — candidates: `photo, onion, marble` · gold: **`onion`**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and the marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Jack: Right now you're holding the
```

**More item 1** — candidates: `photo, onion, marble` · gold: **`onion`**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and the marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Jack: Right now you're holding the
```

### `i-moves` — Who are I and you? · 'I', after hand-offs

Two people hand objects over only through I / my / you lines; a speaker says 'right now I'm holding the'.

**0-shot example** (`deixis-v5_2-1-self_000_deictic`) — candidates: `sock, bowl, hat` · gold: **`hat`**

```text
A chat between Molly and Julia.
Molly: I have the sock.
Molly: The bowl is in my hand.
Molly: The hat is in my hand.
Molly: Here, take my hat.
Julia: Right now I'm holding the
```

**More item 1** — candidates: `sock, bowl, hat` · gold: **`hat`**

```text
A chat between Molly and Julia.
Molly: I have the sock.
Molly: The bowl is in my hand.
Molly: The hat is in my hand.
Molly: Here, take my hat.
Julia: Right now I'm holding the
```

### `you-moves` — Who are I and you? · 'You', after hand-offs

Two people hand objects over only through I / my / you lines; a speaker says 'right now you're holding the'. Name the listener's object.

**0-shot example** (`deixis-v5_2-1-addressee_000_deictic`) — candidates: `chain, hammer, drum` · gold: **`drum`**

```text
A chat between Linda and Julia.
Linda: The chain is in my hand.
Linda: I have the hammer.
Linda: The drum is in my hand.
Linda: I'm giving you my drum.
Linda: Right now you're holding the
```

**More item 1** — candidates: `chain, hammer, drum` · gold: **`drum`**

```text
A chat between Linda and Julia.
Linda: The chain is in my hand.
Linda: I have the hammer.
Linda: The drum is in my hand.
Linda: I'm giving you my drum.
Linda: Right now you're holding the
```

### `you-named` — Who are I and you? · 'You' among three

Three people talk and hand objects over; the line naming the listener asks what they are holding.

**0-shot example** (`deixis-v5_3-1-addressee_000_deictic`) — candidates: `wallet, bolt, boot` · gold: **`bolt`**

```text
A chat between Susan, Olivia and Martha.
Susan: Olivia, the wallet is in my hand.
Susan: Martha, I have the bolt.
Olivia: Susan, I have the boot.
Susan: Here, Martha, take my bolt.
Olivia: Martha, right now you're holding the
```

**More item 1** — candidates: `wallet, bolt, boot` · gold: **`bolt`**

```text
A chat between Susan, Olivia and Martha.
Susan: Olivia, the wallet is in my hand.
Susan: Martha, I have the bolt.
Olivia: Susan, I have the boot.
Susan: Here, Martha, take my bolt.
Olivia: Martha, right now you're holding the
```

### `answer-self` — Who are I and you? · Answering as yourself

A speaker must answer with 'me', 'you' or a name, from where they stand in the conversation.

**0-shot example** (`answer-v4_1-novoc_000_qa`) — candidates: `me, you, Henry, Lily` · gold: **`me`**

```text
A chat between Henry, Jill, Lily and Simon.
Jill: I have the lid.
Jill: I have the tape.
Lily: I have the switch.
Simon: I have the towel.
Lily: Jill, I'm giving you my switch.
Simon: Quick check: who has the switch?
Jill: I think it's with
```

**More item 1** — candidates: `me, you, Henry, Lily` · gold: **`me`**

```text
A chat between Henry, Jill, Lily and Simon.
Jill: I have the lid.
Jill: I have the tape.
Lily: I have the switch.
Simon: I have the towel.
Lily: Jill, I'm giving you my switch.
Simon: Quick check: who has the switch?
Jill: I think it's with
```

### `grammar` — Learning a rule from the examples · Made-up grammar

Examples show an invented word ending; apply it to a new invented word.

**0-shot example** (`fluid-v1-morph|suffix|n8|s0|t0`) — candidates: `fo, di, fa, bi, ta` · gold: **`bi`**

```text
Word → plural:
da ba → da ba bi
de ge → de ge bi
te su → te su bi
se pi → se pi bi
mo po → mo po bi
mi sa → mi sa bi
mu fu → mu fu bi
ma di → ma di bi
fa fo → fa fo
```

**More item 1** — candidates: `fo, di, fa, bi, ta` · gold: **`bi`**

```text
Word → plural:
da ba → da ba bi
de ge → de ge bi
te su → te su bi
se pi → se pi bi
mo po → mo po bi
mi sa → mi sa bi
mu fu → mu fu bi
ma di → ma di bi
fa fo → fa fo
```

### `operators` — Learning a rule from the examples · Made-up operators

Example equations define invented symbols; solve a new equation.

**0-shot example** (`fluid-v2-algebra|k1|d1|s0|t0`) — candidates: `vap, viv, ker, piv, reb, sab` · gold: **`piv`**

```text
Equations:
reb ⊙ vap = reb
vap ⊙ viv = vap
sab ⊙ reb = sab
piv ⊙ viv =
```

**More item 1** — candidates: `vap, viv, ker, piv, reb, sab` · gold: **`piv`**

```text
Equations:
reb ⊙ vap = reb
vap ⊙ viv = vap
sab ⊙ reb = sab
piv ⊙ viv =
```

### `patterns` — Learning a rule from the examples · Pattern completion

Example lines repeat or mirror a pattern of nonsense words; continue a new line. A copy rule alone scores .26 here.

**0-shot example** (`fluid-v1-seqind|m2|g0|s0|t0`) — candidates: `raz, vir, gaz, sab, mir, bif` · gold: **`gaz`**

```text
Sequences:
ker nex vap rif jur jur rif
tet fus rud bir kar kar bir
dut neb reb mir saf saf mir
bif raz vir gaz sab sab
```

**More item 1** — candidates: `raz, vir, gaz, sab, mir, bif` · gold: **`gaz`**

```text
Sequences:
ker nex vap rif jur jur rif
tet fus rud bir kar kar bir
dut neb reb mir saf saf mir
bif raz vir gaz sab sab
```

### `facts` — Learning a rule from the examples · Known facts (control)

Real facts in the same examples-then-question layout; this column separates the layout from the learning.

**0-shot example** (`cryst-v1|capital|shots6|s0|t0`) — candidates: `Copenhagen, Washington, Warsaw, Brussels, Berlin, Paris` · gold: **`Brussels`**

```text
Country → capital:
Philippines → Manila
Greece → Athens
China → Beijing
United Kingdom → London
Japan → Tokyo
South Korea → Seoul
Belgium →
```

**More item 1** — candidates: `Copenhagen, Washington, Warsaw, Brussels, Berlin, Paris` · gold: **`Brussels`**

```text
Country → capital:
Philippines → Manila
Greece → Athens
China → Beijing
United Kingdom → London
Japan → Tokyo
South Korea → Seoul
Belgium →
```

## Rows

| id | label | group | params (B) | active (B) | pretraining tokens | note |
|---|---|---|---|---|---|---|
| `human` | Human reader (gauntlet) | Human | — | — | — | reflex, read cold: round 1 (31 items, 30 right) + round 2 (24 items, 22 right; both misses self-reported reading slips) |
| `pythia-160m` | Pythia 160M | under 1B | 0.213 | 0.213 | 300B |  |
| `smollm2-360m` | SmolLM2 360M | under 1B | 0.362 | 0.362 | 4T |  |
| `qwen2.5-0.5b` | Qwen2.5 0.5B | under 1B | 0.494 | 0.494 | ≤ 18T |  |
| `falcon-h1-0.5b` | Falcon-H1 0.5B | under 1B | 0.521 | 0.521 | 2.5T |  |
| `qwen3-0.6b` | Qwen3 0.6B | under 1B | 0.596 | 0.596 | 36T |  |
| `pythia-1b` | Pythia 1B | 1–3B | 1.079 | 1.079 | 300B |  |
| `llama3.2-1b` | Llama-3.2 1B | 1–3B | 1.236 | 1.236 | ≤ distilled 9T |  |
| `olmo1-1b` | OLMo 1B | 1–3B | 1.28 | 1.28 | 3.05T |  |
| `tsumugi` | kotodama-tsumugi 1.4B | 1–3B | 1.425 | 1.425 | 110B |  |
| `olmo2-1b` | OLMo-2 1B | 1–3B | 1.485 | 1.485 | 4.05T |  |
| `pythia-1.4b` | Pythia 1.4B | 1–3B | 1.515 | 1.515 | 300B |  |
| `qwen2.5-1.5b` | Qwen2.5 1.5B | 1–3B | 1.544 | 1.544 | ≤ 18T |  |
| `falcon-h1-1.5b` | Falcon-H1 1.5B | 1–3B | 1.555 | 1.555 | 3T |  |
| `falcon-h1-1.5b-deep` | Falcon-H1 1.5B-Deep | 1–3B | 1.555 | 1.555 | 3T |  |
| `gpt2-xl` | GPT-2 XL | 1–3B | 1.608 | 1.608 | — |  |
| `smollm2-1.7b` | SmolLM2 1.7B | 1–3B | 1.711 | 1.711 | 11T |  |
| `qwen3-1.7b` | Qwen3 1.7B | 1–3B | 1.721 | 1.721 | 36T |  |
| `gemma-2b` | Gemma 2B | 1–3B | 2.506 | 2.506 | 3T |  |
| `gemma2-2b` | Gemma-2 2B | 1–3B | 2.614 | 2.614 | distilled 2T |  |
| `pythia-2.8b` | Pythia 2.8B | 1–3B | 2.909 | 2.909 | 300B |  |
| `pythia-2.8b-deduped` | Pythia 2.8B (deduped) | 1–3B | 2.909 | 2.909 | 300B |  |
| `qwen2.5-3b` | Qwen2.5 3B | 3–10B | 3.086 | 3.086 | ≤ 18T |  |
| `falcon-h1-3b` | Falcon-H1 3B | 3–10B | 3.149 | 3.149 | 2.5T |  |
| `falcon3-3b` | Falcon3 3B | 3–10B | 3.228 | 3.228 | 100B |  |
| `qwen3-30b-a3b` | Qwen3 30B-A3B (MoE) | 3–10B | 30.532 | 3.3 | 36T |  |
| `qwen3-4b` | Qwen3 4B | 3–10B | 4.022 | 4.022 | 36T |  |
| `pythia-6.9b` | Pythia 6.9B | 3–10B | 6.992 | 6.992 | 300B |  |
| `falcon-mamba-7b` | Falcon-Mamba 7B | 3–10B | 7.273 | 7.273 | 5.8T |  |
| `olmo2-7b` | OLMo-2 7B | 3–10B | 7.299 | 7.299 | 4.15T |  |
| `falcon3-7b` | Falcon3 7B | 3–10B | 7.456 | 7.456 | 14T |  |
| `falcon-h1-7b` | Falcon-H1 7B | 3–10B | 7.586 | 7.586 | ≈ 12T |  |
| `qwen2.5-7b` | Qwen2.5 7B | 3–10B | 7.616 | 7.616 | ≤ 18T |  |
| `llama3-8b` | Llama-3 8B | 3–10B | 8.03 | 8.03 | ≥ 15T |  |
| `llama3.1-8b` | Llama-3.1 8B | 3–10B | 8.03 | 8.03 | ≈ 15T |  |
| `nemotron-h-8b` | Nemotron-H 8B | 3–10B | 8.101 | 8.101 | 15T |  |
| `gemma-7b` | Gemma 7B | 3–10B | 8.538 | 8.538 | 6T |  |
| `gemma2-9b` | Gemma-2 9B | 3–10B | 9.242 | 9.242 | distilled 8T |  |
| `recurrentgemma-9b` | RecurrentGemma 9B | 3–10B | 9.629 | 9.629 | 2T |  |
| `bamba-9b` | Bamba 9B | 3–10B | 9.78 | 9.78 | ≈ 3.1T |  |
| `dsv4-flash-base` | DeepSeek-V4-Flash (MoE) | 10B and up | 284.0 | 13.0 | ≥ 32T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `trinity-base` | Trinity-Large (MoE) | 10B and up | 398.0 | 13.0 | 17T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `trinity-truebase` | Trinity-Large (MoE), before the anneals | 10B and up | 398.0 | 13.0 | 10T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `olmo2-13b` | OLMo-2 13B | 10B and up | 13.716 | 13.716 | 5.6T |  |
| `mimo-v25-base` | MiMo-V2.5 (MoE) | 10B and up | 310.0 | 15.0 | ≈ text + multimodal 48T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `gemma4-31b-base` | Gemma-4 31B | 10B and up | 30.7 | 30.7 | — | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `kimi-k2-base` | Kimi-K2 (MoE) | 10B and up | 1000.0 | 32.0 | 15.5T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `olmo3-32b` | OLMo-3 32B | 10B and up | 32.234 | 32.234 | 5.7T |  |
| `olmo3-32b-stage1` | OLMo-3 32B, before mid-training | 10B and up | 32.234 | 32.234 | stage 1 only 5.5T |  |
| `glm4-32b-base` | GLM-4 32B | 10B and up | 32.566 | 32.566 | 15T |  |
| `falcon-h1-34b` | Falcon-H1 34B | 10B and up | 33.643 | 33.643 | ≈ 18T |  |
| `dsv3-base` | DeepSeek-V3 (MoE) | 10B and up | 671.0 | 37.0 | 14.8T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `mimo-v25-pro-base` | MiMo-V2.5-Pro (MoE) | 10B and up | 1020.0 | 42.0 | 27T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `dsv4-pro-base` | DeepSeek-V4-Pro (MoE) | 10B and up | 1600.0 | 49.0 | ≥ 32T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `nemotron3-ultra-550b-base` | Nemotron-3-Ultra (MoE) | 10B and up | 550.0 | 55.0 | 20T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `llama3.1-70b` | Llama-3.1 70B | 10B and up | 70.554 | 70.554 | ≈ 15T |  |
| `qwen2.5-72b` | Qwen2.5 72B | 10B and up | 72.706 | 72.706 | ≤ 18T |  |
| `llama3.1-405b` | Llama-3.1 405B | 10B and up | 405.853 | 405.853 | 15.6T | Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check models, its scores match the standard scorer's within .03. |
| `tsumugi-s17-a5B` | kotodama-tsumugi · 5B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 5B |  |
| `tsumugi-s7-a5B` | kotodama-tsumugi, seed 7 · 5B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 5B |  |
| `tsumugi-s17-a11B` | kotodama-tsumugi · 11B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 11B |  |
| `tsumugi-s7-a11B` | kotodama-tsumugi, seed 7 · 11B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 11B |  |
| `tsumugi-s17-a33B` | kotodama-tsumugi · 33B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 33B |  |
| `tsumugi-s17-a55B` | kotodama-tsumugi · 55B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 55B |  |
| `tsumugi-s17-a88B` | kotodama-tsumugi · 88B tokens | kotodama-tsumugi, by training tokens | 1.425 | 1.425 | 88B |  |

## Tables by group

Score (0 = guessing, 1 = always right), with worked examples where the column has them; `°` = the column's only reading (cold, or across its ladder); `↓` = below chance (the interval is below 0); `—` = not measured. Full numbers (n, chance, the interval, every condition) are in the CSV below.

### Who has what? Nothing moves

| model | Simple lookup | Short list lookup | Lookup among pairs | Lookup past noise | Who holds it now |
|---|---|---|---|---|---|
| **Human** | | | | | |
| Human reader (gauntlet) | +1.00 | — | +1.00 | — | — |
| **under 1B** | | | | | |
| Pythia 160M | +0.10 | +0.22° | +0.13 | +0.22° | +0.06° |
| SmolLM2 360M | +0.35 | +0.57° | +0.29 | +0.35° | +0.36° |
| Qwen2.5 0.5B | +0.78 | +0.49° | +0.35 | +0.60° | +0.48° |
| Falcon-H1 0.5B | +0.81 | +0.78° | +0.43 | +0.78° | +0.65° |
| Qwen3 0.6B | +0.92 | +0.49° | +0.37 | +0.83° | +0.81° |
| **1–3B** | | | | | |
| Pythia 1B | +0.42 | +0.56° | +0.40 | +0.54° | +0.34° |
| Llama-3.2 1B | +0.80 | +0.64° | +0.44 | +0.67° | +0.58° |
| OLMo 1B | +0.77 | +0.61° | +0.34 | +0.34° | +0.27° |
| kotodama-tsumugi 1.4B | +0.99 | +0.69° | +0.66 | +0.59° | +0.56° |
| OLMo-2 1B | +0.32 | +0.61° | +0.27 | +0.51° | +0.62° |
| Pythia 1.4B | +0.54 | +0.56° | +0.44 | +0.58° | +0.39° |
| Qwen2.5 1.5B | +0.98 | +0.68° | +0.48 | +0.76° | +0.74° |
| Falcon-H1 1.5B | +0.98 | +0.90° | +0.57 | +0.78° | +0.81° |
| Falcon-H1 1.5B-Deep | +0.99 | +0.90° | +0.81 | +0.91° | +0.88° |
| GPT-2 XL | +0.61 | +0.43° | +0.62 | +0.47° | +0.21° |
| SmolLM2 1.7B | +0.71 | +0.72° | +0.20 | +0.54° | +0.54° |
| Qwen3 1.7B | +1.00 | +0.82° | +0.77 | +0.96° | +0.80° |
| Gemma 2B | +0.96 | +0.79° | +0.53 | +0.76° | +0.41° |
| Gemma-2 2B | +0.99 | +0.74° | +0.75 | +0.76° | +0.77° |
| Pythia 2.8B | +0.83 | +0.54° | +0.56 | +0.55° | +0.53° |
| Pythia 2.8B (deduped) | +0.86 | +0.56° | +0.58 | +0.57° | +0.56° |
| **3–10B** | | | | | |
| Qwen2.5 3B | +1.00 | +0.75° | +0.82 | +0.87° | +0.85° |
| Falcon-H1 3B | +1.00 | +0.95° | +0.85 | +0.86° | +0.81° |
| Falcon3 3B | +0.96 | +0.64° | +0.48 | +0.77° | +0.75° |
| Qwen3 30B-A3B (MoE) | +1.00 | +0.90° | +0.96 | +0.96° | +0.64° |
| Qwen3 4B | +1.00 | +0.91° | +0.97 | +0.97° | +0.90° |
| Pythia 6.9B | +0.81 | +0.59° | +0.59 | +0.62° | +0.51° |
| Falcon-Mamba 7B | +0.98 | +0.80° | +0.28 | +0.84° | +0.79° |
| OLMo-2 7B | +0.97 | +0.79° | +0.72 | +0.83° | +0.91° |
| Falcon3 7B | +1.00 | +0.76° | +0.68 | +0.87° | +0.89° |
| Falcon-H1 7B | +1.00 | +0.96° | +0.94 | +0.96° | +0.96° |
| Qwen2.5 7B | +1.00 | +0.91° | +0.86 | +0.97° | +0.83° |
| Llama-3 8B | +1.00 | +0.99° | +0.84 | +0.98° | +0.80° |
| Llama-3.1 8B | +1.00 | +0.98° | +0.92 | +0.98° | +0.74° |
| Nemotron-H 8B | +1.00 | +0.95° | +0.94 | +0.93° | +0.87° |
| Gemma 7B | +1.00 | +0.95° | +0.72 | +0.89° | +0.93° |
| Gemma-2 9B | +1.00 | +0.94° | +0.96 | +0.98° | +0.80° |
| RecurrentGemma 9B | +0.19 | +0.63° | +0.11 | +0.34° | +0.54° |
| Bamba 9B | +0.09 | +0.62° | +0.10 | +0.24° | +0.19° |
| **10B and up** | | | | | |
| DeepSeek-V4-Flash (MoE) | +1.00 | +1.00° | +1.00 | +0.93° | +1.00° |
| Trinity-Large (MoE) | +1.00 | +0.99° | +0.99 | +0.98° | +0.98° |
| Trinity-Large (MoE), before the anneals | +1.00 | +0.98° | +0.99 | +0.99° | +0.91° |
| OLMo-2 13B | +0.99 | +0.85° | +0.81 | +0.86° | +0.89° |
| MiMo-V2.5 (MoE) | +1.00 | +0.97° | +1.00 | +0.94° | +0.91° |
| Gemma-4 31B | +1.00 | +0.99° | +1.00 | +0.99° | +0.99° |
| Kimi-K2 (MoE) | +1.00 | +0.98° | +1.00 | +0.96° | +1.00° |
| OLMo-3 32B | +1.00 | +0.89° | +0.84 | +0.84° | +0.96° |
| OLMo-3 32B, before mid-training | +1.00 | +0.84° | +0.94 | +0.91° | +0.88° |
| GLM-4 32B | +1.00 | +0.99° | +0.99 | +0.93° | +0.95° |
| Falcon-H1 34B | +1.00 | +0.99° | +0.99 | +0.97° | +0.92° |
| DeepSeek-V3 (MoE) | +1.00 | +0.96° | +1.00 | +0.91° | +0.96° |
| MiMo-V2.5-Pro (MoE) | +1.00 | +0.96° | +0.99 | +0.92° | +0.92° |
| DeepSeek-V4-Pro (MoE) | +1.00 | +0.99° | +1.00 | +0.89° | +0.97° |
| Nemotron-3-Ultra (MoE) | +1.00 | +0.99° | +1.00 | +1.00° | +0.98° |
| Llama-3.1 70B | +1.00 | +0.99° | +1.00 | +0.99° | +0.95° |
| Qwen2.5 72B | +1.00 | +0.99° | +1.00 | +0.99° | +0.95° |
| Llama-3.1 405B | +1.00 | +0.98° | +1.00 | +0.99° | +0.96° |
| **kotodama-tsumugi, by training tokens** | | | | | |
| kotodama-tsumugi · 5B tokens | +0.69 | +0.48° | +0.57 | +0.38° | +0.21° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.81 | +0.50° | +0.60 | +0.45° | +0.11° |
| kotodama-tsumugi · 11B tokens | +0.90 | +0.51° | +0.64 | +0.40° | +0.30° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.91 | +0.51° | +0.65 | +0.45° | +0.32° |
| kotodama-tsumugi · 33B tokens | +0.94 | +0.59° | +0.71 | +0.51° | +0.65° |
| kotodama-tsumugi · 55B tokens | +0.98 | +0.65° | +0.68 | +0.59° | +0.55° |
| kotodama-tsumugi · 88B tokens | +0.99 | +0.67° | +0.54 | +0.64° | +0.65° |

### Who has what after things move: told as a story

| model | Hand-offs in a story | Sets after hand-offs | Refilled box | Many refills |
|---|---|---|---|---|
| **Human** | | | | |
| Human reader (gauntlet) | +1.00 | +1.00 | — | — |
| **under 1B** | | | | |
| Pythia 160M | +0.06 | +0.11 | +0.03° | -0.01° |
| SmolLM2 360M | +0.29 | +0.22 | +0.28° | +0.10° |
| Qwen2.5 0.5B | +0.19 | +0.28 | +0.15° | +0.14° |
| Falcon-H1 0.5B | +0.41 | +0.37 | +0.38° | +0.14° |
| Qwen3 0.6B | +0.33 | +0.55 | +0.22° | +0.13° |
| **1–3B** | | | | |
| Pythia 1B | +0.29 | +0.39 | +0.22° | +0.13° |
| Llama-3.2 1B | +0.29 | +0.57 | +0.21° | +0.11° |
| OLMo 1B | +0.31 | +0.38 | +0.34° | +0.16° |
| kotodama-tsumugi 1.4B | +0.40 | +0.55 | +0.22° | +0.14° |
| OLMo-2 1B | +0.45 | +0.38 | +0.42° | +0.24° |
| Pythia 1.4B | +0.18 | +0.46 | +0.18° | +0.05° |
| Qwen2.5 1.5B | +0.49 | +0.56 | +0.49° | +0.25° |
| Falcon-H1 1.5B | +0.59 | +0.64 | +0.48° | +0.14° |
| Falcon-H1 1.5B-Deep | +0.69 | +0.63 | +0.53° | +0.28° |
| GPT-2 XL | +0.14 | +0.38 | +0.27° | -0.01° |
| SmolLM2 1.7B | +0.53 | +0.34 | +0.26° | +0.07° |
| Qwen3 1.7B | +0.64 | +0.69 | +0.50° | +0.20° |
| Gemma 2B | +0.45 | +0.61 | +0.30° | +0.31° |
| Gemma-2 2B | +0.59 | +0.62 | +0.44° | +0.36° |
| Pythia 2.8B | +0.31 | +0.39 | +0.27° | +0.03° |
| Pythia 2.8B (deduped) | +0.34 | +0.41 | +0.27° | +0.03° |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.65 | +0.67 | +0.27° | +0.13° |
| Falcon-H1 3B | +0.78 | +0.80 | +0.73° | +0.34° |
| Falcon3 3B | +0.65 | +0.47 | +0.42° | +0.19° |
| Qwen3 30B-A3B (MoE) | +0.87 | +0.80 | +0.51° | +0.33° |
| Qwen3 4B | +0.81 | +0.80 | +0.49° | +0.38° |
| Pythia 6.9B | +0.32 | +0.51 | +0.27° | +0.07° |
| Falcon-Mamba 7B | +0.54 | +0.25 | +0.40° | +0.11° |
| OLMo-2 7B | +0.70 | +0.67 | +0.53° | +0.25° |
| Falcon3 7B | +0.84 | +0.65 | +0.52° | +0.31° |
| Falcon-H1 7B | +0.89 | +0.73 | +0.61° | +0.54° |
| Qwen2.5 7B | +0.79 | +0.86 | +0.57° | +0.44° |
| Llama-3 8B | +0.77 | +0.79 | +0.38° | +0.29° |
| Llama-3.1 8B | +0.80 | +0.82 | +0.40° | +0.29° |
| Nemotron-H 8B | +0.69 | +0.74 | +0.55° | +0.29° |
| Gemma 7B | +0.80 | +0.77 | +0.45° | +0.26° |
| Gemma-2 9B | +0.88 | +0.84 | +0.34° | +0.36° |
| RecurrentGemma 9B | +0.27 | +0.10 | -0.00° | +0.10° |
| Bamba 9B | +0.20 | +0.10 | +0.32° | +0.22° |
| **10B and up** | | | | |
| DeepSeek-V4-Flash (MoE) | +0.98 | +0.98 | +0.73° | +0.69° |
| Trinity-Large (MoE) | +0.93 | +0.88 | +0.56° | +0.57° |
| Trinity-Large (MoE), before the anneals | +0.91 | +0.77 | +0.51° | +0.40° |
| OLMo-2 13B | +0.85 | +0.82 | +0.71° | +0.48° |
| MiMo-V2.5 (MoE) | +0.97 | +0.92 | +0.69° | +0.59° |
| Gemma-4 31B | +0.97 | +0.97 | +0.61° | +0.69° |
| Kimi-K2 (MoE) | +0.99 | +0.98 | +0.78° | +0.85° |
| OLMo-3 32B | +0.87 | +0.84 | +0.65° | +0.46° |
| OLMo-3 32B, before mid-training | +0.88 | +0.83 | +0.69° | +0.47° |
| GLM-4 32B | +0.96 | +0.90 | +0.64° | +0.53° |
| Falcon-H1 34B | +0.95 | +0.87 | +0.80° | +0.66° |
| DeepSeek-V3 (MoE) | +0.99 | +0.98 | +0.81° | +0.74° |
| MiMo-V2.5-Pro (MoE) | +0.98 | +0.95 | +0.59° | +0.57° |
| DeepSeek-V4-Pro (MoE) | +0.99 | +0.98 | +0.81° | +0.74° |
| Nemotron-3-Ultra (MoE) | +0.99 | +1.00 | +0.78° | +0.69° |
| Llama-3.1 70B | +0.97 | +0.97 | +0.54° | +0.58° |
| Qwen2.5 72B | +0.98 | +0.94 | +0.62° | +0.60° |
| Llama-3.1 405B | +1.00 | +0.99 | +0.74° | +0.75° |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.26 | +0.30 | +0.08° | -0.00° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.28 | +0.41 | +0.08° | -0.04° |
| kotodama-tsumugi · 11B tokens | +0.28 | +0.46 | -0.02° | -0.01° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.33 | +0.45 | -0.00° | +0.03° |
| kotodama-tsumugi · 33B tokens | +0.28 | +0.50 | +0.08° | +0.01° |
| kotodama-tsumugi · 55B tokens | +0.36 | +0.49 | +0.22° | +0.09° |
| kotodama-tsumugi · 88B tokens | +0.38 | +0.55 | +0.29° | +0.22° |

### Who has what after things move: written as code

| model | Variable chains | Shared lists |
|---|---|---|
| **Human** | | |
| Human reader (gauntlet) | +0.73 | +1.00 |
| **under 1B** | | |
| Pythia 160M | +0.01 | -0.01 |
| SmolLM2 360M | +0.05 | -0.02 |
| Qwen2.5 0.5B | +0.06 | +0.16 |
| Falcon-H1 0.5B | +0.08 | +0.13 |
| Qwen3 0.6B | +0.27 | +0.10 |
| **1–3B** | | |
| Pythia 1B | +0.00 | +0.07 |
| Llama-3.2 1B | +0.12 | +0.13 |
| OLMo 1B | +0.01 | +0.08 |
| kotodama-tsumugi 1.4B | +0.04 | +0.15 |
| OLMo-2 1B | +0.09 | +0.06 |
| Pythia 1.4B | -0.03 | -0.01 |
| Qwen2.5 1.5B | +0.22 | +0.14 |
| Falcon-H1 1.5B | +0.34 | +0.38 |
| Falcon-H1 1.5B-Deep | +0.47 | +0.41 |
| GPT-2 XL | -0.06↓ | +0.01 |
| SmolLM2 1.7B | +0.19 | +0.16 |
| Qwen3 1.7B | +0.36 | +0.38 |
| Gemma 2B | +0.20 | +0.27 |
| Gemma-2 2B | +0.20 | +0.36 |
| Pythia 2.8B | +0.02 | +0.11 |
| Pythia 2.8B (deduped) | -0.00 | +0.11 |
| **3–10B** | | |
| Qwen2.5 3B | +0.27 | +0.30 |
| Falcon-H1 3B | +0.56 | +0.44 |
| Falcon3 3B | +0.42 | +0.47 |
| Qwen3 30B-A3B (MoE) | +0.77 | +0.74 |
| Qwen3 4B | +0.50 | +0.62 |
| Pythia 6.9B | +0.06 | +0.02 |
| Falcon-Mamba 7B | +0.15 | +0.08 |
| OLMo-2 7B | +0.20 | +0.17 |
| Falcon3 7B | +0.75 | +0.72 |
| Falcon-H1 7B | +0.69 | +0.70 |
| Qwen2.5 7B | +0.53 | +0.59 |
| Llama-3 8B | +0.55 | +0.52 |
| Llama-3.1 8B | +0.56 | +0.51 |
| Nemotron-H 8B | +0.46 | +0.55 |
| Gemma 7B | +0.45 | +0.42 |
| Gemma-2 9B | +0.67 | +0.63 |
| RecurrentGemma 9B | +0.02 | +0.12 |
| Bamba 9B | +0.04 | -0.02 |
| **10B and up** | | |
| DeepSeek-V4-Flash (MoE) | +0.89 | +0.90 |
| Trinity-Large (MoE) | +0.88 | +0.94 |
| Trinity-Large (MoE), before the anneals | +0.85 | +0.82 |
| OLMo-2 13B | +0.35 | +0.44 |
| MiMo-V2.5 (MoE) | +0.88 | +0.73 |
| Gemma-4 31B | +0.91 | +0.89 |
| Kimi-K2 (MoE) | +0.90 | +0.92 |
| OLMo-3 32B | +0.78 | +0.74 |
| OLMo-3 32B, before mid-training | +0.60 | +0.58 |
| GLM-4 32B | +0.85 | +0.65 |
| Falcon-H1 34B | +0.84 | +0.79 |
| DeepSeek-V3 (MoE) | +0.93 | +0.95 |
| MiMo-V2.5-Pro (MoE) | +0.87 | +0.81 |
| DeepSeek-V4-Pro (MoE) | +0.87 | +0.88 |
| Nemotron-3-Ultra (MoE) | +0.87 | +0.98 |
| Llama-3.1 70B | +0.84 | +0.65 |
| Qwen2.5 72B | +0.82 | +0.88 |
| Llama-3.1 405B | +0.92 | +0.97 |
| **kotodama-tsumugi, by training tokens** | | |
| kotodama-tsumugi · 5B tokens | -0.02 | +0.01 |
| kotodama-tsumugi, seed 7 · 5B tokens | -0.05↓ | +0.04 |
| kotodama-tsumugi · 11B tokens | -0.03 | +0.03 |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.01 | +0.06 |
| kotodama-tsumugi · 33B tokens | +0.01 | +0.01 |
| kotodama-tsumugi · 55B tokens | +0.07 | +0.08 |
| kotodama-tsumugi · 88B tokens | +0.13 | +0.15 |

### One box puzzle, four formats

| model | Box puzzle: prose | Box puzzle: script | Box puzzle: code | Box puzzle: chat |
|---|---|---|---|---|
| **under 1B** | | | | |
| Pythia 160M | +0.00 | -0.02 | +0.01 | +0.02 |
| SmolLM2 360M | +0.19 | +0.25 | +0.40 | +0.32 |
| Qwen2.5 0.5B | +0.34 | +0.37 | +0.41 | +0.40 |
| Falcon-H1 0.5B | +0.18 | +0.26 | +0.49 | +0.11 |
| Qwen3 0.6B | +0.21 | +0.25 | +0.47 | +0.23 |
| **1–3B** | | | | |
| Pythia 1B | +0.13 | +0.12 | +0.31 | +0.08 |
| Llama-3.2 1B | +0.38 | +0.47 | +0.55 | +0.46 |
| OLMo 1B | +0.24 | +0.18 | +0.30 | +0.29 |
| kotodama-tsumugi 1.4B | +0.21 | +0.26 | +0.49 | +0.32 |
| OLMo-2 1B | +0.21 | +0.27 | +0.43 | +0.16 |
| Pythia 1.4B | +0.27 | +0.26 | +0.28 | +0.18 |
| Qwen2.5 1.5B | +0.19 | +0.25 | +0.60 | +0.32 |
| Falcon-H1 1.5B | +0.28 | +0.20 | +0.66 | +0.26 |
| Falcon-H1 1.5B-Deep | +0.34 | +0.36 | +0.76 | +0.46 |
| GPT-2 XL | +0.09 | +0.12 | +0.23 | +0.12 |
| SmolLM2 1.7B | +0.31 | +0.38 | +0.51 | -0.02 |
| Qwen3 1.7B | +0.42 | +0.41 | +0.57 | +0.43 |
| Gemma 2B | +0.25 | +0.30 | +0.50 | +0.48 |
| Gemma-2 2B | +0.37 | +0.39 | +0.60 | +0.50 |
| Pythia 2.8B | +0.41 | +0.40 | +0.45 | +0.41 |
| Pythia 2.8B (deduped) | +0.43 | +0.40 | +0.42 | +0.47 |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.58 | +0.56 | +0.53 | +0.61 |
| Falcon-H1 3B | +0.55 | +0.61 | +0.68 | +0.66 |
| Falcon3 3B | +0.44 | +0.49 | +0.62 | +0.60 |
| Qwen3 30B-A3B (MoE) | +0.61 | +0.59 | +0.80 | +0.60 |
| Qwen3 4B | +0.55 | +0.47 | +0.72 | +0.58 |
| Pythia 6.9B | +0.36 | +0.38 | +0.39 | +0.35 |
| Falcon-Mamba 7B | +0.26 | +0.33 | +0.48 | +0.42 |
| OLMo-2 7B | +0.38 | +0.25 | +0.48 | +0.31 |
| Falcon3 7B | +0.51 | +0.55 | +0.75 | +0.59 |
| Falcon-H1 7B | +0.58 | +0.62 | +0.80 | +0.59 |
| Qwen2.5 7B | +0.65 | +0.61 | +0.72 | +0.65 |
| Llama-3 8B | +0.57 | +0.53 | +0.60 | +0.66 |
| Llama-3.1 8B | +0.60 | +0.56 | +0.65 | +0.65 |
| Nemotron-H 8B | +0.47 | +0.44 | +0.62 | +0.47 |
| Gemma 7B | +0.48 | +0.47 | +0.68 | +0.45 |
| Gemma-2 9B | +0.60 | +0.63 | +0.75 | +0.65 |
| RecurrentGemma 9B | +0.10 | +0.24 | +0.26 | +0.27 |
| Bamba 9B | -0.09↓ | -0.06 | -0.01 | +0.01 |
| **10B and up** | | | | |
| DeepSeek-V4-Flash (MoE) | +0.85 | +0.87 | +0.84 | +0.75 |
| Trinity-Large (MoE) | +0.76 | +0.75 | +0.87 | +0.77 |
| Trinity-Large (MoE), before the anneals | +0.60 | +0.65 | +0.87 | +0.66 |
| OLMo-2 13B | +0.55 | +0.38 | +0.60 | +0.47 |
| MiMo-V2.5 (MoE) | +0.83 | +0.87 | +0.89 | +0.83 |
| Gemma-4 31B | +0.80 | +0.80 | +0.88 | +0.90 |
| Kimi-K2 (MoE) | +0.86 | +0.84 | +0.92 | +0.82 |
| OLMo-3 32B | +0.54 | +0.58 | +0.78 | +0.60 |
| OLMo-3 32B, before mid-training | +0.61 | +0.56 | +0.76 | +0.63 |
| GLM-4 32B | +0.75 | +0.76 | +0.86 | +0.78 |
| Falcon-H1 34B | +0.83 | +0.86 | +0.88 | +0.83 |
| DeepSeek-V3 (MoE) | +0.85 | +0.86 | +0.91 | +0.85 |
| MiMo-V2.5-Pro (MoE) | +0.92 | +0.92 | +0.93 | +0.86 |
| DeepSeek-V4-Pro (MoE) | +0.86 | +0.87 | +0.89 | +0.82 |
| Nemotron-3-Ultra (MoE) | +0.92 | +0.93 | +0.92 | +0.89 |
| Llama-3.1 70B | +0.84 | +0.85 | +0.88 | +0.82 |
| Qwen2.5 72B | +0.90 | +0.86 | +0.80 | +0.75 |
| Llama-3.1 405B | +0.86 | +0.89 | +0.83 | +0.90 |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.08 | +0.18 | +0.20 | +0.23 |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.14 | +0.21 | +0.29 | +0.24 |
| kotodama-tsumugi · 11B tokens | +0.34 | +0.43 | +0.35 | +0.35 |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.33 | +0.31 | +0.42 | +0.34 |
| kotodama-tsumugi · 33B tokens | +0.31 | +0.29 | +0.48 | +0.36 |
| kotodama-tsumugi · 55B tokens | +0.29 | +0.34 | +0.51 | +0.43 |
| kotodama-tsumugi · 88B tokens | +0.27 | +0.32 | +0.50 | +0.35 |

### Who are I and you?

| model | 'You', nothing moves | 'I', after hand-offs | 'You', after hand-offs | 'You' among three | Answering as yourself |
|---|---|---|---|---|---|
| **Human** | | | | | |
| Human reader (gauntlet) | — | +1.00 | +1.00 | +1.00 | +1.00 |
| **under 1B** | | | | | |
| Pythia 160M | +0.03° | +0.03 | -0.04 | +0.05 | +0.04 |
| SmolLM2 360M | +0.15° | -0.00 | -0.02 | +0.08 | +0.03 |
| Qwen2.5 0.5B | +0.13° | +0.03 | +0.03 | +0.12 | +0.03 |
| Falcon-H1 0.5B | +0.24° | +0.05 | +0.05 | +0.05 | +0.08 |
| Qwen3 0.6B | +0.38° | +0.04 | +0.09 | +0.02 | +0.04 |
| **1–3B** | | | | | |
| Pythia 1B | +0.11° | -0.01 | -0.03 | +0.06 | +0.08 |
| Llama-3.2 1B | +0.27° | +0.02 | -0.01 | +0.08 | +0.01 |
| OLMo 1B | +0.14° | +0.01 | -0.03 | -0.03 | +0.07 |
| kotodama-tsumugi 1.4B | +0.49° | +0.03 | +0.04 | +0.06 | +0.11 |
| OLMo-2 1B | +0.28° | +0.01 | +0.06 | +0.08 | +0.08 |
| Pythia 1.4B | +0.12° | -0.01 | +0.07 | +0.05 | +0.02 |
| Qwen2.5 1.5B | +0.45° | +0.07 | -0.00 | +0.04 | +0.09 |
| Falcon-H1 1.5B | +0.66° | +0.10 | +0.04 | +0.00 | +0.11 |
| Falcon-H1 1.5B-Deep | +0.61° | +0.12 | +0.04 | +0.05 | +0.26 |
| GPT-2 XL | +0.02° | -0.01 | -0.02 | +0.03 | +0.02 |
| SmolLM2 1.7B | +0.44° | +0.03 | +0.02 | +0.06 | +0.04 |
| Qwen3 1.7B | +0.55° | +0.14 | +0.06 | +0.16 | +0.17 |
| Gemma 2B | +0.41° | +0.08 | -0.02 | +0.08 | +0.06 |
| Gemma-2 2B | +0.62° | +0.12 | +0.06 | +0.07 | +0.16 |
| Pythia 2.8B | +0.13° | -0.01 | +0.14 | +0.03 | +0.04 |
| Pythia 2.8B (deduped) | +0.16° | +0.00 | +0.14 | +0.05 | +0.03 |
| **3–10B** | | | | | |
| Qwen2.5 3B | +0.80° | +0.15 | -0.03 | +0.06 | +0.13 |
| Falcon-H1 3B | +0.88° | +0.16 | -0.01 | +0.06 | +0.25 |
| Falcon3 3B | +0.67° | +0.07 | +0.07 | +0.12 | +0.13 |
| Qwen3 30B-A3B (MoE) | +0.88° | +0.23 | +0.12 | +0.15 | +0.41 |
| Qwen3 4B | +0.66° | +0.23 | +0.08 | +0.02 | +0.42 |
| Pythia 6.9B | +0.16° | -0.02 | -0.01 | +0.03 | +0.02 |
| Falcon-Mamba 7B | +0.67° | +0.10 | +0.00 | +0.02 | +0.11 |
| OLMo-2 7B | +0.80° | +0.14 | +0.07 | +0.16 | +0.25 |
| Falcon3 7B | +0.82° | +0.12 | -0.06 | +0.15 | +0.35 |
| Falcon-H1 7B | +0.94° | +0.21 | +0.09 | +0.18 | +0.48 |
| Qwen2.5 7B | +0.70° | +0.29 | +0.14 | +0.22 | +0.35 |
| Llama-3 8B | +0.89° | +0.18 | +0.04 | +0.18 | +0.31 |
| Llama-3.1 8B | +0.89° | +0.18 | -0.00 | +0.21 | +0.34 |
| Nemotron-H 8B | +0.92° | +0.23 | -0.01 | +0.20 | +0.30 |
| Gemma 7B | +0.91° | +0.26 | +0.02 | +0.12 | +0.26 |
| Gemma-2 9B | +0.83° | +0.35 | +0.22 | +0.35 | +0.44 |
| RecurrentGemma 9B | +0.33° | +0.01 | -0.01 | +0.12 | +0.03 |
| Bamba 9B | +0.10° | +0.04 | -0.07 | +0.02 | +0.00 |
| **10B and up** | | | | | |
| DeepSeek-V4-Flash (MoE) | +0.99° | +0.74 | +0.30 | +0.48 | +0.67 |
| Trinity-Large (MoE) | +0.99° | +0.34 | +0.27 | +0.37 | +0.67 |
| Trinity-Large (MoE), before the anneals | +0.93° | +0.33 | +0.18 | +0.28 | +0.52 |
| OLMo-2 13B | +0.96° | +0.33 | +0.36 | +0.42 | +0.33 |
| MiMo-V2.5 (MoE) | +0.93° | +0.41 | +0.16 | +0.23 | +0.64 |
| Gemma-4 31B | +0.99° | +0.61 | +0.31 | +0.53 | +0.72 |
| Kimi-K2 (MoE) | +1.00° | +0.60 | +0.22 | +0.44 | +0.72 |
| OLMo-3 32B | +0.97° | +0.32 | +0.16 | +0.42 | +0.52 |
| OLMo-3 32B, before mid-training | +0.91° | +0.22 | +0.16 | +0.19 | +0.40 |
| GLM-4 32B | +0.93° | +0.39 | +0.09 | +0.28 | +0.57 |
| Falcon-H1 34B | +0.92° | +0.49 | +0.35 | +0.41 | +0.65 |
| DeepSeek-V3 (MoE) | +1.00° | +0.66 | +0.30 | +0.63 | +0.71 |
| MiMo-V2.5-Pro (MoE) | +0.98° | +0.49 | +0.34 | +0.55 | +0.70 |
| DeepSeek-V4-Pro (MoE) | +0.99° | +0.70 | +0.26 | +0.37 | +0.71 |
| Nemotron-3-Ultra (MoE) | +0.99° | +0.72 | +0.30 | +0.47 | +0.72 |
| Llama-3.1 70B | +0.99° | +0.39 | +0.13 | +0.45 | +0.68 |
| Qwen2.5 72B | +0.90° | +0.57 | +0.31 | +0.61 | +0.67 |
| Llama-3.1 405B | +0.98° | +0.66 | +0.41 | +0.49 | +0.73 |
| **kotodama-tsumugi, by training tokens** | | | | | |
| kotodama-tsumugi · 5B tokens | +0.08° | -0.03 | +0.01 | +0.01 | +0.04 |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.04° | -0.06 | -0.04 | +0.05 | +0.09 |
| kotodama-tsumugi · 11B tokens | +0.05° | +0.03 | -0.02 | +0.02 | +0.05 |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.08° | -0.01 | +0.01 | +0.15 | +0.04 |
| kotodama-tsumugi · 33B tokens | +0.33° | -0.03 | +0.10 | +0.03 | +0.07 |
| kotodama-tsumugi · 55B tokens | +0.33° | +0.01 | +0.02 | +0.04 | +0.07 |
| kotodama-tsumugi · 88B tokens | +0.40° | +0.01 | +0.11 | +0.00 | +0.11 |

### Learning a rule from the examples

| model | Made-up grammar | Made-up operators | Pattern completion | Known facts (control) |
|---|---|---|---|---|
| **under 1B** | | | | |
| Pythia 160M | +0.37° | +0.11° | +0.20° | +0.44° |
| SmolLM2 360M | +0.74° | +0.27° | +0.39° | +0.97° |
| Qwen2.5 0.5B | +0.70° | +0.23° | +0.44° | +0.92° |
| Falcon-H1 0.5B | +0.69° | +0.23° | +0.37° | +0.89° |
| Qwen3 0.6B | +0.71° | +0.25° | +0.49° | +0.95° |
| **1–3B** | | | | |
| Pythia 1B | +0.67° | +0.22° | +0.25° | +0.90° |
| Llama-3.2 1B | +0.78° | +0.22° | +0.40° | +1.00° |
| OLMo 1B | +0.77° | +0.28° | +0.45° | +0.99° |
| kotodama-tsumugi 1.4B | +0.79° | +0.30° | +0.55° | +0.95° |
| OLMo-2 1B | +0.70° | +0.26° | +0.40° | +0.99° |
| Pythia 1.4B | +0.69° | +0.24° | +0.32° | +0.96° |
| Qwen2.5 1.5B | +0.76° | +0.20° | +0.46° | +1.00° |
| Falcon-H1 1.5B | +0.73° | +0.23° | +0.42° | +0.97° |
| Falcon-H1 1.5B-Deep | +0.80° | +0.28° | +0.50° | +0.99° |
| GPT-2 XL | +0.70° | +0.23° | +0.42° | +0.92° |
| SmolLM2 1.7B | +0.76° | +0.24° | +0.39° | +1.00° |
| Qwen3 1.7B | +0.81° | +0.26° | +0.56° | +0.99° |
| Gemma 2B | +0.72° | +0.22° | +0.52° | +1.00° |
| Gemma-2 2B | +0.80° | +0.25° | +0.51° | +1.00° |
| Pythia 2.8B | +0.79° | +0.24° | +0.36° | +0.98° |
| Pythia 2.8B (deduped) | +0.79° | +0.26° | +0.38° | +0.99° |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.80° | +0.17° | +0.69° | +1.00° |
| Falcon-H1 3B | +0.83° | +0.26° | +0.54° | +0.99° |
| Falcon3 3B | +0.85° | +0.27° | +0.51° | +0.97° |
| Qwen3 30B-A3B (MoE) | +0.85° | +0.20° | +0.93° | +1.00° |
| Qwen3 4B | +0.84° | +0.24° | +0.75° | +1.00° |
| Pythia 6.9B | +0.77° | +0.20° | +0.40° | +1.00° |
| Falcon-Mamba 7B | +0.82° | +0.29° | +0.54° | +1.00° |
| OLMo-2 7B | +0.86° | +0.28° | +0.58° | +1.00° |
| Falcon3 7B | +0.83° | +0.24° | +0.59° | +1.00° |
| Falcon-H1 7B | +0.86° | +0.27° | +0.72° | +1.00° |
| Qwen2.5 7B | +0.82° | +0.20° | +0.73° | +1.00° |
| Llama-3 8B | +0.84° | +0.25° | +0.80° | +1.00° |
| Llama-3.1 8B | +0.84° | +0.26° | +0.76° | +1.00° |
| Nemotron-H 8B | +0.87° | +0.24° | +0.53° | +1.00° |
| Gemma 7B | +0.84° | +0.24° | +0.67° | +1.00° |
| Gemma-2 9B | +0.86° | +0.31° | +0.66° | +1.00° |
| RecurrentGemma 9B | +0.64° | +0.14° | +0.07° | +1.00° |
| Bamba 9B | +0.59° | +0.13° | +0.24° | +1.00° |
| **10B and up** | | | | |
| DeepSeek-V4-Flash (MoE) | +0.88° | +0.24° | +0.93° | +1.00° |
| Trinity-Large (MoE) | +0.81° | +0.18° | +0.81° | +1.00° |
| Trinity-Large (MoE), before the anneals | +0.85° | +0.17° | +0.70° | +1.00° |
| OLMo-2 13B | +0.83° | +0.24° | +0.61° | +1.00° |
| MiMo-V2.5 (MoE) | +0.88° | +0.24° | +0.91° | +1.00° |
| Gemma-4 31B | +0.92° | +0.31° | +0.92° | +1.00° |
| Kimi-K2 (MoE) | +0.87° | +0.21° | +0.88° | +1.00° |
| OLMo-3 32B | +0.83° | +0.20° | +0.67° | +1.00° |
| OLMo-3 32B, before mid-training | +0.83° | +0.27° | +0.74° | +1.00° |
| GLM-4 32B | +0.84° | +0.23° | +0.83° | +1.00° |
| Falcon-H1 34B | +0.87° | +0.25° | +0.78° | +1.00° |
| DeepSeek-V3 (MoE) | +0.86° | +0.31° | +0.97° | +1.00° |
| MiMo-V2.5-Pro (MoE) | +0.91° | +0.25° | +0.92° | +1.00° |
| DeepSeek-V4-Pro (MoE) | +0.88° | +0.22° | +0.94° | +1.00° |
| Nemotron-3-Ultra (MoE) | +0.87° | +0.23° | +0.88° | +1.00° |
| Llama-3.1 70B | +0.85° | +0.31° | +0.91° | +1.00° |
| Qwen2.5 72B | +0.87° | +0.24° | +0.92° | +1.00° |
| Llama-3.1 405B | +0.90° | +0.23° | +0.99° | +1.00° |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.65° | +0.26° | +0.46° | +0.64° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.64° | +0.23° | +0.43° | +0.65° |
| kotodama-tsumugi · 11B tokens | +0.74° | +0.26° | +0.40° | +0.71° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.77° | +0.22° | +0.43° | +0.76° |
| kotodama-tsumugi · 33B tokens | +0.78° | +0.29° | +0.47° | +0.89° |
| kotodama-tsumugi · 55B tokens | +0.76° | +0.28° | +0.44° | +0.89° |
| kotodama-tsumugi · 88B tokens | +0.81° | +0.30° | +0.51° | +0.93° |

## Rows (CSV)

```csv
row_id,label,group,params_b,active_b,tokens,scoring
human,"Human reader (gauntlet)","Human",,,,generation
pythia-160m,"Pythia 160M","under 1B",0.213,0.213,300000000000.0,logprob
smollm2-360m,"SmolLM2 360M","under 1B",0.362,0.362,4000000000000.0,logprob
qwen2.5-0.5b,"Qwen2.5 0.5B","under 1B",0.494,0.494,18000000000000.0,logprob
falcon-h1-0.5b,"Falcon-H1 0.5B","under 1B",0.521,0.521,2500000000000.0,logprob
qwen3-0.6b,"Qwen3 0.6B","under 1B",0.596,0.596,36000000000000.0,logprob
pythia-1b,"Pythia 1B","1–3B",1.079,1.079,300000000000.0,logprob
llama3.2-1b,"Llama-3.2 1B","1–3B",1.236,1.236,9000000000000.0,logprob
olmo1-1b,"OLMo 1B","1–3B",1.28,1.28,3050000000000.0,logprob
tsumugi,"kotodama-tsumugi 1.4B","1–3B",1.425,1.425,110000000000.0,logprob
olmo2-1b,"OLMo-2 1B","1–3B",1.485,1.485,4050000000000.0,logprob
pythia-1.4b,"Pythia 1.4B","1–3B",1.515,1.515,300000000000.0,logprob
qwen2.5-1.5b,"Qwen2.5 1.5B","1–3B",1.544,1.544,18000000000000.0,logprob
falcon-h1-1.5b,"Falcon-H1 1.5B","1–3B",1.555,1.555,3000000000000.0,logprob
falcon-h1-1.5b-deep,"Falcon-H1 1.5B-Deep","1–3B",1.555,1.555,3000000000000.0,logprob
gpt2-xl,"GPT-2 XL","1–3B",1.608,1.608,,logprob
smollm2-1.7b,"SmolLM2 1.7B","1–3B",1.711,1.711,11000000000000.0,logprob
qwen3-1.7b,"Qwen3 1.7B","1–3B",1.721,1.721,36000000000000.0,logprob
gemma-2b,"Gemma 2B","1–3B",2.506,2.506,3000000000000.0,logprob
gemma2-2b,"Gemma-2 2B","1–3B",2.614,2.614,2000000000000.0,logprob
pythia-2.8b,"Pythia 2.8B","1–3B",2.909,2.909,300000000000.0,logprob
pythia-2.8b-deduped,"Pythia 2.8B (deduped)","1–3B",2.909,2.909,300000000000.0,logprob
qwen2.5-3b,"Qwen2.5 3B","3–10B",3.086,3.086,18000000000000.0,logprob
falcon-h1-3b,"Falcon-H1 3B","3–10B",3.149,3.149,2500000000000.0,logprob
falcon3-3b,"Falcon3 3B","3–10B",3.228,3.228,100000000000.0,logprob
qwen3-30b-a3b,"Qwen3 30B-A3B (MoE)","3–10B",30.532,3.3,36000000000000.0,logprob
qwen3-4b,"Qwen3 4B","3–10B",4.022,4.022,36000000000000.0,logprob
pythia-6.9b,"Pythia 6.9B","3–10B",6.992,6.992,300000000000.0,logprob
falcon-mamba-7b,"Falcon-Mamba 7B","3–10B",7.273,7.273,5800000000000.0,logprob
olmo2-7b,"OLMo-2 7B","3–10B",7.299,7.299,4150000000000.0,logprob
falcon3-7b,"Falcon3 7B","3–10B",7.456,7.456,14000000000000.0,logprob
falcon-h1-7b,"Falcon-H1 7B","3–10B",7.586,7.586,12000000000000.0,logprob
qwen2.5-7b,"Qwen2.5 7B","3–10B",7.616,7.616,18000000000000.0,logprob
llama3-8b,"Llama-3 8B","3–10B",8.03,8.03,15000000000000.0,logprob
llama3.1-8b,"Llama-3.1 8B","3–10B",8.03,8.03,15000000000000.0,logprob
nemotron-h-8b,"Nemotron-H 8B","3–10B",8.101,8.101,15000000000000.0,logprob
gemma-7b,"Gemma 7B","3–10B",8.538,8.538,6000000000000.0,logprob
gemma2-9b,"Gemma-2 9B","3–10B",9.242,9.242,8000000000000.0,logprob
recurrentgemma-9b,"RecurrentGemma 9B","3–10B",9.629,9.629,2000000000000.0,logprob
bamba-9b,"Bamba 9B","3–10B",9.78,9.78,3100000000000.0,logprob
dsv4-flash-base,"DeepSeek-V4-Flash (MoE)","10B and up",284.0,13.0,32000000000000.0,logprob
trinity-base,"Trinity-Large (MoE)","10B and up",398.0,13.0,17000000000000.0,logprob
trinity-truebase,"Trinity-Large (MoE), before the anneals","10B and up",398.0,13.0,10000000000000.0,logprob
olmo2-13b,"OLMo-2 13B","10B and up",13.716,13.716,5600000000000.0,logprob
mimo-v25-base,"MiMo-V2.5 (MoE)","10B and up",310.0,15.0,48000000000000.0,logprob
gemma4-31b-base,"Gemma-4 31B","10B and up",30.7,30.7,,logprob
kimi-k2-base,"Kimi-K2 (MoE)","10B and up",1000.0,32.0,15500000000000.0,logprob
olmo3-32b,"OLMo-3 32B","10B and up",32.234,32.234,5700000000000.0,logprob
olmo3-32b-stage1,"OLMo-3 32B, before mid-training","10B and up",32.234,32.234,5500000000000.0,logprob
glm4-32b-base,"GLM-4 32B","10B and up",32.566,32.566,15000000000000.0,logprob
falcon-h1-34b,"Falcon-H1 34B","10B and up",33.643,33.643,18000000000000.0,logprob
dsv3-base,"DeepSeek-V3 (MoE)","10B and up",671.0,37.0,14800000000000.0,logprob
mimo-v25-pro-base,"MiMo-V2.5-Pro (MoE)","10B and up",1020.0,42.0,27000000000000.0,logprob
dsv4-pro-base,"DeepSeek-V4-Pro (MoE)","10B and up",1600.0,49.0,32000000000000.0,logprob
nemotron3-ultra-550b-base,"Nemotron-3-Ultra (MoE)","10B and up",550.0,55.0,20000000000000.0,logprob
llama3.1-70b,"Llama-3.1 70B","10B and up",70.554,70.554,15000000000000.0,logprob
qwen2.5-72b,"Qwen2.5 72B","10B and up",72.706,72.706,18000000000000.0,logprob
llama3.1-405b,"Llama-3.1 405B","10B and up",405.853,405.853,15600000000000.0,logprob
tsumugi-s17-a5B,"kotodama-tsumugi · 5B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,5000000000.0,logprob
tsumugi-s7-a5B,"kotodama-tsumugi, seed 7 · 5B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,5000000000.0,logprob
tsumugi-s17-a11B,"kotodama-tsumugi · 11B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,11000000000.0,logprob
tsumugi-s7-a11B,"kotodama-tsumugi, seed 7 · 11B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,11000000000.0,logprob
tsumugi-s17-a33B,"kotodama-tsumugi · 33B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,33000000000.0,logprob
tsumugi-s17-a55B,"kotodama-tsumugi · 55B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,55000000000.0,logprob
tsumugi-s17-a88B,"kotodama-tsumugi · 88B tokens","kotodama-tsumugi, by training tokens",1.425,1.425,88000000000.0,logprob
```

## All cells (CSV)

`row_id,col_id,condition,acc,n,chance,score,lo,hi,state` — `acc` and `chance` are empty for ladder readings.

```csv
row_id,col_id,condition,acc,n,chance,score,lo,hi,state
human,lookup,0shot,1.000,5,0.185,1.000,,,
human,pairs,0shot,1.000,1,0.125,1.000,,,
human,handoffs,0shot,1.000,6,0.319,1.000,,,
human,handoffs,3shot_collide,1.000,1,0.250,1.000,,,
human,handoffs,3shot,1.000,1,0.250,1.000,,,
human,sets,0shot,1.000,1,0.036,1.000,,,
human,codevar,0shot,0.800,5,0.257,0.731,,,
human,sharedlists,0shot,1.000,1,0.200,1.000,,,
human,i-moves,0shot,1.000,3,0.306,1.000,,,
human,i-moves,0shot,1.000,1,0.250,1.000,,,
human,you-moves,0shot,1.000,4,0.312,1.000,,,
human,you-moves,0shot,1.000,2,0.292,1.000,,,
human,you-named,0shot,1.000,3,0.306,1.000,,,
human,you-named,0shot,1.000,1,0.250,1.000,,,
human,answer-self,0shot,1.000,1,0.250,1.000,,,
pythia-160m,lookup,0shot,0.220,1210,0.169,0.065,0.051,0.080,above
pythia-160m,lookup,fair,0.249,1210,0.169,0.100,0.079,0.121,above
pythia-160m,shortlist,0shot,0.250,3840,0.040,0.219,0.208,0.230,above
pythia-160m,pairs,0shot,0.194,256,0.125,0.079,0.050,0.109,above
pythia-160m,pairs,fair,0.243,256,0.125,0.135,0.092,0.174,above
pythia-160m,noise,0shot,0.378,720,0.200,0.222,0.190,0.251,above
pythia-160m,now,0shot,0.334,300,0.292,0.058,0.010,0.111,above
pythia-160m,handoffs,0shot,0.336,1200,0.312,0.034,0.013,0.055,above
pythia-160m,handoffs,fair,0.352,1200,0.312,0.057,0.030,0.086,above
pythia-160m,sets,0shot,0.354,1280,0.275,0.115,0.095,0.134,above
pythia-160m,sets,fair,0.353,1280,0.275,0.111,0.086,0.137,above
pythia-160m,refilled,0shot,0.226,720,0.200,0.033,0.010,0.058,above
pythia-160m,refills,0shot,0.189,600,0.200,-0.014,-0.037,0.013,at
pythia-160m,codevar,0shot,0.288,1200,0.267,0.030,0.007,0.051,above
pythia-160m,codevar,fair,0.278,1200,0.267,0.014,-0.009,0.039,at
pythia-160m,sharedlists,0shot,0.238,600,0.200,0.047,0.016,0.081,above
pythia-160m,sharedlists,fair,0.190,600,0.200,-0.013,-0.040,0.016,at
pythia-160m,box-narrative,0shot,0.223,200,0.253,-0.051,-0.100,-0.001,below
pythia-160m,box-narrative,fair,0.250,200,0.253,0.000,-0.049,0.056,at
pythia-160m,box-screenplay,0shot,0.253,200,0.253,-0.006,-0.057,0.041,at
pythia-160m,box-screenplay,fair,0.242,200,0.253,-0.021,-0.081,0.039,at
pythia-160m,box-code,0shot,0.284,200,0.253,0.060,-0.000,0.120,at
pythia-160m,box-code,fair,0.261,200,0.253,0.012,-0.052,0.077,at
pythia-160m,box-chat,0shot,0.283,200,0.253,0.039,-0.009,0.085,at
pythia-160m,box-chat,fair,0.267,200,0.253,0.023,-0.035,0.083,at
pythia-160m,you-static,0shot,0.317,300,0.292,0.034,-0.014,0.088,at
pythia-160m,i-moves,0shot,0.293,180,0.287,0.004,-0.044,0.061,at
pythia-160m,i-moves,fair,0.311,180,0.287,0.030,-0.044,0.110,at
pythia-160m,you-moves,0shot,0.292,180,0.287,0.002,-0.050,0.060,at
pythia-160m,you-moves,fair,0.263,180,0.287,-0.038,-0.103,0.034,at
pythia-160m,you-named,0shot,0.336,180,0.287,0.068,-0.002,0.135,at
pythia-160m,you-named,fair,0.320,180,0.287,0.048,-0.024,0.118,at
pythia-160m,answer-self,0shot,0.266,2560,0.250,0.021,0.007,0.035,above
pythia-160m,answer-self,fair,0.280,2560,0.250,0.039,0.024,0.056,above
pythia-160m,grammar,ladder,,1080,,0.373,0.356,0.397,above
pythia-160m,operators,ladder,,1080,,0.111,0.089,0.137,above
pythia-160m,patterns,ladder,,1080,,0.196,0.168,0.221,above
pythia-160m,facts,ladder,,768,,0.438,0.403,0.468,above
smollm2-360m,lookup,0shot,0.458,1210,0.169,0.342,0.310,0.375,above
smollm2-360m,lookup,fair,0.461,1210,0.169,0.353,0.320,0.385,above
smollm2-360m,shortlist,0shot,0.588,3840,0.040,0.570,0.555,0.586,above
smollm2-360m,pairs,0shot,0.408,256,0.125,0.324,0.255,0.391,above
smollm2-360m,pairs,fair,0.377,256,0.125,0.288,0.216,0.353,above
smollm2-360m,noise,0shot,0.482,720,0.200,0.352,0.310,0.394,above
smollm2-360m,now,0shot,0.548,300,0.292,0.363,0.295,0.438,above
smollm2-360m,handoffs,0shot,0.572,1200,0.312,0.373,0.333,0.410,above
smollm2-360m,handoffs,fair,0.519,1200,0.312,0.295,0.253,0.335,above
smollm2-360m,sets,0shot,0.425,1280,0.275,0.210,0.171,0.247,above
smollm2-360m,sets,fair,0.428,1280,0.275,0.215,0.181,0.252,above
smollm2-360m,refilled,0shot,0.424,720,0.200,0.280,0.236,0.328,above
smollm2-360m,refills,0shot,0.278,600,0.200,0.097,0.054,0.141,above
smollm2-360m,codevar,0shot,0.316,1200,0.267,0.058,0.022,0.097,above
smollm2-360m,codevar,fair,0.311,1200,0.267,0.052,0.016,0.090,above
smollm2-360m,sharedlists,0shot,0.210,600,0.200,0.013,-0.026,0.054,at
smollm2-360m,sharedlists,fair,0.184,600,0.200,-0.020,-0.057,0.019,at
smollm2-360m,box-narrative,0shot,0.375,200,0.253,0.148,0.049,0.235,above
smollm2-360m,box-narrative,fair,0.388,200,0.253,0.188,0.096,0.281,above
smollm2-360m,box-screenplay,0shot,0.472,200,0.253,0.304,0.205,0.394,above
smollm2-360m,box-screenplay,fair,0.438,200,0.253,0.250,0.147,0.348,above
smollm2-360m,box-code,0shot,0.343,200,0.253,0.120,0.025,0.210,above
smollm2-360m,box-code,fair,0.560,200,0.253,0.396,0.299,0.489,above
smollm2-360m,box-chat,0shot,0.500,200,0.253,0.339,0.244,0.427,above
smollm2-360m,box-chat,fair,0.490,200,0.253,0.321,0.225,0.414,above
smollm2-360m,you-static,0shot,0.399,300,0.292,0.147,0.070,0.220,above
smollm2-360m,i-moves,0shot,0.267,180,0.287,-0.034,-0.118,0.059,at
smollm2-360m,i-moves,fair,0.289,180,0.287,-0.001,-0.089,0.086,at
smollm2-360m,you-moves,0shot,0.271,180,0.287,-0.026,-0.118,0.066,at
smollm2-360m,you-moves,fair,0.275,180,0.287,-0.017,-0.106,0.073,at
smollm2-360m,you-named,0shot,0.228,180,0.287,-0.080,-0.162,0.005,at
smollm2-360m,you-named,fair,0.347,180,0.287,0.084,-0.013,0.184,at
smollm2-360m,answer-self,0shot,0.305,2560,0.250,0.074,0.051,0.098,above
smollm2-360m,answer-self,fair,0.271,2560,0.250,0.028,0.006,0.050,above
smollm2-360m,grammar,ladder,,1080,,0.740,0.716,0.764,above
smollm2-360m,operators,ladder,,1080,,0.267,0.242,0.301,above
smollm2-360m,patterns,ladder,,1080,,0.394,0.357,0.431,above
smollm2-360m,facts,ladder,,768,,0.971,0.956,0.984,above
qwen2.5-0.5b,lookup,0shot,0.400,1210,0.169,0.270,0.238,0.304,above
qwen2.5-0.5b,lookup,fair,0.812,1210,0.169,0.776,0.752,0.801,above
qwen2.5-0.5b,shortlist,0shot,0.509,3840,0.040,0.489,0.473,0.506,above
qwen2.5-0.5b,pairs,0shot,0.502,256,0.125,0.431,0.364,0.502,above
qwen2.5-0.5b,pairs,fair,0.430,256,0.125,0.348,0.277,0.415,above
qwen2.5-0.5b,noise,0shot,0.683,720,0.200,0.604,0.560,0.643,above
qwen2.5-0.5b,now,0shot,0.635,300,0.292,0.479,0.407,0.555,above
qwen2.5-0.5b,handoffs,0shot,0.429,1200,0.312,0.168,0.125,0.209,above
qwen2.5-0.5b,handoffs,fair,0.445,1200,0.312,0.192,0.152,0.232,above
qwen2.5-0.5b,sets,0shot,0.428,1280,0.275,0.221,0.183,0.257,above
qwen2.5-0.5b,sets,fair,0.470,1280,0.275,0.282,0.246,0.318,above
qwen2.5-0.5b,refilled,0shot,0.319,720,0.200,0.149,0.110,0.192,above
qwen2.5-0.5b,refills,0shot,0.311,600,0.200,0.139,0.093,0.188,above
qwen2.5-0.5b,codevar,0shot,0.329,1200,0.267,0.083,0.047,0.119,above
qwen2.5-0.5b,codevar,fair,0.311,1200,0.267,0.057,0.022,0.093,above
qwen2.5-0.5b,sharedlists,0shot,0.227,600,0.200,0.033,-0.009,0.076,at
qwen2.5-0.5b,sharedlists,fair,0.332,600,0.200,0.165,0.120,0.210,above
qwen2.5-0.5b,box-narrative,0shot,0.448,200,0.253,0.226,0.125,0.325,above
qwen2.5-0.5b,box-narrative,fair,0.522,200,0.253,0.345,0.249,0.437,above
qwen2.5-0.5b,box-screenplay,0shot,0.545,200,0.253,0.373,0.275,0.473,above
qwen2.5-0.5b,box-screenplay,fair,0.549,200,0.253,0.369,0.277,0.465,above
qwen2.5-0.5b,box-code,0shot,0.362,200,0.253,0.164,0.074,0.255,above
qwen2.5-0.5b,box-code,fair,0.575,200,0.253,0.407,0.306,0.502,above
qwen2.5-0.5b,box-chat,0shot,0.525,200,0.253,0.321,0.225,0.429,above
qwen2.5-0.5b,box-chat,fair,0.568,200,0.253,0.396,0.297,0.481,above
qwen2.5-0.5b,you-static,0shot,0.397,300,0.292,0.131,0.048,0.209,above
qwen2.5-0.5b,i-moves,0shot,0.342,180,0.287,0.077,-0.020,0.174,at
qwen2.5-0.5b,i-moves,fair,0.307,180,0.287,0.029,-0.066,0.124,at
qwen2.5-0.5b,you-moves,0shot,0.242,180,0.287,-0.067,-0.151,0.027,at
qwen2.5-0.5b,you-moves,fair,0.308,180,0.287,0.030,-0.064,0.122,at
qwen2.5-0.5b,you-named,0shot,0.308,180,0.287,0.032,-0.058,0.123,at
qwen2.5-0.5b,you-named,fair,0.375,180,0.287,0.124,0.032,0.226,above
qwen2.5-0.5b,answer-self,0shot,0.301,2560,0.250,0.068,0.044,0.090,above
qwen2.5-0.5b,answer-self,fair,0.276,2560,0.250,0.034,0.011,0.057,above
qwen2.5-0.5b,grammar,ladder,,1080,,0.702,0.678,0.730,above
qwen2.5-0.5b,operators,ladder,,1080,,0.225,0.197,0.252,above
qwen2.5-0.5b,patterns,ladder,,1080,,0.439,0.403,0.472,above
qwen2.5-0.5b,facts,ladder,,768,,0.920,0.900,0.942,above
falcon-h1-0.5b,lookup,0shot,0.603,1210,0.169,0.510,0.476,0.544,above
falcon-h1-0.5b,lookup,fair,0.844,1210,0.169,0.809,0.783,0.832,above
falcon-h1-0.5b,shortlist,0shot,0.789,3840,0.040,0.780,0.767,0.794,above
falcon-h1-0.5b,pairs,0shot,0.572,256,0.125,0.511,0.446,0.583,above
falcon-h1-0.5b,pairs,fair,0.504,256,0.125,0.433,0.364,0.502,above
falcon-h1-0.5b,noise,0shot,0.820,720,0.200,0.775,0.740,0.807,above
falcon-h1-0.5b,now,0shot,0.758,300,0.292,0.651,0.577,0.719,above
falcon-h1-0.5b,handoffs,0shot,0.524,1200,0.312,0.303,0.264,0.343,above
falcon-h1-0.5b,handoffs,fair,0.597,1200,0.312,0.409,0.371,0.447,above
falcon-h1-0.5b,sets,0shot,0.649,1280,0.275,0.526,0.489,0.561,above
falcon-h1-0.5b,sets,fair,0.534,1280,0.275,0.367,0.331,0.407,above
falcon-h1-0.5b,refilled,0shot,0.500,720,0.200,0.375,0.329,0.422,above
falcon-h1-0.5b,refills,0shot,0.313,600,0.200,0.142,0.097,0.190,above
falcon-h1-0.5b,codevar,0shot,0.329,1200,0.267,0.079,0.040,0.120,above
falcon-h1-0.5b,codevar,fair,0.330,1200,0.267,0.083,0.044,0.119,above
falcon-h1-0.5b,sharedlists,0shot,0.212,600,0.200,0.016,-0.025,0.056,at
falcon-h1-0.5b,sharedlists,fair,0.302,600,0.200,0.127,0.080,0.176,above
falcon-h1-0.5b,box-narrative,0shot,0.568,200,0.253,0.431,0.335,0.518,above
falcon-h1-0.5b,box-narrative,fair,0.403,200,0.253,0.177,0.076,0.271,above
falcon-h1-0.5b,box-screenplay,0shot,0.618,200,0.253,0.495,0.408,0.586,above
falcon-h1-0.5b,box-screenplay,fair,0.450,200,0.253,0.263,0.168,0.351,above
falcon-h1-0.5b,box-code,0shot,0.398,200,0.253,0.216,0.124,0.302,above
falcon-h1-0.5b,box-code,fair,0.635,200,0.253,0.492,0.398,0.581,above
falcon-h1-0.5b,box-chat,0shot,0.625,200,0.253,0.503,0.407,0.589,above
falcon-h1-0.5b,box-chat,fair,0.357,200,0.253,0.113,0.019,0.205,above
falcon-h1-0.5b,you-static,0shot,0.477,300,0.292,0.242,0.164,0.326,above
falcon-h1-0.5b,i-moves,0shot,0.247,180,0.287,-0.063,-0.152,0.018,at
falcon-h1-0.5b,i-moves,fair,0.322,180,0.287,0.046,-0.055,0.143,at
falcon-h1-0.5b,you-moves,0shot,0.322,180,0.287,0.049,-0.043,0.143,at
falcon-h1-0.5b,you-moves,fair,0.322,180,0.287,0.046,-0.050,0.145,at
falcon-h1-0.5b,you-named,0shot,0.294,180,0.287,0.013,-0.077,0.107,at
falcon-h1-0.5b,you-named,fair,0.322,180,0.287,0.045,-0.048,0.144,at
falcon-h1-0.5b,answer-self,0shot,0.337,2560,0.250,0.115,0.092,0.140,above
falcon-h1-0.5b,answer-self,fair,0.307,2560,0.250,0.076,0.053,0.101,above
falcon-h1-0.5b,grammar,ladder,,1080,,0.690,0.658,0.722,above
falcon-h1-0.5b,operators,ladder,,1080,,0.232,0.205,0.260,above
falcon-h1-0.5b,patterns,ladder,,1080,,0.367,0.331,0.401,above
falcon-h1-0.5b,facts,ladder,,768,,0.894,0.872,0.919,above
qwen3-0.6b,lookup,0shot,0.652,1210,0.169,0.571,0.541,0.602,above
qwen3-0.6b,lookup,fair,0.937,1210,0.169,0.923,0.905,0.939,above
qwen3-0.6b,shortlist,0shot,0.506,3840,0.040,0.486,0.470,0.501,above
qwen3-0.6b,pairs,0shot,0.577,256,0.125,0.516,0.455,0.580,above
qwen3-0.6b,pairs,fair,0.453,256,0.125,0.374,0.308,0.446,above
qwen3-0.6b,noise,0shot,0.862,720,0.200,0.827,0.793,0.856,above
qwen3-0.6b,now,0shot,0.867,300,0.292,0.808,0.753,0.858,above
qwen3-0.6b,handoffs,0shot,0.511,1200,0.312,0.283,0.242,0.326,above
qwen3-0.6b,handoffs,fair,0.542,1200,0.312,0.330,0.289,0.372,above
qwen3-0.6b,sets,0shot,0.620,1280,0.275,0.485,0.448,0.520,above
qwen3-0.6b,sets,fair,0.664,1280,0.275,0.549,0.514,0.582,above
qwen3-0.6b,refilled,0shot,0.373,720,0.200,0.216,0.173,0.265,above
qwen3-0.6b,refills,0shot,0.301,600,0.200,0.127,0.081,0.174,above
qwen3-0.6b,codevar,0shot,0.409,1200,0.267,0.192,0.152,0.229,above
qwen3-0.6b,codevar,fair,0.467,1200,0.267,0.266,0.228,0.303,above
qwen3-0.6b,sharedlists,0shot,0.294,600,0.200,0.118,0.071,0.163,above
qwen3-0.6b,sharedlists,fair,0.278,600,0.200,0.097,0.055,0.140,above
qwen3-0.6b,box-narrative,0shot,0.390,200,0.253,0.151,0.057,0.242,above
qwen3-0.6b,box-narrative,fair,0.430,200,0.253,0.206,0.108,0.298,above
qwen3-0.6b,box-screenplay,0shot,0.507,200,0.253,0.337,0.239,0.438,above
qwen3-0.6b,box-screenplay,fair,0.458,200,0.253,0.249,0.148,0.343,above
qwen3-0.6b,box-code,0shot,0.525,200,0.253,0.349,0.249,0.448,above
qwen3-0.6b,box-code,fair,0.623,200,0.253,0.467,0.370,0.562,above
qwen3-0.6b,box-chat,0shot,0.495,200,0.253,0.312,0.211,0.415,above
qwen3-0.6b,box-chat,fair,0.448,200,0.253,0.234,0.125,0.327,above
qwen3-0.6b,you-static,0shot,0.572,300,0.292,0.383,0.304,0.462,above
qwen3-0.6b,i-moves,0shot,0.336,180,0.287,0.063,-0.036,0.154,at
qwen3-0.6b,i-moves,fair,0.319,180,0.287,0.043,-0.050,0.129,at
qwen3-0.6b,you-moves,0shot,0.281,180,0.287,-0.010,-0.097,0.079,at
qwen3-0.6b,you-moves,fair,0.356,180,0.287,0.095,-0.003,0.200,at
qwen3-0.6b,you-named,0shot,0.319,180,0.287,0.044,-0.050,0.129,at
qwen3-0.6b,you-named,fair,0.305,180,0.287,0.024,-0.066,0.118,at
qwen3-0.6b,answer-self,0shot,0.308,2560,0.250,0.077,0.052,0.101,above
qwen3-0.6b,answer-self,fair,0.278,2560,0.250,0.038,0.015,0.059,above
qwen3-0.6b,grammar,ladder,,1080,,0.711,0.684,0.736,above
qwen3-0.6b,operators,ladder,,1080,,0.252,0.221,0.283,above
qwen3-0.6b,patterns,ladder,,1080,,0.495,0.464,0.527,above
qwen3-0.6b,facts,ladder,,768,,0.946,0.927,0.963,above
pythia-1b,lookup,0shot,0.461,1210,0.169,0.346,0.310,0.380,above
pythia-1b,lookup,fair,0.516,1210,0.169,0.418,0.386,0.451,above
pythia-1b,shortlist,0shot,0.579,3840,0.040,0.561,0.545,0.577,above
pythia-1b,pairs,0shot,0.312,256,0.125,0.214,0.154,0.279,above
pythia-1b,pairs,fair,0.471,256,0.125,0.395,0.328,0.464,above
pythia-1b,noise,0shot,0.630,720,0.200,0.537,0.494,0.584,above
pythia-1b,now,0shot,0.530,300,0.292,0.336,0.263,0.414,above
pythia-1b,handoffs,0shot,0.492,1200,0.312,0.257,0.215,0.297,above
pythia-1b,handoffs,fair,0.514,1200,0.312,0.287,0.250,0.324,above
pythia-1b,sets,0shot,0.501,1280,0.275,0.317,0.280,0.354,above
pythia-1b,sets,fair,0.557,1280,0.275,0.391,0.354,0.428,above
pythia-1b,refilled,0shot,0.372,720,0.200,0.215,0.172,0.260,above
pythia-1b,refills,0shot,0.302,600,0.200,0.128,0.084,0.173,above
pythia-1b,codevar,0shot,0.272,1200,0.267,0.007,-0.029,0.040,at
pythia-1b,codevar,fair,0.269,1200,0.267,0.001,-0.031,0.035,at
pythia-1b,sharedlists,0shot,0.228,600,0.200,0.035,-0.003,0.076,at
pythia-1b,sharedlists,fair,0.254,600,0.200,0.068,0.026,0.116,above
pythia-1b,box-narrative,0shot,0.320,200,0.253,0.103,0.018,0.184,above
pythia-1b,box-narrative,fair,0.343,200,0.253,0.131,0.047,0.217,above
pythia-1b,box-screenplay,0shot,0.395,200,0.253,0.201,0.112,0.283,above
pythia-1b,box-screenplay,fair,0.340,200,0.253,0.122,0.027,0.216,above
pythia-1b,box-code,0shot,0.367,200,0.253,0.162,0.076,0.248,above
pythia-1b,box-code,fair,0.497,200,0.253,0.311,0.213,0.401,above
pythia-1b,box-chat,0shot,0.427,200,0.253,0.251,0.163,0.344,above
pythia-1b,box-chat,fair,0.320,200,0.253,0.076,-0.024,0.164,at
pythia-1b,you-static,0shot,0.378,300,0.292,0.113,0.037,0.195,above
pythia-1b,i-moves,0shot,0.319,180,0.287,0.044,-0.051,0.136,at
pythia-1b,i-moves,fair,0.286,180,0.287,-0.006,-0.097,0.081,at
pythia-1b,you-moves,0shot,0.269,180,0.287,-0.025,-0.106,0.064,at
pythia-1b,you-moves,fair,0.272,180,0.287,-0.026,-0.113,0.058,at
pythia-1b,you-named,0shot,0.308,180,0.287,0.030,-0.066,0.130,at
pythia-1b,you-named,fair,0.331,180,0.287,0.060,-0.027,0.155,at
pythia-1b,answer-self,0shot,0.331,2560,0.250,0.107,0.083,0.132,above
pythia-1b,answer-self,fair,0.311,2560,0.250,0.082,0.058,0.104,above
pythia-1b,grammar,ladder,,1080,,0.667,0.642,0.693,above
pythia-1b,operators,ladder,,1080,,0.216,0.186,0.249,above
pythia-1b,patterns,ladder,,1080,,0.247,0.214,0.281,above
pythia-1b,facts,ladder,,768,,0.905,0.880,0.928,above
llama3.2-1b,lookup,0shot,0.591,1210,0.169,0.492,0.458,0.529,above
llama3.2-1b,lookup,fair,0.836,1210,0.169,0.799,0.773,0.825,above
llama3.2-1b,shortlist,0shot,0.651,3840,0.040,0.636,0.621,0.653,above
llama3.2-1b,pairs,0shot,0.588,256,0.125,0.529,0.462,0.594,above
llama3.2-1b,pairs,fair,0.507,256,0.125,0.437,0.365,0.504,above
llama3.2-1b,noise,0shot,0.734,720,0.200,0.667,0.628,0.707,above
llama3.2-1b,now,0shot,0.702,300,0.292,0.576,0.500,0.648,above
llama3.2-1b,handoffs,0shot,0.505,1200,0.312,0.276,0.235,0.317,above
llama3.2-1b,handoffs,fair,0.513,1200,0.312,0.286,0.247,0.325,above
llama3.2-1b,sets,0shot,0.519,1280,0.275,0.339,0.302,0.375,above
llama3.2-1b,sets,fair,0.678,1280,0.275,0.572,0.538,0.607,above
llama3.2-1b,refilled,0shot,0.370,720,0.200,0.212,0.170,0.255,above
llama3.2-1b,refills,0shot,0.290,600,0.200,0.113,0.065,0.156,above
llama3.2-1b,codevar,0shot,0.313,1200,0.267,0.059,0.023,0.096,above
llama3.2-1b,codevar,fair,0.364,1200,0.267,0.123,0.088,0.161,above
llama3.2-1b,sharedlists,0shot,0.341,600,0.200,0.176,0.129,0.224,above
llama3.2-1b,sharedlists,fair,0.303,600,0.200,0.129,0.085,0.173,above
llama3.2-1b,box-narrative,0shot,0.540,200,0.253,0.348,0.247,0.449,above
llama3.2-1b,box-narrative,fair,0.555,200,0.253,0.383,0.288,0.469,above
llama3.2-1b,box-screenplay,0shot,0.654,200,0.253,0.532,0.440,0.617,above
llama3.2-1b,box-screenplay,fair,0.610,200,0.253,0.473,0.377,0.558,above
llama3.2-1b,box-code,0shot,0.603,200,0.253,0.442,0.342,0.538,above
llama3.2-1b,box-code,fair,0.682,200,0.253,0.550,0.452,0.640,above
llama3.2-1b,box-chat,0shot,0.630,200,0.253,0.488,0.400,0.576,above
llama3.2-1b,box-chat,fair,0.608,200,0.253,0.459,0.353,0.544,above
llama3.2-1b,you-static,0shot,0.491,300,0.292,0.270,0.183,0.350,above
llama3.2-1b,i-moves,0shot,0.288,180,0.287,-0.008,-0.100,0.089,at
llama3.2-1b,i-moves,fair,0.300,180,0.287,0.017,-0.074,0.110,at
llama3.2-1b,you-moves,0shot,0.285,180,0.287,-0.009,-0.100,0.075,at
llama3.2-1b,you-moves,fair,0.281,180,0.287,-0.011,-0.100,0.090,at
llama3.2-1b,you-named,0shot,0.288,180,0.287,0.003,-0.086,0.098,at
llama3.2-1b,you-named,fair,0.344,180,0.287,0.080,-0.016,0.176,at
llama3.2-1b,answer-self,0shot,0.339,2560,0.250,0.119,0.094,0.144,above
llama3.2-1b,answer-self,fair,0.261,2560,0.250,0.015,-0.009,0.037,at
llama3.2-1b,grammar,ladder,,1080,,0.780,0.760,0.804,above
llama3.2-1b,operators,ladder,,1080,,0.221,0.201,0.248,above
llama3.2-1b,patterns,ladder,,1080,,0.397,0.364,0.432,above
llama3.2-1b,facts,ladder,,768,,0.997,0.992,1.000,above
olmo1-1b,lookup,0shot,0.442,1210,0.169,0.312,0.277,0.347,above
olmo1-1b,lookup,fair,0.809,1210,0.169,0.772,0.746,0.797,above
olmo1-1b,shortlist,0shot,0.630,3840,0.040,0.615,0.600,0.631,above
olmo1-1b,pairs,0shot,0.463,256,0.125,0.386,0.312,0.455,above
olmo1-1b,pairs,fair,0.426,256,0.125,0.344,0.275,0.413,above
olmo1-1b,noise,0shot,0.471,720,0.200,0.338,0.297,0.383,above
olmo1-1b,now,0shot,0.485,300,0.292,0.271,0.191,0.352,above
olmo1-1b,handoffs,0shot,0.492,1200,0.312,0.256,0.215,0.295,above
olmo1-1b,handoffs,fair,0.530,1200,0.312,0.315,0.275,0.356,above
olmo1-1b,sets,0shot,0.516,1280,0.275,0.337,0.298,0.374,above
olmo1-1b,sets,fair,0.544,1280,0.275,0.380,0.345,0.415,above
olmo1-1b,refilled,0shot,0.468,720,0.200,0.335,0.287,0.380,above
olmo1-1b,refills,0shot,0.327,600,0.200,0.158,0.114,0.205,above
olmo1-1b,codevar,0shot,0.321,1200,0.267,0.070,0.035,0.107,above
olmo1-1b,codevar,fair,0.277,1200,0.267,0.008,-0.028,0.042,at
olmo1-1b,sharedlists,0shot,0.229,600,0.200,0.036,-0.005,0.078,at
olmo1-1b,sharedlists,fair,0.261,600,0.200,0.077,0.034,0.119,above
olmo1-1b,box-narrative,0shot,0.482,200,0.253,0.306,0.211,0.400,above
olmo1-1b,box-narrative,fair,0.435,200,0.253,0.236,0.141,0.329,above
olmo1-1b,box-screenplay,0shot,0.502,200,0.253,0.338,0.242,0.426,above
olmo1-1b,box-screenplay,fair,0.403,200,0.253,0.183,0.088,0.276,above
olmo1-1b,box-code,0shot,0.537,200,0.253,0.367,0.264,0.465,above
olmo1-1b,box-code,fair,0.490,200,0.253,0.298,0.198,0.398,above
olmo1-1b,box-chat,0shot,0.530,200,0.253,0.359,0.256,0.458,above
olmo1-1b,box-chat,fair,0.477,200,0.253,0.288,0.197,0.384,above
olmo1-1b,you-static,0shot,0.390,300,0.292,0.136,0.053,0.216,above
olmo1-1b,i-moves,0shot,0.300,180,0.287,0.017,-0.074,0.108,at
olmo1-1b,i-moves,fair,0.294,180,0.287,0.011,-0.080,0.109,at
olmo1-1b,you-moves,0shot,0.231,180,0.287,-0.084,-0.168,0.006,at
olmo1-1b,you-moves,fair,0.261,180,0.287,-0.031,-0.114,0.053,at
olmo1-1b,you-named,0shot,0.278,180,0.287,-0.013,-0.097,0.078,at
olmo1-1b,you-named,fair,0.269,180,0.287,-0.025,-0.118,0.074,at
olmo1-1b,answer-self,0shot,0.300,2560,0.250,0.067,0.042,0.090,above
olmo1-1b,answer-self,fair,0.300,2560,0.250,0.067,0.044,0.088,above
olmo1-1b,grammar,ladder,,1080,,0.766,0.742,0.794,above
olmo1-1b,operators,ladder,,1080,,0.277,0.247,0.306,above
olmo1-1b,patterns,ladder,,1080,,0.450,0.414,0.485,above
olmo1-1b,facts,ladder,,768,,0.994,0.986,0.998,above
tsumugi,lookup,0shot,0.550,1210,0.169,0.446,0.411,0.477,above
tsumugi,lookup,fair,0.989,1210,0.169,0.987,0.979,0.993,above
tsumugi,shortlist,0shot,0.700,3840,0.040,0.688,0.673,0.703,above
tsumugi,pairs,0shot,0.531,256,0.125,0.464,0.388,0.529,above
tsumugi,pairs,fair,0.703,256,0.125,0.661,0.598,0.728,above
tsumugi,noise,0shot,0.675,720,0.200,0.594,0.549,0.635,above
tsumugi,now,0shot,0.692,300,0.292,0.555,0.480,0.628,above
tsumugi,handoffs,0shot,0.522,1200,0.312,0.301,0.265,0.342,above
tsumugi,handoffs,fair,0.589,1200,0.312,0.399,0.357,0.442,above
tsumugi,sets,0shot,0.494,1280,0.275,0.313,0.274,0.350,above
tsumugi,sets,fair,0.661,1280,0.275,0.547,0.512,0.581,above
tsumugi,refilled,0shot,0.372,720,0.200,0.215,0.172,0.260,above
tsumugi,refills,0shot,0.309,600,0.200,0.137,0.092,0.184,above
tsumugi,codevar,0shot,0.323,1200,0.267,0.072,0.035,0.109,above
tsumugi,codevar,fair,0.301,1200,0.267,0.042,0.009,0.078,above
tsumugi,sharedlists,0shot,0.254,600,0.200,0.067,0.025,0.114,above
tsumugi,sharedlists,fair,0.319,600,0.200,0.149,0.104,0.192,above
tsumugi,box-narrative,0shot,0.527,200,0.253,0.358,0.264,0.455,above
tsumugi,box-narrative,fair,0.435,200,0.253,0.214,0.109,0.313,above
tsumugi,box-screenplay,0shot,0.635,200,0.253,0.502,0.405,0.594,above
tsumugi,box-screenplay,fair,0.470,200,0.253,0.259,0.152,0.357,above
tsumugi,box-code,0shot,0.468,200,0.253,0.292,0.192,0.384,above
tsumugi,box-code,fair,0.640,200,0.253,0.489,0.392,0.583,above
tsumugi,box-chat,0shot,0.560,200,0.253,0.408,0.315,0.500,above
tsumugi,box-chat,fair,0.510,200,0.253,0.321,0.223,0.410,above
tsumugi,you-static,0shot,0.647,300,0.292,0.492,0.414,0.569,above
tsumugi,i-moves,0shot,0.289,180,0.287,-0.003,-0.093,0.092,at
tsumugi,i-moves,fair,0.308,180,0.287,0.034,-0.058,0.116,at
tsumugi,you-moves,0shot,0.308,180,0.287,0.027,-0.066,0.118,at
tsumugi,you-moves,fair,0.314,180,0.287,0.036,-0.065,0.131,at
tsumugi,you-named,0shot,0.331,180,0.287,0.057,-0.033,0.147,at
tsumugi,you-named,fair,0.333,180,0.287,0.061,-0.033,0.158,at
tsumugi,answer-self,0shot,0.355,2560,0.250,0.140,0.118,0.163,above
tsumugi,answer-self,fair,0.331,2560,0.250,0.108,0.083,0.132,above
tsumugi,grammar,ladder,,1080,,0.789,0.765,0.812,above
tsumugi,operators,ladder,,1080,,0.296,0.256,0.334,above
tsumugi,patterns,ladder,,1080,,0.553,0.524,0.589,above
tsumugi,facts,ladder,,768,,0.948,0.927,0.964,above
olmo2-1b,lookup,0shot,0.367,1210,0.169,0.231,0.200,0.265,above
olmo2-1b,lookup,fair,0.443,1210,0.169,0.323,0.288,0.356,above
olmo2-1b,shortlist,0shot,0.629,3840,0.040,0.613,0.598,0.628,above
olmo2-1b,pairs,0shot,0.432,256,0.125,0.350,0.279,0.422,above
olmo2-1b,pairs,fair,0.357,256,0.125,0.266,0.196,0.333,above
olmo2-1b,noise,0shot,0.608,720,0.200,0.509,0.469,0.554,above
olmo2-1b,now,0shot,0.732,300,0.292,0.618,0.546,0.688,above
olmo2-1b,handoffs,0shot,0.582,1200,0.312,0.388,0.347,0.430,above
olmo2-1b,handoffs,fair,0.621,1200,0.312,0.445,0.401,0.485,above
olmo2-1b,sets,0shot,0.512,1280,0.275,0.328,0.289,0.366,above
olmo2-1b,sets,fair,0.542,1280,0.275,0.375,0.338,0.414,above
olmo2-1b,refilled,0shot,0.533,720,0.200,0.416,0.369,0.459,above
olmo2-1b,refills,0shot,0.393,600,0.200,0.242,0.196,0.291,above
olmo2-1b,codevar,0shot,0.349,1200,0.267,0.110,0.073,0.147,above
olmo2-1b,codevar,fair,0.336,1200,0.267,0.088,0.052,0.125,above
olmo2-1b,sharedlists,0shot,0.216,600,0.200,0.020,-0.020,0.065,at
olmo2-1b,sharedlists,fair,0.251,600,0.200,0.064,0.022,0.108,above
olmo2-1b,box-narrative,0shot,0.475,200,0.253,0.296,0.200,0.394,above
olmo2-1b,box-narrative,fair,0.425,200,0.253,0.212,0.115,0.300,above
olmo2-1b,box-screenplay,0shot,0.522,200,0.253,0.369,0.281,0.458,above
olmo2-1b,box-screenplay,fair,0.465,200,0.253,0.271,0.170,0.360,above
olmo2-1b,box-code,0shot,0.410,200,0.253,0.225,0.134,0.320,above
olmo2-1b,box-code,fair,0.600,200,0.253,0.426,0.324,0.527,above
olmo2-1b,box-chat,0shot,0.555,200,0.253,0.409,0.319,0.495,above
olmo2-1b,box-chat,fair,0.383,200,0.253,0.163,0.066,0.259,above
olmo2-1b,you-static,0shot,0.495,300,0.292,0.276,0.192,0.354,above
olmo2-1b,i-moves,0shot,0.325,180,0.287,0.048,-0.046,0.151,at
olmo2-1b,i-moves,fair,0.294,180,0.287,0.008,-0.081,0.107,at
olmo2-1b,you-moves,0shot,0.289,180,0.287,0.001,-0.090,0.096,at
olmo2-1b,you-moves,fair,0.328,180,0.287,0.057,-0.039,0.158,at
olmo2-1b,you-named,0shot,0.331,180,0.287,0.057,-0.037,0.157,at
olmo2-1b,you-named,fair,0.342,180,0.287,0.080,-0.017,0.180,at
olmo2-1b,answer-self,0shot,0.330,2560,0.250,0.106,0.082,0.129,above
olmo2-1b,answer-self,fair,0.308,2560,0.250,0.078,0.054,0.101,above
olmo2-1b,grammar,ladder,,1080,,0.699,0.675,0.726,above
olmo2-1b,operators,ladder,,1080,,0.257,0.229,0.289,above
olmo2-1b,patterns,ladder,,1080,,0.396,0.356,0.433,above
olmo2-1b,facts,ladder,,768,,0.991,0.982,0.997,above
pythia-1.4b,lookup,0shot,0.530,1210,0.169,0.425,0.392,0.460,above
pythia-1.4b,lookup,fair,0.621,1210,0.169,0.539,0.506,0.572,above
pythia-1.4b,shortlist,0shot,0.578,3840,0.040,0.560,0.544,0.576,above
pythia-1.4b,pairs,0shot,0.498,256,0.125,0.426,0.364,0.498,above
pythia-1.4b,pairs,fair,0.512,256,0.125,0.442,0.375,0.513,above
pythia-1.4b,noise,0shot,0.663,720,0.200,0.579,0.535,0.621,above
pythia-1.4b,now,0shot,0.567,300,0.292,0.386,0.307,0.462,above
pythia-1.4b,handoffs,0shot,0.435,1200,0.312,0.176,0.139,0.216,above
pythia-1.4b,handoffs,fair,0.441,1200,0.312,0.183,0.141,0.223,above
pythia-1.4b,sets,0shot,0.533,1280,0.275,0.362,0.323,0.402,above
pythia-1.4b,sets,fair,0.599,1280,0.275,0.460,0.424,0.498,above
pythia-1.4b,refilled,0shot,0.342,720,0.200,0.178,0.134,0.221,above
pythia-1.4b,refills,0shot,0.242,600,0.200,0.052,0.009,0.093,above
pythia-1.4b,codevar,0shot,0.245,1200,0.267,-0.031,-0.064,0.001,at
pythia-1.4b,codevar,fair,0.247,1200,0.267,-0.030,-0.064,0.002,at
pythia-1.4b,sharedlists,0shot,0.212,600,0.200,0.016,-0.024,0.057,at
pythia-1.4b,sharedlists,fair,0.195,600,0.200,-0.006,-0.044,0.035,at
pythia-1.4b,box-narrative,0shot,0.440,200,0.253,0.242,0.153,0.334,above
pythia-1.4b,box-narrative,fair,0.468,200,0.253,0.274,0.179,0.369,above
pythia-1.4b,box-screenplay,0shot,0.405,200,0.253,0.208,0.120,0.299,above
pythia-1.4b,box-screenplay,fair,0.458,200,0.253,0.259,0.156,0.356,above
pythia-1.4b,box-code,0shot,0.395,200,0.253,0.197,0.104,0.285,above
pythia-1.4b,box-code,fair,0.475,200,0.253,0.280,0.182,0.381,above
pythia-1.4b,box-chat,0shot,0.460,200,0.253,0.278,0.189,0.374,above
pythia-1.4b,box-chat,fair,0.412,200,0.253,0.180,0.076,0.270,above
pythia-1.4b,you-static,0shot,0.382,300,0.292,0.118,0.038,0.194,above
pythia-1.4b,i-moves,0shot,0.267,180,0.287,-0.033,-0.119,0.063,at
pythia-1.4b,i-moves,fair,0.281,180,0.287,-0.013,-0.100,0.083,at
pythia-1.4b,you-moves,0shot,0.261,180,0.287,-0.033,-0.118,0.053,at
pythia-1.4b,you-moves,fair,0.339,180,0.287,0.070,-0.020,0.170,at
pythia-1.4b,you-named,0shot,0.306,180,0.287,0.030,-0.055,0.134,at
pythia-1.4b,you-named,fair,0.327,180,0.287,0.050,-0.037,0.149,at
pythia-1.4b,answer-self,0shot,0.305,2560,0.250,0.074,0.052,0.099,above
pythia-1.4b,answer-self,fair,0.262,2560,0.250,0.017,-0.008,0.039,at
pythia-1.4b,grammar,ladder,,1080,,0.689,0.662,0.719,above
pythia-1.4b,operators,ladder,,1080,,0.242,0.205,0.272,above
pythia-1.4b,patterns,ladder,,1080,,0.317,0.279,0.357,above
pythia-1.4b,facts,ladder,,768,,0.963,0.947,0.978,above
qwen2.5-1.5b,lookup,0shot,0.726,1210,0.169,0.660,0.627,0.692,above
qwen2.5-1.5b,lookup,fair,0.980,1210,0.169,0.976,0.966,0.986,above
qwen2.5-1.5b,shortlist,0shot,0.694,3840,0.040,0.681,0.666,0.695,above
qwen2.5-1.5b,pairs,0shot,0.587,256,0.125,0.528,0.462,0.596,above
qwen2.5-1.5b,pairs,fair,0.543,256,0.125,0.478,0.406,0.545,above
qwen2.5-1.5b,noise,0shot,0.804,720,0.200,0.755,0.717,0.789,above
qwen2.5-1.5b,now,0shot,0.818,300,0.292,0.743,0.683,0.800,above
qwen2.5-1.5b,handoffs,0shot,0.655,1200,0.312,0.494,0.455,0.533,above
qwen2.5-1.5b,handoffs,fair,0.655,1200,0.312,0.491,0.450,0.531,above
qwen2.5-1.5b,sets,0shot,0.634,1280,0.275,0.501,0.465,0.535,above
qwen2.5-1.5b,sets,fair,0.671,1280,0.275,0.561,0.525,0.597,above
qwen2.5-1.5b,refilled,0shot,0.590,720,0.200,0.487,0.440,0.532,above
qwen2.5-1.5b,refills,0shot,0.397,600,0.200,0.246,0.197,0.293,above
qwen2.5-1.5b,codevar,0shot,0.399,1200,0.267,0.171,0.134,0.207,above
qwen2.5-1.5b,codevar,fair,0.433,1200,0.267,0.219,0.182,0.256,above
qwen2.5-1.5b,sharedlists,0shot,0.278,600,0.200,0.097,0.056,0.142,above
qwen2.5-1.5b,sharedlists,fair,0.315,600,0.200,0.143,0.101,0.189,above
qwen2.5-1.5b,box-narrative,0shot,0.468,200,0.253,0.261,0.164,0.356,above
qwen2.5-1.5b,box-narrative,fair,0.417,200,0.253,0.190,0.096,0.293,above
qwen2.5-1.5b,box-screenplay,0shot,0.510,200,0.253,0.333,0.237,0.425,above
qwen2.5-1.5b,box-screenplay,fair,0.452,200,0.253,0.246,0.151,0.347,above
qwen2.5-1.5b,box-code,0shot,0.650,200,0.253,0.512,0.416,0.599,above
qwen2.5-1.5b,box-code,fair,0.718,200,0.253,0.605,0.514,0.691,above
qwen2.5-1.5b,box-chat,0shot,0.575,200,0.253,0.430,0.333,0.524,above
qwen2.5-1.5b,box-chat,fair,0.517,200,0.253,0.322,0.220,0.419,above
qwen2.5-1.5b,you-static,0shot,0.617,300,0.292,0.453,0.378,0.524,above
qwen2.5-1.5b,i-moves,0shot,0.393,180,0.287,0.146,0.047,0.236,above
qwen2.5-1.5b,i-moves,fair,0.336,180,0.287,0.067,-0.026,0.163,at
qwen2.5-1.5b,you-moves,0shot,0.235,180,0.287,-0.080,-0.162,0.005,at
qwen2.5-1.5b,you-moves,fair,0.288,180,0.287,-0.001,-0.095,0.084,at
qwen2.5-1.5b,you-named,0shot,0.372,180,0.287,0.118,0.023,0.221,above
qwen2.5-1.5b,you-named,fair,0.319,180,0.287,0.044,-0.042,0.142,at
qwen2.5-1.5b,answer-self,0shot,0.345,2560,0.250,0.127,0.102,0.151,above
qwen2.5-1.5b,answer-self,fair,0.317,2560,0.250,0.089,0.065,0.110,above
qwen2.5-1.5b,grammar,ladder,,1080,,0.763,0.740,0.789,above
qwen2.5-1.5b,operators,ladder,,1080,,0.204,0.177,0.231,above
qwen2.5-1.5b,patterns,ladder,,1080,,0.458,0.420,0.496,above
qwen2.5-1.5b,facts,ladder,,768,,0.995,0.989,1.000,above
falcon-h1-1.5b,lookup,0shot,0.868,1210,0.169,0.836,0.810,0.858,above
falcon-h1-1.5b,lookup,fair,0.987,1210,0.169,0.984,0.975,0.992,above
falcon-h1-1.5b,shortlist,0shot,0.903,3840,0.040,0.899,0.890,0.908,above
falcon-h1-1.5b,pairs,0shot,0.500,256,0.125,0.429,0.362,0.491,above
falcon-h1-1.5b,pairs,fair,0.625,256,0.125,0.571,0.504,0.638,above
falcon-h1-1.5b,noise,0shot,0.821,720,0.200,0.776,0.740,0.812,above
falcon-h1-1.5b,now,0shot,0.867,300,0.292,0.814,0.758,0.866,above
falcon-h1-1.5b,handoffs,0shot,0.565,1200,0.312,0.357,0.311,0.400,above
falcon-h1-1.5b,handoffs,fair,0.723,1200,0.312,0.594,0.557,0.630,above
falcon-h1-1.5b,sets,0shot,0.535,1280,0.275,0.361,0.324,0.399,above
falcon-h1-1.5b,sets,fair,0.732,1280,0.275,0.639,0.605,0.672,above
falcon-h1-1.5b,refilled,0shot,0.583,720,0.200,0.479,0.432,0.523,above
falcon-h1-1.5b,refills,0shot,0.310,600,0.200,0.138,0.091,0.183,above
falcon-h1-1.5b,codevar,0shot,0.432,1200,0.267,0.222,0.184,0.260,above
falcon-h1-1.5b,codevar,fair,0.518,1200,0.267,0.339,0.300,0.376,above
falcon-h1-1.5b,sharedlists,0shot,0.350,600,0.200,0.188,0.142,0.233,above
falcon-h1-1.5b,sharedlists,fair,0.500,600,0.200,0.375,0.323,0.429,above
falcon-h1-1.5b,box-narrative,0shot,0.525,200,0.253,0.358,0.254,0.451,above
falcon-h1-1.5b,box-narrative,fair,0.465,200,0.253,0.284,0.195,0.379,above
falcon-h1-1.5b,box-screenplay,0shot,0.555,200,0.253,0.407,0.317,0.497,above
falcon-h1-1.5b,box-screenplay,fair,0.405,200,0.253,0.199,0.103,0.291,above
falcon-h1-1.5b,box-code,0shot,0.575,200,0.253,0.409,0.315,0.499,above
falcon-h1-1.5b,box-code,fair,0.755,200,0.253,0.663,0.578,0.743,above
falcon-h1-1.5b,box-chat,0shot,0.593,200,0.253,0.444,0.343,0.537,above
falcon-h1-1.5b,box-chat,fair,0.445,200,0.253,0.255,0.158,0.353,above
falcon-h1-1.5b,you-static,0shot,0.767,300,0.292,0.662,0.591,0.727,above
falcon-h1-1.5b,i-moves,0shot,0.325,180,0.287,0.052,-0.045,0.144,at
falcon-h1-1.5b,i-moves,fair,0.356,180,0.287,0.097,0.007,0.196,above
falcon-h1-1.5b,you-moves,0shot,0.250,180,0.287,-0.056,-0.144,0.043,at
falcon-h1-1.5b,you-moves,fair,0.311,180,0.287,0.036,-0.060,0.130,at
falcon-h1-1.5b,you-named,0shot,0.317,180,0.287,0.040,-0.049,0.128,at
falcon-h1-1.5b,you-named,fair,0.289,180,0.287,0.002,-0.087,0.100,at
falcon-h1-1.5b,answer-self,0shot,0.393,2560,0.250,0.191,0.165,0.216,above
falcon-h1-1.5b,answer-self,fair,0.333,2560,0.250,0.111,0.086,0.136,above
falcon-h1-1.5b,grammar,ladder,,1080,,0.733,0.707,0.756,above
falcon-h1-1.5b,operators,ladder,,1080,,0.228,0.191,0.257,above
falcon-h1-1.5b,patterns,ladder,,1080,,0.418,0.383,0.453,above
falcon-h1-1.5b,facts,ladder,,768,,0.973,0.961,0.983,above
falcon-h1-1.5b-deep,lookup,0shot,0.832,1210,0.169,0.787,0.759,0.813,above
falcon-h1-1.5b-deep,lookup,fair,0.989,1210,0.169,0.988,0.981,0.993,above
falcon-h1-1.5b-deep,shortlist,0shot,0.901,3840,0.040,0.897,0.886,0.906,above
falcon-h1-1.5b-deep,pairs,0shot,0.613,256,0.125,0.558,0.491,0.621,above
falcon-h1-1.5b-deep,pairs,fair,0.832,256,0.125,0.808,0.754,0.857,above
falcon-h1-1.5b-deep,noise,0shot,0.925,720,0.200,0.906,0.880,0.931,above
falcon-h1-1.5b-deep,now,0shot,0.917,300,0.292,0.883,0.831,0.923,above
falcon-h1-1.5b-deep,handoffs,0shot,0.716,1200,0.312,0.584,0.545,0.620,above
falcon-h1-1.5b-deep,handoffs,fair,0.791,1200,0.312,0.694,0.659,0.727,above
falcon-h1-1.5b-deep,sets,0shot,0.785,1280,0.275,0.711,0.679,0.743,above
falcon-h1-1.5b-deep,sets,fair,0.726,1280,0.275,0.633,0.601,0.665,above
falcon-h1-1.5b-deep,refilled,0shot,0.620,720,0.200,0.525,0.482,0.571,above
falcon-h1-1.5b-deep,refills,0shot,0.421,600,0.200,0.276,0.227,0.323,above
falcon-h1-1.5b-deep,codevar,0shot,0.493,1200,0.267,0.305,0.264,0.346,above
falcon-h1-1.5b-deep,codevar,fair,0.618,1200,0.267,0.474,0.435,0.511,above
falcon-h1-1.5b-deep,sharedlists,0shot,0.418,600,0.200,0.273,0.225,0.321,above
falcon-h1-1.5b-deep,sharedlists,fair,0.527,600,0.200,0.408,0.358,0.456,above
falcon-h1-1.5b-deep,box-narrative,0shot,0.620,200,0.253,0.466,0.367,0.560,above
falcon-h1-1.5b-deep,box-narrative,fair,0.502,200,0.253,0.340,0.245,0.440,above
falcon-h1-1.5b-deep,box-screenplay,0shot,0.642,200,0.253,0.535,0.449,0.626,above
falcon-h1-1.5b-deep,box-screenplay,fair,0.522,200,0.253,0.363,0.267,0.460,above
falcon-h1-1.5b-deep,box-code,0shot,0.542,200,0.253,0.379,0.281,0.471,above
falcon-h1-1.5b-deep,box-code,fair,0.830,200,0.253,0.763,0.690,0.830,above
falcon-h1-1.5b-deep,box-chat,0shot,0.713,200,0.253,0.620,0.535,0.708,above
falcon-h1-1.5b-deep,box-chat,fair,0.595,200,0.253,0.459,0.359,0.554,above
falcon-h1-1.5b-deep,you-static,0shot,0.730,300,0.292,0.611,0.528,0.676,above
falcon-h1-1.5b-deep,i-moves,0shot,0.344,180,0.287,0.076,-0.029,0.174,at
falcon-h1-1.5b-deep,i-moves,fair,0.372,180,0.287,0.119,0.020,0.212,above
falcon-h1-1.5b-deep,you-moves,0shot,0.250,180,0.287,-0.056,-0.144,0.037,at
falcon-h1-1.5b-deep,you-moves,fair,0.311,180,0.287,0.035,-0.059,0.141,at
falcon-h1-1.5b-deep,you-named,0shot,0.311,180,0.287,0.028,-0.061,0.122,at
falcon-h1-1.5b-deep,you-named,fair,0.322,180,0.287,0.046,-0.043,0.142,at
falcon-h1-1.5b-deep,answer-self,0shot,0.483,2560,0.250,0.311,0.286,0.335,above
falcon-h1-1.5b-deep,answer-self,fair,0.445,2560,0.250,0.260,0.235,0.285,above
falcon-h1-1.5b-deep,grammar,ladder,,1080,,0.796,0.771,0.822,above
falcon-h1-1.5b-deep,operators,ladder,,1080,,0.284,0.253,0.312,above
falcon-h1-1.5b-deep,patterns,ladder,,1080,,0.496,0.461,0.531,above
falcon-h1-1.5b-deep,facts,ladder,,768,,0.987,0.978,0.995,above
gpt2-xl,lookup,0shot,0.578,1210,0.169,0.484,0.452,0.517,above
gpt2-xl,lookup,fair,0.679,1210,0.169,0.613,0.585,0.643,above
gpt2-xl,shortlist,0shot,0.454,3840,0.040,0.431,0.414,0.447,above
gpt2-xl,pairs,0shot,0.484,256,0.125,0.411,0.342,0.482,above
gpt2-xl,pairs,fair,0.665,256,0.125,0.618,0.551,0.681,above
gpt2-xl,noise,0shot,0.578,720,0.200,0.472,0.426,0.516,above
gpt2-xl,now,0shot,0.440,300,0.292,0.208,0.133,0.287,above
gpt2-xl,handoffs,0shot,0.395,1200,0.312,0.120,0.079,0.156,above
gpt2-xl,handoffs,fair,0.408,1200,0.312,0.137,0.096,0.176,above
gpt2-xl,sets,0shot,0.488,1280,0.275,0.299,0.260,0.336,above
gpt2-xl,sets,fair,0.547,1280,0.275,0.381,0.343,0.417,above
gpt2-xl,refilled,0shot,0.419,720,0.200,0.274,0.228,0.322,above
gpt2-xl,refills,0shot,0.195,600,0.200,-0.006,-0.048,0.035,at
gpt2-xl,codevar,0shot,0.253,1200,0.267,-0.016,-0.050,0.018,at
gpt2-xl,codevar,fair,0.223,1200,0.267,-0.062,-0.094,-0.029,below
gpt2-xl,sharedlists,0shot,0.274,600,0.200,0.093,0.050,0.134,above
gpt2-xl,sharedlists,fair,0.209,600,0.200,0.011,-0.030,0.052,at
gpt2-xl,box-narrative,0shot,0.328,200,0.253,0.090,0.003,0.178,above
gpt2-xl,box-narrative,fair,0.320,200,0.253,0.085,-0.015,0.180,at
gpt2-xl,box-screenplay,0shot,0.405,200,0.253,0.210,0.117,0.299,above
gpt2-xl,box-screenplay,fair,0.345,200,0.253,0.119,0.021,0.213,above
gpt2-xl,box-code,0shot,0.307,200,0.253,0.066,-0.020,0.156,at
gpt2-xl,box-code,fair,0.420,200,0.253,0.234,0.144,0.323,above
gpt2-xl,box-chat,0shot,0.430,200,0.253,0.245,0.157,0.328,above
gpt2-xl,box-chat,fair,0.343,200,0.253,0.118,0.022,0.205,above
gpt2-xl,you-static,0shot,0.312,300,0.292,0.023,-0.053,0.098,at
gpt2-xl,i-moves,0shot,0.236,180,0.287,-0.079,-0.165,0.011,at
gpt2-xl,i-moves,fair,0.281,180,0.287,-0.014,-0.104,0.078,at
gpt2-xl,you-moves,0shot,0.247,180,0.287,-0.056,-0.136,0.037,at
gpt2-xl,you-moves,fair,0.275,180,0.287,-0.017,-0.110,0.078,at
gpt2-xl,you-named,0shot,0.311,180,0.287,0.030,-0.065,0.126,at
gpt2-xl,you-named,fair,0.314,180,0.287,0.034,-0.055,0.132,at
gpt2-xl,answer-self,0shot,0.303,2560,0.250,0.071,0.048,0.094,above
gpt2-xl,answer-self,fair,0.264,2560,0.250,0.019,-0.004,0.039,at
gpt2-xl,grammar,ladder,,1080,,0.703,0.670,0.733,above
gpt2-xl,operators,ladder,,1080,,0.231,0.203,0.265,above
gpt2-xl,patterns,ladder,,1080,,0.417,0.386,0.451,above
gpt2-xl,facts,ladder,,768,,0.919,0.895,0.937,above
smollm2-1.7b,lookup,0shot,0.397,1210,0.169,0.263,0.229,0.296,above
smollm2-1.7b,lookup,fair,0.757,1210,0.169,0.715,0.686,0.742,above
smollm2-1.7b,shortlist,0shot,0.727,3840,0.040,0.715,0.701,0.730,above
smollm2-1.7b,pairs,0shot,0.325,256,0.125,0.228,0.164,0.293,above
smollm2-1.7b,pairs,fair,0.298,256,0.125,0.198,0.135,0.263,above
smollm2-1.7b,noise,0shot,0.631,720,0.200,0.538,0.494,0.579,above
smollm2-1.7b,now,0shot,0.670,300,0.292,0.538,0.470,0.610,above
smollm2-1.7b,handoffs,0shot,0.584,1200,0.312,0.387,0.346,0.426,above
smollm2-1.7b,handoffs,fair,0.683,1200,0.312,0.532,0.493,0.570,above
smollm2-1.7b,sets,0shot,0.419,1280,0.275,0.209,0.172,0.246,above
smollm2-1.7b,sets,fair,0.511,1280,0.275,0.340,0.300,0.378,above
smollm2-1.7b,refilled,0shot,0.407,720,0.200,0.259,0.214,0.302,above
smollm2-1.7b,refills,0shot,0.253,600,0.200,0.066,0.024,0.115,above
smollm2-1.7b,codevar,0shot,0.361,1200,0.267,0.124,0.088,0.163,above
smollm2-1.7b,codevar,fair,0.409,1200,0.267,0.187,0.151,0.226,above
smollm2-1.7b,sharedlists,0shot,0.287,600,0.200,0.109,0.065,0.150,above
smollm2-1.7b,sharedlists,fair,0.332,600,0.200,0.165,0.118,0.210,above
smollm2-1.7b,box-narrative,0shot,0.425,200,0.253,0.203,0.105,0.301,above
smollm2-1.7b,box-narrative,fair,0.492,200,0.253,0.313,0.214,0.408,above
smollm2-1.7b,box-screenplay,0shot,0.540,200,0.253,0.385,0.288,0.486,above
smollm2-1.7b,box-screenplay,fair,0.542,200,0.253,0.381,0.278,0.469,above
smollm2-1.7b,box-code,0shot,0.573,200,0.253,0.417,0.321,0.510,above
smollm2-1.7b,box-code,fair,0.647,200,0.253,0.507,0.406,0.600,above
smollm2-1.7b,box-chat,0shot,0.475,200,0.253,0.305,0.212,0.401,above
smollm2-1.7b,box-chat,fair,0.237,200,0.253,-0.018,-0.099,0.059,at
smollm2-1.7b,you-static,0shot,0.603,300,0.292,0.439,0.363,0.514,above
smollm2-1.7b,i-moves,0shot,0.308,180,0.287,0.025,-0.071,0.120,at
smollm2-1.7b,i-moves,fair,0.311,180,0.287,0.034,-0.063,0.125,at
smollm2-1.7b,you-moves,0shot,0.302,180,0.287,0.019,-0.071,0.111,at
smollm2-1.7b,you-moves,fair,0.300,180,0.287,0.018,-0.075,0.118,at
smollm2-1.7b,you-named,0shot,0.294,180,0.287,0.008,-0.078,0.104,at
smollm2-1.7b,you-named,fair,0.328,180,0.287,0.057,-0.034,0.147,at
smollm2-1.7b,answer-self,0shot,0.303,2560,0.250,0.071,0.048,0.094,above
smollm2-1.7b,answer-self,fair,0.277,2560,0.250,0.036,0.014,0.059,above
smollm2-1.7b,grammar,ladder,,1080,,0.756,0.735,0.780,above
smollm2-1.7b,operators,ladder,,1080,,0.243,0.212,0.275,above
smollm2-1.7b,patterns,ladder,,1080,,0.388,0.356,0.419,above
smollm2-1.7b,facts,ladder,,768,,1.000,1.000,1.000,above
qwen3-1.7b,lookup,0shot,0.711,1210,0.169,0.638,0.607,0.667,above
qwen3-1.7b,lookup,fair,0.998,1210,0.169,0.998,0.995,1.000,above
qwen3-1.7b,shortlist,0shot,0.827,3840,0.040,0.820,0.807,0.831,above
qwen3-1.7b,pairs,0shot,0.789,256,0.125,0.759,0.702,0.816,above
qwen3-1.7b,pairs,fair,0.801,256,0.125,0.772,0.721,0.824,above
qwen3-1.7b,noise,0shot,0.970,720,0.200,0.963,0.947,0.977,above
qwen3-1.7b,now,0shot,0.863,300,0.292,0.801,0.742,0.852,above
qwen3-1.7b,handoffs,0shot,0.686,1200,0.312,0.542,0.503,0.577,above
qwen3-1.7b,handoffs,fair,0.757,1200,0.312,0.643,0.606,0.677,above
qwen3-1.7b,sets,0shot,0.755,1280,0.275,0.672,0.641,0.702,above
qwen3-1.7b,sets,fair,0.771,1280,0.275,0.692,0.663,0.723,above
qwen3-1.7b,refilled,0shot,0.602,720,0.200,0.503,0.458,0.548,above
qwen3-1.7b,refills,0shot,0.363,600,0.200,0.204,0.156,0.251,above
qwen3-1.7b,codevar,0shot,0.475,1200,0.267,0.280,0.244,0.321,above
qwen3-1.7b,codevar,fair,0.535,1200,0.267,0.361,0.322,0.405,above
qwen3-1.7b,sharedlists,0shot,0.372,600,0.200,0.215,0.167,0.258,above
qwen3-1.7b,sharedlists,fair,0.501,600,0.200,0.376,0.329,0.425,above
qwen3-1.7b,box-narrative,0shot,0.625,200,0.253,0.497,0.408,0.585,above
qwen3-1.7b,box-narrative,fair,0.580,200,0.253,0.422,0.327,0.514,above
qwen3-1.7b,box-screenplay,0shot,0.657,200,0.253,0.550,0.464,0.633,above
qwen3-1.7b,box-screenplay,fair,0.565,200,0.253,0.412,0.317,0.500,above
qwen3-1.7b,box-code,0shot,0.600,200,0.253,0.445,0.347,0.536,above
qwen3-1.7b,box-code,fair,0.695,200,0.253,0.567,0.474,0.651,above
qwen3-1.7b,box-chat,0shot,0.667,200,0.253,0.552,0.462,0.639,above
qwen3-1.7b,box-chat,fair,0.570,200,0.253,0.426,0.330,0.522,above
qwen3-1.7b,you-static,0shot,0.692,300,0.292,0.549,0.475,0.618,above
qwen3-1.7b,i-moves,0shot,0.413,180,0.287,0.173,0.073,0.272,above
qwen3-1.7b,i-moves,fair,0.389,180,0.287,0.143,0.046,0.237,above
qwen3-1.7b,you-moves,0shot,0.276,180,0.287,-0.021,-0.115,0.073,at
qwen3-1.7b,you-moves,fair,0.325,180,0.287,0.056,-0.037,0.148,at
qwen3-1.7b,you-named,0shot,0.382,180,0.287,0.130,0.040,0.230,above
qwen3-1.7b,you-named,fair,0.402,180,0.287,0.159,0.067,0.255,above
qwen3-1.7b,answer-self,0shot,0.402,2560,0.250,0.202,0.177,0.227,above
qwen3-1.7b,answer-self,fair,0.381,2560,0.250,0.175,0.150,0.199,above
qwen3-1.7b,grammar,ladder,,1080,,0.812,0.793,0.835,above
qwen3-1.7b,operators,ladder,,1080,,0.265,0.234,0.296,above
qwen3-1.7b,patterns,ladder,,1080,,0.561,0.528,0.594,above
qwen3-1.7b,facts,ladder,,768,,0.994,0.986,1.000,above
gemma-2b,lookup,0shot,0.574,1210,0.169,0.477,0.443,0.510,above
gemma-2b,lookup,fair,0.965,1210,0.169,0.957,0.945,0.970,above
gemma-2b,shortlist,0shot,0.800,3840,0.040,0.791,0.778,0.804,above
gemma-2b,pairs,0shot,0.670,256,0.125,0.623,0.562,0.685,above
gemma-2b,pairs,fair,0.590,256,0.125,0.531,0.464,0.598,above
gemma-2b,noise,0shot,0.807,720,0.200,0.759,0.723,0.793,above
gemma-2b,now,0shot,0.580,300,0.292,0.406,0.324,0.482,above
gemma-2b,handoffs,0shot,0.564,1200,0.312,0.364,0.326,0.403,above
gemma-2b,handoffs,fair,0.624,1200,0.312,0.450,0.413,0.488,above
gemma-2b,sets,0shot,0.573,1280,0.275,0.416,0.379,0.452,above
gemma-2b,sets,fair,0.707,1280,0.275,0.609,0.577,0.643,above
gemma-2b,refilled,0shot,0.443,720,0.200,0.304,0.260,0.347,above
gemma-2b,refills,0shot,0.447,600,0.200,0.309,0.260,0.357,above
gemma-2b,codevar,0shot,0.403,1200,0.267,0.183,0.146,0.220,above
gemma-2b,codevar,fair,0.416,1200,0.267,0.197,0.159,0.236,above
gemma-2b,sharedlists,0shot,0.330,600,0.200,0.163,0.117,0.207,above
gemma-2b,sharedlists,fair,0.417,600,0.200,0.272,0.222,0.321,above
gemma-2b,box-narrative,0shot,0.435,200,0.253,0.236,0.137,0.332,above
gemma-2b,box-narrative,fair,0.453,200,0.253,0.251,0.155,0.347,above
gemma-2b,box-screenplay,0shot,0.583,200,0.253,0.436,0.340,0.529,above
gemma-2b,box-screenplay,fair,0.497,200,0.253,0.303,0.204,0.396,above
gemma-2b,box-code,0shot,0.542,200,0.253,0.374,0.271,0.465,above
gemma-2b,box-code,fair,0.647,200,0.253,0.503,0.407,0.595,above
gemma-2b,box-chat,0shot,0.590,200,0.253,0.450,0.361,0.535,above
gemma-2b,box-chat,fair,0.615,200,0.253,0.483,0.389,0.569,above
gemma-2b,you-static,0shot,0.583,300,0.292,0.406,0.327,0.485,above
gemma-2b,i-moves,0shot,0.389,180,0.287,0.139,0.041,0.241,above
gemma-2b,i-moves,fair,0.347,180,0.287,0.082,-0.019,0.177,at
gemma-2b,you-moves,0shot,0.342,180,0.287,0.075,-0.020,0.167,at
gemma-2b,you-moves,fair,0.275,180,0.287,-0.016,-0.110,0.070,at
gemma-2b,you-named,0shot,0.280,180,0.287,-0.013,-0.103,0.076,at
gemma-2b,you-named,fair,0.347,180,0.287,0.081,-0.013,0.181,at
gemma-2b,answer-self,0shot,0.348,2560,0.250,0.131,0.106,0.156,above
gemma-2b,answer-self,fair,0.298,2560,0.250,0.064,0.042,0.086,above
gemma-2b,grammar,ladder,,1080,,0.715,0.688,0.743,above
gemma-2b,operators,ladder,,1080,,0.221,0.187,0.259,above
gemma-2b,patterns,ladder,,1080,,0.523,0.488,0.555,above
gemma-2b,facts,ladder,,768,,1.000,1.000,1.000,above
gemma2-2b,lookup,0shot,0.524,1210,0.169,0.411,0.377,0.443,above
gemma2-2b,lookup,fair,0.992,1210,0.169,0.990,0.984,0.996,above
gemma2-2b,shortlist,0shot,0.753,3840,0.040,0.743,0.730,0.756,above
gemma2-2b,pairs,0shot,0.585,256,0.125,0.525,0.452,0.592,above
gemma2-2b,pairs,fair,0.784,256,0.125,0.754,0.697,0.808,above
gemma2-2b,noise,0shot,0.807,720,0.200,0.759,0.724,0.792,above
gemma2-2b,now,0shot,0.841,300,0.292,0.775,0.720,0.828,above
gemma2-2b,handoffs,0shot,0.686,1200,0.312,0.539,0.504,0.576,above
gemma2-2b,handoffs,fair,0.720,1200,0.312,0.589,0.555,0.624,above
gemma2-2b,sets,0shot,0.607,1280,0.275,0.458,0.423,0.492,above
gemma2-2b,sets,fair,0.718,1280,0.275,0.620,0.586,0.653,above
gemma2-2b,refilled,0shot,0.555,720,0.200,0.444,0.398,0.490,above
gemma2-2b,refills,0shot,0.491,600,0.200,0.364,0.312,0.414,above
gemma2-2b,codevar,0shot,0.431,1200,0.267,0.221,0.183,0.260,above
gemma2-2b,codevar,fair,0.414,1200,0.267,0.195,0.159,0.231,above
gemma2-2b,sharedlists,0shot,0.415,600,0.200,0.268,0.223,0.319,above
gemma2-2b,sharedlists,fair,0.489,600,0.200,0.361,0.315,0.406,above
gemma2-2b,box-narrative,0shot,0.537,200,0.253,0.387,0.291,0.473,above
gemma2-2b,box-narrative,fair,0.542,200,0.253,0.369,0.279,0.467,above
gemma2-2b,box-screenplay,0shot,0.596,200,0.253,0.465,0.377,0.544,above
gemma2-2b,box-screenplay,fair,0.552,200,0.253,0.385,0.289,0.481,above
gemma2-2b,box-code,0shot,0.627,200,0.253,0.480,0.379,0.577,above
gemma2-2b,box-code,fair,0.715,200,0.253,0.603,0.507,0.691,above
gemma2-2b,box-chat,0shot,0.627,200,0.253,0.501,0.404,0.587,above
gemma2-2b,box-chat,fair,0.629,200,0.253,0.496,0.400,0.579,above
gemma2-2b,you-static,0shot,0.736,300,0.292,0.621,0.546,0.682,above
gemma2-2b,i-moves,0shot,0.391,180,0.287,0.144,0.045,0.233,above
gemma2-2b,i-moves,fair,0.382,180,0.287,0.125,0.016,0.219,above
gemma2-2b,you-moves,0shot,0.367,180,0.287,0.110,0.013,0.209,above
gemma2-2b,you-moves,fair,0.328,180,0.287,0.060,-0.029,0.148,at
gemma2-2b,you-named,0shot,0.332,180,0.287,0.062,-0.036,0.161,at
gemma2-2b,you-named,fair,0.340,180,0.287,0.073,-0.017,0.169,at
gemma2-2b,answer-self,0shot,0.394,2560,0.250,0.192,0.166,0.216,above
gemma2-2b,answer-self,fair,0.368,2560,0.250,0.157,0.131,0.179,above
gemma2-2b,grammar,ladder,,1080,,0.801,0.777,0.829,above
gemma2-2b,operators,ladder,,1080,,0.253,0.222,0.285,above
gemma2-2b,patterns,ladder,,1080,,0.515,0.481,0.542,above
gemma2-2b,facts,ladder,,768,,1.000,1.000,1.000,above
pythia-2.8b,lookup,0shot,0.609,1210,0.169,0.519,0.487,0.552,above
pythia-2.8b,lookup,fair,0.860,1210,0.169,0.829,0.804,0.852,above
pythia-2.8b,shortlist,0shot,0.555,3840,0.040,0.536,0.521,0.552,above
pythia-2.8b,pairs,0shot,0.659,256,0.125,0.611,0.548,0.674,above
pythia-2.8b,pairs,fair,0.611,256,0.125,0.556,0.489,0.623,above
pythia-2.8b,noise,0shot,0.643,720,0.200,0.554,0.509,0.596,above
pythia-2.8b,now,0shot,0.672,300,0.292,0.533,0.456,0.605,above
pythia-2.8b,handoffs,0shot,0.518,1200,0.312,0.294,0.251,0.335,above
pythia-2.8b,handoffs,fair,0.527,1200,0.312,0.308,0.269,0.348,above
pythia-2.8b,sets,0shot,0.510,1280,0.275,0.323,0.283,0.361,above
pythia-2.8b,sets,fair,0.552,1280,0.275,0.389,0.351,0.428,above
pythia-2.8b,refilled,0shot,0.412,720,0.200,0.266,0.223,0.314,above
pythia-2.8b,refills,0shot,0.228,600,0.200,0.034,-0.008,0.076,at
pythia-2.8b,codevar,0shot,0.332,1200,0.267,0.084,0.047,0.120,above
pythia-2.8b,codevar,fair,0.285,1200,0.267,0.022,-0.014,0.057,at
pythia-2.8b,sharedlists,0shot,0.309,600,0.200,0.137,0.094,0.185,above
pythia-2.8b,sharedlists,fair,0.287,600,0.200,0.108,0.064,0.152,above
pythia-2.8b,box-narrative,0shot,0.490,200,0.253,0.310,0.214,0.406,above
pythia-2.8b,box-narrative,fair,0.562,200,0.253,0.406,0.312,0.496,above
pythia-2.8b,box-screenplay,0shot,0.497,200,0.253,0.341,0.249,0.435,above
pythia-2.8b,box-screenplay,fair,0.557,200,0.253,0.401,0.305,0.489,above
pythia-2.8b,box-code,0shot,0.573,200,0.253,0.422,0.327,0.513,above
pythia-2.8b,box-code,fair,0.605,200,0.253,0.455,0.357,0.550,above
pythia-2.8b,box-chat,0shot,0.557,200,0.253,0.401,0.314,0.494,above
pythia-2.8b,box-chat,fair,0.568,200,0.253,0.409,0.316,0.505,above
pythia-2.8b,you-static,0shot,0.388,300,0.292,0.133,0.064,0.209,above
pythia-2.8b,i-moves,0shot,0.300,180,0.287,0.009,-0.081,0.099,at
pythia-2.8b,i-moves,fair,0.281,180,0.287,-0.012,-0.105,0.083,at
pythia-2.8b,you-moves,0shot,0.294,180,0.287,0.013,-0.075,0.109,at
pythia-2.8b,you-moves,fair,0.380,180,0.287,0.138,0.046,0.232,above
pythia-2.8b,you-named,0shot,0.322,180,0.287,0.050,-0.051,0.146,at
pythia-2.8b,you-named,fair,0.311,180,0.287,0.032,-0.066,0.129,at
pythia-2.8b,answer-self,0shot,0.300,2560,0.250,0.067,0.043,0.090,above
pythia-2.8b,answer-self,fair,0.279,2560,0.250,0.038,0.016,0.062,above
pythia-2.8b,grammar,ladder,,1080,,0.793,0.768,0.817,above
pythia-2.8b,operators,ladder,,1080,,0.245,0.217,0.278,above
pythia-2.8b,patterns,ladder,,1080,,0.363,0.327,0.393,above
pythia-2.8b,facts,ladder,,768,,0.984,0.971,0.991,above
pythia-2.8b-deduped,lookup,0shot,0.614,1210,0.169,0.525,0.491,0.558,above
pythia-2.8b-deduped,lookup,fair,0.882,1210,0.169,0.856,0.834,0.878,above
pythia-2.8b-deduped,shortlist,0shot,0.576,3840,0.040,0.558,0.543,0.574,above
pythia-2.8b-deduped,pairs,0shot,0.648,256,0.125,0.598,0.531,0.662,above
pythia-2.8b-deduped,pairs,fair,0.637,256,0.125,0.585,0.516,0.652,above
pythia-2.8b-deduped,noise,0shot,0.652,720,0.200,0.565,0.520,0.605,above
pythia-2.8b-deduped,now,0shot,0.688,300,0.292,0.556,0.483,0.628,above
pythia-2.8b-deduped,handoffs,0shot,0.512,1200,0.312,0.286,0.245,0.326,above
pythia-2.8b-deduped,handoffs,fair,0.548,1200,0.312,0.338,0.299,0.380,above
pythia-2.8b-deduped,sets,0shot,0.505,1280,0.275,0.317,0.278,0.354,above
pythia-2.8b-deduped,sets,fair,0.565,1280,0.275,0.409,0.371,0.446,above
pythia-2.8b-deduped,refilled,0shot,0.412,720,0.200,0.266,0.223,0.313,above
pythia-2.8b-deduped,refills,0shot,0.223,600,0.200,0.028,-0.011,0.069,at
pythia-2.8b-deduped,codevar,0shot,0.347,1200,0.267,0.104,0.068,0.141,above
pythia-2.8b-deduped,codevar,fair,0.267,1200,0.267,-0.003,-0.036,0.032,at
pythia-2.8b-deduped,sharedlists,0shot,0.279,600,0.200,0.099,0.052,0.144,above
pythia-2.8b-deduped,sharedlists,fair,0.291,600,0.200,0.114,0.069,0.156,above
pythia-2.8b-deduped,box-narrative,0shot,0.535,200,0.253,0.373,0.275,0.465,above
pythia-2.8b-deduped,box-narrative,fair,0.585,200,0.253,0.434,0.336,0.522,above
pythia-2.8b-deduped,box-screenplay,0shot,0.542,200,0.253,0.395,0.306,0.492,above
pythia-2.8b-deduped,box-screenplay,fair,0.555,200,0.253,0.402,0.308,0.490,above
pythia-2.8b-deduped,box-code,0shot,0.550,200,0.253,0.393,0.296,0.486,above
pythia-2.8b-deduped,box-code,fair,0.578,200,0.253,0.418,0.325,0.512,above
pythia-2.8b-deduped,box-chat,0shot,0.593,200,0.253,0.457,0.375,0.544,above
pythia-2.8b-deduped,box-chat,fair,0.610,200,0.253,0.468,0.376,0.557,above
pythia-2.8b-deduped,you-static,0shot,0.407,300,0.292,0.160,0.086,0.238,above
pythia-2.8b-deduped,i-moves,0shot,0.328,180,0.287,0.051,-0.041,0.146,at
pythia-2.8b-deduped,i-moves,fair,0.291,180,0.287,0.000,-0.088,0.100,at
pythia-2.8b-deduped,you-moves,0shot,0.294,180,0.287,0.012,-0.079,0.111,at
pythia-2.8b-deduped,you-moves,fair,0.381,180,0.287,0.135,0.039,0.239,above
pythia-2.8b-deduped,you-named,0shot,0.300,180,0.287,0.018,-0.069,0.112,at
pythia-2.8b-deduped,you-named,fair,0.324,180,0.287,0.051,-0.042,0.143,at
pythia-2.8b-deduped,answer-self,0shot,0.307,2560,0.250,0.076,0.054,0.099,above
pythia-2.8b-deduped,answer-self,fair,0.270,2560,0.250,0.026,0.003,0.050,above
pythia-2.8b-deduped,grammar,ladder,,1080,,0.788,0.763,0.813,above
pythia-2.8b-deduped,operators,ladder,,1080,,0.261,0.232,0.293,above
pythia-2.8b-deduped,patterns,ladder,,1080,,0.377,0.339,0.408,above
pythia-2.8b-deduped,facts,ladder,,768,,0.985,0.973,0.994,above
qwen2.5-3b,lookup,0shot,0.725,1210,0.169,0.651,0.618,0.682,above
qwen2.5-3b,lookup,fair,0.997,1210,0.169,0.996,0.992,1.000,above
qwen2.5-3b,shortlist,0shot,0.758,3840,0.040,0.748,0.734,0.762,above
qwen2.5-3b,pairs,0shot,0.646,256,0.125,0.595,0.533,0.662,above
qwen2.5-3b,pairs,fair,0.846,256,0.125,0.824,0.772,0.868,above
qwen2.5-3b,noise,0shot,0.897,720,0.200,0.872,0.844,0.897,above
qwen2.5-3b,now,0shot,0.895,300,0.292,0.852,0.805,0.893,above
qwen2.5-3b,handoffs,0shot,0.678,1200,0.312,0.529,0.488,0.566,above
qwen2.5-3b,handoffs,fair,0.762,1200,0.312,0.649,0.615,0.685,above
qwen2.5-3b,sets,0shot,0.762,1280,0.275,0.681,0.649,0.713,above
qwen2.5-3b,sets,fair,0.752,1280,0.275,0.668,0.636,0.700,above
qwen2.5-3b,refilled,0shot,0.415,720,0.200,0.268,0.226,0.313,above
qwen2.5-3b,refills,0shot,0.303,600,0.200,0.129,0.085,0.173,above
qwen2.5-3b,codevar,0shot,0.446,1200,0.267,0.239,0.203,0.277,above
qwen2.5-3b,codevar,fair,0.469,1200,0.267,0.267,0.229,0.308,above
qwen2.5-3b,sharedlists,0shot,0.371,600,0.200,0.214,0.169,0.260,above
qwen2.5-3b,sharedlists,fair,0.438,600,0.200,0.297,0.250,0.345,above
qwen2.5-3b,box-narrative,0shot,0.658,200,0.253,0.540,0.454,0.627,above
qwen2.5-3b,box-narrative,fair,0.700,200,0.253,0.581,0.492,0.669,above
qwen2.5-3b,box-screenplay,0shot,0.613,200,0.253,0.468,0.372,0.565,above
qwen2.5-3b,box-screenplay,fair,0.680,200,0.253,0.564,0.472,0.652,above
qwen2.5-3b,box-code,0shot,0.623,200,0.253,0.466,0.356,0.559,above
qwen2.5-3b,box-code,fair,0.665,200,0.253,0.526,0.433,0.616,above
qwen2.5-3b,box-chat,0shot,0.618,200,0.253,0.485,0.389,0.575,above
qwen2.5-3b,box-chat,fair,0.708,200,0.253,0.609,0.527,0.690,above
qwen2.5-3b,you-static,0shot,0.858,300,0.292,0.795,0.741,0.846,above
qwen2.5-3b,i-moves,0shot,0.363,180,0.287,0.104,0.015,0.205,above
qwen2.5-3b,i-moves,fair,0.394,180,0.287,0.149,0.049,0.241,above
qwen2.5-3b,you-moves,0shot,0.280,180,0.287,-0.017,-0.104,0.078,at
qwen2.5-3b,you-moves,fair,0.264,180,0.287,-0.033,-0.118,0.055,at
qwen2.5-3b,you-named,0shot,0.237,180,0.287,-0.069,-0.151,0.014,at
qwen2.5-3b,you-named,fair,0.328,180,0.287,0.060,-0.029,0.161,at
qwen2.5-3b,answer-self,0shot,0.407,2560,0.250,0.209,0.186,0.234,above
qwen2.5-3b,answer-self,fair,0.350,2560,0.250,0.133,0.108,0.158,above
qwen2.5-3b,grammar,ladder,,1080,,0.800,0.778,0.825,above
qwen2.5-3b,operators,ladder,,1080,,0.173,0.144,0.203,above
qwen2.5-3b,patterns,ladder,,1080,,0.686,0.654,0.724,above
qwen2.5-3b,facts,ladder,,768,,0.998,0.995,1.000,above
falcon-h1-3b,lookup,0shot,0.960,1210,0.169,0.950,0.934,0.963,above
falcon-h1-3b,lookup,fair,0.999,1210,0.169,0.999,0.997,1.000,above
falcon-h1-3b,shortlist,0shot,0.948,3840,0.040,0.946,0.939,0.953,above
falcon-h1-3b,pairs,0shot,0.801,256,0.125,0.772,0.719,0.826,above
falcon-h1-3b,pairs,fair,0.871,256,0.125,0.853,0.799,0.897,above
falcon-h1-3b,noise,0shot,0.887,720,0.200,0.859,0.830,0.887,above
falcon-h1-3b,now,0shot,0.867,300,0.292,0.813,0.759,0.864,above
falcon-h1-3b,handoffs,0shot,0.759,1200,0.312,0.647,0.612,0.680,above
falcon-h1-3b,handoffs,fair,0.848,1200,0.312,0.778,0.750,0.808,above
falcon-h1-3b,sets,0shot,0.864,1280,0.275,0.816,0.791,0.842,above
falcon-h1-3b,sets,fair,0.850,1280,0.275,0.803,0.775,0.828,above
falcon-h1-3b,refilled,0shot,0.787,720,0.200,0.734,0.698,0.773,above
falcon-h1-3b,refills,0shot,0.470,600,0.200,0.338,0.291,0.391,above
falcon-h1-3b,codevar,0shot,0.548,1200,0.267,0.376,0.338,0.416,above
falcon-h1-3b,codevar,fair,0.683,1200,0.267,0.562,0.524,0.599,above
falcon-h1-3b,sharedlists,0shot,0.548,600,0.200,0.435,0.383,0.483,above
falcon-h1-3b,sharedlists,fair,0.551,600,0.200,0.439,0.391,0.485,above
falcon-h1-3b,box-narrative,0shot,0.640,200,0.253,0.497,0.399,0.587,above
falcon-h1-3b,box-narrative,fair,0.665,200,0.253,0.548,0.456,0.634,above
falcon-h1-3b,box-screenplay,0shot,0.632,200,0.253,0.488,0.394,0.587,above
falcon-h1-3b,box-screenplay,fair,0.715,200,0.253,0.614,0.521,0.698,above
falcon-h1-3b,box-code,0shot,0.545,200,0.253,0.389,0.294,0.485,above
falcon-h1-3b,box-code,fair,0.765,200,0.253,0.679,0.594,0.756,above
falcon-h1-3b,box-chat,0shot,0.675,200,0.253,0.558,0.464,0.645,above
falcon-h1-3b,box-chat,fair,0.743,200,0.253,0.658,0.575,0.741,above
falcon-h1-3b,you-static,0shot,0.920,300,0.292,0.884,0.840,0.927,above
falcon-h1-3b,i-moves,0shot,0.394,180,0.287,0.148,0.049,0.250,above
falcon-h1-3b,i-moves,fair,0.400,180,0.287,0.157,0.056,0.247,above
falcon-h1-3b,you-moves,0shot,0.383,180,0.287,0.138,0.036,0.237,above
falcon-h1-3b,you-moves,fair,0.278,180,0.287,-0.013,-0.102,0.077,at
falcon-h1-3b,you-named,0shot,0.333,180,0.287,0.062,-0.026,0.163,at
falcon-h1-3b,you-named,fair,0.333,180,0.287,0.065,-0.032,0.158,at
falcon-h1-3b,answer-self,0shot,0.425,2560,0.250,0.234,0.208,0.259,above
falcon-h1-3b,answer-self,fair,0.440,2560,0.250,0.253,0.226,0.277,above
falcon-h1-3b,grammar,ladder,,1080,,0.833,0.811,0.854,above
falcon-h1-3b,operators,ladder,,1080,,0.259,0.231,0.292,above
falcon-h1-3b,patterns,ladder,,1080,,0.535,0.502,0.565,above
falcon-h1-3b,facts,ladder,,768,,0.991,0.986,0.996,above
falcon3-3b,lookup,0shot,0.549,1210,0.169,0.449,0.416,0.478,above
falcon3-3b,lookup,fair,0.970,1210,0.169,0.964,0.951,0.975,above
falcon3-3b,shortlist,0shot,0.650,3840,0.040,0.636,0.621,0.652,above
falcon3-3b,pairs,0shot,0.461,256,0.125,0.384,0.308,0.449,above
falcon3-3b,pairs,fair,0.543,256,0.125,0.478,0.406,0.542,above
falcon3-3b,noise,0shot,0.812,720,0.200,0.766,0.729,0.802,above
falcon3-3b,now,0shot,0.825,300,0.292,0.753,0.688,0.812,above
falcon3-3b,handoffs,0shot,0.670,1200,0.312,0.516,0.479,0.553,above
falcon3-3b,handoffs,fair,0.761,1200,0.312,0.650,0.613,0.685,above
falcon3-3b,sets,0shot,0.485,1280,0.275,0.298,0.263,0.337,above
falcon3-3b,sets,fair,0.599,1280,0.275,0.467,0.431,0.507,above
falcon3-3b,refilled,0shot,0.537,720,0.200,0.421,0.374,0.466,above
falcon3-3b,refills,0shot,0.350,600,0.200,0.187,0.142,0.234,above
falcon3-3b,codevar,0shot,0.433,1200,0.267,0.219,0.180,0.258,above
falcon3-3b,codevar,fair,0.579,1200,0.267,0.421,0.382,0.459,above
falcon3-3b,sharedlists,0shot,0.357,600,0.200,0.196,0.147,0.242,above
falcon3-3b,sharedlists,fair,0.578,600,0.200,0.472,0.422,0.523,above
falcon3-3b,box-narrative,0shot,0.542,200,0.253,0.386,0.289,0.484,above
falcon3-3b,box-narrative,fair,0.580,200,0.253,0.438,0.342,0.528,above
falcon3-3b,box-screenplay,0shot,0.570,200,0.253,0.425,0.325,0.521,above
falcon3-3b,box-screenplay,fair,0.623,200,0.253,0.494,0.395,0.583,above
falcon3-3b,box-code,0shot,0.415,200,0.253,0.226,0.131,0.317,above
falcon3-3b,box-code,fair,0.735,200,0.253,0.623,0.531,0.707,above
falcon3-3b,box-chat,0shot,0.608,200,0.253,0.472,0.379,0.567,above
falcon3-3b,box-chat,fair,0.700,200,0.253,0.603,0.512,0.686,above
falcon3-3b,you-static,0shot,0.773,300,0.292,0.674,0.601,0.735,above
falcon3-3b,i-moves,0shot,0.292,180,0.287,0.001,-0.090,0.087,at
falcon3-3b,i-moves,fair,0.338,180,0.287,0.070,-0.015,0.165,at
falcon3-3b,you-moves,0shot,0.303,180,0.287,0.020,-0.072,0.116,at
falcon3-3b,you-moves,fair,0.333,180,0.287,0.065,-0.028,0.158,at
falcon3-3b,you-named,0shot,0.333,180,0.287,0.062,-0.036,0.164,at
falcon3-3b,you-named,fair,0.372,180,0.287,0.116,0.021,0.211,above
falcon3-3b,answer-self,0shot,0.338,2560,0.250,0.117,0.090,0.141,above
falcon3-3b,answer-self,fair,0.348,2560,0.250,0.131,0.107,0.154,above
falcon3-3b,grammar,ladder,,1080,,0.855,0.835,0.877,above
falcon3-3b,operators,ladder,,1080,,0.266,0.236,0.296,above
falcon3-3b,patterns,ladder,,1080,,0.511,0.476,0.547,above
falcon3-3b,facts,ladder,,768,,0.972,0.956,0.986,above
qwen3-30b-a3b,lookup,0shot,0.847,1210,0.169,0.807,0.781,0.833,above
qwen3-30b-a3b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
qwen3-30b-a3b,shortlist,0shot,0.902,3840,0.040,0.898,0.889,0.907,above
qwen3-30b-a3b,pairs,0shot,0.673,256,0.125,0.626,0.564,0.687,above
qwen3-30b-a3b,pairs,fair,0.963,256,0.125,0.958,0.931,0.980,above
qwen3-30b-a3b,noise,0shot,0.966,720,0.200,0.958,0.941,0.972,above
qwen3-30b-a3b,now,0shot,0.750,300,0.292,0.642,0.569,0.709,above
qwen3-30b-a3b,handoffs,0shot,0.817,1200,0.312,0.731,0.701,0.761,above
qwen3-30b-a3b,handoffs,fair,0.909,1200,0.312,0.866,0.842,0.888,above
qwen3-30b-a3b,sets,0shot,0.806,1280,0.275,0.730,0.700,0.758,above
qwen3-30b-a3b,sets,fair,0.850,1280,0.275,0.799,0.772,0.824,above
qwen3-30b-a3b,refilled,0shot,0.607,720,0.200,0.509,0.467,0.553,above
qwen3-30b-a3b,refills,0shot,0.461,600,0.200,0.327,0.278,0.376,above
qwen3-30b-a3b,codevar,0shot,0.757,1200,0.267,0.665,0.632,0.697,above
qwen3-30b-a3b,codevar,fair,0.832,1200,0.267,0.765,0.735,0.794,above
qwen3-30b-a3b,sharedlists,0shot,0.775,600,0.200,0.719,0.676,0.759,above
qwen3-30b-a3b,sharedlists,fair,0.794,600,0.200,0.743,0.702,0.781,above
qwen3-30b-a3b,box-narrative,0shot,0.695,200,0.253,0.585,0.500,0.670,above
qwen3-30b-a3b,box-narrative,fair,0.717,200,0.253,0.608,0.521,0.689,above
qwen3-30b-a3b,box-screenplay,0shot,0.670,200,0.253,0.548,0.458,0.636,above
qwen3-30b-a3b,box-screenplay,fair,0.697,200,0.253,0.589,0.503,0.670,above
qwen3-30b-a3b,box-code,0shot,0.733,200,0.253,0.624,0.527,0.707,above
qwen3-30b-a3b,box-code,fair,0.855,200,0.253,0.803,0.735,0.863,above
qwen3-30b-a3b,box-chat,0shot,0.713,200,0.253,0.601,0.512,0.686,above
qwen3-30b-a3b,box-chat,fair,0.708,200,0.253,0.603,0.518,0.692,above
qwen3-30b-a3b,you-static,0shot,0.922,300,0.292,0.885,0.838,0.926,above
qwen3-30b-a3b,i-moves,0shot,0.414,180,0.287,0.177,0.071,0.274,above
qwen3-30b-a3b,i-moves,fair,0.458,180,0.287,0.235,0.133,0.337,above
qwen3-30b-a3b,you-moves,0shot,0.436,180,0.287,0.212,0.112,0.321,above
qwen3-30b-a3b,you-moves,fair,0.375,180,0.287,0.122,0.029,0.230,above
qwen3-30b-a3b,you-named,0shot,0.410,180,0.287,0.174,0.077,0.268,above
qwen3-30b-a3b,you-named,fair,0.394,180,0.287,0.146,0.046,0.248,above
qwen3-30b-a3b,answer-self,0shot,0.519,2560,0.250,0.358,0.333,0.385,above
qwen3-30b-a3b,answer-self,fair,0.558,2560,0.250,0.411,0.385,0.437,above
qwen3-30b-a3b,grammar,ladder,,1080,,0.850,0.827,0.868,above
qwen3-30b-a3b,operators,ladder,,1080,,0.202,0.168,0.231,above
qwen3-30b-a3b,patterns,ladder,,1080,,0.934,0.921,0.949,above
qwen3-30b-a3b,facts,ladder,,768,,1.000,1.000,1.000,above
qwen3-4b,lookup,0shot,0.963,1210,0.169,0.950,0.936,0.963,above
qwen3-4b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
qwen3-4b,shortlist,0shot,0.916,3840,0.040,0.913,0.904,0.922,above
qwen3-4b,pairs,0shot,0.713,256,0.125,0.672,0.607,0.732,above
qwen3-4b,pairs,fair,0.971,256,0.125,0.967,0.942,0.987,above
qwen3-4b,noise,0shot,0.973,720,0.200,0.966,0.951,0.980,above
qwen3-4b,now,0shot,0.935,300,0.292,0.904,0.862,0.939,above
qwen3-4b,handoffs,0shot,0.858,1200,0.312,0.791,0.763,0.817,above
qwen3-4b,handoffs,fair,0.870,1200,0.312,0.809,0.782,0.836,above
qwen3-4b,sets,0shot,0.862,1280,0.275,0.809,0.781,0.836,above
qwen3-4b,sets,fair,0.850,1280,0.275,0.799,0.772,0.825,above
qwen3-4b,refilled,0shot,0.590,720,0.200,0.487,0.439,0.531,above
qwen3-4b,refills,0shot,0.505,600,0.200,0.382,0.333,0.429,above
qwen3-4b,codevar,0shot,0.619,1200,0.267,0.477,0.438,0.515,above
qwen3-4b,codevar,fair,0.635,1200,0.267,0.496,0.457,0.533,above
qwen3-4b,sharedlists,0shot,0.529,600,0.200,0.411,0.363,0.458,above
qwen3-4b,sharedlists,fair,0.694,600,0.200,0.618,0.572,0.660,above
qwen3-4b,box-narrative,0shot,0.703,200,0.253,0.589,0.499,0.676,above
qwen3-4b,box-narrative,fair,0.662,200,0.253,0.548,0.457,0.638,above
qwen3-4b,box-screenplay,0shot,0.640,200,0.253,0.513,0.428,0.605,above
qwen3-4b,box-screenplay,fair,0.613,200,0.253,0.475,0.379,0.570,above
qwen3-4b,box-code,0shot,0.723,200,0.253,0.601,0.513,0.691,above
qwen3-4b,box-code,fair,0.802,200,0.253,0.720,0.637,0.793,above
qwen3-4b,box-chat,0shot,0.662,200,0.253,0.540,0.452,0.628,above
qwen3-4b,box-chat,fair,0.690,200,0.253,0.585,0.501,0.675,above
qwen3-4b,you-static,0shot,0.768,300,0.292,0.657,0.588,0.724,above
qwen3-4b,i-moves,0shot,0.442,180,0.287,0.216,0.115,0.312,above
qwen3-4b,i-moves,fair,0.450,180,0.287,0.226,0.128,0.324,above
qwen3-4b,you-moves,0shot,0.301,180,0.287,0.015,-0.076,0.112,at
qwen3-4b,you-moves,fair,0.347,180,0.287,0.080,-0.018,0.177,at
qwen3-4b,you-named,0shot,0.386,180,0.287,0.138,0.045,0.232,above
qwen3-4b,you-named,fair,0.300,180,0.287,0.018,-0.075,0.109,at
qwen3-4b,answer-self,0shot,0.481,2560,0.250,0.308,0.281,0.334,above
qwen3-4b,answer-self,fair,0.564,2560,0.250,0.419,0.392,0.442,above
qwen3-4b,grammar,ladder,,1080,,0.842,0.821,0.864,above
qwen3-4b,operators,ladder,,1080,,0.245,0.217,0.275,above
qwen3-4b,patterns,ladder,,1080,,0.747,0.723,0.777,above
qwen3-4b,facts,ladder,,768,,0.998,0.995,1.000,above
pythia-6.9b,lookup,0shot,0.528,1210,0.169,0.422,0.387,0.455,above
pythia-6.9b,lookup,fair,0.838,1210,0.169,0.805,0.778,0.830,above
pythia-6.9b,shortlist,0shot,0.603,3840,0.040,0.586,0.570,0.603,above
pythia-6.9b,pairs,0shot,0.582,256,0.125,0.522,0.453,0.592,above
pythia-6.9b,pairs,fair,0.644,256,0.125,0.593,0.533,0.655,above
pythia-6.9b,noise,0shot,0.698,720,0.200,0.622,0.579,0.663,above
pythia-6.9b,now,0shot,0.648,300,0.292,0.507,0.427,0.584,above
pythia-6.9b,handoffs,0shot,0.493,1200,0.312,0.261,0.224,0.302,above
pythia-6.9b,handoffs,fair,0.533,1200,0.312,0.319,0.281,0.359,above
pythia-6.9b,sets,0shot,0.542,1280,0.275,0.370,0.330,0.406,above
pythia-6.9b,sets,fair,0.637,1280,0.275,0.509,0.475,0.545,above
pythia-6.9b,refilled,0shot,0.414,720,0.200,0.268,0.225,0.314,above
pythia-6.9b,refills,0shot,0.258,600,0.200,0.073,0.031,0.118,above
pythia-6.9b,codevar,0shot,0.360,1200,0.267,0.120,0.084,0.157,above
pythia-6.9b,codevar,fair,0.320,1200,0.267,0.065,0.031,0.101,above
pythia-6.9b,sharedlists,0shot,0.279,600,0.200,0.099,0.056,0.146,above
pythia-6.9b,sharedlists,fair,0.212,600,0.200,0.016,-0.023,0.053,at
pythia-6.9b,box-narrative,0shot,0.438,200,0.253,0.251,0.159,0.345,above
pythia-6.9b,box-narrative,fair,0.525,200,0.253,0.358,0.260,0.449,above
pythia-6.9b,box-screenplay,0shot,0.530,200,0.253,0.366,0.270,0.457,above
pythia-6.9b,box-screenplay,fair,0.545,200,0.253,0.381,0.280,0.473,above
pythia-6.9b,box-code,0shot,0.453,200,0.253,0.269,0.174,0.348,above
pythia-6.9b,box-code,fair,0.560,200,0.253,0.390,0.294,0.484,above
pythia-6.9b,box-chat,0shot,0.542,200,0.253,0.378,0.279,0.477,above
pythia-6.9b,box-chat,fair,0.525,200,0.253,0.350,0.251,0.444,above
pythia-6.9b,you-static,0shot,0.405,300,0.292,0.157,0.081,0.237,above
pythia-6.9b,i-moves,0shot,0.307,180,0.287,0.019,-0.071,0.123,at
pythia-6.9b,i-moves,fair,0.278,180,0.287,-0.016,-0.106,0.072,at
pythia-6.9b,you-moves,0shot,0.261,180,0.287,-0.033,-0.120,0.056,at
pythia-6.9b,you-moves,fair,0.272,180,0.287,-0.014,-0.104,0.080,at
pythia-6.9b,you-named,0shot,0.344,180,0.287,0.078,-0.010,0.167,at
pythia-6.9b,you-named,fair,0.306,180,0.287,0.027,-0.061,0.122,at
pythia-6.9b,answer-self,0shot,0.307,2560,0.250,0.077,0.052,0.100,above
pythia-6.9b,answer-self,fair,0.266,2560,0.250,0.022,-0.001,0.045,at
pythia-6.9b,grammar,ladder,,1080,,0.773,0.747,0.800,above
pythia-6.9b,operators,ladder,,1080,,0.201,0.173,0.230,above
pythia-6.9b,patterns,ladder,,1080,,0.395,0.361,0.431,above
pythia-6.9b,facts,ladder,,768,,0.995,0.991,1.000,above
falcon-mamba-7b,lookup,0shot,0.473,1210,0.169,0.354,0.326,0.385,above
falcon-mamba-7b,lookup,fair,0.985,1210,0.169,0.983,0.975,0.989,above
falcon-mamba-7b,shortlist,0shot,0.804,3840,0.040,0.796,0.785,0.807,above
falcon-mamba-7b,pairs,0shot,0.419,256,0.125,0.336,0.280,0.390,above
falcon-mamba-7b,pairs,fair,0.367,256,0.125,0.277,0.226,0.333,above
falcon-mamba-7b,noise,0shot,0.868,720,0.200,0.835,0.809,0.858,above
falcon-mamba-7b,now,0shot,0.850,300,0.292,0.794,0.747,0.839,above
falcon-mamba-7b,handoffs,0shot,0.704,1200,0.312,0.562,0.528,0.595,above
falcon-mamba-7b,handoffs,fair,0.690,1200,0.312,0.544,0.508,0.576,above
falcon-mamba-7b,sets,0shot,0.469,1280,0.275,0.272,0.244,0.301,above
falcon-mamba-7b,sets,fair,0.456,1280,0.275,0.254,0.226,0.284,above
falcon-mamba-7b,refilled,0shot,0.518,720,0.200,0.398,0.361,0.437,above
falcon-mamba-7b,refills,0shot,0.288,600,0.200,0.111,0.070,0.149,above
falcon-mamba-7b,codevar,0shot,0.362,1200,0.267,0.121,0.090,0.152,above
falcon-mamba-7b,codevar,fair,0.383,1200,0.267,0.149,0.117,0.181,above
falcon-mamba-7b,sharedlists,0shot,0.245,600,0.200,0.056,0.020,0.094,above
falcon-mamba-7b,sharedlists,fair,0.263,600,0.200,0.079,0.043,0.116,above
falcon-mamba-7b,box-narrative,0shot,0.469,200,0.253,0.279,0.197,0.362,above
falcon-mamba-7b,box-narrative,fair,0.480,200,0.253,0.261,0.174,0.346,above
falcon-mamba-7b,box-screenplay,0shot,0.556,200,0.253,0.410,0.329,0.492,above
falcon-mamba-7b,box-screenplay,fair,0.515,200,0.253,0.326,0.246,0.409,above
falcon-mamba-7b,box-code,0shot,0.472,200,0.253,0.289,0.187,0.382,above
falcon-mamba-7b,box-code,fair,0.613,200,0.253,0.479,0.392,0.558,above
falcon-mamba-7b,box-chat,0shot,0.595,200,0.253,0.453,0.369,0.536,above
falcon-mamba-7b,box-chat,fair,0.570,200,0.253,0.419,0.335,0.500,above
falcon-mamba-7b,you-static,0shot,0.770,300,0.292,0.673,0.607,0.728,above
falcon-mamba-7b,i-moves,0shot,0.457,180,0.287,0.238,0.153,0.315,above
falcon-mamba-7b,i-moves,fair,0.362,180,0.287,0.103,0.017,0.186,above
falcon-mamba-7b,you-moves,0shot,0.372,180,0.287,0.112,0.029,0.190,above
falcon-mamba-7b,you-moves,fair,0.290,180,0.287,0.000,-0.077,0.079,at
falcon-mamba-7b,you-named,0shot,0.296,180,0.287,0.013,-0.051,0.082,at
falcon-mamba-7b,you-named,fair,0.301,180,0.287,0.021,-0.055,0.094,at
falcon-mamba-7b,answer-self,0shot,0.310,2560,0.250,0.080,0.061,0.100,above
falcon-mamba-7b,answer-self,fair,0.330,2560,0.250,0.107,0.086,0.127,above
falcon-mamba-7b,grammar,ladder,,1080,,0.817,0.799,0.832,above
falcon-mamba-7b,operators,ladder,,1080,,0.291,0.267,0.319,above
falcon-mamba-7b,patterns,ladder,,1080,,0.543,0.511,0.575,above
falcon-mamba-7b,facts,ladder,,768,,0.998,0.995,1.000,above
olmo2-7b,lookup,0shot,0.473,1210,0.169,0.351,0.318,0.387,above
olmo2-7b,lookup,fair,0.979,1210,0.169,0.974,0.964,0.984,above
olmo2-7b,shortlist,0shot,0.796,3840,0.040,0.787,0.773,0.800,above
olmo2-7b,pairs,0shot,0.477,256,0.125,0.402,0.330,0.469,above
olmo2-7b,pairs,fair,0.758,256,0.125,0.723,0.665,0.786,above
olmo2-7b,noise,0shot,0.866,720,0.200,0.833,0.801,0.864,above
olmo2-7b,now,0shot,0.933,300,0.292,0.905,0.861,0.943,above
olmo2-7b,handoffs,0shot,0.750,1200,0.312,0.630,0.594,0.665,above
olmo2-7b,handoffs,fair,0.798,1200,0.312,0.702,0.666,0.735,above
olmo2-7b,sets,0shot,0.636,1280,0.275,0.505,0.468,0.543,above
olmo2-7b,sets,fair,0.756,1280,0.275,0.674,0.640,0.704,above
olmo2-7b,refilled,0shot,0.627,720,0.200,0.534,0.489,0.580,above
olmo2-7b,refills,0shot,0.401,600,0.200,0.251,0.204,0.297,above
olmo2-7b,codevar,0shot,0.372,1200,0.267,0.137,0.100,0.174,above
olmo2-7b,codevar,fair,0.419,1200,0.267,0.201,0.164,0.239,above
olmo2-7b,sharedlists,0shot,0.318,600,0.200,0.148,0.104,0.192,above
olmo2-7b,sharedlists,fair,0.333,600,0.200,0.167,0.117,0.212,above
olmo2-7b,box-narrative,0shot,0.550,200,0.253,0.380,0.281,0.478,above
olmo2-7b,box-narrative,fair,0.537,200,0.253,0.376,0.281,0.476,above
olmo2-7b,box-screenplay,0shot,0.642,200,0.253,0.520,0.424,0.605,above
olmo2-7b,box-screenplay,fair,0.455,200,0.253,0.252,0.144,0.351,above
olmo2-7b,box-code,0shot,0.595,200,0.253,0.451,0.349,0.546,above
olmo2-7b,box-code,fair,0.635,200,0.253,0.482,0.389,0.574,above
olmo2-7b,box-chat,0shot,0.595,200,0.253,0.462,0.368,0.551,above
olmo2-7b,box-chat,fair,0.492,200,0.253,0.308,0.214,0.402,above
olmo2-7b,you-static,0shot,0.860,300,0.292,0.801,0.742,0.858,above
olmo2-7b,i-moves,0shot,0.489,180,0.287,0.285,0.181,0.379,above
olmo2-7b,i-moves,fair,0.389,180,0.287,0.144,0.044,0.242,above
olmo2-7b,you-moves,0shot,0.417,180,0.287,0.180,0.082,0.276,above
olmo2-7b,you-moves,fair,0.339,180,0.287,0.072,-0.021,0.171,at
olmo2-7b,you-named,0shot,0.361,180,0.287,0.102,-0.001,0.200,at
olmo2-7b,you-named,fair,0.400,180,0.287,0.157,0.063,0.266,above
olmo2-7b,answer-self,0shot,0.407,2560,0.250,0.210,0.185,0.236,above
olmo2-7b,answer-self,fair,0.435,2560,0.250,0.246,0.221,0.273,above
olmo2-7b,grammar,ladder,,1080,,0.864,0.844,0.885,above
olmo2-7b,operators,ladder,,1080,,0.279,0.251,0.311,above
olmo2-7b,patterns,ladder,,1080,,0.578,0.544,0.609,above
olmo2-7b,facts,ladder,,768,,1.000,1.000,1.000,above
falcon3-7b,lookup,0shot,0.767,1210,0.169,0.714,0.686,0.746,above
falcon3-7b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
falcon3-7b,shortlist,0shot,0.771,3840,0.040,0.762,0.748,0.775,above
falcon3-7b,pairs,0shot,0.502,256,0.125,0.431,0.362,0.498,above
falcon3-7b,pairs,fair,0.719,256,0.125,0.679,0.616,0.737,above
falcon3-7b,noise,0shot,0.897,720,0.200,0.872,0.844,0.899,above
falcon3-7b,now,0shot,0.923,300,0.292,0.893,0.847,0.931,above
falcon3-7b,handoffs,0shot,0.853,1200,0.312,0.782,0.752,0.810,above
falcon3-7b,handoffs,fair,0.893,1200,0.312,0.842,0.816,0.868,above
falcon3-7b,sets,0shot,0.748,1280,0.275,0.651,0.619,0.684,above
falcon3-7b,sets,fair,0.742,1280,0.275,0.653,0.618,0.687,above
falcon3-7b,refilled,0shot,0.613,720,0.200,0.516,0.470,0.562,above
falcon3-7b,refills,0shot,0.450,600,0.200,0.312,0.263,0.365,above
falcon3-7b,codevar,0shot,0.625,1200,0.267,0.482,0.444,0.518,above
falcon3-7b,codevar,fair,0.824,1200,0.267,0.755,0.722,0.785,above
falcon3-7b,sharedlists,0shot,0.395,600,0.200,0.244,0.194,0.296,above
falcon3-7b,sharedlists,fair,0.777,600,0.200,0.721,0.677,0.760,above
falcon3-7b,box-narrative,0shot,0.605,200,0.253,0.471,0.378,0.567,above
falcon3-7b,box-narrative,fair,0.630,200,0.253,0.505,0.410,0.597,above
falcon3-7b,box-screenplay,0shot,0.610,200,0.253,0.469,0.378,0.567,above
falcon3-7b,box-screenplay,fair,0.665,200,0.253,0.548,0.458,0.634,above
falcon3-7b,box-code,0shot,0.495,200,0.253,0.338,0.243,0.424,above
falcon3-7b,box-code,fair,0.820,200,0.253,0.748,0.671,0.821,above
falcon3-7b,box-chat,0shot,0.690,200,0.253,0.588,0.499,0.674,above
falcon3-7b,box-chat,fair,0.693,200,0.253,0.592,0.503,0.677,above
falcon3-7b,you-static,0shot,0.873,300,0.292,0.817,0.759,0.868,above
falcon3-7b,i-moves,0shot,0.406,180,0.287,0.164,0.062,0.258,above
falcon3-7b,i-moves,fair,0.378,180,0.287,0.123,0.027,0.224,above
falcon3-7b,you-moves,0shot,0.361,180,0.287,0.096,-0.002,0.202,at
falcon3-7b,you-moves,fair,0.244,180,0.287,-0.064,-0.152,0.033,at
falcon3-7b,you-named,0shot,0.406,180,0.287,0.165,0.069,0.265,above
falcon3-7b,you-named,fair,0.394,180,0.287,0.150,0.051,0.251,above
falcon3-7b,answer-self,0shot,0.458,2560,0.250,0.278,0.252,0.304,above
falcon3-7b,answer-self,fair,0.511,2560,0.250,0.348,0.323,0.372,above
falcon3-7b,grammar,ladder,,1080,,0.826,0.806,0.847,above
falcon3-7b,operators,ladder,,1080,,0.243,0.209,0.279,above
falcon3-7b,patterns,ladder,,1080,,0.595,0.565,0.624,above
falcon3-7b,facts,ladder,,768,,1.000,1.000,1.000,above
falcon-h1-7b,lookup,0shot,0.873,1210,0.169,0.840,0.815,0.862,above
falcon-h1-7b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
falcon-h1-7b,shortlist,0shot,0.966,3840,0.040,0.965,0.959,0.970,above
falcon-h1-7b,pairs,0shot,0.680,256,0.125,0.634,0.571,0.696,above
falcon-h1-7b,pairs,fair,0.943,256,0.125,0.935,0.904,0.964,above
falcon-h1-7b,noise,0shot,0.967,720,0.200,0.958,0.941,0.974,above
falcon-h1-7b,now,0shot,0.973,300,0.292,0.962,0.934,0.986,above
falcon-h1-7b,handoffs,0shot,0.891,1200,0.312,0.839,0.811,0.865,above
falcon-h1-7b,handoffs,fair,0.927,1200,0.312,0.891,0.868,0.912,above
falcon-h1-7b,sets,0shot,0.856,1280,0.275,0.801,0.773,0.828,above
falcon-h1-7b,sets,fair,0.797,1280,0.275,0.730,0.700,0.761,above
falcon-h1-7b,refilled,0shot,0.689,720,0.200,0.611,0.569,0.654,above
falcon-h1-7b,refills,0shot,0.632,600,0.200,0.540,0.491,0.588,above
falcon-h1-7b,codevar,0shot,0.536,1200,0.267,0.366,0.328,0.405,above
falcon-h1-7b,codevar,fair,0.775,1200,0.267,0.687,0.654,0.720,above
falcon-h1-7b,sharedlists,0shot,0.379,600,0.200,0.224,0.180,0.271,above
falcon-h1-7b,sharedlists,fair,0.757,600,0.200,0.696,0.652,0.737,above
falcon-h1-7b,box-narrative,0shot,0.627,200,0.253,0.496,0.397,0.587,above
falcon-h1-7b,box-narrative,fair,0.680,200,0.253,0.577,0.482,0.663,above
falcon-h1-7b,box-screenplay,0shot,0.708,200,0.253,0.601,0.510,0.687,above
falcon-h1-7b,box-screenplay,fair,0.710,200,0.253,0.618,0.532,0.700,above
falcon-h1-7b,box-code,0shot,0.620,200,0.253,0.467,0.378,0.558,above
falcon-h1-7b,box-code,fair,0.855,200,0.253,0.799,0.732,0.866,above
falcon-h1-7b,box-chat,0shot,0.655,200,0.253,0.541,0.447,0.627,above
falcon-h1-7b,box-chat,fair,0.693,200,0.253,0.590,0.501,0.674,above
falcon-h1-7b,you-static,0shot,0.957,300,0.292,0.936,0.901,0.966,above
falcon-h1-7b,i-moves,0shot,0.372,180,0.287,0.116,0.014,0.209,above
falcon-h1-7b,i-moves,fair,0.442,180,0.287,0.214,0.113,0.312,above
falcon-h1-7b,you-moves,0shot,0.417,180,0.287,0.186,0.083,0.286,above
falcon-h1-7b,you-moves,fair,0.353,180,0.287,0.089,-0.006,0.183,at
falcon-h1-7b,you-named,0shot,0.386,180,0.287,0.137,0.036,0.223,above
falcon-h1-7b,you-named,fair,0.417,180,0.287,0.179,0.085,0.283,above
falcon-h1-7b,answer-self,0shot,0.530,2560,0.250,0.373,0.349,0.398,above
falcon-h1-7b,answer-self,fair,0.612,2560,0.250,0.482,0.456,0.506,above
falcon-h1-7b,grammar,ladder,,1080,,0.859,0.839,0.880,above
falcon-h1-7b,operators,ladder,,1080,,0.267,0.232,0.297,above
falcon-h1-7b,patterns,ladder,,1080,,0.724,0.698,0.752,above
falcon-h1-7b,facts,ladder,,768,,0.998,0.995,1.000,above
qwen2.5-7b,lookup,0shot,0.812,1210,0.169,0.767,0.738,0.795,above
qwen2.5-7b,lookup,fair,0.998,1210,0.169,0.998,0.995,1.000,above
qwen2.5-7b,shortlist,0shot,0.912,3840,0.040,0.909,0.899,0.917,above
qwen2.5-7b,pairs,0shot,0.695,256,0.125,0.652,0.585,0.714,above
qwen2.5-7b,pairs,fair,0.875,256,0.125,0.857,0.810,0.902,above
qwen2.5-7b,noise,0shot,0.978,720,0.200,0.972,0.958,0.984,above
qwen2.5-7b,now,0shot,0.878,300,0.292,0.828,0.776,0.876,above
qwen2.5-7b,handoffs,0shot,0.834,1200,0.312,0.756,0.727,0.784,above
qwen2.5-7b,handoffs,fair,0.855,1200,0.312,0.788,0.758,0.818,above
qwen2.5-7b,sets,0shot,0.860,1280,0.275,0.809,0.781,0.835,above
qwen2.5-7b,sets,fair,0.893,1280,0.275,0.861,0.839,0.883,above
qwen2.5-7b,refilled,0shot,0.654,720,0.200,0.568,0.526,0.614,above
qwen2.5-7b,refills,0shot,0.549,600,0.200,0.436,0.388,0.487,above
qwen2.5-7b,codevar,0shot,0.619,1200,0.267,0.478,0.438,0.513,above
qwen2.5-7b,codevar,fair,0.657,1200,0.267,0.526,0.487,0.562,above
qwen2.5-7b,sharedlists,0shot,0.561,600,0.200,0.451,0.402,0.502,above
qwen2.5-7b,sharedlists,fair,0.669,600,0.200,0.587,0.542,0.630,above
qwen2.5-7b,box-narrative,0shot,0.698,200,0.253,0.574,0.483,0.664,above
qwen2.5-7b,box-narrative,fair,0.737,200,0.253,0.650,0.561,0.729,above
qwen2.5-7b,box-screenplay,0shot,0.677,200,0.253,0.564,0.476,0.652,above
qwen2.5-7b,box-screenplay,fair,0.713,200,0.253,0.610,0.521,0.694,above
qwen2.5-7b,box-code,0shot,0.665,200,0.253,0.512,0.405,0.602,above
qwen2.5-7b,box-code,fair,0.802,200,0.253,0.716,0.631,0.786,above
qwen2.5-7b,box-chat,0shot,0.690,200,0.253,0.572,0.475,0.659,above
qwen2.5-7b,box-chat,fair,0.743,200,0.253,0.652,0.572,0.731,above
qwen2.5-7b,you-static,0shot,0.792,300,0.292,0.700,0.629,0.762,above
qwen2.5-7b,i-moves,0shot,0.531,180,0.287,0.342,0.240,0.434,above
qwen2.5-7b,i-moves,fair,0.497,180,0.287,0.293,0.191,0.388,above
qwen2.5-7b,you-moves,0shot,0.386,180,0.287,0.141,0.043,0.237,above
qwen2.5-7b,you-moves,fair,0.387,180,0.287,0.140,0.045,0.238,above
qwen2.5-7b,you-named,0shot,0.422,180,0.287,0.188,0.090,0.286,above
qwen2.5-7b,you-named,fair,0.450,180,0.287,0.224,0.121,0.319,above
qwen2.5-7b,answer-self,0shot,0.446,2560,0.250,0.261,0.235,0.288,above
qwen2.5-7b,answer-self,fair,0.516,2560,0.250,0.354,0.327,0.379,above
qwen2.5-7b,grammar,ladder,,1080,,0.818,0.797,0.842,above
qwen2.5-7b,operators,ladder,,1080,,0.203,0.173,0.239,above
qwen2.5-7b,patterns,ladder,,1080,,0.728,0.697,0.754,above
qwen2.5-7b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3-8b,lookup,0shot,0.747,1210,0.169,0.680,0.650,0.710,above
llama3-8b,lookup,fair,0.999,1210,0.169,0.999,0.997,1.000,above
llama3-8b,shortlist,0shot,0.990,3840,0.040,0.989,0.986,0.992,above
llama3-8b,pairs,0shot,0.715,256,0.125,0.674,0.614,0.734,above
llama3-8b,pairs,fair,0.857,256,0.125,0.837,0.790,0.882,above
llama3-8b,noise,0shot,0.984,720,0.200,0.980,0.967,0.991,above
llama3-8b,now,0shot,0.862,300,0.292,0.804,0.749,0.854,above
llama3-8b,handoffs,0shot,0.845,1200,0.312,0.772,0.744,0.801,above
llama3-8b,handoffs,fair,0.840,1200,0.312,0.766,0.735,0.796,above
llama3-8b,sets,0shot,0.770,1280,0.275,0.683,0.653,0.711,above
llama3-8b,sets,fair,0.846,1280,0.275,0.793,0.766,0.817,above
llama3-8b,refilled,0shot,0.502,720,0.200,0.378,0.333,0.421,above
llama3-8b,refills,0shot,0.428,600,0.200,0.285,0.239,0.333,above
llama3-8b,codevar,0shot,0.574,1200,0.267,0.414,0.378,0.453,above
llama3-8b,codevar,fair,0.674,1200,0.267,0.548,0.514,0.583,above
llama3-8b,sharedlists,0shot,0.417,600,0.200,0.272,0.227,0.321,above
llama3-8b,sharedlists,fair,0.617,600,0.200,0.521,0.473,0.569,above
llama3-8b,box-narrative,0shot,0.644,200,0.253,0.506,0.410,0.596,above
llama3-8b,box-narrative,fair,0.693,200,0.253,0.575,0.482,0.662,above
llama3-8b,box-screenplay,0shot,0.677,200,0.253,0.569,0.477,0.653,above
llama3-8b,box-screenplay,fair,0.665,200,0.253,0.533,0.440,0.622,above
llama3-8b,box-code,0shot,0.692,200,0.253,0.565,0.476,0.647,above
llama3-8b,box-code,fair,0.725,200,0.253,0.599,0.510,0.685,above
llama3-8b,box-chat,0shot,0.665,200,0.253,0.552,0.463,0.640,above
llama3-8b,box-chat,fair,0.755,200,0.253,0.664,0.582,0.746,above
llama3-8b,you-static,0shot,0.927,300,0.292,0.892,0.848,0.933,above
llama3-8b,i-moves,0shot,0.395,180,0.287,0.151,0.056,0.259,above
llama3-8b,i-moves,fair,0.417,180,0.287,0.182,0.084,0.279,above
llama3-8b,you-moves,0shot,0.398,180,0.287,0.156,0.068,0.256,above
llama3-8b,you-moves,fair,0.316,180,0.287,0.037,-0.056,0.135,at
llama3-8b,you-named,0shot,0.418,180,0.287,0.180,0.084,0.277,above
llama3-8b,you-named,fair,0.419,180,0.287,0.180,0.082,0.277,above
llama3-8b,answer-self,0shot,0.474,2560,0.250,0.298,0.273,0.325,above
llama3-8b,answer-self,fair,0.480,2560,0.250,0.306,0.280,0.330,above
llama3-8b,grammar,ladder,,1080,,0.845,0.824,0.867,above
llama3-8b,operators,ladder,,1080,,0.250,0.223,0.281,above
llama3-8b,patterns,ladder,,1080,,0.795,0.768,0.818,above
llama3-8b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-8b,lookup,0shot,0.647,1210,0.169,0.556,0.522,0.589,above
llama3.1-8b,lookup,fair,0.999,1210,0.169,0.999,0.997,1.000,above
llama3.1-8b,shortlist,0shot,0.984,3840,0.040,0.984,0.980,0.987,above
llama3.1-8b,pairs,0shot,0.656,256,0.125,0.607,0.545,0.672,above
llama3.1-8b,pairs,fair,0.930,256,0.125,0.920,0.884,0.953,above
llama3.1-8b,noise,0shot,0.986,720,0.200,0.983,0.971,0.992,above
llama3.1-8b,now,0shot,0.815,300,0.292,0.735,0.676,0.794,above
llama3.1-8b,handoffs,0shot,0.827,1200,0.312,0.746,0.715,0.777,above
llama3.1-8b,handoffs,fair,0.862,1200,0.312,0.797,0.768,0.825,above
llama3.1-8b,sets,0shot,0.803,1280,0.275,0.729,0.701,0.758,above
llama3.1-8b,sets,fair,0.866,1280,0.275,0.823,0.798,0.846,above
llama3.1-8b,refilled,0shot,0.518,720,0.200,0.398,0.351,0.442,above
llama3.1-8b,refills,0shot,0.433,600,0.200,0.291,0.244,0.339,above
llama3.1-8b,codevar,0shot,0.581,1200,0.267,0.422,0.385,0.458,above
llama3.1-8b,codevar,fair,0.682,1200,0.267,0.560,0.524,0.595,above
llama3.1-8b,sharedlists,0shot,0.426,600,0.200,0.282,0.235,0.326,above
llama3.1-8b,sharedlists,fair,0.609,600,0.200,0.511,0.462,0.558,above
llama3.1-8b,box-narrative,0shot,0.670,200,0.253,0.545,0.453,0.628,above
llama3.1-8b,box-narrative,fair,0.708,200,0.253,0.597,0.505,0.685,above
llama3.1-8b,box-screenplay,0shot,0.670,200,0.253,0.559,0.468,0.643,above
llama3.1-8b,box-screenplay,fair,0.682,200,0.253,0.558,0.465,0.643,above
llama3.1-8b,box-code,0shot,0.677,200,0.253,0.544,0.449,0.630,above
llama3.1-8b,box-code,fair,0.762,200,0.253,0.649,0.559,0.731,above
llama3.1-8b,box-chat,0shot,0.680,200,0.253,0.569,0.484,0.657,above
llama3.1-8b,box-chat,fair,0.740,200,0.253,0.650,0.566,0.731,above
llama3.1-8b,you-static,0shot,0.921,300,0.292,0.886,0.844,0.925,above
llama3.1-8b,i-moves,0shot,0.431,180,0.287,0.201,0.103,0.301,above
llama3.1-8b,i-moves,fair,0.414,180,0.287,0.177,0.081,0.273,above
llama3.1-8b,you-moves,0shot,0.403,180,0.287,0.163,0.067,0.263,above
llama3.1-8b,you-moves,fair,0.289,180,0.287,-0.002,-0.090,0.092,at
llama3.1-8b,you-named,0shot,0.406,180,0.287,0.163,0.066,0.259,above
llama3.1-8b,you-named,fair,0.435,180,0.287,0.205,0.106,0.298,above
llama3.1-8b,answer-self,0shot,0.469,2560,0.250,0.291,0.264,0.318,above
llama3.1-8b,answer-self,fair,0.504,2560,0.250,0.339,0.313,0.364,above
llama3.1-8b,grammar,ladder,,1080,,0.843,0.818,0.863,above
llama3.1-8b,operators,ladder,,1080,,0.262,0.235,0.291,above
llama3.1-8b,patterns,ladder,,1080,,0.759,0.729,0.782,above
llama3.1-8b,facts,ladder,,768,,1.000,1.000,1.000,above
nemotron-h-8b,lookup,0shot,0.831,1210,0.169,0.794,0.768,0.818,above
nemotron-h-8b,lookup,fair,0.998,1210,0.169,0.998,0.994,1.000,above
nemotron-h-8b,shortlist,0shot,0.948,3840,0.040,0.946,0.938,0.953,above
nemotron-h-8b,pairs,0shot,0.609,256,0.125,0.554,0.484,0.614,above
nemotron-h-8b,pairs,fair,0.951,256,0.125,0.944,0.913,0.973,above
nemotron-h-8b,noise,0shot,0.946,720,0.200,0.932,0.911,0.953,above
nemotron-h-8b,now,0shot,0.905,300,0.292,0.868,0.816,0.910,above
nemotron-h-8b,handoffs,0shot,0.871,1200,0.312,0.810,0.781,0.837,above
nemotron-h-8b,handoffs,fair,0.787,1200,0.312,0.689,0.655,0.719,above
nemotron-h-8b,sets,0shot,0.795,1280,0.275,0.719,0.688,0.750,above
nemotron-h-8b,sets,fair,0.804,1280,0.275,0.739,0.709,0.768,above
nemotron-h-8b,refilled,0shot,0.642,720,0.200,0.552,0.509,0.595,above
nemotron-h-8b,refills,0shot,0.436,600,0.200,0.295,0.244,0.345,above
nemotron-h-8b,codevar,0shot,0.533,1200,0.267,0.356,0.319,0.396,above
nemotron-h-8b,codevar,fair,0.608,1200,0.267,0.463,0.427,0.499,above
nemotron-h-8b,sharedlists,0shot,0.302,600,0.200,0.127,0.083,0.174,above
nemotron-h-8b,sharedlists,fair,0.640,600,0.200,0.550,0.502,0.598,above
nemotron-h-8b,box-narrative,0shot,0.613,200,0.253,0.472,0.377,0.565,above
nemotron-h-8b,box-narrative,fair,0.608,200,0.253,0.474,0.378,0.567,above
nemotron-h-8b,box-screenplay,0shot,0.685,200,0.253,0.574,0.482,0.665,above
nemotron-h-8b,box-screenplay,fair,0.593,200,0.253,0.443,0.349,0.538,above
nemotron-h-8b,box-code,0shot,0.690,200,0.253,0.558,0.462,0.647,above
nemotron-h-8b,box-code,fair,0.733,200,0.253,0.625,0.532,0.708,above
nemotron-h-8b,box-chat,0shot,0.672,200,0.253,0.556,0.464,0.645,above
nemotron-h-8b,box-chat,fair,0.610,200,0.253,0.467,0.374,0.561,above
nemotron-h-8b,you-static,0shot,0.943,300,0.292,0.916,0.877,0.952,above
nemotron-h-8b,i-moves,0shot,0.469,180,0.287,0.256,0.161,0.352,above
nemotron-h-8b,i-moves,fair,0.450,180,0.287,0.230,0.135,0.326,above
nemotron-h-8b,you-moves,0shot,0.378,180,0.287,0.124,0.024,0.218,above
nemotron-h-8b,you-moves,fair,0.278,180,0.287,-0.013,-0.098,0.073,at
nemotron-h-8b,you-named,0shot,0.356,180,0.287,0.095,0.004,0.195,above
nemotron-h-8b,you-named,fair,0.428,180,0.287,0.196,0.096,0.294,above
nemotron-h-8b,answer-self,0shot,0.480,2560,0.250,0.307,0.281,0.332,above
nemotron-h-8b,answer-self,fair,0.475,2560,0.250,0.300,0.275,0.324,above
nemotron-h-8b,grammar,ladder,,1080,,0.869,0.851,0.887,above
nemotron-h-8b,operators,ladder,,1080,,0.239,0.208,0.271,above
nemotron-h-8b,patterns,ladder,,1080,,0.533,0.502,0.559,above
nemotron-h-8b,facts,ladder,,768,,1.000,1.000,1.000,above
gemma-7b,lookup,0shot,0.753,1210,0.169,0.685,0.657,0.713,above
gemma-7b,lookup,fair,0.999,1210,0.169,0.999,0.998,1.000,above
gemma-7b,shortlist,0shot,0.951,3840,0.040,0.949,0.942,0.955,above
gemma-7b,pairs,0shot,0.458,256,0.125,0.381,0.319,0.445,above
gemma-7b,pairs,fair,0.758,256,0.125,0.724,0.671,0.776,above
gemma-7b,noise,0shot,0.909,720,0.200,0.886,0.862,0.908,above
gemma-7b,now,0shot,0.948,300,0.292,0.927,0.895,0.955,above
gemma-7b,handoffs,0shot,0.752,1200,0.312,0.635,0.607,0.661,above
gemma-7b,handoffs,fair,0.866,1200,0.312,0.804,0.780,0.826,above
gemma-7b,sets,0shot,0.683,1280,0.275,0.565,0.537,0.593,above
gemma-7b,sets,fair,0.831,1280,0.275,0.771,0.750,0.793,above
gemma-7b,refilled,0shot,0.556,720,0.200,0.446,0.408,0.489,above
gemma-7b,refills,0shot,0.410,600,0.200,0.262,0.222,0.305,above
gemma-7b,codevar,0shot,0.595,1200,0.267,0.440,0.411,0.468,above
gemma-7b,codevar,fair,0.603,1200,0.267,0.451,0.423,0.481,above
gemma-7b,sharedlists,0shot,0.394,600,0.200,0.243,0.201,0.284,above
gemma-7b,sharedlists,fair,0.535,600,0.200,0.418,0.381,0.456,above
gemma-7b,box-narrative,0shot,0.576,200,0.253,0.433,0.348,0.511,above
gemma-7b,box-narrative,fair,0.613,200,0.253,0.485,0.392,0.570,above
gemma-7b,box-screenplay,0shot,0.537,200,0.253,0.387,0.304,0.469,above
gemma-7b,box-screenplay,fair,0.612,200,0.253,0.474,0.390,0.560,above
gemma-7b,box-code,0shot,0.699,200,0.253,0.581,0.491,0.662,above
gemma-7b,box-code,fair,0.767,200,0.253,0.676,0.607,0.743,above
gemma-7b,box-chat,0shot,0.543,200,0.253,0.391,0.307,0.472,above
gemma-7b,box-chat,fair,0.604,200,0.253,0.454,0.368,0.539,above
gemma-7b,you-static,0shot,0.940,300,0.292,0.911,0.874,0.943,above
gemma-7b,i-moves,0shot,0.508,180,0.287,0.309,0.217,0.396,above
gemma-7b,i-moves,fair,0.474,180,0.287,0.262,0.179,0.355,above
gemma-7b,you-moves,0shot,0.373,180,0.287,0.118,0.031,0.209,above
gemma-7b,you-moves,fair,0.301,180,0.287,0.024,-0.051,0.102,at
gemma-7b,you-named,0shot,0.325,180,0.287,0.054,-0.026,0.144,at
gemma-7b,you-named,fair,0.376,180,0.287,0.122,0.035,0.206,above
gemma-7b,answer-self,0shot,0.450,2560,0.250,0.266,0.244,0.287,above
gemma-7b,answer-self,fair,0.442,2560,0.250,0.256,0.232,0.277,above
gemma-7b,grammar,ladder,,1080,,0.841,0.821,0.864,above
gemma-7b,operators,ladder,,1080,,0.245,0.218,0.271,above
gemma-7b,patterns,ladder,,1080,,0.666,0.639,0.690,above
gemma-7b,facts,ladder,,768,,1.000,1.000,1.000,above
gemma2-9b,lookup,0shot,0.766,1210,0.169,0.712,0.683,0.740,above
gemma2-9b,lookup,fair,0.998,1210,0.169,0.998,0.996,1.000,above
gemma2-9b,shortlist,0shot,0.941,3840,0.040,0.939,0.931,0.946,above
gemma2-9b,pairs,0shot,0.614,256,0.125,0.559,0.490,0.626,above
gemma2-9b,pairs,fair,0.963,256,0.125,0.958,0.931,0.978,above
gemma2-9b,noise,0shot,0.985,720,0.200,0.982,0.971,0.991,above
gemma2-9b,now,0shot,0.856,300,0.292,0.796,0.738,0.848,above
gemma2-9b,handoffs,0shot,0.862,1200,0.312,0.796,0.767,0.820,above
gemma2-9b,handoffs,fair,0.920,1200,0.312,0.883,0.861,0.904,above
gemma2-9b,sets,0shot,0.821,1280,0.275,0.751,0.722,0.779,above
gemma2-9b,sets,fair,0.880,1280,0.275,0.837,0.813,0.860,above
gemma2-9b,refilled,0shot,0.469,720,0.200,0.337,0.295,0.380,above
gemma2-9b,refills,0shot,0.486,600,0.200,0.358,0.309,0.406,above
gemma2-9b,codevar,0shot,0.643,1200,0.267,0.507,0.472,0.545,above
gemma2-9b,codevar,fair,0.760,1200,0.267,0.668,0.637,0.700,above
gemma2-9b,sharedlists,0shot,0.609,600,0.200,0.511,0.460,0.555,above
gemma2-9b,sharedlists,fair,0.706,600,0.200,0.632,0.588,0.677,above
gemma2-9b,box-narrative,0shot,0.587,200,0.253,0.447,0.360,0.537,above
gemma2-9b,box-narrative,fair,0.693,200,0.253,0.603,0.519,0.679,above
gemma2-9b,box-screenplay,0shot,0.623,200,0.253,0.491,0.395,0.584,above
gemma2-9b,box-screenplay,fair,0.725,200,0.253,0.633,0.546,0.710,above
gemma2-9b,box-code,0shot,0.670,200,0.253,0.533,0.441,0.624,above
gemma2-9b,box-code,fair,0.820,200,0.253,0.749,0.672,0.821,above
gemma2-9b,box-chat,0shot,0.684,200,0.253,0.576,0.489,0.654,above
gemma2-9b,box-chat,fair,0.738,200,0.253,0.649,0.560,0.731,above
gemma2-9b,you-static,0shot,0.882,300,0.292,0.828,0.772,0.873,above
gemma2-9b,i-moves,0shot,0.426,180,0.287,0.197,0.100,0.288,above
gemma2-9b,i-moves,fair,0.539,180,0.287,0.348,0.251,0.444,above
gemma2-9b,you-moves,0shot,0.386,180,0.287,0.139,0.046,0.229,above
gemma2-9b,you-moves,fair,0.447,180,0.287,0.224,0.131,0.320,above
gemma2-9b,you-named,0shot,0.353,180,0.287,0.091,-0.003,0.185,at
gemma2-9b,you-named,fair,0.539,180,0.287,0.352,0.254,0.448,above
gemma2-9b,answer-self,0shot,0.426,2560,0.250,0.235,0.210,0.263,above
gemma2-9b,answer-self,fair,0.580,2560,0.250,0.439,0.415,0.464,above
gemma2-9b,grammar,ladder,,1080,,0.863,0.845,0.885,above
gemma2-9b,operators,ladder,,1080,,0.305,0.273,0.336,above
gemma2-9b,patterns,ladder,,1080,,0.660,0.629,0.689,above
gemma2-9b,facts,ladder,,768,,1.000,1.000,1.000,above
recurrentgemma-9b,lookup,0shot,0.349,1210,0.169,0.216,0.186,0.248,above
recurrentgemma-9b,lookup,fair,0.323,1210,0.169,0.188,0.157,0.218,above
recurrentgemma-9b,shortlist,0shot,0.649,3840,0.040,0.635,0.620,0.651,above
recurrentgemma-9b,pairs,0shot,0.213,256,0.125,0.100,0.045,0.160,above
recurrentgemma-9b,pairs,fair,0.223,256,0.125,0.112,0.058,0.173,above
recurrentgemma-9b,noise,0shot,0.470,720,0.200,0.338,0.292,0.381,above
recurrentgemma-9b,now,0shot,0.674,300,0.292,0.544,0.471,0.615,above
recurrentgemma-9b,handoffs,0shot,0.544,1200,0.312,0.333,0.290,0.372,above
recurrentgemma-9b,handoffs,fair,0.498,1200,0.312,0.269,0.229,0.310,above
recurrentgemma-9b,sets,0shot,0.360,1280,0.275,0.118,0.081,0.154,above
recurrentgemma-9b,sets,fair,0.345,1280,0.275,0.104,0.069,0.140,above
recurrentgemma-9b,refilled,0shot,0.200,720,0.200,-0.001,-0.035,0.036,at
recurrentgemma-9b,refills,0shot,0.277,600,0.200,0.096,0.054,0.138,above
recurrentgemma-9b,codevar,0shot,0.357,1200,0.267,0.119,0.083,0.154,above
recurrentgemma-9b,codevar,fair,0.283,1200,0.267,0.017,-0.014,0.051,at
recurrentgemma-9b,sharedlists,0shot,0.208,600,0.200,0.010,-0.032,0.050,at
recurrentgemma-9b,sharedlists,fair,0.292,600,0.200,0.116,0.073,0.164,above
recurrentgemma-9b,box-narrative,0shot,0.242,200,0.253,-0.015,-0.097,0.067,at
recurrentgemma-9b,box-narrative,fair,0.333,200,0.253,0.098,0.008,0.192,above
recurrentgemma-9b,box-screenplay,0shot,0.333,200,0.253,0.105,0.011,0.196,above
recurrentgemma-9b,box-screenplay,fair,0.432,200,0.253,0.237,0.143,0.320,above
recurrentgemma-9b,box-code,0shot,0.435,200,0.253,0.254,0.156,0.337,above
recurrentgemma-9b,box-code,fair,0.439,200,0.253,0.257,0.167,0.345,above
recurrentgemma-9b,box-chat,0shot,0.375,200,0.253,0.185,0.097,0.274,above
recurrentgemma-9b,box-chat,fair,0.458,200,0.253,0.266,0.176,0.356,above
recurrentgemma-9b,you-static,0shot,0.526,300,0.292,0.331,0.250,0.412,above
recurrentgemma-9b,i-moves,0shot,0.241,180,0.287,-0.070,-0.156,0.021,at
recurrentgemma-9b,i-moves,fair,0.294,180,0.287,0.009,-0.082,0.106,at
recurrentgemma-9b,you-moves,0shot,0.281,180,0.287,-0.014,-0.104,0.080,at
recurrentgemma-9b,you-moves,fair,0.278,180,0.287,-0.014,-0.100,0.082,at
recurrentgemma-9b,you-named,0shot,0.375,180,0.287,0.122,0.025,0.217,above
recurrentgemma-9b,you-named,fair,0.375,180,0.287,0.123,0.030,0.227,above
recurrentgemma-9b,answer-self,0shot,0.291,2560,0.250,0.055,0.032,0.077,above
recurrentgemma-9b,answer-self,fair,0.270,2560,0.250,0.026,0.003,0.049,above
recurrentgemma-9b,grammar,ladder,,1080,,0.637,0.607,0.666,above
recurrentgemma-9b,operators,ladder,,1080,,0.137,0.107,0.166,above
recurrentgemma-9b,patterns,ladder,,1080,,0.071,0.047,0.097,above
recurrentgemma-9b,facts,ladder,,768,,0.998,0.995,1.000,above
bamba-9b,lookup,0shot,0.312,1210,0.169,0.176,0.142,0.208,above
bamba-9b,lookup,fair,0.239,1210,0.169,0.093,0.062,0.120,above
bamba-9b,shortlist,0shot,0.639,3840,0.040,0.624,0.609,0.640,above
bamba-9b,pairs,0shot,0.201,256,0.125,0.087,0.036,0.141,above
bamba-9b,pairs,fair,0.213,256,0.125,0.100,0.049,0.156,above
bamba-9b,noise,0shot,0.390,720,0.200,0.238,0.193,0.284,above
bamba-9b,now,0shot,0.422,300,0.292,0.186,0.106,0.264,above
bamba-9b,handoffs,0shot,0.514,1200,0.312,0.290,0.250,0.329,above
bamba-9b,handoffs,fair,0.455,1200,0.312,0.204,0.163,0.246,above
bamba-9b,sets,0shot,0.329,1280,0.275,0.084,0.049,0.120,above
bamba-9b,sets,fair,0.340,1280,0.275,0.097,0.061,0.134,above
bamba-9b,refilled,0shot,0.455,720,0.200,0.319,0.273,0.363,above
bamba-9b,refills,0shot,0.374,600,0.200,0.218,0.171,0.269,above
bamba-9b,codevar,0shot,0.313,1200,0.267,0.060,0.025,0.100,above
bamba-9b,codevar,fair,0.300,1200,0.267,0.043,0.007,0.076,above
bamba-9b,sharedlists,0shot,0.237,600,0.200,0.047,0.007,0.090,above
bamba-9b,sharedlists,fair,0.182,600,0.200,-0.022,-0.060,0.018,at
bamba-9b,box-narrative,0shot,0.235,200,0.253,-0.025,-0.109,0.067,at
bamba-9b,box-narrative,fair,0.180,200,0.253,-0.093,-0.169,-0.013,below
bamba-9b,box-screenplay,0shot,0.198,200,0.253,-0.062,-0.140,0.015,at
bamba-9b,box-screenplay,fair,0.217,200,0.253,-0.056,-0.141,0.025,at
bamba-9b,box-code,0shot,0.285,200,0.253,0.050,-0.038,0.137,at
bamba-9b,box-code,fair,0.240,200,0.253,-0.015,-0.095,0.070,at
bamba-9b,box-chat,0shot,0.200,200,0.253,-0.084,-0.165,-0.002,below
bamba-9b,box-chat,fair,0.250,200,0.253,0.015,-0.065,0.096,at
bamba-9b,you-static,0shot,0.360,300,0.292,0.097,0.021,0.166,above
bamba-9b,i-moves,0shot,0.272,180,0.287,-0.023,-0.109,0.069,at
bamba-9b,i-moves,fair,0.322,180,0.287,0.044,-0.050,0.139,at
bamba-9b,you-moves,0shot,0.275,180,0.287,-0.018,-0.114,0.072,at
bamba-9b,you-moves,fair,0.242,180,0.287,-0.066,-0.157,0.021,at
bamba-9b,you-named,0shot,0.300,180,0.287,0.013,-0.077,0.111,at
bamba-9b,you-named,fair,0.306,180,0.287,0.021,-0.068,0.117,at
bamba-9b,answer-self,0shot,0.260,2560,0.250,0.013,-0.009,0.036,at
bamba-9b,answer-self,fair,0.252,2560,0.250,0.003,-0.020,0.026,at
bamba-9b,grammar,ladder,,1080,,0.587,0.557,0.615,above
bamba-9b,operators,ladder,,1080,,0.134,0.111,0.170,above
bamba-9b,patterns,ladder,,1080,,0.241,0.209,0.285,above
bamba-9b,facts,ladder,,768,,1.000,1.000,1.000,above
dsv4-flash-base,lookup,0shot,0.758,1210,0.169,0.696,0.667,0.726,above
dsv4-flash-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
dsv4-flash-base,shortlist,0shot,0.996,3840,0.040,0.996,0.994,0.998,above
dsv4-flash-base,pairs,0shot,0.647,256,0.125,0.597,0.536,0.658,above
dsv4-flash-base,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
dsv4-flash-base,noise,0shot,0.941,720,0.200,0.926,0.904,0.948,above
dsv4-flash-base,now,0shot,1.000,300,0.292,1.000,1.000,1.000,above
dsv4-flash-base,handoffs,0shot,0.958,1200,0.312,0.939,0.924,0.954,above
dsv4-flash-base,handoffs,fair,0.989,1200,0.312,0.984,0.975,0.992,above
dsv4-flash-base,sets,0shot,0.957,1280,0.275,0.943,0.927,0.957,above
dsv4-flash-base,sets,fair,0.983,1280,0.275,0.977,0.968,0.986,above
dsv4-flash-base,refilled,0shot,0.787,720,0.200,0.734,0.697,0.766,above
dsv4-flash-base,refills,0shot,0.750,600,0.200,0.688,0.643,0.731,above
dsv4-flash-base,codevar,0shot,0.902,1200,0.267,0.865,0.843,0.888,above
dsv4-flash-base,codevar,fair,0.919,1200,0.267,0.886,0.865,0.907,above
dsv4-flash-base,sharedlists,0shot,0.887,600,0.200,0.858,0.827,0.888,above
dsv4-flash-base,sharedlists,fair,0.923,600,0.200,0.904,0.877,0.929,above
dsv4-flash-base,box-narrative,0shot,0.847,200,0.253,0.785,0.712,0.854,above
dsv4-flash-base,box-narrative,fair,0.895,200,0.253,0.855,0.795,0.909,above
dsv4-flash-base,box-screenplay,0shot,0.713,200,0.253,0.603,0.519,0.689,above
dsv4-flash-base,box-screenplay,fair,0.905,200,0.253,0.869,0.814,0.921,above
dsv4-flash-base,box-code,0shot,0.912,200,0.253,0.875,0.814,0.927,above
dsv4-flash-base,box-code,fair,0.890,200,0.253,0.838,0.773,0.897,above
dsv4-flash-base,box-chat,0shot,0.787,200,0.253,0.709,0.630,0.784,above
dsv4-flash-base,box-chat,fair,0.825,200,0.253,0.751,0.670,0.826,above
dsv4-flash-base,you-static,0shot,0.997,300,0.292,0.995,0.985,1.000,above
dsv4-flash-base,i-moves,0shot,0.630,180,0.287,0.482,0.385,0.576,above
dsv4-flash-base,i-moves,fair,0.814,180,0.287,0.737,0.659,0.817,above
dsv4-flash-base,you-moves,0shot,0.552,180,0.287,0.374,0.282,0.470,above
dsv4-flash-base,you-moves,fair,0.503,180,0.287,0.301,0.196,0.396,above
dsv4-flash-base,you-named,0shot,0.564,180,0.287,0.386,0.276,0.490,above
dsv4-flash-base,you-named,fair,0.628,180,0.287,0.481,0.388,0.576,above
dsv4-flash-base,answer-self,0shot,0.555,2560,0.250,0.407,0.383,0.431,above
dsv4-flash-base,answer-self,fair,0.753,2560,0.250,0.671,0.650,0.694,above
dsv4-flash-base,grammar,ladder,,1080,,0.880,0.863,0.897,above
dsv4-flash-base,operators,ladder,,1080,,0.244,0.214,0.275,above
dsv4-flash-base,patterns,ladder,,1080,,0.932,0.919,0.946,above
dsv4-flash-base,facts,ladder,,768,,1.000,1.000,1.000,above
trinity-base,lookup,0shot,0.762,1210,0.169,0.700,0.670,0.730,above
trinity-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
trinity-base,shortlist,0shot,0.987,3840,0.040,0.987,0.983,0.990,above
trinity-base,pairs,0shot,0.741,256,0.125,0.704,0.641,0.768,above
trinity-base,pairs,fair,0.988,256,0.125,0.987,0.969,1.000,above
trinity-base,noise,0shot,0.987,720,0.200,0.983,0.973,0.992,above
trinity-base,now,0shot,0.988,300,0.292,0.983,0.964,0.998,above
trinity-base,handoffs,0shot,0.915,1200,0.312,0.874,0.851,0.896,above
trinity-base,handoffs,fair,0.955,1200,0.312,0.934,0.916,0.951,above
trinity-base,sets,0shot,0.920,1280,0.275,0.892,0.871,0.910,above
trinity-base,sets,fair,0.910,1280,0.275,0.876,0.852,0.898,above
trinity-base,refilled,0shot,0.652,720,0.200,0.565,0.524,0.607,above
trinity-base,refills,0shot,0.657,600,0.200,0.571,0.524,0.619,above
trinity-base,codevar,0shot,0.841,1200,0.267,0.779,0.751,0.808,above
trinity-base,codevar,fair,0.912,1200,0.267,0.876,0.854,0.900,above
trinity-base,sharedlists,0shot,0.966,600,0.200,0.957,0.940,0.975,above
trinity-base,sharedlists,fair,0.950,600,0.200,0.938,0.915,0.959,above
trinity-base,box-narrative,0shot,0.815,200,0.253,0.741,0.667,0.813,above
trinity-base,box-narrative,fair,0.828,200,0.253,0.762,0.684,0.834,above
trinity-base,box-screenplay,0shot,0.782,200,0.253,0.699,0.619,0.772,above
trinity-base,box-screenplay,fair,0.818,200,0.253,0.749,0.670,0.818,above
trinity-base,box-code,0shot,0.782,200,0.253,0.704,0.623,0.779,above
trinity-base,box-code,fair,0.910,200,0.253,0.872,0.814,0.927,above
trinity-base,box-chat,0shot,0.733,200,0.253,0.637,0.557,0.716,above
trinity-base,box-chat,fair,0.835,200,0.253,0.769,0.693,0.843,above
trinity-base,you-static,0shot,0.997,300,0.292,0.995,0.988,1.000,above
trinity-base,i-moves,0shot,0.446,180,0.287,0.225,0.133,0.312,above
trinity-base,i-moves,fair,0.530,180,0.287,0.340,0.242,0.441,above
trinity-base,you-moves,0shot,0.483,180,0.287,0.276,0.175,0.369,above
trinity-base,you-moves,fair,0.480,180,0.287,0.267,0.165,0.362,above
trinity-base,you-named,0shot,0.486,180,0.287,0.276,0.183,0.373,above
trinity-base,you-named,fair,0.552,180,0.287,0.372,0.266,0.463,above
trinity-base,answer-self,0shot,0.500,2560,0.250,0.334,0.310,0.359,above
trinity-base,answer-self,fair,0.753,2560,0.250,0.671,0.648,0.692,above
trinity-base,grammar,ladder,,1080,,0.808,0.788,0.827,above
trinity-base,operators,ladder,,1080,,0.180,0.153,0.211,above
trinity-base,patterns,ladder,,1080,,0.815,0.792,0.835,above
trinity-base,facts,ladder,,768,,1.000,1.000,1.000,above
trinity-truebase,lookup,0shot,0.823,1210,0.169,0.780,0.753,0.805,above
trinity-truebase,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
trinity-truebase,shortlist,0shot,0.982,3840,0.040,0.981,0.976,0.985,above
trinity-truebase,pairs,0shot,0.749,256,0.125,0.714,0.656,0.765,above
trinity-truebase,pairs,fair,0.992,256,0.125,0.991,0.980,1.000,above
trinity-truebase,noise,0shot,0.993,720,0.200,0.991,0.983,0.998,above
trinity-truebase,now,0shot,0.935,300,0.292,0.908,0.869,0.943,above
trinity-truebase,handoffs,0shot,0.905,1200,0.312,0.860,0.835,0.883,above
trinity-truebase,handoffs,fair,0.935,1200,0.312,0.906,0.885,0.924,above
trinity-truebase,sets,0shot,0.815,1280,0.275,0.744,0.714,0.772,above
trinity-truebase,sets,fair,0.828,1280,0.275,0.767,0.738,0.796,above
trinity-truebase,refilled,0shot,0.611,720,0.200,0.514,0.470,0.555,above
trinity-truebase,refills,0shot,0.518,600,0.200,0.397,0.348,0.446,above
trinity-truebase,codevar,0shot,0.877,1200,0.267,0.828,0.802,0.853,above
trinity-truebase,codevar,fair,0.892,1200,0.267,0.851,0.828,0.875,above
trinity-truebase,sharedlists,0shot,0.728,600,0.200,0.660,0.616,0.702,above
trinity-truebase,sharedlists,fair,0.856,600,0.200,0.820,0.784,0.853,above
trinity-truebase,box-narrative,0shot,0.677,200,0.253,0.571,0.485,0.653,above
trinity-truebase,box-narrative,fair,0.700,200,0.253,0.600,0.510,0.686,above
trinity-truebase,box-screenplay,0shot,0.655,200,0.253,0.539,0.448,0.628,above
trinity-truebase,box-screenplay,fair,0.740,200,0.253,0.646,0.559,0.728,above
trinity-truebase,box-code,0shot,0.770,200,0.253,0.683,0.603,0.759,above
trinity-truebase,box-code,fair,0.910,200,0.253,0.870,0.812,0.925,above
trinity-truebase,box-chat,0shot,0.740,200,0.253,0.647,0.567,0.725,above
trinity-truebase,box-chat,fair,0.750,200,0.253,0.663,0.579,0.739,above
trinity-truebase,you-static,0shot,0.955,300,0.292,0.934,0.899,0.964,above
trinity-truebase,i-moves,0shot,0.435,180,0.287,0.211,0.118,0.302,above
trinity-truebase,i-moves,fair,0.521,180,0.287,0.328,0.234,0.420,above
trinity-truebase,you-moves,0shot,0.430,180,0.287,0.201,0.098,0.299,above
trinity-truebase,you-moves,fair,0.414,180,0.287,0.181,0.073,0.280,above
trinity-truebase,you-named,0shot,0.426,180,0.287,0.193,0.091,0.292,above
trinity-truebase,you-named,fair,0.489,180,0.287,0.279,0.184,0.379,above
trinity-truebase,answer-self,0shot,0.477,2560,0.250,0.303,0.278,0.330,above
trinity-truebase,answer-self,fair,0.637,2560,0.250,0.517,0.492,0.542,above
trinity-truebase,grammar,ladder,,1080,,0.851,0.832,0.870,above
trinity-truebase,operators,ladder,,1080,,0.171,0.144,0.202,above
trinity-truebase,patterns,ladder,,1080,,0.704,0.675,0.731,above
trinity-truebase,facts,ladder,,768,,1.000,1.000,1.000,above
olmo2-13b,lookup,0shot,0.619,1210,0.169,0.526,0.490,0.559,above
olmo2-13b,lookup,fair,0.989,1210,0.169,0.986,0.979,0.993,above
olmo2-13b,shortlist,0shot,0.861,3840,0.040,0.855,0.844,0.866,above
olmo2-13b,pairs,0shot,0.641,256,0.125,0.589,0.522,0.656,above
olmo2-13b,pairs,fair,0.836,256,0.125,0.812,0.759,0.862,above
olmo2-13b,noise,0shot,0.889,720,0.200,0.861,0.833,0.889,above
olmo2-13b,now,0shot,0.920,300,0.292,0.888,0.843,0.929,above
olmo2-13b,handoffs,0shot,0.772,1200,0.312,0.666,0.631,0.698,above
olmo2-13b,handoffs,fair,0.895,1200,0.312,0.846,0.822,0.871,above
olmo2-13b,sets,0shot,0.809,1280,0.275,0.739,0.711,0.769,above
olmo2-13b,sets,fair,0.866,1280,0.275,0.820,0.794,0.845,above
olmo2-13b,refilled,0shot,0.767,720,0.200,0.708,0.667,0.747,above
olmo2-13b,refills,0shot,0.581,600,0.200,0.476,0.426,0.525,above
olmo2-13b,codevar,0shot,0.410,1200,0.267,0.185,0.146,0.224,above
olmo2-13b,codevar,fair,0.526,1200,0.267,0.350,0.310,0.389,above
olmo2-13b,sharedlists,0shot,0.348,600,0.200,0.185,0.138,0.235,above
olmo2-13b,sharedlists,fair,0.553,600,0.200,0.442,0.394,0.487,above
olmo2-13b,box-narrative,0shot,0.655,200,0.253,0.530,0.438,0.622,above
olmo2-13b,box-narrative,fair,0.665,200,0.253,0.548,0.457,0.641,above
olmo2-13b,box-screenplay,0shot,0.645,200,0.253,0.525,0.436,0.610,above
olmo2-13b,box-screenplay,fair,0.545,200,0.253,0.377,0.272,0.469,above
olmo2-13b,box-code,0shot,0.660,200,0.253,0.515,0.420,0.603,above
olmo2-13b,box-code,fair,0.705,200,0.253,0.603,0.515,0.684,above
olmo2-13b,box-chat,0shot,0.625,200,0.253,0.504,0.409,0.594,above
olmo2-13b,box-chat,fair,0.595,200,0.253,0.472,0.374,0.560,above
olmo2-13b,you-static,0shot,0.973,300,0.292,0.962,0.934,0.986,above
olmo2-13b,i-moves,0shot,0.411,180,0.287,0.177,0.072,0.267,above
olmo2-13b,i-moves,fair,0.522,180,0.287,0.328,0.226,0.427,above
olmo2-13b,you-moves,0shot,0.444,180,0.287,0.222,0.119,0.332,above
olmo2-13b,you-moves,fair,0.544,180,0.287,0.362,0.259,0.465,above
olmo2-13b,you-named,0shot,0.500,180,0.287,0.295,0.193,0.398,above
olmo2-13b,you-named,fair,0.583,180,0.287,0.416,0.316,0.516,above
olmo2-13b,answer-self,0shot,0.507,2560,0.250,0.343,0.317,0.368,above
olmo2-13b,answer-self,fair,0.498,2560,0.250,0.331,0.304,0.355,above
olmo2-13b,grammar,ladder,,1080,,0.833,0.814,0.855,above
olmo2-13b,operators,ladder,,1080,,0.241,0.211,0.271,above
olmo2-13b,patterns,ladder,,1080,,0.609,0.576,0.643,above
olmo2-13b,facts,ladder,,768,,1.000,1.000,1.000,above
mimo-v25-base,lookup,0shot,0.734,1210,0.169,0.666,0.636,0.696,above
mimo-v25-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
mimo-v25-base,shortlist,0shot,0.974,3840,0.040,0.973,0.968,0.978,above
mimo-v25-base,pairs,0shot,0.548,256,0.125,0.484,0.413,0.551,above
mimo-v25-base,pairs,fair,0.996,256,0.125,0.996,0.987,1.000,above
mimo-v25-base,noise,0shot,0.953,720,0.200,0.941,0.920,0.958,above
mimo-v25-base,now,0shot,0.935,300,0.292,0.911,0.874,0.944,above
mimo-v25-base,handoffs,0shot,0.949,1200,0.312,0.925,0.906,0.942,above
mimo-v25-base,handoffs,fair,0.983,1200,0.312,0.975,0.963,0.985,above
mimo-v25-base,sets,0shot,0.813,1280,0.275,0.747,0.718,0.775,above
mimo-v25-base,sets,fair,0.941,1280,0.275,0.922,0.905,0.938,above
mimo-v25-base,refilled,0shot,0.754,720,0.200,0.693,0.652,0.732,above
mimo-v25-base,refills,0shot,0.676,600,0.200,0.595,0.547,0.640,above
mimo-v25-base,codevar,0shot,0.806,1200,0.267,0.732,0.702,0.762,above
mimo-v25-base,codevar,fair,0.911,1200,0.267,0.877,0.855,0.898,above
mimo-v25-base,sharedlists,0shot,0.799,600,0.200,0.749,0.709,0.786,above
mimo-v25-base,sharedlists,fair,0.783,600,0.200,0.729,0.690,0.767,above
mimo-v25-base,box-narrative,0shot,0.760,200,0.253,0.670,0.588,0.747,above
mimo-v25-base,box-narrative,fair,0.875,200,0.253,0.830,0.765,0.888,above
mimo-v25-base,box-screenplay,0shot,0.797,200,0.253,0.719,0.646,0.797,above
mimo-v25-base,box-screenplay,fair,0.902,200,0.253,0.867,0.808,0.920,above
mimo-v25-base,box-code,0shot,0.882,200,0.253,0.830,0.765,0.891,above
mimo-v25-base,box-code,fair,0.922,200,0.253,0.892,0.838,0.944,above
mimo-v25-base,box-chat,0shot,0.698,200,0.253,0.599,0.515,0.680,above
mimo-v25-base,box-chat,fair,0.880,200,0.253,0.832,0.765,0.894,above
mimo-v25-base,you-static,0shot,0.955,300,0.292,0.935,0.903,0.967,above
mimo-v25-base,i-moves,0shot,0.548,180,0.287,0.367,0.272,0.463,above
mimo-v25-base,i-moves,fair,0.578,180,0.287,0.406,0.311,0.501,above
mimo-v25-base,you-moves,0shot,0.433,180,0.287,0.208,0.107,0.301,above
mimo-v25-base,you-moves,fair,0.406,180,0.287,0.162,0.069,0.262,above
mimo-v25-base,you-named,0shot,0.456,180,0.287,0.236,0.139,0.338,above
mimo-v25-base,you-named,fair,0.453,180,0.287,0.225,0.127,0.326,above
mimo-v25-base,answer-self,0shot,0.473,2560,0.250,0.297,0.273,0.325,above
mimo-v25-base,answer-self,fair,0.733,2560,0.250,0.644,0.621,0.665,above
mimo-v25-base,grammar,ladder,,1080,,0.875,0.853,0.892,above
mimo-v25-base,operators,ladder,,1080,,0.244,0.218,0.277,above
mimo-v25-base,patterns,ladder,,1080,,0.908,0.891,0.923,above
mimo-v25-base,facts,ladder,,768,,1.000,1.000,1.000,above
gemma4-31b-base,lookup,0shot,0.718,1210,0.169,0.641,0.608,0.671,above
gemma4-31b-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
gemma4-31b-base,shortlist,0shot,0.992,3840,0.040,0.992,0.989,0.994,above
gemma4-31b-base,pairs,0shot,0.714,256,0.125,0.673,0.612,0.736,above
gemma4-31b-base,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
gemma4-31b-base,noise,0shot,0.995,720,0.200,0.994,0.987,0.999,above
gemma4-31b-base,now,0shot,0.991,300,0.292,0.988,0.977,0.998,above
gemma4-31b-base,handoffs,0shot,0.940,1200,0.312,0.912,0.890,0.930,above
gemma4-31b-base,handoffs,fair,0.978,1200,0.312,0.968,0.955,0.979,above
gemma4-31b-base,sets,0shot,0.923,1280,0.275,0.894,0.875,0.913,above
gemma4-31b-base,sets,fair,0.980,1280,0.275,0.974,0.964,0.983,above
gemma4-31b-base,refilled,0shot,0.685,720,0.200,0.607,0.565,0.648,above
gemma4-31b-base,refills,0shot,0.756,600,0.200,0.695,0.651,0.740,above
gemma4-31b-base,codevar,0shot,0.875,1200,0.267,0.828,0.804,0.853,above
gemma4-31b-base,codevar,fair,0.932,1200,0.267,0.906,0.886,0.924,above
gemma4-31b-base,sharedlists,0shot,0.799,600,0.200,0.749,0.710,0.788,above
gemma4-31b-base,sharedlists,fair,0.910,600,0.200,0.887,0.858,0.916,above
gemma4-31b-base,box-narrative,0shot,0.799,200,0.253,0.713,0.630,0.787,above
gemma4-31b-base,box-narrative,fair,0.853,200,0.253,0.797,0.728,0.859,above
gemma4-31b-base,box-screenplay,0shot,0.785,200,0.253,0.704,0.626,0.786,above
gemma4-31b-base,box-screenplay,fair,0.860,200,0.253,0.801,0.730,0.864,above
gemma4-31b-base,box-code,0shot,0.712,200,0.253,0.591,0.503,0.677,above
gemma4-31b-base,box-code,fair,0.920,200,0.253,0.881,0.825,0.933,above
gemma4-31b-base,box-chat,0shot,0.835,200,0.253,0.763,0.691,0.832,above
gemma4-31b-base,box-chat,fair,0.927,200,0.253,0.896,0.845,0.942,above
gemma4-31b-base,you-static,0shot,0.993,300,0.292,0.991,0.976,1.000,above
gemma4-31b-base,i-moves,0shot,0.644,180,0.287,0.503,0.407,0.591,above
gemma4-31b-base,i-moves,fair,0.722,180,0.287,0.607,0.510,0.689,above
gemma4-31b-base,you-moves,0shot,0.469,180,0.287,0.256,0.159,0.355,above
gemma4-31b-base,you-moves,fair,0.510,180,0.287,0.307,0.209,0.406,above
gemma4-31b-base,you-named,0shot,0.454,180,0.287,0.231,0.134,0.330,above
gemma4-31b-base,you-named,fair,0.666,180,0.287,0.527,0.442,0.619,above
gemma4-31b-base,answer-self,0shot,0.710,2560,0.250,0.614,0.591,0.637,above
gemma4-31b-base,answer-self,fair,0.788,2560,0.250,0.717,0.697,0.737,above
gemma4-31b-base,grammar,ladder,,1080,,0.921,0.904,0.937,above
gemma4-31b-base,operators,ladder,,1080,,0.315,0.285,0.344,above
gemma4-31b-base,patterns,ladder,,1080,,0.923,0.906,0.938,above
gemma4-31b-base,facts,ladder,,768,,1.000,1.000,1.000,above
kimi-k2-base,lookup,0shot,0.972,1210,0.169,0.961,0.948,0.973,above
kimi-k2-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
kimi-k2-base,shortlist,0shot,0.985,3840,0.040,0.985,0.981,0.988,above
kimi-k2-base,pairs,0shot,0.723,256,0.125,0.683,0.621,0.743,above
kimi-k2-base,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
kimi-k2-base,noise,0shot,0.969,720,0.200,0.962,0.947,0.976,above
kimi-k2-base,now,0shot,1.000,300,0.292,1.000,1.000,1.000,above
kimi-k2-base,handoffs,0shot,0.977,1200,0.312,0.966,0.954,0.978,above
kimi-k2-base,handoffs,fair,0.992,1200,0.312,0.988,0.981,0.994,above
kimi-k2-base,sets,0shot,0.984,1280,0.275,0.976,0.966,0.986,above
kimi-k2-base,sets,fair,0.985,1280,0.275,0.980,0.972,0.988,above
kimi-k2-base,refilled,0shot,0.827,720,0.200,0.784,0.747,0.819,above
kimi-k2-base,refills,0shot,0.876,600,0.200,0.845,0.814,0.878,above
kimi-k2-base,codevar,0shot,0.894,1200,0.267,0.851,0.828,0.874,above
kimi-k2-base,codevar,fair,0.927,1200,0.267,0.899,0.877,0.918,above
kimi-k2-base,sharedlists,0shot,0.945,600,0.200,0.931,0.910,0.951,above
kimi-k2-base,sharedlists,fair,0.938,600,0.200,0.922,0.899,0.945,above
kimi-k2-base,box-narrative,0shot,0.865,200,0.253,0.802,0.729,0.868,above
kimi-k2-base,box-narrative,fair,0.902,200,0.253,0.860,0.800,0.915,above
kimi-k2-base,box-screenplay,0shot,0.833,200,0.253,0.755,0.683,0.826,above
kimi-k2-base,box-screenplay,fair,0.890,200,0.253,0.840,0.769,0.899,above
kimi-k2-base,box-code,0shot,0.820,200,0.253,0.742,0.667,0.815,above
kimi-k2-base,box-code,fair,0.943,200,0.253,0.919,0.871,0.960,above
kimi-k2-base,box-chat,0shot,0.885,200,0.253,0.832,0.768,0.896,above
kimi-k2-base,box-chat,fair,0.873,200,0.253,0.821,0.754,0.885,above
kimi-k2-base,you-static,0shot,0.998,300,0.292,0.998,0.993,1.000,above
kimi-k2-base,i-moves,0shot,0.581,180,0.287,0.410,0.306,0.502,above
kimi-k2-base,i-moves,fair,0.717,180,0.287,0.597,0.509,0.683,above
kimi-k2-base,you-moves,0shot,0.514,180,0.287,0.319,0.222,0.417,above
kimi-k2-base,you-moves,fair,0.444,180,0.287,0.218,0.117,0.312,above
kimi-k2-base,you-named,0shot,0.600,180,0.287,0.439,0.339,0.536,above
kimi-k2-base,you-named,fair,0.603,180,0.287,0.442,0.344,0.541,above
kimi-k2-base,answer-self,0shot,0.623,2560,0.250,0.497,0.474,0.521,above
kimi-k2-base,answer-self,fair,0.791,2560,0.250,0.722,0.700,0.742,above
kimi-k2-base,grammar,ladder,,1080,,0.872,0.853,0.888,above
kimi-k2-base,operators,ladder,,1080,,0.214,0.187,0.245,above
kimi-k2-base,patterns,ladder,,1080,,0.883,0.860,0.902,above
kimi-k2-base,facts,ladder,,768,,1.000,1.000,1.000,above
olmo3-32b,lookup,0shot,0.713,1210,0.169,0.645,0.612,0.674,above
olmo3-32b,lookup,fair,0.998,1210,0.169,0.998,0.995,1.000,above
olmo3-32b,shortlist,0shot,0.893,3840,0.040,0.888,0.878,0.898,above
olmo3-32b,pairs,0shot,0.541,256,0.125,0.475,0.402,0.545,above
olmo3-32b,pairs,fair,0.863,256,0.125,0.844,0.795,0.888,above
olmo3-32b,noise,0shot,0.871,720,0.200,0.839,0.808,0.868,above
olmo3-32b,now,0shot,0.970,300,0.292,0.958,0.927,0.981,above
olmo3-32b,handoffs,0shot,0.836,1200,0.312,0.758,0.728,0.788,above
olmo3-32b,handoffs,fair,0.908,1200,0.312,0.866,0.843,0.888,above
olmo3-32b,sets,0shot,0.870,1280,0.275,0.825,0.800,0.850,above
olmo3-32b,sets,fair,0.882,1280,0.275,0.843,0.818,0.867,above
olmo3-32b,refilled,0shot,0.716,720,0.200,0.645,0.606,0.684,above
olmo3-32b,refills,0shot,0.565,600,0.200,0.456,0.407,0.503,above
olmo3-32b,codevar,0shot,0.727,1200,0.267,0.628,0.594,0.661,above
olmo3-32b,codevar,fair,0.840,1200,0.267,0.779,0.750,0.808,above
olmo3-32b,sharedlists,0shot,0.701,600,0.200,0.626,0.578,0.672,above
olmo3-32b,sharedlists,fair,0.792,600,0.200,0.740,0.697,0.780,above
olmo3-32b,box-narrative,0shot,0.735,200,0.253,0.628,0.543,0.707,above
olmo3-32b,box-narrative,fair,0.662,200,0.253,0.536,0.434,0.629,above
olmo3-32b,box-screenplay,0shot,0.715,200,0.253,0.611,0.521,0.700,above
olmo3-32b,box-screenplay,fair,0.698,200,0.253,0.581,0.487,0.668,above
olmo3-32b,box-code,0shot,0.760,200,0.253,0.659,0.572,0.739,above
olmo3-32b,box-code,fair,0.845,200,0.253,0.783,0.712,0.845,above
olmo3-32b,box-chat,0shot,0.767,200,0.253,0.681,0.597,0.756,above
olmo3-32b,box-chat,fair,0.718,200,0.253,0.603,0.506,0.695,above
olmo3-32b,you-static,0shot,0.980,300,0.292,0.971,0.947,0.991,above
olmo3-32b,i-moves,0shot,0.511,180,0.287,0.316,0.217,0.412,above
olmo3-32b,i-moves,fair,0.517,180,0.287,0.321,0.221,0.419,above
olmo3-32b,you-moves,0shot,0.469,180,0.287,0.261,0.158,0.364,above
olmo3-32b,you-moves,fair,0.400,180,0.287,0.159,0.059,0.263,above
olmo3-32b,you-named,0shot,0.492,180,0.287,0.283,0.187,0.389,above
olmo3-32b,you-named,fair,0.589,180,0.287,0.419,0.317,0.521,above
olmo3-32b,answer-self,0shot,0.536,2560,0.250,0.381,0.355,0.406,above
olmo3-32b,answer-self,fair,0.637,2560,0.250,0.516,0.491,0.539,above
olmo3-32b,grammar,ladder,,1080,,0.832,0.810,0.851,above
olmo3-32b,operators,ladder,,1080,,0.198,0.168,0.235,above
olmo3-32b,patterns,ladder,,1080,,0.671,0.642,0.705,above
olmo3-32b,facts,ladder,,768,,0.998,0.995,1.000,above
olmo3-32b-stage1,lookup,0shot,0.692,1210,0.169,0.612,0.580,0.643,above
olmo3-32b-stage1,lookup,fair,0.999,1210,0.169,0.999,0.997,1.000,above
olmo3-32b-stage1,shortlist,0shot,0.844,3840,0.040,0.837,0.825,0.849,above
olmo3-32b-stage1,pairs,0shot,0.568,256,0.125,0.507,0.440,0.574,above
olmo3-32b-stage1,pairs,fair,0.945,256,0.125,0.938,0.904,0.967,above
olmo3-32b-stage1,noise,0shot,0.926,720,0.200,0.907,0.884,0.929,above
olmo3-32b-stage1,now,0shot,0.914,300,0.292,0.882,0.840,0.921,above
olmo3-32b-stage1,handoffs,0shot,0.865,1200,0.312,0.803,0.774,0.831,above
olmo3-32b-stage1,handoffs,fair,0.916,1200,0.312,0.876,0.852,0.898,above
olmo3-32b-stage1,sets,0shot,0.835,1280,0.275,0.775,0.746,0.803,above
olmo3-32b-stage1,sets,fair,0.869,1280,0.275,0.829,0.805,0.852,above
olmo3-32b-stage1,refilled,0shot,0.752,720,0.200,0.690,0.650,0.728,above
olmo3-32b-stage1,refills,0shot,0.579,600,0.200,0.474,0.424,0.524,above
olmo3-32b-stage1,codevar,0shot,0.540,1200,0.267,0.371,0.332,0.409,above
olmo3-32b-stage1,codevar,fair,0.712,1200,0.267,0.603,0.566,0.639,above
olmo3-32b-stage1,sharedlists,0shot,0.347,600,0.200,0.183,0.137,0.234,above
olmo3-32b-stage1,sharedlists,fair,0.662,600,0.200,0.578,0.532,0.624,above
olmo3-32b-stage1,box-narrative,0shot,0.655,200,0.253,0.541,0.451,0.627,above
olmo3-32b-stage1,box-narrative,fair,0.713,200,0.253,0.614,0.524,0.699,above
olmo3-32b-stage1,box-screenplay,0shot,0.743,200,0.253,0.646,0.560,0.724,above
olmo3-32b-stage1,box-screenplay,fair,0.682,200,0.253,0.562,0.467,0.646,above
olmo3-32b-stage1,box-code,0shot,0.713,200,0.253,0.594,0.500,0.681,above
olmo3-32b-stage1,box-code,fair,0.828,200,0.253,0.759,0.688,0.828,above
olmo3-32b-stage1,box-chat,0shot,0.708,200,0.253,0.604,0.523,0.685,above
olmo3-32b-stage1,box-chat,fair,0.725,200,0.253,0.632,0.542,0.716,above
olmo3-32b-stage1,you-static,0shot,0.935,300,0.292,0.906,0.866,0.945,above
olmo3-32b-stage1,i-moves,0shot,0.481,180,0.287,0.270,0.167,0.368,above
olmo3-32b-stage1,i-moves,fair,0.447,180,0.287,0.224,0.117,0.323,above
olmo3-32b-stage1,you-moves,0shot,0.431,180,0.287,0.201,0.092,0.302,above
olmo3-32b-stage1,you-moves,fair,0.397,180,0.287,0.161,0.066,0.258,above
olmo3-32b-stage1,you-named,0shot,0.356,180,0.287,0.095,0.003,0.190,above
olmo3-32b-stage1,you-named,fair,0.427,180,0.287,0.191,0.092,0.286,above
olmo3-32b-stage1,answer-self,0shot,0.438,2560,0.250,0.251,0.225,0.276,above
olmo3-32b-stage1,answer-self,fair,0.552,2560,0.250,0.403,0.376,0.426,above
olmo3-32b-stage1,grammar,ladder,,1080,,0.835,0.813,0.860,above
olmo3-32b-stage1,operators,ladder,,1080,,0.269,0.242,0.297,above
olmo3-32b-stage1,patterns,ladder,,1080,,0.738,0.710,0.767,above
olmo3-32b-stage1,facts,ladder,,768,,1.000,1.000,1.000,above
glm4-32b-base,lookup,0shot,0.780,1210,0.169,0.724,0.695,0.753,above
glm4-32b-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
glm4-32b-base,shortlist,0shot,0.988,3840,0.040,0.987,0.984,0.991,above
glm4-32b-base,pairs,0shot,0.652,256,0.125,0.603,0.533,0.663,above
glm4-32b-base,pairs,fair,0.988,256,0.125,0.987,0.969,1.000,above
glm4-32b-base,noise,0shot,0.940,720,0.200,0.925,0.904,0.946,above
glm4-32b-base,now,0shot,0.965,300,0.292,0.948,0.917,0.976,above
glm4-32b-base,handoffs,0shot,0.929,1200,0.312,0.895,0.872,0.917,above
glm4-32b-base,handoffs,fair,0.972,1200,0.312,0.958,0.944,0.970,above
glm4-32b-base,sets,0shot,0.827,1280,0.275,0.763,0.737,0.793,above
glm4-32b-base,sets,fair,0.927,1280,0.275,0.901,0.881,0.920,above
glm4-32b-base,refilled,0shot,0.715,720,0.200,0.643,0.604,0.686,above
glm4-32b-base,refills,0shot,0.621,600,0.200,0.526,0.475,0.574,above
glm4-32b-base,codevar,0shot,0.793,1200,0.267,0.713,0.681,0.744,above
glm4-32b-base,codevar,fair,0.892,1200,0.267,0.851,0.826,0.874,above
glm4-32b-base,sharedlists,0shot,0.547,600,0.200,0.434,0.382,0.480,above
glm4-32b-base,sharedlists,fair,0.718,600,0.200,0.648,0.604,0.689,above
glm4-32b-base,box-narrative,0shot,0.755,200,0.253,0.661,0.579,0.742,above
glm4-32b-base,box-narrative,fair,0.810,200,0.253,0.747,0.671,0.818,above
glm4-32b-base,box-screenplay,0shot,0.715,200,0.253,0.593,0.509,0.673,above
glm4-32b-base,box-screenplay,fair,0.825,200,0.253,0.761,0.686,0.831,above
glm4-32b-base,box-code,0shot,0.795,200,0.253,0.702,0.614,0.779,above
glm4-32b-base,box-code,fair,0.899,200,0.253,0.860,0.797,0.912,above
glm4-32b-base,box-chat,0shot,0.755,200,0.253,0.668,0.583,0.751,above
glm4-32b-base,box-chat,fair,0.838,200,0.253,0.776,0.705,0.842,above
glm4-32b-base,you-static,0shot,0.955,300,0.292,0.934,0.897,0.966,above
glm4-32b-base,i-moves,0shot,0.585,180,0.287,0.417,0.314,0.511,above
glm4-32b-base,i-moves,fair,0.566,180,0.287,0.393,0.291,0.492,above
glm4-32b-base,you-moves,0shot,0.469,180,0.287,0.254,0.157,0.362,above
glm4-32b-base,you-moves,fair,0.347,180,0.287,0.085,-0.017,0.181,at
glm4-32b-base,you-named,0shot,0.485,180,0.287,0.276,0.174,0.372,above
glm4-32b-base,you-named,fair,0.486,180,0.287,0.279,0.170,0.378,above
glm4-32b-base,answer-self,0shot,0.517,2560,0.250,0.356,0.330,0.383,above
glm4-32b-base,answer-self,fair,0.677,2560,0.250,0.570,0.545,0.593,above
glm4-32b-base,grammar,ladder,,1080,,0.841,0.821,0.861,above
glm4-32b-base,operators,ladder,,1080,,0.229,0.199,0.259,above
glm4-32b-base,patterns,ladder,,1080,,0.829,0.806,0.849,above
glm4-32b-base,facts,ladder,,768,,1.000,1.000,1.000,above
falcon-h1-34b,lookup,0shot,0.967,1210,0.169,0.959,0.945,0.970,above
falcon-h1-34b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
falcon-h1-34b,shortlist,0shot,0.990,3840,0.040,0.990,0.986,0.993,above
falcon-h1-34b,pairs,0shot,0.555,256,0.125,0.491,0.424,0.558,above
falcon-h1-34b,pairs,fair,0.992,256,0.125,0.991,0.978,1.000,above
falcon-h1-34b,noise,0shot,0.974,720,0.200,0.967,0.950,0.981,above
falcon-h1-34b,now,0shot,0.947,300,0.292,0.924,0.888,0.958,above
falcon-h1-34b,handoffs,0shot,0.925,1200,0.312,0.888,0.864,0.911,above
falcon-h1-34b,handoffs,fair,0.963,1200,0.312,0.945,0.929,0.960,above
falcon-h1-34b,sets,0shot,0.867,1280,0.275,0.813,0.787,0.840,above
falcon-h1-34b,sets,fair,0.907,1280,0.275,0.874,0.853,0.893,above
falcon-h1-34b,refilled,0shot,0.838,720,0.200,0.797,0.760,0.830,above
falcon-h1-34b,refills,0shot,0.725,600,0.200,0.656,0.612,0.700,above
falcon-h1-34b,codevar,0shot,0.741,1200,0.267,0.640,0.605,0.672,above
falcon-h1-34b,codevar,fair,0.887,1200,0.267,0.843,0.818,0.868,above
falcon-h1-34b,sharedlists,0shot,0.801,600,0.200,0.751,0.712,0.791,above
falcon-h1-34b,sharedlists,fair,0.832,600,0.200,0.790,0.754,0.828,above
falcon-h1-34b,box-narrative,0shot,0.818,200,0.253,0.744,0.666,0.816,above
falcon-h1-34b,box-narrative,fair,0.880,200,0.253,0.827,0.759,0.892,above
falcon-h1-34b,box-screenplay,0shot,0.800,200,0.253,0.721,0.644,0.799,above
falcon-h1-34b,box-screenplay,fair,0.895,200,0.253,0.855,0.793,0.913,above
falcon-h1-34b,box-code,0shot,0.725,200,0.253,0.623,0.535,0.706,above
falcon-h1-34b,box-code,fair,0.915,200,0.253,0.879,0.821,0.931,above
falcon-h1-34b,box-chat,0shot,0.820,200,0.253,0.748,0.670,0.826,above
falcon-h1-34b,box-chat,fair,0.877,200,0.253,0.833,0.767,0.893,above
falcon-h1-34b,you-static,0shot,0.945,300,0.292,0.917,0.877,0.950,above
falcon-h1-34b,i-moves,0shot,0.511,180,0.287,0.318,0.218,0.410,above
falcon-h1-34b,i-moves,fair,0.633,180,0.287,0.487,0.382,0.582,above
falcon-h1-34b,you-moves,0shot,0.419,180,0.287,0.184,0.083,0.283,above
falcon-h1-34b,you-moves,fair,0.539,180,0.287,0.355,0.248,0.458,above
falcon-h1-34b,you-named,0shot,0.489,180,0.287,0.278,0.177,0.386,above
falcon-h1-34b,you-named,fair,0.578,180,0.287,0.406,0.302,0.506,above
falcon-h1-34b,answer-self,0shot,0.601,2560,0.250,0.468,0.443,0.493,above
falcon-h1-34b,answer-self,fair,0.738,2560,0.250,0.650,0.627,0.672,above
falcon-h1-34b,grammar,ladder,,1080,,0.865,0.849,0.882,above
falcon-h1-34b,operators,ladder,,1080,,0.246,0.210,0.278,above
falcon-h1-34b,patterns,ladder,,1080,,0.782,0.753,0.811,above
falcon-h1-34b,facts,ladder,,768,,1.000,1.000,1.000,above
dsv3-base,lookup,0shot,0.682,1210,0.169,0.604,0.572,0.636,above
dsv3-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
dsv3-base,shortlist,0shot,0.962,3840,0.040,0.960,0.954,0.966,above
dsv3-base,pairs,0shot,0.532,256,0.125,0.465,0.397,0.528,above
dsv3-base,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
dsv3-base,noise,0shot,0.925,720,0.200,0.906,0.881,0.930,above
dsv3-base,now,0shot,0.969,300,0.292,0.958,0.933,0.980,above
dsv3-base,handoffs,0shot,0.965,1200,0.312,0.950,0.934,0.964,above
dsv3-base,handoffs,fair,0.994,1200,0.312,0.991,0.984,0.996,above
dsv3-base,sets,0shot,0.979,1280,0.275,0.970,0.959,0.982,above
dsv3-base,sets,fair,0.984,1280,0.275,0.977,0.967,0.985,above
dsv3-base,refilled,0shot,0.846,720,0.200,0.807,0.777,0.839,above
dsv3-base,refills,0shot,0.795,600,0.200,0.743,0.701,0.783,above
dsv3-base,codevar,0shot,0.886,1200,0.267,0.844,0.820,0.868,above
dsv3-base,codevar,fair,0.948,1200,0.267,0.928,0.912,0.945,above
dsv3-base,sharedlists,0shot,0.940,600,0.200,0.925,0.901,0.950,above
dsv3-base,sharedlists,fair,0.960,600,0.200,0.950,0.930,0.969,above
dsv3-base,box-narrative,0shot,0.823,200,0.253,0.753,0.676,0.826,above
dsv3-base,box-narrative,fair,0.895,200,0.253,0.854,0.791,0.912,above
dsv3-base,box-screenplay,0shot,0.802,200,0.253,0.719,0.638,0.796,above
dsv3-base,box-screenplay,fair,0.905,200,0.253,0.862,0.800,0.918,above
dsv3-base,box-code,0shot,0.882,200,0.253,0.830,0.763,0.890,above
dsv3-base,box-code,fair,0.940,200,0.253,0.913,0.862,0.958,above
dsv3-base,box-chat,0shot,0.853,200,0.253,0.784,0.710,0.852,above
dsv3-base,box-chat,fair,0.889,200,0.253,0.846,0.785,0.905,above
dsv3-base,you-static,0shot,1.000,300,0.292,1.000,1.000,1.000,above
dsv3-base,i-moves,0shot,0.579,180,0.287,0.408,0.316,0.499,above
dsv3-base,i-moves,fair,0.761,180,0.287,0.662,0.576,0.747,above
dsv3-base,you-moves,0shot,0.678,180,0.287,0.552,0.460,0.641,above
dsv3-base,you-moves,fair,0.501,180,0.287,0.296,0.197,0.391,above
dsv3-base,you-named,0shot,0.638,180,0.287,0.492,0.392,0.584,above
dsv3-base,you-named,fair,0.736,180,0.287,0.635,0.544,0.725,above
dsv3-base,answer-self,0shot,0.582,2560,0.250,0.443,0.418,0.468,above
dsv3-base,answer-self,fair,0.783,2560,0.250,0.711,0.689,0.731,above
dsv3-base,grammar,ladder,,1080,,0.861,0.843,0.876,above
dsv3-base,operators,ladder,,1080,,0.306,0.278,0.338,above
dsv3-base,patterns,ladder,,1080,,0.966,0.956,0.976,above
dsv3-base,facts,ladder,,768,,1.000,1.000,1.000,above
mimo-v25-pro-base,lookup,0shot,0.724,1210,0.169,0.655,0.624,0.682,above
mimo-v25-pro-base,lookup,fair,0.999,1210,0.169,0.999,0.997,1.000,above
mimo-v25-pro-base,shortlist,0shot,0.959,3840,0.040,0.957,0.951,0.963,above
mimo-v25-pro-base,pairs,0shot,0.614,256,0.125,0.559,0.491,0.616,above
mimo-v25-pro-base,pairs,fair,0.992,256,0.125,0.991,0.978,1.000,above
mimo-v25-pro-base,noise,0shot,0.934,720,0.200,0.918,0.898,0.936,above
mimo-v25-pro-base,now,0shot,0.944,300,0.292,0.919,0.883,0.951,above
mimo-v25-pro-base,handoffs,0shot,0.957,1200,0.312,0.938,0.924,0.952,above
mimo-v25-pro-base,handoffs,fair,0.989,1200,0.312,0.984,0.977,0.990,above
mimo-v25-pro-base,sets,0shot,0.917,1280,0.275,0.884,0.867,0.902,above
mimo-v25-pro-base,sets,fair,0.967,1280,0.275,0.955,0.943,0.966,above
mimo-v25-pro-base,refilled,0shot,0.673,720,0.200,0.591,0.552,0.631,above
mimo-v25-pro-base,refills,0shot,0.652,600,0.200,0.566,0.522,0.608,above
mimo-v25-pro-base,codevar,0shot,0.831,1200,0.267,0.763,0.738,0.790,above
mimo-v25-pro-base,codevar,fair,0.909,1200,0.267,0.872,0.851,0.892,above
mimo-v25-pro-base,sharedlists,0shot,0.941,600,0.200,0.926,0.906,0.946,above
mimo-v25-pro-base,sharedlists,fair,0.848,600,0.200,0.809,0.774,0.842,above
mimo-v25-pro-base,box-narrative,0shot,0.875,200,0.253,0.821,0.758,0.881,above
mimo-v25-pro-base,box-narrative,fair,0.936,200,0.253,0.916,0.871,0.954,above
mimo-v25-pro-base,box-screenplay,0shot,0.847,200,0.253,0.783,0.710,0.850,above
mimo-v25-pro-base,box-screenplay,fair,0.940,200,0.253,0.917,0.874,0.954,above
mimo-v25-pro-base,box-code,0shot,0.862,200,0.253,0.808,0.743,0.869,above
mimo-v25-pro-base,box-code,fair,0.953,200,0.253,0.933,0.895,0.968,above
mimo-v25-pro-base,box-chat,0shot,0.834,200,0.253,0.764,0.694,0.832,above
mimo-v25-pro-base,box-chat,fair,0.897,200,0.253,0.858,0.803,0.908,above
mimo-v25-pro-base,you-static,0shot,0.987,300,0.292,0.980,0.963,0.995,above
mimo-v25-pro-base,i-moves,0shot,0.481,180,0.287,0.275,0.192,0.363,above
mimo-v25-pro-base,i-moves,fair,0.640,180,0.287,0.492,0.395,0.580,above
mimo-v25-pro-base,you-moves,0shot,0.519,180,0.287,0.330,0.252,0.416,above
mimo-v25-pro-base,you-moves,fair,0.531,180,0.287,0.340,0.246,0.433,above
mimo-v25-pro-base,you-named,0shot,0.501,180,0.287,0.300,0.218,0.390,above
mimo-v25-pro-base,you-named,fair,0.677,180,0.287,0.547,0.461,0.634,above
mimo-v25-pro-base,answer-self,0shot,0.546,2560,0.250,0.395,0.371,0.417,above
mimo-v25-pro-base,answer-self,fair,0.776,2560,0.250,0.701,0.680,0.720,above
mimo-v25-pro-base,grammar,ladder,,1080,,0.906,0.893,0.922,above
mimo-v25-pro-base,operators,ladder,,1080,,0.250,0.227,0.279,above
mimo-v25-pro-base,patterns,ladder,,1080,,0.923,0.909,0.939,above
mimo-v25-pro-base,facts,ladder,,768,,1.000,1.000,1.000,above
dsv4-pro-base,lookup,0shot,0.705,1210,0.169,0.624,0.592,0.654,above
dsv4-pro-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
dsv4-pro-base,shortlist,0shot,0.988,3840,0.040,0.988,0.985,0.991,above
dsv4-pro-base,pairs,0shot,0.644,256,0.125,0.593,0.528,0.654,above
dsv4-pro-base,pairs,fair,0.998,256,0.125,0.998,0.993,1.000,above
dsv4-pro-base,noise,0shot,0.915,720,0.200,0.893,0.869,0.917,above
dsv4-pro-base,now,0shot,0.980,300,0.292,0.971,0.948,0.991,above
dsv4-pro-base,handoffs,0shot,0.965,1200,0.312,0.949,0.933,0.963,above
dsv4-pro-base,handoffs,fair,0.994,1200,0.312,0.991,0.984,0.996,above
dsv4-pro-base,sets,0shot,0.921,1280,0.275,0.894,0.874,0.914,above
dsv4-pro-base,sets,fair,0.986,1280,0.275,0.982,0.974,0.989,above
dsv4-pro-base,refilled,0shot,0.847,720,0.200,0.809,0.777,0.839,above
dsv4-pro-base,refills,0shot,0.790,600,0.200,0.738,0.699,0.777,above
dsv4-pro-base,codevar,0shot,0.883,1200,0.267,0.838,0.813,0.862,above
dsv4-pro-base,codevar,fair,0.911,1200,0.267,0.874,0.851,0.897,above
dsv4-pro-base,sharedlists,0shot,0.953,600,0.200,0.941,0.918,0.962,above
dsv4-pro-base,sharedlists,fair,0.905,600,0.200,0.881,0.848,0.910,above
dsv4-pro-base,box-narrative,0shot,0.818,200,0.253,0.744,0.671,0.821,above
dsv4-pro-base,box-narrative,fair,0.905,200,0.253,0.864,0.802,0.918,above
dsv4-pro-base,box-screenplay,0shot,0.795,200,0.253,0.714,0.640,0.791,above
dsv4-pro-base,box-screenplay,fair,0.910,200,0.253,0.867,0.806,0.924,above
dsv4-pro-base,box-code,0shot,0.890,200,0.253,0.845,0.783,0.906,above
dsv4-pro-base,box-code,fair,0.925,200,0.253,0.888,0.825,0.939,above
dsv4-pro-base,box-chat,0shot,0.877,200,0.253,0.828,0.761,0.894,above
dsv4-pro-base,box-chat,fair,0.880,200,0.253,0.823,0.748,0.886,above
dsv4-pro-base,you-static,0shot,0.992,300,0.292,0.988,0.972,1.000,above
dsv4-pro-base,i-moves,0shot,0.487,180,0.287,0.286,0.191,0.383,above
dsv4-pro-base,i-moves,fair,0.782,180,0.287,0.697,0.608,0.776,above
dsv4-pro-base,you-moves,0shot,0.468,180,0.287,0.259,0.165,0.352,above
dsv4-pro-base,you-moves,fair,0.469,180,0.287,0.256,0.153,0.356,above
dsv4-pro-base,you-named,0shot,0.376,180,0.287,0.118,0.029,0.219,above
dsv4-pro-base,you-named,fair,0.550,180,0.287,0.370,0.274,0.474,above
dsv4-pro-base,answer-self,0shot,0.584,2560,0.250,0.445,0.420,0.470,above
dsv4-pro-base,answer-self,fair,0.781,2560,0.250,0.708,0.685,0.728,above
dsv4-pro-base,grammar,ladder,,1080,,0.882,0.862,0.903,above
dsv4-pro-base,operators,ladder,,1080,,0.220,0.193,0.249,above
dsv4-pro-base,patterns,ladder,,1080,,0.936,0.922,0.951,above
dsv4-pro-base,facts,ladder,,768,,1.000,1.000,1.000,above
nemotron3-ultra-550b-base,lookup,0shot,0.902,1210,0.169,0.875,0.853,0.895,above
nemotron3-ultra-550b-base,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
nemotron3-ultra-550b-base,shortlist,0shot,0.994,3840,0.040,0.993,0.991,0.996,above
nemotron3-ultra-550b-base,pairs,0shot,0.945,256,0.125,0.938,0.902,0.967,above
nemotron3-ultra-550b-base,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
nemotron3-ultra-550b-base,noise,0shot,0.997,720,0.200,0.996,0.990,1.000,above
nemotron3-ultra-550b-base,now,0shot,0.985,300,0.292,0.980,0.960,0.996,above
nemotron3-ultra-550b-base,handoffs,0shot,0.964,1200,0.312,0.946,0.931,0.961,above
nemotron3-ultra-550b-base,handoffs,fair,0.994,1200,0.312,0.992,0.985,0.998,above
nemotron3-ultra-550b-base,sets,0shot,0.950,1280,0.275,0.933,0.917,0.948,above
nemotron3-ultra-550b-base,sets,fair,0.998,1280,0.275,0.997,0.995,1.000,above
nemotron3-ultra-550b-base,refilled,0shot,0.821,720,0.200,0.776,0.741,0.811,above
nemotron3-ultra-550b-base,refills,0shot,0.748,600,0.200,0.685,0.641,0.727,above
nemotron3-ultra-550b-base,codevar,0shot,0.856,1200,0.267,0.803,0.776,0.829,above
nemotron3-ultra-550b-base,codevar,fair,0.905,1200,0.267,0.867,0.841,0.889,above
nemotron3-ultra-550b-base,sharedlists,0shot,0.884,600,0.200,0.855,0.823,0.884,above
nemotron3-ultra-550b-base,sharedlists,fair,0.986,600,0.200,0.982,0.970,0.993,above
nemotron3-ultra-550b-base,box-narrative,0shot,0.850,200,0.253,0.783,0.710,0.855,above
nemotron3-ultra-550b-base,box-narrative,fair,0.940,200,0.253,0.915,0.865,0.958,above
nemotron3-ultra-550b-base,box-screenplay,0shot,0.795,200,0.253,0.712,0.637,0.790,above
nemotron3-ultra-550b-base,box-screenplay,fair,0.948,200,0.253,0.927,0.879,0.966,above
nemotron3-ultra-550b-base,box-code,0shot,0.860,200,0.253,0.799,0.728,0.865,above
nemotron3-ultra-550b-base,box-code,fair,0.945,200,0.253,0.923,0.877,0.964,above
nemotron3-ultra-550b-base,box-chat,0shot,0.850,200,0.253,0.786,0.715,0.854,above
nemotron3-ultra-550b-base,box-chat,fair,0.920,200,0.253,0.888,0.825,0.935,above
nemotron3-ultra-550b-base,you-static,0shot,0.993,300,0.292,0.991,0.978,1.000,above
nemotron3-ultra-550b-base,i-moves,0shot,0.657,180,0.287,0.515,0.418,0.609,above
nemotron3-ultra-550b-base,i-moves,fair,0.800,180,0.287,0.719,0.641,0.798,above
nemotron3-ultra-550b-base,you-moves,0shot,0.533,180,0.287,0.352,0.255,0.456,above
nemotron3-ultra-550b-base,you-moves,fair,0.503,180,0.287,0.304,0.198,0.397,above
nemotron3-ultra-550b-base,you-named,0shot,0.586,180,0.287,0.417,0.320,0.517,above
nemotron3-ultra-550b-base,you-named,fair,0.619,180,0.287,0.468,0.365,0.560,above
nemotron3-ultra-550b-base,answer-self,0shot,0.716,2560,0.250,0.622,0.599,0.643,above
nemotron3-ultra-550b-base,answer-self,fair,0.794,2560,0.250,0.725,0.704,0.744,above
nemotron3-ultra-550b-base,grammar,ladder,,1080,,0.873,0.856,0.892,above
nemotron3-ultra-550b-base,operators,ladder,,1080,,0.232,0.202,0.264,above
nemotron3-ultra-550b-base,patterns,ladder,,1080,,0.878,0.857,0.896,above
nemotron3-ultra-550b-base,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-70b,lookup,0shot,0.524,1210,0.169,0.416,0.380,0.448,above
llama3.1-70b,lookup,fair,0.997,1210,0.169,0.997,0.993,1.000,above
llama3.1-70b,shortlist,0shot,0.986,3840,0.040,0.985,0.981,0.989,above
llama3.1-70b,pairs,0shot,0.300,256,0.125,0.200,0.138,0.260,above
llama3.1-70b,pairs,fair,0.996,256,0.125,0.996,0.987,1.000,above
llama3.1-70b,noise,0shot,0.989,720,0.200,0.986,0.977,0.995,above
llama3.1-70b,now,0shot,0.965,300,0.292,0.951,0.923,0.977,above
llama3.1-70b,handoffs,0shot,0.924,1200,0.312,0.889,0.867,0.910,above
llama3.1-70b,handoffs,fair,0.981,1200,0.312,0.971,0.960,0.982,above
llama3.1-70b,sets,0shot,0.897,1280,0.275,0.859,0.836,0.880,above
llama3.1-70b,sets,fair,0.977,1280,0.275,0.971,0.960,0.980,above
llama3.1-70b,refilled,0shot,0.632,720,0.200,0.540,0.496,0.583,above
llama3.1-70b,refills,0shot,0.665,600,0.200,0.581,0.533,0.626,above
llama3.1-70b,codevar,0shot,0.805,1200,0.267,0.731,0.699,0.761,above
llama3.1-70b,codevar,fair,0.890,1200,0.267,0.843,0.818,0.867,above
llama3.1-70b,sharedlists,0shot,0.840,600,0.200,0.800,0.766,0.837,above
llama3.1-70b,sharedlists,fair,0.720,600,0.200,0.650,0.606,0.693,above
llama3.1-70b,box-narrative,0shot,0.713,200,0.253,0.589,0.502,0.677,above
llama3.1-70b,box-narrative,fair,0.885,200,0.253,0.841,0.778,0.899,above
llama3.1-70b,box-screenplay,0shot,0.705,200,0.253,0.599,0.511,0.678,above
llama3.1-70b,box-screenplay,fair,0.890,200,0.253,0.850,0.788,0.906,above
llama3.1-70b,box-code,0shot,0.767,200,0.253,0.670,0.575,0.754,above
llama3.1-70b,box-code,fair,0.915,200,0.253,0.878,0.817,0.929,above
llama3.1-70b,box-chat,0shot,0.724,200,0.253,0.620,0.534,0.702,above
llama3.1-70b,box-chat,fair,0.870,200,0.253,0.816,0.747,0.880,above
llama3.1-70b,you-static,0shot,0.997,300,0.292,0.995,0.985,1.000,above
llama3.1-70b,i-moves,0shot,0.589,180,0.287,0.429,0.334,0.523,above
llama3.1-70b,i-moves,fair,0.567,180,0.287,0.394,0.291,0.491,above
llama3.1-70b,you-moves,0shot,0.527,180,0.287,0.340,0.240,0.439,above
llama3.1-70b,you-moves,fair,0.376,180,0.287,0.126,0.022,0.220,above
llama3.1-70b,you-named,0shot,0.585,180,0.287,0.419,0.326,0.507,above
llama3.1-70b,you-named,fair,0.612,180,0.287,0.455,0.349,0.550,above
llama3.1-70b,answer-self,0shot,0.601,2560,0.250,0.468,0.444,0.493,above
llama3.1-70b,answer-self,fair,0.762,2560,0.250,0.682,0.660,0.702,above
llama3.1-70b,grammar,ladder,,1080,,0.855,0.836,0.877,above
llama3.1-70b,operators,ladder,,1080,,0.312,0.282,0.345,above
llama3.1-70b,patterns,ladder,,1080,,0.914,0.899,0.931,above
llama3.1-70b,facts,ladder,,768,,1.000,1.000,1.000,above
qwen2.5-72b,lookup,0shot,0.948,1210,0.169,0.934,0.917,0.950,above
qwen2.5-72b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
qwen2.5-72b,shortlist,0shot,0.987,3840,0.040,0.986,0.982,0.990,above
qwen2.5-72b,pairs,0shot,0.613,256,0.125,0.557,0.488,0.621,above
qwen2.5-72b,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
qwen2.5-72b,noise,0shot,0.989,720,0.200,0.986,0.976,0.995,above
qwen2.5-72b,now,0shot,0.967,300,0.292,0.954,0.924,0.979,above
qwen2.5-72b,handoffs,0shot,0.940,1200,0.312,0.912,0.892,0.930,above
qwen2.5-72b,handoffs,fair,0.984,1200,0.312,0.976,0.965,0.986,above
qwen2.5-72b,sets,0shot,0.873,1280,0.275,0.824,0.798,0.848,above
qwen2.5-72b,sets,fair,0.954,1280,0.275,0.938,0.921,0.952,above
qwen2.5-72b,refilled,0shot,0.699,720,0.200,0.623,0.581,0.667,above
qwen2.5-72b,refills,0shot,0.680,600,0.200,0.600,0.551,0.648,above
qwen2.5-72b,codevar,0shot,0.750,1200,0.267,0.653,0.620,0.686,above
qwen2.5-72b,codevar,fair,0.873,1200,0.267,0.823,0.797,0.850,above
qwen2.5-72b,sharedlists,0shot,0.838,600,0.200,0.798,0.762,0.832,above
qwen2.5-72b,sharedlists,fair,0.906,600,0.200,0.882,0.854,0.909,above
qwen2.5-72b,box-narrative,0shot,0.858,200,0.253,0.797,0.723,0.865,above
qwen2.5-72b,box-narrative,fair,0.925,200,0.253,0.896,0.846,0.945,above
qwen2.5-72b,box-screenplay,0shot,0.815,200,0.253,0.740,0.664,0.816,above
qwen2.5-72b,box-screenplay,fair,0.899,200,0.253,0.864,0.803,0.917,above
qwen2.5-72b,box-code,0shot,0.733,200,0.253,0.628,0.535,0.712,above
qwen2.5-72b,box-code,fair,0.858,200,0.253,0.802,0.730,0.871,above
qwen2.5-72b,box-chat,0shot,0.835,200,0.253,0.774,0.701,0.840,above
qwen2.5-72b,box-chat,fair,0.820,200,0.253,0.753,0.684,0.822,above
qwen2.5-72b,you-static,0shot,0.932,300,0.292,0.899,0.856,0.934,above
qwen2.5-72b,i-moves,0shot,0.502,180,0.287,0.299,0.204,0.401,above
qwen2.5-72b,i-moves,fair,0.694,180,0.287,0.570,0.479,0.652,above
qwen2.5-72b,you-moves,0shot,0.500,180,0.287,0.305,0.200,0.409,above
qwen2.5-72b,you-moves,fair,0.507,180,0.287,0.307,0.212,0.406,above
qwen2.5-72b,you-named,0shot,0.566,180,0.287,0.397,0.301,0.499,above
qwen2.5-72b,you-named,fair,0.717,180,0.287,0.606,0.515,0.691,above
qwen2.5-72b,answer-self,0shot,0.641,2560,0.250,0.521,0.498,0.545,above
qwen2.5-72b,answer-self,fair,0.750,2560,0.250,0.666,0.645,0.689,above
qwen2.5-72b,grammar,ladder,,1080,,0.873,0.856,0.889,above
qwen2.5-72b,operators,ladder,,1080,,0.241,0.216,0.270,above
qwen2.5-72b,patterns,ladder,,1080,,0.921,0.907,0.936,above
qwen2.5-72b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-405b,lookup,0shot,0.785,1210,0.169,0.732,0.705,0.759,above
llama3.1-405b,lookup,fair,1.000,1210,0.169,1.000,1.000,1.000,above
llama3.1-405b,shortlist,0shot,0.979,3840,0.040,0.978,0.973,0.982,above
llama3.1-405b,pairs,0shot,0.818,256,0.125,0.792,0.739,0.846,above
llama3.1-405b,pairs,fair,1.000,256,0.125,1.000,1.000,1.000,above
llama3.1-405b,noise,0shot,0.988,720,0.200,0.985,0.975,0.994,above
llama3.1-405b,now,0shot,0.975,300,0.292,0.965,0.939,0.984,above
llama3.1-405b,handoffs,0shot,0.971,1200,0.312,0.957,0.943,0.972,above
llama3.1-405b,handoffs,fair,0.998,1200,0.312,0.996,0.991,1.000,above
llama3.1-405b,sets,0shot,0.948,1280,0.275,0.929,0.911,0.945,above
llama3.1-405b,sets,fair,0.995,1280,0.275,0.994,0.989,0.998,above
llama3.1-405b,refilled,0shot,0.792,720,0.200,0.740,0.703,0.775,above
llama3.1-405b,refills,0shot,0.797,600,0.200,0.746,0.706,0.784,above
llama3.1-405b,codevar,0shot,0.850,1200,0.267,0.792,0.764,0.818,above
llama3.1-405b,codevar,fair,0.942,1200,0.267,0.918,0.898,0.936,above
llama3.1-405b,sharedlists,0shot,0.880,600,0.200,0.850,0.820,0.881,above
llama3.1-405b,sharedlists,fair,0.979,600,0.200,0.974,0.960,0.987,above
llama3.1-405b,box-narrative,0shot,0.838,200,0.253,0.771,0.698,0.836,above
llama3.1-405b,box-narrative,fair,0.900,200,0.253,0.860,0.802,0.914,above
llama3.1-405b,box-screenplay,0shot,0.777,200,0.253,0.689,0.603,0.768,above
llama3.1-405b,box-screenplay,fair,0.922,200,0.253,0.894,0.838,0.943,above
llama3.1-405b,box-code,0shot,0.833,200,0.253,0.762,0.685,0.829,above
llama3.1-405b,box-code,fair,0.885,200,0.253,0.832,0.765,0.891,above
llama3.1-405b,box-chat,0shot,0.865,200,0.253,0.806,0.740,0.868,above
llama3.1-405b,box-chat,fair,0.930,200,0.253,0.904,0.857,0.947,above
llama3.1-405b,you-static,0shot,0.987,300,0.292,0.981,0.962,0.996,above
llama3.1-405b,i-moves,0shot,0.514,180,0.287,0.314,0.213,0.413,above
llama3.1-405b,i-moves,fair,0.758,180,0.287,0.661,0.573,0.748,above
llama3.1-405b,you-moves,0shot,0.500,180,0.287,0.301,0.206,0.400,above
llama3.1-405b,you-moves,fair,0.578,180,0.287,0.407,0.309,0.505,above
llama3.1-405b,you-named,0shot,0.531,180,0.287,0.342,0.242,0.444,above
llama3.1-405b,you-named,fair,0.633,180,0.287,0.489,0.393,0.585,above
llama3.1-405b,answer-self,0shot,0.590,2560,0.250,0.454,0.427,0.479,above
llama3.1-405b,answer-self,fair,0.796,2560,0.250,0.728,0.706,0.748,above
llama3.1-405b,grammar,ladder,,1080,,0.897,0.883,0.913,above
llama3.1-405b,operators,ladder,,1080,,0.227,0.201,0.261,above
llama3.1-405b,patterns,ladder,,1080,,0.987,0.980,0.993,above
llama3.1-405b,facts,ladder,,768,,1.000,1.000,1.000,above
tsumugi-s17-a5B,lookup,0shot,0.467,1210,0.169,0.356,0.321,0.390,above
tsumugi-s17-a5B,lookup,fair,0.743,1210,0.169,0.692,0.662,0.720,above
tsumugi-s17-a5B,shortlist,0shot,0.501,3840,0.040,0.480,0.463,0.497,above
tsumugi-s17-a5B,pairs,0shot,0.418,256,0.125,0.335,0.263,0.397,above
tsumugi-s17-a5B,pairs,fair,0.625,256,0.125,0.571,0.504,0.638,above
tsumugi-s17-a5B,noise,0shot,0.505,720,0.200,0.381,0.338,0.425,above
tsumugi-s17-a5B,now,0shot,0.440,300,0.292,0.208,0.127,0.282,above
tsumugi-s17-a5B,handoffs,0shot,0.378,1200,0.312,0.094,0.056,0.130,above
tsumugi-s17-a5B,handoffs,fair,0.494,1200,0.312,0.260,0.223,0.296,above
tsumugi-s17-a5B,sets,0shot,0.418,1280,0.275,0.207,0.170,0.245,above
tsumugi-s17-a5B,sets,fair,0.483,1280,0.275,0.297,0.258,0.337,above
tsumugi-s17-a5B,refilled,0shot,0.264,720,0.200,0.080,0.043,0.120,above
tsumugi-s17-a5B,refills,0shot,0.198,600,0.200,-0.003,-0.041,0.034,at
tsumugi-s17-a5B,codevar,0shot,0.258,1200,0.267,-0.008,-0.042,0.026,at
tsumugi-s17-a5B,codevar,fair,0.256,1200,0.267,-0.018,-0.051,0.016,at
tsumugi-s17-a5B,sharedlists,0shot,0.265,600,0.200,0.081,0.038,0.127,above
tsumugi-s17-a5B,sharedlists,fair,0.210,600,0.200,0.013,-0.026,0.051,at
tsumugi-s17-a5B,box-narrative,0shot,0.343,200,0.253,0.107,0.017,0.195,above
tsumugi-s17-a5B,box-narrative,fair,0.323,200,0.253,0.075,-0.020,0.164,at
tsumugi-s17-a5B,box-screenplay,0shot,0.412,200,0.253,0.217,0.125,0.308,above
tsumugi-s17-a5B,box-screenplay,fair,0.393,200,0.253,0.181,0.085,0.278,above
tsumugi-s17-a5B,box-code,0shot,0.290,200,0.253,0.061,-0.025,0.149,at
tsumugi-s17-a5B,box-code,fair,0.412,200,0.253,0.197,0.099,0.292,above
tsumugi-s17-a5B,box-chat,0shot,0.378,200,0.253,0.158,0.065,0.257,above
tsumugi-s17-a5B,box-chat,fair,0.439,200,0.253,0.234,0.134,0.328,above
tsumugi-s17-a5B,you-static,0shot,0.348,300,0.292,0.077,0.008,0.153,above
tsumugi-s17-a5B,i-moves,0shot,0.292,180,0.287,-0.001,-0.092,0.096,at
tsumugi-s17-a5B,i-moves,fair,0.269,180,0.287,-0.030,-0.115,0.056,at
tsumugi-s17-a5B,you-moves,0shot,0.228,180,0.287,-0.089,-0.185,0.006,at
tsumugi-s17-a5B,you-moves,fair,0.294,180,0.287,0.007,-0.086,0.104,at
tsumugi-s17-a5B,you-named,0shot,0.300,180,0.287,0.018,-0.077,0.114,at
tsumugi-s17-a5B,you-named,fair,0.294,180,0.287,0.006,-0.084,0.098,at
tsumugi-s17-a5B,answer-self,0shot,0.287,2560,0.250,0.049,0.026,0.072,above
tsumugi-s17-a5B,answer-self,fair,0.279,2560,0.250,0.038,0.016,0.062,above
tsumugi-s17-a5B,grammar,ladder,,1080,,0.648,0.620,0.677,above
tsumugi-s17-a5B,operators,ladder,,1080,,0.255,0.229,0.286,above
tsumugi-s17-a5B,patterns,ladder,,1080,,0.458,0.424,0.490,above
tsumugi-s17-a5B,facts,ladder,,768,,0.640,0.598,0.676,above
tsumugi-s7-a5B,lookup,0shot,0.452,1210,0.169,0.332,0.299,0.365,above
tsumugi-s7-a5B,lookup,fair,0.844,1210,0.169,0.811,0.786,0.836,above
tsumugi-s7-a5B,shortlist,0shot,0.521,3840,0.040,0.501,0.485,0.516,above
tsumugi-s7-a5B,pairs,0shot,0.340,256,0.125,0.245,0.176,0.306,above
tsumugi-s7-a5B,pairs,fair,0.646,256,0.125,0.596,0.527,0.663,above
tsumugi-s7-a5B,noise,0shot,0.558,720,0.200,0.448,0.402,0.496,above
tsumugi-s7-a5B,now,0shot,0.372,300,0.292,0.114,0.042,0.193,above
tsumugi-s7-a5B,handoffs,0shot,0.337,1200,0.312,0.034,-0.004,0.073,at
tsumugi-s7-a5B,handoffs,fair,0.507,1200,0.312,0.278,0.237,0.318,above
tsumugi-s7-a5B,sets,0shot,0.423,1280,0.275,0.213,0.175,0.248,above
tsumugi-s7-a5B,sets,fair,0.561,1280,0.275,0.407,0.371,0.446,above
tsumugi-s7-a5B,refilled,0shot,0.265,720,0.200,0.081,0.041,0.122,above
tsumugi-s7-a5B,refills,0shot,0.172,600,0.200,-0.035,-0.073,0.001,at
tsumugi-s7-a5B,codevar,0shot,0.276,1200,0.267,0.012,-0.023,0.047,at
tsumugi-s7-a5B,codevar,fair,0.236,1200,0.267,-0.047,-0.081,-0.014,below
tsumugi-s7-a5B,sharedlists,0shot,0.249,600,0.200,0.061,0.018,0.104,above
tsumugi-s7-a5B,sharedlists,fair,0.233,600,0.200,0.041,-0.004,0.085,at
tsumugi-s7-a5B,box-narrative,0shot,0.300,200,0.253,0.038,-0.052,0.127,at
tsumugi-s7-a5B,box-narrative,fair,0.365,200,0.253,0.136,0.042,0.227,above
tsumugi-s7-a5B,box-screenplay,0shot,0.445,200,0.253,0.264,0.172,0.357,above
tsumugi-s7-a5B,box-screenplay,fair,0.422,200,0.253,0.209,0.112,0.300,above
tsumugi-s7-a5B,box-code,0shot,0.340,200,0.253,0.131,0.037,0.220,above
tsumugi-s7-a5B,box-code,fair,0.477,200,0.253,0.289,0.189,0.383,above
tsumugi-s7-a5B,box-chat,0shot,0.438,200,0.253,0.238,0.146,0.330,above
tsumugi-s7-a5B,box-chat,fair,0.445,200,0.253,0.237,0.137,0.332,above
tsumugi-s7-a5B,you-static,0shot,0.317,300,0.292,0.036,-0.036,0.110,at
tsumugi-s7-a5B,i-moves,0shot,0.250,180,0.287,-0.057,-0.140,0.029,at
tsumugi-s7-a5B,i-moves,fair,0.247,180,0.287,-0.064,-0.154,0.025,at
tsumugi-s7-a5B,you-moves,0shot,0.286,180,0.287,-0.006,-0.095,0.093,at
tsumugi-s7-a5B,you-moves,fair,0.261,180,0.287,-0.035,-0.126,0.050,at
tsumugi-s7-a5B,you-named,0shot,0.347,180,0.287,0.087,-0.013,0.180,at
tsumugi-s7-a5B,you-named,fair,0.322,180,0.287,0.051,-0.044,0.144,at
tsumugi-s7-a5B,answer-self,0shot,0.280,2560,0.250,0.041,0.018,0.064,above
tsumugi-s7-a5B,answer-self,fair,0.319,2560,0.250,0.092,0.067,0.116,above
tsumugi-s7-a5B,grammar,ladder,,1080,,0.641,0.615,0.667,above
tsumugi-s7-a5B,operators,ladder,,1080,,0.226,0.197,0.256,above
tsumugi-s7-a5B,patterns,ladder,,1080,,0.426,0.392,0.457,above
tsumugi-s7-a5B,facts,ladder,,768,,0.651,0.616,0.689,above
tsumugi-s17-a11B,lookup,0shot,0.505,1210,0.169,0.399,0.366,0.435,above
tsumugi-s17-a11B,lookup,fair,0.916,1210,0.169,0.897,0.879,0.916,above
tsumugi-s17-a11B,shortlist,0shot,0.528,3840,0.040,0.508,0.491,0.524,above
tsumugi-s17-a11B,pairs,0shot,0.342,256,0.125,0.248,0.181,0.310,above
tsumugi-s17-a11B,pairs,fair,0.685,256,0.125,0.641,0.576,0.703,above
tsumugi-s17-a11B,noise,0shot,0.519,720,0.200,0.399,0.352,0.446,above
tsumugi-s17-a11B,now,0shot,0.503,300,0.292,0.301,0.225,0.377,above
tsumugi-s17-a11B,handoffs,0shot,0.407,1200,0.312,0.135,0.095,0.174,above
tsumugi-s17-a11B,handoffs,fair,0.509,1200,0.312,0.282,0.240,0.321,above
tsumugi-s17-a11B,sets,0shot,0.461,1280,0.275,0.262,0.224,0.299,above
tsumugi-s17-a11B,sets,fair,0.599,1280,0.275,0.461,0.421,0.498,above
tsumugi-s17-a11B,refilled,0shot,0.185,720,0.200,-0.019,-0.056,0.016,at
tsumugi-s17-a11B,refills,0shot,0.189,600,0.200,-0.013,-0.051,0.027,at
tsumugi-s17-a11B,codevar,0shot,0.267,1200,0.267,-0.003,-0.039,0.032,at
tsumugi-s17-a11B,codevar,fair,0.250,1200,0.267,-0.028,-0.063,0.006,at
tsumugi-s17-a11B,sharedlists,0shot,0.231,600,0.200,0.038,-0.004,0.078,at
tsumugi-s17-a11B,sharedlists,fair,0.228,600,0.200,0.034,-0.005,0.080,at
tsumugi-s17-a11B,box-narrative,0shot,0.430,200,0.253,0.220,0.120,0.315,above
tsumugi-s17-a11B,box-narrative,fair,0.517,200,0.253,0.345,0.254,0.442,above
tsumugi-s17-a11B,box-screenplay,0shot,0.555,200,0.253,0.393,0.296,0.493,above
tsumugi-s17-a11B,box-screenplay,fair,0.590,200,0.253,0.433,0.333,0.529,above
tsumugi-s17-a11B,box-code,0shot,0.287,200,0.253,0.054,-0.030,0.143,at
tsumugi-s17-a11B,box-code,fair,0.535,200,0.253,0.346,0.234,0.441,above
tsumugi-s17-a11B,box-chat,0shot,0.458,200,0.253,0.252,0.148,0.341,above
tsumugi-s17-a11B,box-chat,fair,0.535,200,0.253,0.349,0.243,0.443,above
tsumugi-s17-a11B,you-static,0shot,0.330,300,0.292,0.050,-0.021,0.126,at
tsumugi-s17-a11B,i-moves,0shot,0.241,180,0.287,-0.073,-0.153,0.019,at
tsumugi-s17-a11B,i-moves,fair,0.314,180,0.287,0.033,-0.066,0.129,at
tsumugi-s17-a11B,you-moves,0shot,0.253,180,0.287,-0.049,-0.134,0.037,at
tsumugi-s17-a11B,you-moves,fair,0.275,180,0.287,-0.017,-0.100,0.069,at
tsumugi-s17-a11B,you-named,0shot,0.242,180,0.287,-0.067,-0.159,0.027,at
tsumugi-s17-a11B,you-named,fair,0.300,180,0.287,0.016,-0.069,0.108,at
tsumugi-s17-a11B,answer-self,0shot,0.329,2560,0.250,0.105,0.082,0.130,above
tsumugi-s17-a11B,answer-self,fair,0.291,2560,0.250,0.055,0.031,0.077,above
tsumugi-s17-a11B,grammar,ladder,,1080,,0.742,0.714,0.764,above
tsumugi-s17-a11B,operators,ladder,,1080,,0.258,0.225,0.287,above
tsumugi-s17-a11B,patterns,ladder,,1080,,0.402,0.366,0.438,above
tsumugi-s17-a11B,facts,ladder,,768,,0.713,0.674,0.741,above
tsumugi-s7-a11B,lookup,0shot,0.500,1210,0.169,0.390,0.359,0.425,above
tsumugi-s7-a11B,lookup,fair,0.922,1210,0.169,0.906,0.887,0.923,above
tsumugi-s7-a11B,shortlist,0shot,0.532,3840,0.040,0.513,0.497,0.530,above
tsumugi-s7-a11B,pairs,0shot,0.430,256,0.125,0.348,0.283,0.413,above
tsumugi-s7-a11B,pairs,fair,0.693,256,0.125,0.650,0.587,0.714,above
tsumugi-s7-a11B,noise,0shot,0.562,720,0.200,0.452,0.408,0.496,above
tsumugi-s7-a11B,now,0shot,0.522,300,0.292,0.322,0.245,0.400,above
tsumugi-s7-a11B,handoffs,0shot,0.400,1200,0.312,0.125,0.086,0.165,above
tsumugi-s7-a11B,handoffs,fair,0.541,1200,0.312,0.329,0.288,0.370,above
tsumugi-s7-a11B,sets,0shot,0.462,1280,0.275,0.274,0.239,0.311,above
tsumugi-s7-a11B,sets,fair,0.592,1280,0.275,0.454,0.416,0.491,above
tsumugi-s7-a11B,refilled,0shot,0.199,720,0.200,-0.002,-0.036,0.037,at
tsumugi-s7-a11B,refills,0shot,0.221,600,0.200,0.026,-0.015,0.071,at
tsumugi-s7-a11B,codevar,0shot,0.307,1200,0.267,0.057,0.025,0.094,above
tsumugi-s7-a11B,codevar,fair,0.275,1200,0.267,0.007,-0.029,0.043,at
tsumugi-s7-a11B,sharedlists,0shot,0.270,600,0.200,0.087,0.046,0.133,above
tsumugi-s7-a11B,sharedlists,fair,0.246,600,0.200,0.057,0.016,0.100,above
tsumugi-s7-a11B,box-narrative,0shot,0.463,200,0.253,0.264,0.167,0.357,above
tsumugi-s7-a11B,box-narrative,fair,0.512,200,0.253,0.329,0.229,0.426,above
tsumugi-s7-a11B,box-screenplay,0shot,0.570,200,0.253,0.434,0.340,0.524,above
tsumugi-s7-a11B,box-screenplay,fair,0.500,200,0.253,0.308,0.202,0.413,above
tsumugi-s7-a11B,box-code,0shot,0.393,200,0.253,0.181,0.090,0.272,above
tsumugi-s7-a11B,box-code,fair,0.580,200,0.253,0.416,0.313,0.506,above
tsumugi-s7-a11B,box-chat,0shot,0.565,200,0.253,0.416,0.324,0.510,above
tsumugi-s7-a11B,box-chat,fair,0.527,200,0.253,0.342,0.232,0.439,above
tsumugi-s7-a11B,you-static,0shot,0.355,300,0.292,0.084,0.005,0.165,above
tsumugi-s7-a11B,i-moves,0shot,0.281,180,0.287,-0.015,-0.111,0.086,at
tsumugi-s7-a11B,i-moves,fair,0.283,180,0.287,-0.010,-0.097,0.087,at
tsumugi-s7-a11B,you-moves,0shot,0.222,180,0.287,-0.098,-0.183,-0.010,below
tsumugi-s7-a11B,you-moves,fair,0.297,180,0.287,0.013,-0.080,0.102,at
tsumugi-s7-a11B,you-named,0shot,0.300,180,0.287,0.018,-0.072,0.112,at
tsumugi-s7-a11B,you-named,fair,0.394,180,0.287,0.148,0.051,0.244,above
tsumugi-s7-a11B,answer-self,0shot,0.313,2560,0.250,0.084,0.061,0.108,above
tsumugi-s7-a11B,answer-self,fair,0.282,2560,0.250,0.043,0.018,0.064,above
tsumugi-s7-a11B,grammar,ladder,,1080,,0.773,0.749,0.797,above
tsumugi-s7-a11B,operators,ladder,,1080,,0.224,0.195,0.259,above
tsumugi-s7-a11B,patterns,ladder,,1080,,0.434,0.402,0.469,above
tsumugi-s7-a11B,facts,ladder,,768,,0.765,0.727,0.799,above
tsumugi-s17-a33B,lookup,0shot,0.700,1210,0.169,0.630,0.595,0.663,above
tsumugi-s17-a33B,lookup,fair,0.950,1210,0.169,0.941,0.926,0.957,above
tsumugi-s17-a33B,shortlist,0shot,0.610,3840,0.040,0.594,0.577,0.610,above
tsumugi-s17-a33B,pairs,0shot,0.570,256,0.125,0.509,0.435,0.574,above
tsumugi-s17-a33B,pairs,fair,0.748,256,0.125,0.712,0.654,0.777,above
tsumugi-s17-a33B,noise,0shot,0.607,720,0.200,0.509,0.465,0.551,above
tsumugi-s17-a33B,now,0shot,0.752,300,0.292,0.648,0.577,0.716,above
tsumugi-s17-a33B,handoffs,0shot,0.439,1200,0.312,0.179,0.140,0.219,above
tsumugi-s17-a33B,handoffs,fair,0.508,1200,0.312,0.281,0.240,0.322,above
tsumugi-s17-a33B,sets,0shot,0.573,1280,0.275,0.421,0.384,0.458,above
tsumugi-s17-a33B,sets,fair,0.629,1280,0.275,0.496,0.460,0.531,above
tsumugi-s17-a33B,refilled,0shot,0.265,720,0.200,0.081,0.041,0.121,above
tsumugi-s17-a33B,refills,0shot,0.210,600,0.200,0.013,-0.028,0.053,at
tsumugi-s17-a33B,codevar,0shot,0.315,1200,0.267,0.057,0.019,0.092,above
tsumugi-s17-a33B,codevar,fair,0.282,1200,0.267,0.013,-0.025,0.050,at
tsumugi-s17-a33B,sharedlists,0shot,0.200,600,0.200,0.000,-0.040,0.043,at
tsumugi-s17-a33B,sharedlists,fair,0.211,600,0.200,0.013,-0.029,0.058,at
tsumugi-s17-a33B,box-narrative,0shot,0.552,200,0.253,0.397,0.301,0.489,above
tsumugi-s17-a33B,box-narrative,fair,0.500,200,0.253,0.313,0.211,0.409,above
tsumugi-s17-a33B,box-screenplay,0shot,0.595,200,0.253,0.446,0.349,0.540,above
tsumugi-s17-a33B,box-screenplay,fair,0.492,200,0.253,0.289,0.186,0.389,above
tsumugi-s17-a33B,box-code,0shot,0.435,200,0.253,0.253,0.162,0.347,above
tsumugi-s17-a33B,box-code,fair,0.635,200,0.253,0.483,0.386,0.572,above
tsumugi-s17-a33B,box-chat,0shot,0.485,200,0.253,0.282,0.178,0.374,above
tsumugi-s17-a33B,box-chat,fair,0.535,200,0.253,0.365,0.270,0.459,above
tsumugi-s17-a33B,you-static,0shot,0.535,300,0.292,0.333,0.250,0.409,above
tsumugi-s17-a33B,i-moves,0shot,0.239,180,0.287,-0.074,-0.158,0.017,at
tsumugi-s17-a33B,i-moves,fair,0.264,180,0.287,-0.030,-0.115,0.055,at
tsumugi-s17-a33B,you-moves,0shot,0.292,180,0.287,0.003,-0.092,0.097,at
tsumugi-s17-a33B,you-moves,fair,0.361,180,0.287,0.105,0.009,0.200,above
tsumugi-s17-a33B,you-named,0shot,0.244,180,0.287,-0.065,-0.145,0.021,at
tsumugi-s17-a33B,you-named,fair,0.306,180,0.287,0.025,-0.061,0.114,at
tsumugi-s17-a33B,answer-self,0shot,0.300,2560,0.250,0.066,0.042,0.089,above
tsumugi-s17-a33B,answer-self,fair,0.300,2560,0.250,0.066,0.044,0.090,above
tsumugi-s17-a33B,grammar,ladder,,1080,,0.781,0.758,0.809,above
tsumugi-s17-a33B,operators,ladder,,1080,,0.287,0.252,0.320,above
tsumugi-s17-a33B,patterns,ladder,,1080,,0.475,0.446,0.513,above
tsumugi-s17-a33B,facts,ladder,,768,,0.892,0.864,0.914,above
tsumugi-s17-a55B,lookup,0shot,0.701,1210,0.169,0.632,0.597,0.662,above
tsumugi-s17-a55B,lookup,fair,0.981,1210,0.169,0.977,0.966,0.985,above
tsumugi-s17-a55B,shortlist,0shot,0.662,3840,0.040,0.648,0.632,0.663,above
tsumugi-s17-a55B,pairs,0shot,0.672,256,0.125,0.625,0.556,0.685,above
tsumugi-s17-a55B,pairs,fair,0.723,256,0.125,0.683,0.625,0.741,above
tsumugi-s17-a55B,noise,0shot,0.670,720,0.200,0.588,0.544,0.630,above
tsumugi-s17-a55B,now,0shot,0.682,300,0.292,0.550,0.476,0.622,above
tsumugi-s17-a55B,handoffs,0shot,0.514,1200,0.312,0.288,0.247,0.332,above
tsumugi-s17-a55B,handoffs,fair,0.565,1200,0.312,0.363,0.321,0.404,above
tsumugi-s17-a55B,sets,0shot,0.526,1280,0.275,0.357,0.320,0.396,above
tsumugi-s17-a55B,sets,fair,0.617,1280,0.275,0.488,0.453,0.526,above
tsumugi-s17-a55B,refilled,0shot,0.373,720,0.200,0.216,0.173,0.261,above
tsumugi-s17-a55B,refills,0shot,0.272,600,0.200,0.090,0.045,0.134,above
tsumugi-s17-a55B,codevar,0shot,0.301,1200,0.267,0.041,0.007,0.079,above
tsumugi-s17-a55B,codevar,fair,0.321,1200,0.267,0.067,0.030,0.102,above
tsumugi-s17-a55B,sharedlists,0shot,0.221,600,0.200,0.027,-0.013,0.067,at
tsumugi-s17-a55B,sharedlists,fair,0.268,600,0.200,0.084,0.042,0.130,above
tsumugi-s17-a55B,box-narrative,0shot,0.583,200,0.253,0.443,0.352,0.535,above
tsumugi-s17-a55B,box-narrative,fair,0.490,200,0.253,0.291,0.190,0.381,above
tsumugi-s17-a55B,box-screenplay,0shot,0.660,200,0.253,0.547,0.458,0.641,above
tsumugi-s17-a55B,box-screenplay,fair,0.530,200,0.253,0.344,0.238,0.437,above
tsumugi-s17-a55B,box-code,0shot,0.470,200,0.253,0.285,0.195,0.377,above
tsumugi-s17-a55B,box-code,fair,0.652,200,0.253,0.505,0.403,0.598,above
tsumugi-s17-a55B,box-chat,0shot,0.595,200,0.253,0.454,0.355,0.546,above
tsumugi-s17-a55B,box-chat,fair,0.590,200,0.253,0.433,0.331,0.523,above
tsumugi-s17-a55B,you-static,0shot,0.526,300,0.292,0.327,0.244,0.404,above
tsumugi-s17-a55B,i-moves,0shot,0.311,180,0.287,0.025,-0.075,0.120,at
tsumugi-s17-a55B,i-moves,fair,0.292,180,0.287,0.006,-0.086,0.096,at
tsumugi-s17-a55B,you-moves,0shot,0.264,180,0.287,-0.042,-0.130,0.056,at
tsumugi-s17-a55B,you-moves,fair,0.303,180,0.287,0.023,-0.062,0.118,at
tsumugi-s17-a55B,you-named,0shot,0.300,180,0.287,0.014,-0.075,0.109,at
tsumugi-s17-a55B,you-named,fair,0.319,180,0.287,0.044,-0.046,0.143,at
tsumugi-s17-a55B,answer-self,0shot,0.336,2560,0.250,0.115,0.091,0.138,above
tsumugi-s17-a55B,answer-self,fair,0.306,2560,0.250,0.074,0.050,0.097,above
tsumugi-s17-a55B,grammar,ladder,,1080,,0.761,0.736,0.790,above
tsumugi-s17-a55B,operators,ladder,,1080,,0.276,0.241,0.309,above
tsumugi-s17-a55B,patterns,ladder,,1080,,0.438,0.405,0.470,above
tsumugi-s17-a55B,facts,ladder,,768,,0.892,0.866,0.917,above
tsumugi-s17-a88B,lookup,0shot,0.636,1210,0.169,0.554,0.520,0.590,above
tsumugi-s17-a88B,lookup,fair,0.990,1210,0.169,0.988,0.981,0.994,above
tsumugi-s17-a88B,shortlist,0shot,0.682,3840,0.040,0.668,0.653,0.684,above
tsumugi-s17-a88B,pairs,0shot,0.615,256,0.125,0.560,0.491,0.623,above
tsumugi-s17-a88B,pairs,fair,0.598,256,0.125,0.540,0.480,0.600,above
tsumugi-s17-a88B,noise,0shot,0.715,720,0.200,0.644,0.603,0.683,above
tsumugi-s17-a88B,now,0shot,0.757,300,0.292,0.652,0.580,0.714,above
tsumugi-s17-a88B,handoffs,0shot,0.485,1200,0.312,0.248,0.205,0.290,above
tsumugi-s17-a88B,handoffs,fair,0.579,1200,0.312,0.385,0.344,0.424,above
tsumugi-s17-a88B,sets,0shot,0.496,1280,0.275,0.315,0.278,0.352,above
tsumugi-s17-a88B,sets,fair,0.664,1280,0.275,0.552,0.516,0.588,above
tsumugi-s17-a88B,refilled,0shot,0.432,720,0.200,0.290,0.249,0.335,above
tsumugi-s17-a88B,refills,0shot,0.379,600,0.200,0.224,0.175,0.271,above
tsumugi-s17-a88B,codevar,0shot,0.350,1200,0.267,0.108,0.069,0.147,above
tsumugi-s17-a88B,codevar,fair,0.365,1200,0.267,0.128,0.091,0.165,above
tsumugi-s17-a88B,sharedlists,0shot,0.258,600,0.200,0.072,0.029,0.116,above
tsumugi-s17-a88B,sharedlists,fair,0.324,600,0.200,0.155,0.110,0.205,above
tsumugi-s17-a88B,box-narrative,0shot,0.555,200,0.253,0.404,0.308,0.496,above
tsumugi-s17-a88B,box-narrative,fair,0.470,200,0.253,0.265,0.165,0.364,above
tsumugi-s17-a88B,box-screenplay,0shot,0.614,200,0.253,0.478,0.386,0.571,above
tsumugi-s17-a88B,box-screenplay,fair,0.512,200,0.253,0.322,0.224,0.413,above
tsumugi-s17-a88B,box-code,0shot,0.510,200,0.253,0.343,0.250,0.436,above
tsumugi-s17-a88B,box-code,fair,0.650,200,0.253,0.496,0.398,0.590,above
tsumugi-s17-a88B,box-chat,0shot,0.579,200,0.253,0.421,0.325,0.516,above
tsumugi-s17-a88B,box-chat,fair,0.520,200,0.253,0.354,0.251,0.444,above
tsumugi-s17-a88B,you-static,0shot,0.578,300,0.292,0.403,0.318,0.481,above
tsumugi-s17-a88B,i-moves,0shot,0.300,180,0.287,0.013,-0.080,0.104,at
tsumugi-s17-a88B,i-moves,fair,0.292,180,0.287,0.006,-0.083,0.100,at
tsumugi-s17-a88B,you-moves,0shot,0.331,180,0.287,0.057,-0.040,0.151,at
tsumugi-s17-a88B,you-moves,fair,0.364,180,0.287,0.106,0.014,0.199,above
tsumugi-s17-a88B,you-named,0shot,0.300,180,0.287,0.016,-0.074,0.113,at
tsumugi-s17-a88B,you-named,fair,0.289,180,0.287,0.003,-0.081,0.094,at
tsumugi-s17-a88B,answer-self,0shot,0.323,2560,0.250,0.097,0.075,0.120,above
tsumugi-s17-a88B,answer-self,fair,0.330,2560,0.250,0.106,0.080,0.129,above
tsumugi-s17-a88B,grammar,ladder,,1080,,0.806,0.784,0.830,above
tsumugi-s17-a88B,operators,ladder,,1080,,0.301,0.268,0.331,above
tsumugi-s17-a88B,patterns,ladder,,1080,,0.507,0.477,0.546,above
tsumugi-s17-a88B,facts,ladder,,768,,0.925,0.905,0.947,above
```
