# Binding Atlas — agent view

Plain-text twin of the [Binding Atlas page](https://aetherawi.red/kotodama/binding-atlas). Same data, same build. Built 2026-10-29 13:32 UTC: 117 rows (models + one human), 28 columns, 7,919 cells, 1,285,239 scored items.

## Summary

Base language models acquire the referent of *you* in five ordered stages. Every base of 0.6B parameters or more trained on trillions of tokens reaches the first three. The fifth stage, tracking *you* after objects change hands, is **never reached zero-shot** (with no worked examples) by a public model up to 8B, at any token count measured up to 36T; its swap shift is negative in every one of them. From 14B to 1T parameters, stage 5 is weak and nearly flat. In these models, parameter count and depth grow together. **kotodama-tsumugi**, a 1.4B model run as 72 effective layers (looped depth, boundary-fitted attention residuals), separates them: it reaches stage 3 by **33B tokens** (annealed; on the constant-LR trunk by 40B), the token cost of OLMo-2 13B, against about 420B for OLMo-2 1B. Trained to 110B tokens, it still does not reach stage 5, but its zero-shot reading (−.021) is not inverted under the gauge rule, where every 1–2B public base whose static *you* binds, except Llama-3.2-1B, inverts.

Swap shift = mean[p(gold | original) − p(gold | speaker swapped)] over the candidates: above +.02 a pronoun binds (its answer moves with the speaker), below −.02 it inverts (it resolves to the speaker).

## Part I · Where base models fail

### The staircase

| stage | name | what it means | example | where public models stand | OLMo-2 1B | kotodama-tsumugi 1.4B |
|---|---|---|---|---|---|---|
| 1 | I binds | "I" resolves to whoever is speaking. | `Jack: I'm holding the ___` | every base ≥ 0.5B | ~21–63B tokens | ~3–3.75B tokens |
| 2 | you = me | "you" is read as the speaker too: the pronoun-reversal phase young children go through. | `Jack: you're holding the ___  → Jack's object` | transient in every trajectory | ~63B → ~500B tokens | transient, ~4–20B |
| 3 | static you binds | "you" resolves to the addressee, when ownership is simply stated. | `Carol has the onion. … Jack: you're holding the ___` | every base ≥ 0.6B trained on trillions of tokens | ~420B tokens | by 33B tokens annealed (trunk: 40B); +.235 at 110B |
| 4 | you = me, again | once objects change hands in the scene, "you" collapses back onto the speaker. | `Erin gives Paul her cup. … Paul: now you're holding the ___` | where almost every base stops | from ~63B, still there at 4T | from ~4B; shallowing (−.047 at 11B → −.021 at 110B, no longer inverted under the gauge rule) |
| 5 | dynamic you binds | "you" tracked through transfers. In-context first (3-shot), zero-shot last. | `same scene, answered correctly without examples` | zero-shot: never binds at ≤ 8B (any token count measured); weak and nearly flat from 14B to 1T. Token-reachability unknown | never (flat to 4T) | not reached by 110B |

### Dynamic *you* at each model's final checkpoint (OLMo-2 1B: its 4T pre-anneal checkpoint; not reached zero-shot at or below 8B; weak above 14B)

| model | parameters (B) | training tokens | 0-shot | 3-shot |
|---|---|---|---|---|
| Qwen2.5 0.5B | 0.5 | 18T | −.083 | −.013 |
| Qwen3 0.6B | 0.6 | 36T | −.060 | −.003 |
| OLMo-2 1B (4T, pre-anneal) | 1.5 | 4T | −.047 | −.023 |
| Qwen2.5 1.5B | 1.5 | 18T | −.085 | −.030 |
| SmolLM2 1.7B | 1.7 | 11T | −.047 | +.007 |
| Qwen3 1.7B | 1.7 | 36T | −.089 | +.011 |
| SmolLM3 3B | 3.1 | 11T | −.036 | −.002 |
| Qwen2.5 3B | 3.1 | 18T | −.088 | −.017 |
| Qwen3 4B | 4.0 | 36T | −.068 | +.035 |
| OLMo-2 7B | 7.3 | 4T | −.023 | −.027 |
| Qwen2.5 7B | 7.6 | 18T | −.004 | +.061 |
| Qwen3 8B | 8.2 | 36T | −.079 | +.043 |
| Gemma-2 9B | 9.2 | 8T | −.042 | +.009 |
| OLMo-2 13B | 13.7 | 5T | −.014 | −.010 |
| Qwen2.5 14B | 14.7 | 18T | +.011 | +.189 |
| Qwen3 14B | 14.8 | 36T | +.022 | +.242 |
| Qwen3 30B-A3B | 30.5 | 36T | +.026 | +.041 |
| GLM-4 32B | 32 | 15T | +.029 | +.013 |
| OLMo-3 32B | 32 | 6T | −.062 | +.047 |
| Llama-3.1 70B | 70.6 | 15T | +.023 | +.040 |
| Qwen2.5 72B | 72.7 | 18T | +.033 | +.039 |
| Llama-3.1 405B | 405 | 15T | +.054 | +.145 |
| Nemotron-3 Ultra | 550 | — | +.014 | +.068 |
| DeepSeek-V3 | 671 | 15T | +.067 | +.065 |
| Kimi-K2 | 1040 | 15T | +.045 | +.070 |
| DeepSeek-V4 Flash | not public | — | +.102 | +.083 |
| DeepSeek-V4 Pro | not public | — | +.004 | +.032 |
| Trinity-Large (base) | not public | — | .000 | +.110 |
| MiMo-V2.5 Pro (base) | not public | — | +.036 | +.083 |
| kotodama-tsumugi 1.4B @ 110B tokens (release) | 1.42 | 110B | −.021 | +.018 |

No model at or below 8B binds zero-shot dynamic *you* at any token count measured (to 36T). The same holds with each chat rewritten as a scene (8 of 8 models tested at or below 8B; the highest is +.010, Qwen2.5-7B), so the result does not come from the chat format. This is not evidence that the stage needs more parameters. No model at or below 8B has been trained far enough to rule out more tokens. Whether more tokens would bring a small model to stage 5 is unknown. From 14B to 1T parameters, stage 5 is weak and nearly flat. Ten of the 11 bases with a public size read +.01 to +.07 zero-shot. Static *you* reads +.27 to +.82 in the seven of them where it was measured.

### Training tokens until each stage appears (the range spans the checkpoints that bracket the stage)

| stage | model | tokens |
|---|---|---|
| 1 · I | OLMo-2 1B | 21.0–63.0B |
| 1 · I | OLMo-2 7B | 21.0–42.0B |
| 1 · I | OLMo-2 13B | 17.0–42.0B |
| 1 · I | kotodama-tsumugi | 3.0–3.75B |
| 3 · static you | OLMo-2 1B | 294.0–420.0B |
| 3 · static you | OLMo-2 7B | 105.0–210.0B |
| 3 · static you | OLMo-2 13B | 17.0–42.0B |
| 3 · static you | kotodama-tsumugi | 30.0–33.0B |
| 5 · dynamic you | OLMo-2 1B | not reached by 4000B |
| 5 · dynamic you | OLMo-2 7B | not reached by 3900B; recovering, not crossed by 3.9T |
| 5 · dynamic you | OLMo-2 13B | not reached by 2600B; at threshold (touches + at 2.6T) |
| 5 · dynamic you | kotodama-tsumugi | not reached by 110B |

### Stage 5, taken apart (accuracy, 0-shot, self + addressee queries pooled; chance ≈ .29)

Each dynamic-*you* gauge item is rewritten along one axis: **transfer, narrated** (`Erin gives the cup to Paul. … Right now Paul is holding the`); **transfer, spoken** (`Erin: Here, take my cup. … Paul: Right now I'm holding the`); **spoken, transfer narrated** (`[Erin gives the cup to Paul.] … deictic query`); **spoken + vocative** (`Erin: Paul, right now you're holding the`).

| model | tokens | transfer, narrated | transfer, spoken | spoken, transfer narrated | spoken + vocative |
|---|---|---|---|---|---|
| Pythia-1.4B | 134B | .286 | .283 | .253 | .242 |
| kotodama-tsumugi · 33B | 33B | .308 | .297 | .300 | .306 |
| kotodama-tsumugi · 55B | 55B | .297 | .289 | .292 | .283 |
| kotodama-tsumugi · 88B | 88B | .314 | .275 | .319 | .264 |
| kotodama-tsumugi · 110B, second anneal | 110B | .331 | .297 | .339 | .292 |
| kotodama-tsumugi · 110B, release | 110B | .336 | .281 | .339 | .292 |
| OLMo-2 1B | 4T | .306 | .319 | .333 | .308 |
| SmolLM2 1.7B | 11T | .444 | .300 | .347 | .281 |
| Qwen3 1.7B | 36T | .514 | .353 | .258 | .325 |
| Qwen3 4B | 36T | .669 | .400 | .406 | .353 |
| Qwen2.5 7B | 18T | .572 | .400 | .514 | .381 |
| Llama-3.1 8B | 15T | .589 | .289 | .436 | .258 |
| Qwen3 8B | 36T | .714 | .397 | .450 | .319 |
| Qwen2.5 14B | 18T | .725 | .361 | .533 | .317 |
| Qwen3 14B | 36T | .675 | .406 | .542 | .358 |
| Qwen3 30B-A3B | 36T | .592 | .339 | .531 | .297 |
| Llama-3.1 70B | 15T | .797 | .358 | .631 | .331 |

Three layers: (1) possession transfer itself: every 1.4–1.5B model is at chance even narrated (kotodama-tsumugi through 110B, Pythia-1.4B, OLMo-2 1B at 4T); SmolLM2 1.7B at 11T and Qwen3 1.7B at 36T lift off chance; from 4B it is easy. (2) The transfer as a speech act is the mid-size wall: 4B–70B models that pass the narrated version stay at .29–.41 spoken; narrating only the hand-off recovers part of the gap. (3) Naming the addressee does not help: the vocative variant is at or below the spoken one for every public model.

### Findings

1. **Scale solves who holds what; it does not solve who *you* is.** The atlas includes 26 pretrained bases from 1B to ~1T parameters (Llama-3.1 405B, Kimi-K2, DeepSeek-V3 and V4, MiMo-V2.5 and V2.5-Pro, Trinity-Large, Qwen2.5 72B, Gemma-4, GLM-4 and others). On the 3rd-person twin of the shortcut-free *you* query, their accuracy rises with scale, to .72–.99 at ≥30B. On this accuracy test, the 0-shot *you* version stays at .15–.33, around the 29% chance line, with no trend in size. The swap shift of Figure 2 is a different measure. From 14B up, it reads weakly positive in 10 of the 11 bases with a public size.
2. **The items are unambiguous.** An independent solver reads only the prompt text. Its rules: *I* is the speaker, and *you* is the named addressee, or the other person in a two-person chat. It finds a unique answer for every one of the 900 deixis items and 540 twins. Every answer matches the gold label. The cleanest contrast is *I* against *you* on the same transcripts, with the same format, bookkeeping and question: bases ≥30B score .21–.60 on “I'm holding the” and .15–.33 on “you're holding the”. Part of the gap to the 3rd-person twin is the cost of tracking the transcript. The *I* version pays that cost too. The part that belongs to *you* alone is about .2.
3. **Bases bind *I* to the speaker and leave *you* unbound.** A speaker-swap test changes only the name on the last line of a two-person chat (“Neil: Right now you're holding the” → “Kevin: …”), which flips the right answer. Here the test runs on scenes where objects change hands, on 18 bases from 9B to about 1T parameters. For “I'm holding the”, all 18 bases tested move their answer with the speaker given three examples (+.14 to +.58 in gold probability), and 17 of 18 already do at 0-shot (+.06 to +.17; Kimi-K2 +.03). For “you're holding the”, 11 of 18 do not move at 0-shot (within ±.04). Three move the wrong way and read *you* as the speaker (Gemma-2 9B, Gemma-4 31B, OLMo-3 32B). Four of the largest move slightly the right way (DeepSeek-V3 +.07, DeepSeek-V4 Flash +.10, Kimi-K2 +.05, Llama-3.1 405B +.05), a fraction of their *I* shift with examples. In bases of this size, the failure is therefore mostly an unattached *you*, not a misread speaker. Smaller bases read below zero on the same test (Figure 2).
4. **The residual signal comes from the shortcut.** On all structural-*you* items 405B scores 32%, just over chance. Restricted to items where the answer is neither the last-mentioned person nor the last speaker, it scores 21%, and the remaining errors point the wrong way. The Shortcut-free toggle shows the drop.
5. **The reflex/deliberate split holds across vendors, and the reflex error has one direction.** Asked for only the next word, frontier models score 21–48% on structural *you*, against about 29% by chance (all items, 0-shot: GPT-5.5 48%, GPT-5 mini 30%, Grok 4.3 33%, DeepSeek V3.2 21%). Allowed to reason, they reach 100%, 96%, 89% and 70%. Most reflex misses pick **the speaker's own object**, reading “you're holding” as “I'm holding”. For open bases of 9B and up, the speaker-swap test shows mostly non-binding rather than inversion. Each deixis cell opens its error anatomy.
6. **Frontier models resolve the referent and name it.** Asked to finish a line spoken *to* the person holding the object, GPT-5.5 names that person every time instead of saying *you*. The answer columns use **referent scoring** (the name counts as correct) and show how often the seat form (*you/me*) appears. On these columns the answer is a person, not an object. There, frontier models resolve the referent ~100% of the time on reflex. In a frame ablation the deliberate frame (“work out the answer”) biases answers toward names, and GPT-5.5 still names the asker when told “you are Ted”.
7. **More tokens and anneals grow bookkeeping, not perspective.** Three training runs are measured at two or more points: Trinity-Large TrueBase (10T tokens, pre-anneal) → Base (17T, with anneals); OLMo-3 32B stage 1 → final; OLMo-2 7B from 2.9T tokens through mid-training. The later checkpoints gain code aliasing at 0-shot (copy-then-mutate: Trinity 36 → 98, OLMo-3 3 → 35; rebind: OLMo-3 14 → 69) and set binding (OLMo-3 45 → 76). Two-person *you* stays at or below the 29% chance line throughout (Trinity 21 → 17, OLMo-3 23 → 29, OLMo-2 27 → 22).
8. **A human reader solves these on reflex.** The human row reads every item cold, once, without options, and is at ceiling on the perspective columns where 405B is at chance. It is one reader and a handful of items per cell: a reference point, not an estimate.

## Part II · The levers

Each lever measured on 1.4B-parameter arms trained for 4B tokens on one data order; one seed per arm except the fitted boundaries (three seeds). kotodama-tsumugi (Part III) combines the three positive levers in one model.

| lever | effect | result | cost |
|---|---|---|---|
| Depth (52 vs 28 layers, same params) | + | set-binding .70 vs .41; attribution .83 vs .69 | ≈ same loss, −22% throughput |
| Looped depth (shared middle block) | + | matches the deep model at 56–80% of the unique params | loss flat when the loop sits inside one fitted block |
| Data-fitted AttnRes boundaries | + | a 28-layer model reaches 52-layer set-binding in 2 of 3 seeds (.67/.66 vs .41–.44) | no measured cost |
| Attention residuals at all | + | stage 1 arrives 1.5–2× sooner; needed for the depth effect | −20% throughput |
| Dialogue-heavy early data (13–40%) | 0 / − | no binding gain at any dose or window; 25% delays stage 1 ~2.3× | costs code and structure binding |

## Part III · kotodama-tsumugi

A 1.42B-parameter model with a looped core, run as 72 effective layers. Its trunk trains at a constant learning rate to 100B tokens, and annealed reads are cooldown branches forked from its checkpoints (cosine decay to zero over about 10% of the tokens so far); the release is the 110B branch from the 100B fork. The architecture and training run are described in Who Is You?, the companion write-up, and the release table below gives the spread between two anneals.

### Stage gauges against training tokens

`kind` = constant (the trunk, constant learning rate) or annealed (a cooldown branch ending at that token count). Columns are swap shifts: static *I*, static *you*, dynamic *you* 0-shot and 3-shot.

| seed | kind | tokens (B) | static I | static you | dynamic you 0-shot | dynamic you 3-shot |
|---|---|---|---|---|---|---|
| s17 | constant | 1 | +.004 | −.003 | −.004 | −.003 |
| s17 | constant | 2 | +.001 | −.003 | −.004 | +.002 |
| s17 | constant | 3 | +.007 | −.008 | −.005 | −.018 |
| s17 | constant | 3.75 | +.035 | −.020 | −.025 | −.013 |
| s17 | constant | 10 | +.075 | −.013 | −.042 | −.015 |
| s17 | constant | 20 | +.133 | −.005 | −.036 | −.008 |
| s17 | constant | 30 | +.137 | +.017 | −.030 | −.008 |
| s17 | constant | 40 | +.182 | +.077 | −.025 | +.011 |
| s17 | constant | 50 | +.309 | +.040 | −.029 | +.006 |
| s17 | constant | 60 | +.256 | +.072 | −.025 | +.031 |
| s17 | constant | 80 | +.213 | +.084 | −.032 | +.022 |
| s17 | constant | 100 | +.223 | +.105 | −.018 | +.005 |
| s17 | annealed | 5 | +.039 | −.024 | −.023 | −.012 |
| s17 | annealed | 11 | +.077 | −.023 | −.047 | −.015 |
| s17 | annealed | 33 | +.243 | +.073 | −.022 | −.004 |
| s17 | annealed | 55 | +.263 | +.138 | −.029 | +.016 |
| s17 | annealed | 88 | +.191 | +.206 | −.020 | +.010 |
| s17 | annealed (release) | 110 | +.246 | +.235 | −.021 | +.018 |
| s17 | annealed (second anneal from the same 100B fork) | 110 | +.225 | +.142 | −.016 | +.012 |
| s7 | constant | 1 | +.004 | −.006 | −.004 | −.003 |
| s7 | constant | 2 | +.004 | −.004 | −.011 | +.002 |
| s7 | constant | 3 | +.007 | −.004 | −.018 | +.012 |
| s7 | constant | 3.75 | +.011 | −.006 | −.034 | +.001 |
| s7 | constant | 10 | +.099 | −.006 | −.048 | −.001 |
| s7 | annealed | 5 | +.027 | −.008 | −.017 | −.015 |
| s7 | annealed | 11 | +.111 | −.009 | −.043 | +.004 |

For reference, OLMo-2 reaches stage 3 at about 42B tokens (OLMo-2 13B), 210B tokens (OLMo-2 7B), 420B tokens (OLMo-2 1B).

### Binding battery across the annealed reads (accuracy)

| probe | s17 · 5B | s17 · 11B | s17 · 33B | s7 · 5B | s7 · 11B | s17 · 55B | s17 · 88B | 110B · release |
|---|---|---|---|---|---|---|---|---|
| *who holds what* | | | | | | | | |
| set binding, one object (chance .125) | .422 | .492 | .469 | .453 | .484 | .516 | .547 | .562 |
| set binding, 2–3 objects | .555 | .697 | .682 | .596 | .633 | .758 | .744 | .729 |
| attribution | .713 | .904 | .940 | .826 | .915 | .976 | .976 | .986 |
| aliased Python lists (chance .20) | .207 | .223 | .210 | .228 | .240 | .263 | .318 | .317 |
| dialogue tracking | .490 | .534 | .514 | .517 | .551 | .632 | .638 | .618 |
| *who you is* | | | | | | | | |
| deixis v2, owner query | .311 | .411 | .422 | .306 | .367 | .372 | .422 | .383 |
| deixis v2, self (P2) | .244 | .361 | .361 | .239 | .283 | .306 | .306 | .344 |
| deixis v2, addressee (P2) | .250 | .306 | .333 | .228 | .278 | .328 | .283 | .311 |
| reply adjacency, no vocative (chance .33) | .453 | .400 | .445 | .384 | .299 | .461 | .427 | .411 |
| deixis-lite (0-shot) | .430 | .433 | .431 | .421 | .435 | .458 | .475 | .502 |

Scored with one harness (the kotodama binding suite). Each item gets four worked examples, balanced so that none of them shows the item's own answer pattern; deixis-lite has no such condition and is read zero-shot. The bookkeeping probes rise with tokens: attribution from .71 to .99, set binding and dialogue tracking by .13 to .17. Aliased Python lists stays at chance through 33B and lifts only after 55B. The perspective probes stay within .15 of their 5B level. The two seeds differ at 5B (seed 7 forms set binding earlier) and agree within .05 on seven of the ten rows at 11B.

### The release against its trajectory and its class

| probe | 33B | 55B | 88B | 110B · 2nd anneal | 110B · release | OLMo-2 1B · 4T | SmolLM2 1.7B · 11T | Qwen3 1.7B · 36T |
|---|---|---|---|---|---|---|---|---|
| static I (swap shift) | +.243 | +.263 | +.191 | +.225 | +.246 | +.355 | +.378 | +.606 |
| static you (swap shift) | +.073 | +.138 | +.206 | +.142 | +.235 | +.137 | +.197 | +.123 |
| dynamic you, 0-shot (swap shift) | −.022 | −.029 | −.020 | −.016 | −.021 | −.082 | −.047 | −.089 |
| dynamic you, 3-shot (swap shift) | −.004 | +.016 | +.010 | +.012 | +.018 | −.016 | +.007 | +.011 |
| binding capacity K50 (chance .04) | 3.3 | 3.5 | 4.2 | 3.8 | 4.8 | 6.6 | 18.7 | 44.1 |
| one binding past 256 filler tokens | .863 | .983 | .892 | .921 | .908 | .829 | .771 | .958 |
| deixis v2, deictic, 0-shot (acc) | .311 | .307 | .314 | .331 | .339 | .344 | — | — |
| deixis v2, third person, 0-shot (acc) | .368 | .373 | .392 | .420 | .416 | .411 | — | — |

The two 110B columns are two anneals from the same fork with the same training loss. Their spread (static *you* .142 vs .235, K50 3.8 vs 4.8, and .025 bits/byte on held-out text) is the measured anneal-to-anneal variance, and every single-anneal number here carries it. A second initialization seed (seed 7, same data order) was trained to 11B. Capacity (K50, after arXiv 2609.30634: how many name–task pairs fit in context before recall falls halfway to chance) tracks recipe and tokens more than parameters on this harness, so no parameter-equivalent size is read from it.

### Object tracking with fixed vs fair exemplars (riders v2, prose; accuracy, chance .25)

| model · tokens | 0shot | fixed3 | fair1 | fair3 | fair6 |
|---|---|---|---|---|---|
| Pythia-1.4B · 134B | .362 | .415 | .385 | .440 | .417 |
| OLMo-2 1B · 4T | .385 | .275 | .375 | .383 | .453 |
| SmolLM2 1.7B · 10.5T | .398 | .398 | .472 | .432 | .505 |
| kotodama-tsumugi · 11B | .427 | .375 | .443 | .490 | .495 |
| kotodama-tsumugi · 33B | .455 | .325 | .450 | .485 | .530 |
| kotodama-tsumugi · 55B | .490 | .278 | .455 | .453 | .510 |
| kotodama-tsumugi · 110B, second anneal | .470 | .338 | .448 | .393 | .438 |
| kotodama-tsumugi · 110B, release | .482 | .263 | .472 | .388 | .438 |

fixed3 = the same three exemplars for every item; fairK = K exemplars drawn per item. Fixed exemplars push accuracy well below zero-shot (a setup artifact). The penalty is not specific to these models. On a lighter riders slice, fixed exemplars lower prose accuracy for most of an eight-model panel (chat 8 of 8, screenplay 6 of 8, narrative 5 of 8) and raise code accuracy for all eight. Fair exemplars help every public model here and kotodama-tsumugi through 55B, but not the two 110B anneals (second anneal −.03, release −.045). The zero-shot column pools narrative and screenplay prose. Zero-shot riders accuracy depends on the register: on the lighter slice, it spreads by 20 points or more across the four registers for five of the eight panel models. The public pool is three 1–2B models; no larger model has been run on this harness.

## How to read the full atlas

- **A cell** = one model (row) on one probe type (column) under one condition, as `acc` = share of items where the gold answer is the model's top choice among the item's candidates, with `n` items and the `chance` rate (mean 1/#candidates). `score` = (acc − chance)/(1 − chance); `se` = binomial standard error.
- **Conditions:** `0shot` = the item alone; `3shot` = three solved items of the test item's own type first (a diagnostic: on aliased lists it teaches the answer's role; see Caveats); `3shot_collide` = the three exemplars reuse the test item's names/objects in other roles (interference test).
- **Modes:** `all` = every item; `free` = shortcut-free subset (deixis columns only): the answer is neither the last-mentioned candidate nor the last speaker.
- **Open models** (bases, hybrids, post-trained) are scored by log-probability over the candidate set (vLLM or HF transformers, never mixed within a row; parity-checked to ±.01). **API models** are scored by generation: `reflex` rows = single next word with reasoning off (calls that reasoned anyway are excluded); `deliberate` rows = think, then answer. **Human** row = one person reading items cold, typed answers.
- **Answer-deixis columns** (`an-*`) use a calibrated read for open models (argmax of logp(item) − logp(body hidden)); API rows use referent scoring (the person's name counts as correct as well as me/you). Per-role cells mix skill with each model's standing preference — compare a model's answer columns together.
- **Deixis twins:** every `I/you/my` item has a `3rd-person twin` with the same events, gold and candidates, told in third person with names. The gap between the two is the cost of perspective.
- **Parsing:** sections are `##` headings outside code fences (prompts inside ``` fences can contain lines that start with `#`). Two CSV blocks at the end: `rows` (id → label, group, size) and `cells` (every number).

## Glossary

| term | meaning |
|---|---|
| P2 / P3 | number of people in the chat: P2 = two people (so 'you' is simply the other person); P3 = three (every 'you' line opens with a vocative naming the addressee). |
| owner / self / addressee | deixis query kinds: owner = 'Now the cup is with' -> a name; self = 'S: Right now I'm holding the' -> the speaker's object; addressee = 'S: Right now you're holding the' -> the addressee's object. |
| I/you/my vs 3rd-person twin | the same events told in chat with pronouns vs in prose with names; same gold and candidates. Column ids end in -deictic / -third. |
| structural you | a 'you' whose referent is fixed only by the conversation (P2, no vocative). |
| shortcut-free (mode=free) | items where the gold is neither the last-mentioned candidate nor the last speaker, so recency cannot answer them. |
| kind: acc | gold is the top-ranked candidate (logprob rows) or the produced word (API/human rows). |
| kind: hit1 / exactk | set binding: hit1 = top candidate is in the gold set; exactk = the top-k candidates are exactly the gold set (k = set size); chance differs accordingly. |
| kind: cal | answer deixis, open models: argmax of logp(item) − logp(same tail with the body hidden); API rows use referent scoring instead (the person's name counts as right as well as me/you). |
| seat_form | API answer columns: share of referent-correct answers that used me/you rather than the name. |
| errs | deixis I/you cells: shares of items by error type (a wrong pick only). speaker_confusion = the wrong side of I/you (for 'you' queries: an object the SPEAKER holds); previous_owner = someone who held it earlier; other_participant = a third person (owner queries) / their object; other_object = another object; tie = gold tied for top (counted wrong); format = API answer not among the candidates; other = rest. |
| reflex / deliberate | API frames: reflex = answer with the next word, reasoning off (calls that reasoned anyway are excluded); deliberate = think first, then give the word. |
| row ids | or-<provider>/<model>-A = API reflex row, -B = API deliberate row; other ids are open checkpoints (e.g. llama31-405b). 'human' = the human row. |
| swap shift | speaker-swap test: P(original gold | original asker) − P(original gold | swapped asker). |

## Caveats

- In the 3-shot (same type) condition, all three examples share the test item's type. On the aliased-list columns, that shows the model the answer's role. In a check on eight models, the examples' answers were flipped to the wrong role. Every model that had gained 20 points or more from the examples then fell to near zero (Qwen2.5-7B: 98% → 2%). Balanced examples carry almost none of the role. Read those 3-shot cells as copying the shown role. The same check finds that code variables are partly shortcut (balanced examples keep 39% of the 3-shot gain) and static lookup mostly format (85%). Table 3, the battery across kotodama-tsumugi's annealed reads, uses four balanced examples.
- Static lookup at 0-shot is bare prose with no frame. All eight models in a check panel gain 19 to 46 points from three worked examples of the item's type. Balanced examples keep most of that gain (28 of 33 points on average). The 0-shot cells in this column therefore mostly show whether a model infers the task's format. For aliased lists, 0-shot sits close to the balanced reading (7.5 points below it on average) and is a fair floor.
- Open-model rows are scored with either vLLM or HF transformers, noted per row and never mixed within a row. A parity check finds the two engines agree within ±.01 accuracy, with no systematic bias.
- Answer-deixis cells are calibrated reads over four candidates (me / you / two names), and each model has a strong standing preference (one base answers “me” 59% of the time). A single per-role cell mixes skill with that preference, so a model's three answer columns read together, not one alone.
- API rows (Claude and others via OpenRouter) are scored by generation: the first word for reflex, the last word for deliberate. Open models are scored by log-probability over the candidate set. Answer columns use referent scoring for API rows (me/you or the person's name). For open models the name is not a candidate, so me/you is the only way to name the right person.
- Deliberate-frame rows on the answer columns are biased toward naming, because the frame says “work out the answer”.
- Reflex rows count only calls that used **zero** reasoning tokens. Some models reason regardless of settings (Qwen3 14B always; gpt-oss and Qwen3 32B often), so their reflex cells are thinner or absent.
- API rows were served by whichever OpenRouter provider answered; the provider, and therefore the quantization, is recorded per call.
- Set binding with exact-set scoring has no generation version; API rows show single-object sets only.
- The human row is a handful of items per cell (two cold reads of a 31- and a 24-item gauntlet): a reference point, not an estimate.
- Every open base row is a full vLLM grid. Mistral Large has only a partial reflex row, because of provider failures on the other calls.

## Columns

| id | family | label | what it asks | scoring | shortcut-free |
|---|---|---|---|---|---|
| `attr` | Core binding | Static lookup | N people each keep an object; query one. Pure retrieval under distractors. | acc | — |
| `codevar` | Core binding | Code variables | Python assignments, copies, swaps; read a variable after reassignment. | acc | — |
| `dialogue` | Core binding | 3rd-person updates | Prose scene; objects change hands, partly via he/she/it; who has what. | acc | — |
| `dx-owner-3-deictic` | Deixis · owner | I/you/my | Ownership stated only via I/my/you + vocatives; query 'Now the X is with' -> a name. | acc | yes |
| `dx-owner-3-third` | Deixis · owner | 3rd-person twin | The same events narrated with names, no pronouns (the control). | acc | yes |
| `dx-self-3-deictic` | Deixis · self · P3 | I/you/my | Query 'Ana: Right now I'm holding the' -> the speaker's own object. | acc | yes |
| `dx-self-3-third` | Deixis · self · P3 | 3rd-person twin | The same events narrated with names, no pronouns (the control). | acc | yes |
| `dx-self-2-deictic` | Deixis · self · P2 | I/you/my | Query 'Ana: Right now I'm holding the' -> the speaker's own object. | acc | yes |
| `dx-self-2-third` | Deixis · self · P2 | 3rd-person twin | The same events narrated with names, no pronouns (the control). | acc | yes |
| `dx-addressee-3-deictic` | Deixis · you · P3 (vocative) | I/you/my | Query 'Ben: Ana, right now you're holding the' (P3 names 'you'; P2 = the only other person). | acc | yes |
| `dx-addressee-3-third` | Deixis · you · P3 (vocative) | 3rd-person twin | The same events narrated with names, no pronouns (the control). | acc | yes |
| `dx-addressee-2-deictic` | Deixis · you · P2 (structural) | I/you/my | Query 'Ben: Ana, right now you're holding the' (P3 names 'you'; P2 = the only other person). | acc | yes |
| `dx-addressee-2-third` | Deixis · you · P2 (structural) | 3rd-person twin | The same events narrated with names, no pronouns (the control). | acc | yes |
| `sb-k1` | Set binding | 1 object | Static, one object per queried person (baseline). | hit1 | — |
| `sb-k23` | Set binding | 2-3 objects, exact set | Person keeps 2-3 objects (one sentence). Score: top-k == the set. | exactk | — |
| `sb-split` | Set binding | split across sentences | Same, but each object declared in a separate sentence. | exactk | — |
| `sb-dyn` | Set binding | after transfers | Sets change through two gives; read the final set. | exactk | — |
| `an-narr` | Answer deixis | narrator query | Same chat, narrator asks 'Now the X is with' -> a name (comprehension). | acc | — |
| `an-other-any` | Answer deixis | answer: a name | 'Dee: Who has the cup now?' / 'Ben: I think it's with' -> another person's name. Calibrated read (vs body-hidden null). | cal | — |
| `an-me-any` | Answer deixis | answer: me | The answerer holds it -> ' me' (speak from your own seat). Calibrated read (vs body-hidden null). | cal | — |
| `an-you-voc` | Answer deixis | answer: you (named) | The asker holds it, answer opens with the asker's name -> ' you'. Calibrated read (vs body-hidden null). | cal | — |
| `an-you-novoc` | Answer deixis | answer: you (no name) | The asker holds it, no name in the answer -> ' you' (reply structure only). Calibrated read (vs body-hidden null). | cal | — |
| `adj-adj` | Adjacency | 'yours' by reply | 'Ben: Can I have the cup?' / 'Ana: Sure, the cup is yours.' -> Ben. Critical items (object last moved by a grant). | acc | — |
| `adj-voc` | Adjacency | 'yours' with the name | Same, grant names him: 'Sure, Ben, the cup is yours.' Critical items (object last moved by a grant). | acc | — |
| `adj-third` | Adjacency | 3rd-person twin | 'Ben asks Ana for the cup. Ana gives the cup to Ben.' Critical items (object last moved by a grant). | acc | — |
| `cm-mutate` | Aliased lists | mutate via alias | b = a; a.append('x'); print(b[-1]) -> the appended value. | acc | — |
| `cm-rebind` | Aliased lists | rebind after alias | b = a; a = ['new']; print(b[-1]) -> the OLD list's value. | acc | — |
| `cm-copy` | Aliased lists | copy then mutate | b = list(a); a.append('x'); print(b[-1]) -> unaffected. | acc | — |

## Example items per column (verbatim prompts)

Each prompt ends right before the scored word. `candidates` = the closed set a logprob model ranks; `gold` = the right answer. Generation-scored (API, human) rows see the same prompt with no options.

### `attr` — Core binding · Static lookup

N people each keep an object; query one. Pure retrieval under distractors.

**0-shot example** (`attr_4-first_000`) — candidates: switch, needle, clock, towel, pencil · gold: **switch**

```text
Jack keeps the switch. A needle lies on the shelf. Larry keeps the clock. Betty keeps the towel. John keeps the pencil. So Jack keeps the
```

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Core binding column uses this layout)

```text
A book lies on the shelf. Hugo has the shell. Lucy has the stone. A candle lies on the shelf. Jane has the bell. Adam has the button. So Hugo has the shell.

* * *

Kate owns the key. Paul owns the cup. Ben owns the coin. Sarah owns the pen. A glass lies on the shelf. So Kate owns the key.

* * *

Leo owns the apple. A ring lies on the shelf. Ivan owns the brush. Lucy owns the map. Ben owns the thread. So Leo owns the apple.

* * *

Jack keeps the switch. A needle lies on the shelf. Larry keeps the clock. Betty keeps the towel. John keeps the pencil. So Jack keeps the
```

**More item 1** — candidates: knife, hammer, rope, glove, ribbon · gold: **ribbon**

```text
A knife lies on the shelf. Nancy owns the hammer. Joe owns the rope. Greg owns the glove. Lily owns the ribbon. So Lily owns the
```

### `codevar` — Core binding · Code variables

Python assignments, copies, swaps; read a variable after reassignment.

**0-shot example** (`codevar_2-1_000`) — candidates: sock, shoe, magnet, dice · gold: **shoe**

```text
k = "sock"
h = "shoe"
m = "magnet"
n = h
d = n
n = "dice"
print(d)  # output:
```

**More item 1** — candidates: carrot, dice, fork, hammer · gold: **carrot**

```text
h = "carrot"
d = "dice"
k = "fork"
g = k
f = h
h = d
d = f
h = "hammer"
a = g
print(d)  # output:
```

### `dialogue` — Core binding · 3rd-person updates

Prose scene; objects change hands, partly via he/she/it; who has what.

**0-shot example** (`dialogue_3-2-holder_000`) — candidates: Jake, Molly, Erin · gold: **Erin**

```text
Jake, Molly and Erin are sitting at a table. Molly has the towel and the belt. Erin has the bottle. Jake asks Erin for the bottle, so she gives it to him. He passes it to Erin. Now the bottle is with
```

**More item 1** — candidates: Sandra, Joe, Judy · gold: **Joe**

```text
Sandra, Joe and Judy are sitting at a table. Sandra has the bolt and the dice. Joe has the screw. Joe hands the screw to Sandra. Joe asks Sandra for the screw, so she gives it to him. Now the screw is with
```

> **Deixis:** Only the 0-shot and shortcut-free examples are paired with their twin; the 'more item' examples of the I/you and 3rd-person columns are independent draws.

### `dx-owner-3-deictic` — Deixis · owner · I/you/my

Ownership stated only via I/my/you + vocatives; query 'Now the X is with' -> a name.

**0-shot example** (`deixis-v2_3-2-owner_000_deictic`) — candidates: John, Nick, Wendy · gold: **Wendy**

```text
A chat between John, Nick and Wendy.
Nick: I have the rope.
Nick: I have the nail.
Nick: Here, Wendy, take my rope.
Nick: Wendy, my nail is yours now.
Now the rope is with
```

**Shortcut-free example** — candidates: Ross, Jim, Rosa · gold: **Ross**

```text
A chat between Ross, Jim and Rosa.
Ross: The notebook is mine.
Jim: I have the ticket.
Jim: The lantern is mine.
Rosa: I have the marble.
Jim: Here, Ross, take my ticket.
Jim: Rosa, my lantern is yours now.
Now the ticket is with
```

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Deixis column uses this layout)

```text
A chat between Adam, Ruth and Owen.
Adam: The button is mine.
Adam: I have the book.
Owen: I have the bell.
Owen: Ruth, I'm giving you my bell.
Adam: Here, Ruth, take my book.
Now the bell is with Ruth.

* * *

A chat between Mary, Kate and Paul.
Mary: I have the brush.
Kate: The key is mine.
Kate: The glass is mine.
Mary: Kate, my brush is yours now.
Kate: Paul, I'm giving you my key.
Now the brush is with Kate.

* * *

A chat between Hugo, Anna and Emma.
Hugo: I have the stone.
Anna: The cup is mine.
Anna: I have the thread.
Emma: I have the map.
Emma: Hugo, my map is yours now.
Anna: Hugo, I'm giving you my cup.
Now the map is with Hugo.

* * *

A chat between John, Nick and Wendy.
Nick: I have the rope.
Nick: I have the nail.
Nick: Here, Wendy, take my rope.
Nick: Wendy, my nail is yours now.
Now the rope is with
```

**More item 1** — candidates: Bruce, Joe, Julia · gold: **Joe**

```text
A chat between Bruce, Joe and Julia.
Bruce: The needle is mine.
Bruce: I have the feather.
Bruce: I have the funnel.
Bruce: Joe, I'm giving you my feather.
Bruce: Here, Julia, take my funnel.
Now the feather is with
```

### `dx-owner-3-third` — Deixis · owner · 3rd-person twin

The same events narrated with names, no pronouns (the control).

**0-shot example** (`deixis-v2_3-2-owner_000_third`) — candidates: John, Nick, Wendy · gold: **Wendy**

```text
John, Nick and Wendy are sitting at a table. Nick has the rope and the nail. Nick gives the rope to Wendy. Nick gives the nail to Wendy. Now the rope is with
```

**Shortcut-free example** — candidates: Ross, Jim, Rosa · gold: **Ross**

```text
Ross, Jim and Rosa are sitting at a table. Ross has the notebook. Jim has the ticket and the lantern. Rosa has the marble. Jim gives the ticket to Ross. Jim gives the lantern to Rosa. Now the ticket is with
```

**More item 1** — candidates: Susan, Peter, Oscar · gold: **Susan**

```text
Susan, Peter and Oscar are sitting at a table. Susan has the soap. Peter has the clock and the switch. Peter gives the clock to Susan. Peter gives the switch to Oscar. Now the clock is with
```

### `dx-self-3-deictic` — Deixis · self · P3 · I/you/my

Query 'Ana: Right now I'm holding the' -> the speaker's own object.

**0-shot example** (`deixis-v2_3-2-self_000_deictic`) — candidates: basket, chain, dice · gold: **basket**

```text
A chat between Mike, Walt and Laura.
Mike: I have the basket.
Walt: The chain is mine.
Laura: The dice is mine.
Laura: Mike, I'm giving you my dice.
Mike: Here, Laura, take my basket.
Laura: Right now I'm holding the
```

**Shortcut-free example** — candidates: ticket, needle, spoon · gold: **ticket**

```text
A chat between Molly, Bruce and Seth.
Molly: The ticket is mine.
Bruce: The needle is mine.
Seth: I have the spoon.
Molly: Seth, I'm giving you my ticket.
Seth: Here, Molly, take my spoon.
Seth: Right now I'm holding the
```

**More item 1** — candidates: fork, lamp, ribbon, wallet · gold: **ribbon**

```text
A chat between Linda, Ellen and Erin.
Erin: I have the fork.
Erin: I have the lamp.
Erin: The ribbon is mine.
Erin: The wallet is mine.
Erin: Ellen, my ribbon is yours now.
Erin: Linda, I'm giving you my lamp.
Ellen: Right now I'm holding the
```

### `dx-self-3-third` — Deixis · self · P3 · 3rd-person twin

The same events narrated with names, no pronouns (the control).

**0-shot example** (`deixis-v2_3-2-self_000_third`) — candidates: basket, chain, dice · gold: **basket**

```text
Mike, Walt and Laura are sitting at a table. Mike has the basket. Walt has the chain. Laura has the dice. Laura gives the dice to Mike. Mike gives the basket to Laura. Now Laura is holding the
```

**Shortcut-free example** — candidates: ticket, needle, spoon · gold: **ticket**

```text
Molly, Bruce and Seth are sitting at a table. Molly has the ticket. Bruce has the needle. Seth has the spoon. Molly gives the ticket to Seth. Seth gives the spoon to Molly. Now Seth is holding the
```

**More item 1** — candidates: whistle, wire, straw, lid · gold: **lid**

```text
Tony, Linda and Sean are sitting at a table. Tony has the whistle and the wire. Sean has the straw and the lid. Sean gives the straw to Tony. Tony gives the whistle to Linda. Now Sean is holding the
```

### `dx-self-2-deictic` — Deixis · self · P2 · I/you/my

Query 'Ana: Right now I'm holding the' -> the speaker's own object.

**0-shot example** (`deixis-v2_2-2-self_000_deictic`) — candidates: jar, magnet, lemon · gold: **magnet**

```text
A chat between Emily and Kevin.
Emily: The jar is mine.
Kevin: The magnet is mine.
Kevin: The lemon is mine.
Emily: I'm giving you my jar.
Kevin: Here, take my magnet.
Emily: Right now I'm holding the
```

**Shortcut-free example** — candidates: screw, hat, carrot · gold: **screw**

```text
A chat between Alice and Jerry.
Alice: I have the screw.
Alice: I have the hat.
Alice: I have the carrot.
Alice: I'm giving you my hat.
Alice: Here, take my carrot.
Alice: Right now I'm holding the
```

**More item 1** — candidates: glue, bulb, glove, ticket · gold: **glove**

```text
A chat between Jake and Molly.
Jake: I have the glue.
Jake: I have the bulb.
Jake: I have the glove.
Molly: The ticket is mine.
Jake: My glove is yours now.
Molly: I'm giving you my ticket.
Molly: Right now I'm holding the
```

### `dx-self-2-third` — Deixis · self · P2 · 3rd-person twin

The same events narrated with names, no pronouns (the control).

**0-shot example** (`deixis-v2_2-2-self_000_third`) — candidates: jar, magnet, lemon · gold: **magnet**

```text
Emily and Kevin are sitting at a table. Emily has the jar. Kevin has the magnet and the lemon. Emily gives the jar to Kevin. Kevin gives the magnet to Emily. Now Emily is holding the
```

**Shortcut-free example** — candidates: screw, hat, carrot · gold: **screw**

```text
Alice and Jerry are sitting at a table. Alice has the screw, the hat and the carrot. Alice gives the hat to Jerry. Alice gives the carrot to Jerry. Now Alice is holding the
```

**More item 1** — candidates: sock, bottle, soap, whistle · gold: **whistle**

```text
Hal and Alice are sitting at a table. Hal has the sock. Alice has the bottle, the soap and the whistle. Alice gives the soap to Hal. Alice gives the bottle to Hal. Now Alice is holding the
```

### `dx-addressee-3-deictic` — Deixis · you · P3 (vocative) · I/you/my

Query 'Ben: Ana, right now you're holding the' (P3 names 'you'; P2 = the only other person).

**0-shot example** (`deixis-v2_3-2-addressee_000_deictic`) — candidates: dice, banana, notebook · gold: **banana**

```text
A chat between Ada, Ross and Linda.
Ada: I have the dice.
Ada: The banana is mine.
Ross: The notebook is mine.
Ada: Here, Ross, take my banana.
Ross: Linda, my banana is yours now.
Ross: Linda, right now you're holding the
```

**Shortcut-free example** — candidates: plate, dice, pencil · gold: **dice**

```text
A chat between Rosa, Larry and Mike.
Rosa: The plate is mine.
Rosa: I have the dice.
Larry: I have the pencil.
Rosa: Larry, I'm giving you my plate.
Larry: Here, Mike, take my pencil.
Mike: Rosa, right now you're holding the
```

**More item 1** — candidates: magnet, shoe, wire · gold: **shoe**

```text
A chat between Ada, Maya and Laura.
Ada: I have the magnet.
Laura: I have the shoe.
Laura: The wire is mine.
Laura: Ada, I'm giving you my wire.
Ada: Here, Maya, take my wire.
Ada: Laura, right now you're holding the
```

### `dx-addressee-3-third` — Deixis · you · P3 (vocative) · 3rd-person twin

The same events narrated with names, no pronouns (the control).

**0-shot example** (`deixis-v2_3-2-addressee_000_third`) — candidates: dice, banana, notebook · gold: **banana**

```text
Ada, Ross and Linda are sitting at a table. Ada has the dice and the banana. Ross has the notebook. Ada gives the banana to Ross. Ross gives the banana to Linda. Now Linda is holding the
```

**Shortcut-free example** — candidates: plate, dice, pencil · gold: **dice**

```text
Rosa, Larry and Mike are sitting at a table. Rosa has the plate and the dice. Larry has the pencil. Rosa gives the plate to Larry. Larry gives the pencil to Mike. Now Rosa is holding the
```

**More item 1** — candidates: lemon, dice, hat, soap · gold: **soap**

```text
Greg, Jessica and Molly are sitting at a table. Greg has the lemon, the dice and the hat. Molly has the soap. Molly gives the soap to Jessica. Greg gives the hat to Molly. Now Jessica is holding the
```

### `dx-addressee-2-deictic` — Deixis · you · P2 (structural) · I/you/my

Query 'Ben: Ana, right now you're holding the' (P3 names 'you'; P2 = the only other person).

**0-shot example** (`deixis-v2_2-2-addressee_000_deictic`) — candidates: chain, glue, screw · gold: **glue**

```text
A chat between Betty and Lily.
Betty: I have the chain.
Betty: The glue is mine.
Lily: The screw is mine.
Lily: My screw is yours now.
Betty: I'm giving you my glue.
Betty: Right now you're holding the
```

**Shortcut-free example** — candidates: photo, bulb, sponge · gold: **photo**

```text
A chat between Beth and Ryan.
Ryan: I have the photo.
Ryan: The bulb is mine.
Ryan: I have the sponge.
Ryan: Here, take my sponge.
Ryan: My bulb is yours now.
Beth: Right now you're holding the
```

**More item 1** — candidates: battery, hammer, lamp, stamp · gold: **lamp**

```text
A chat between Diana and Martha.
Diana: The battery is mine.
Diana: I have the hammer.
Diana: The lamp is mine.
Martha: The stamp is mine.
Diana: Here, take my hammer.
Diana: My battery is yours now.
Martha: Right now you're holding the
```

### `dx-addressee-2-third` — Deixis · you · P2 (structural) · 3rd-person twin

The same events narrated with names, no pronouns (the control).

**0-shot example** (`deixis-v2_2-2-addressee_000_third`) — candidates: chain, glue, screw · gold: **glue**

```text
Betty and Lily are sitting at a table. Betty has the chain and the glue. Lily has the screw. Lily gives the screw to Betty. Betty gives the glue to Lily. Now Lily is holding the
```

**Shortcut-free example** — candidates: photo, bulb, sponge · gold: **photo**

```text
Beth and Ryan are sitting at a table. Ryan has the photo, the bulb and the sponge. Ryan gives the sponge to Beth. Ryan gives the bulb to Beth. Now Ryan is holding the
```

**More item 1** — candidates: rope, wire, onion · gold: **wire**

```text
Emily and Kevin are sitting at a table. Emily has the rope, the wire and the onion. Emily gives the rope to Kevin. Emily gives the onion to Kevin. Now Emily is holding the
```

> **Set binding:** 3-shot exemplars show completed sentences (often one object); the test item is scored on the exact gold set.

### `sb-k1` — Set binding · 1 object

Static, one object per queried person (baseline).

**0-shot example** (`setbind-v1_1-joint-0_000`) — candidates: tape, nail, carrot, boot, mirror, letter, wallet, onion · gold: **onion**

```text
Kevin has the tape and the nail. Neil has the carrot and the boot. A mirror lies on the shelf. A letter lies on the shelf. Steve has the wallet. Jerry has the onion. So Jerry has the
```

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Set binding column uses this layout)

```text
Paul has the brush and the ring. Ivan has the button. A candle lies on the shelf. Hugo has the map and the cup. Jane has the pen. A stone lies on the shelf. So Jane has the pen.

* * *

Ben has the candle and the glass. A stone lies on the shelf. A pen lies on the shelf. Emma has the brush. Sarah has the shell. Lucy has the book and the ring. So Sarah has the shell.

* * *

A map lies on the shelf. Mary has the button. Adam has the shell and the pen. Leo has the stone and the bell. Kate has the brush. A book lies on the shelf. So Kate has the brush.

* * *

Kevin has the tape and the nail. Neil has the carrot and the boot. A mirror lies on the shelf. A letter lies on the shelf. Steve has the wallet. Jerry has the onion. So Jerry has the
```

**More item 1** — candidates: rope, stamp, boot, marble, potato, belt, fork, shoe · gold: **stamp**

```text
A rope lies on the shelf. Amy has the stamp. A boot lies on the shelf. Bruce has the marble. George has the potato and the belt. Rachel has the fork and the shoe. So Amy has the
```

### `sb-k23` — Set binding · 2-3 objects, exact set

Person keeps 2-3 objects (one sentence). Score: top-k == the set.

**0-shot example** (`setbind-v1_2-joint-0_000`) — candidates: feather, carrot, shoe, spoon, lamp, nail, boot, switch · gold: **boot, switch**

```text
Joe has the feather. A carrot lies on the shelf. Hal has the shoe and the spoon. Sandra has the lamp. A nail lies on the shelf. Amanda has the boot and the switch. So Amanda has the
```

**More item 1** — candidates: towel, feather, sock, envelope, onion, fork, switch, plate · gold: **switch, plate**

```text
Ellen has the towel and the feather. Rachel has the sock. An envelope lies on the shelf. An onion lies on the shelf. Matt has the fork. George has the switch and the plate. So George has the
```

### `sb-split` — Set binding · split across sentences

Same, but each object declared in a separate sentence.

**0-shot example** (`setbind-v1_3-split-0_000`) — candidates: stamp, ribbon, magnet, spoon, bulb, lid, drum, glove · gold: **stamp, spoon, glove**

```text
Oscar has the stamp. Betty has the ribbon. Jessica has the magnet. Oscar also has the spoon. Amy has the bulb. A lid lies on the shelf. A drum lies on the shelf. Oscar also has the glove. So Oscar has the
```

**More item 1** — candidates: onion, boot, feather, marble, banana, mirror, hat, rope · gold: **onion, marble, hat**

```text
Ada has the onion. A boot lies on the shelf. Frank has the feather. Ada also has the marble. Olivia has the banana. A mirror lies on the shelf. Ada also has the hat. Sean has the rope. So Ada has the
```

### `sb-dyn` — Set binding · after transfers

Sets change through two gives; read the final set.

**0-shot example** (`setbind-v1_2-joint-2_000`) — candidates: fan, hammer, screw, magnet, stamp, banana, feather, tape · gold: **hammer, feather**

```text
Diana has the fan and the hammer. Carol has the screw and the magnet. Jill has the stamp. A banana lies on the shelf. Nancy has the feather. A tape lies on the shelf. Diana gives the fan to Nancy. Nancy gives the feather to Diana. So now Diana has the
```

**More item 1** — candidates: letter, jar, knife, switch, photo, screw, towel, boot · gold: **screw, boot**

```text
Ken has the letter. A jar lies on the shelf. George has the knife. Lily has the switch. A photo lies on the shelf. Amanda has the screw, the towel and the boot. Ken gives the letter to Lily. Amanda gives the towel to Ken. So now Amanda has the
```

> **Answer deixis:** Roles are cycled so every constant answer is at chance: holder = the answerer (me) ¼, the asker (you) ¼, one of the two others (a name) ½. For open models the candidates are me / you / the two OTHER people's names; exemplar answers follow the same cycle, so 'you'/'me' exemplars before a name item are expected.

### `an-narr` — Answer deixis · narrator query

Same chat, narrator asks 'Now the X is with' -> a name (comprehension).

**0-shot example** (`answer-v2_1-novoc_000_narr`) — candidates: Tim, Nina, Carl, Erin · gold: **Nina**

```text
A chat between Tim, Nina, Carl and Erin.
Tim: I have the envelope.
Tim: The soap is mine.
Nina: The towel is mine.
Nina: The sock is mine.
Tim: Nina, my soap is yours now.
Now the soap is with
```

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Answer deixis column uses this layout)

```text
A chat between Mary, Emma, Ivan and Kate.
Mary: I have the button.
Emma: The brush is mine.
Ivan: The key is mine.
Emma: Mary, I'm giving you my brush.
Now the brush is with Mary.

* * *

A chat between Tom, Adam, Anna and Paul.
Tom: The ring is mine.
Adam: The glass is mine.
Adam: The bell is mine.
Adam: Tom, my glass is yours now.
Now the glass is with Tom.

* * *

A chat between Ben, Leo, Owen and Sarah.
Ben: I have the stone.
Leo: The pen is mine.
Owen: I have the cup.
Sarah: The apple is mine.
Sarah: Owen, I'm giving you my apple.
Now the apple is with Owen.

* * *

A chat between Tim, Nina, Carl and Erin.
Tim: I have the envelope.
Tim: The soap is mine.
Nina: The towel is mine.
Nina: The sock is mine.
Tim: Nina, my soap is yours now.
Now the soap is with
```

**More item 1** — candidates: Carl, Julia, Alice, Nina · gold: **Carl**

```text
A chat between Carl, Julia, Alice and Nina.
Carl: The hammer is mine.
Alice: I have the feather.
Nina: I have the chain.
Nina: The envelope is mine.
Nina: Here, Julia, take my envelope.
Now the hammer is with
```

### `an-other-any` — Answer deixis · answer: a name

'Dee: Who has the cup now?' / 'Ben: I think it's with' -> another person's name. Calibrated read (vs body-hidden null).

**0-shot example** (`answer-v2_1-novoc_002_qa`) — candidates: me, you, Kevin, Martha · gold: **Kevin**

```text
A chat between Kevin, Helen, Clara and Martha.
Clara: The hat is mine.
Clara: I have the carrot.
Clara: I have the mirror.
Clara: Here, Kevin, take my hat.
Clara: Who has the hat now?
Helen: I think it's with
```

**More item 1** — candidates: me, you, Sandra, Nina · gold: **Sandra**

```text
A chat between Sandra, Eric, Nina and James.
Sandra: I have the sponge.
Nina: I have the drum.
Nina: I have the potato.
Nina: The whistle is mine.
Nina: Here, James, take my drum.
Eric: Who has the sponge now?
James: Eric, I think it's with
```

### `an-me-any` — Answer deixis · answer: me

The answerer holds it -> ' me' (speak from your own seat). Calibrated read (vs body-hidden null).

**0-shot example** (`answer-v2_1-novoc_000_qa`) — candidates: me, you, Tim, Carl · gold: **me**

```text
A chat between Tim, Nina, Carl and Erin.
Tim: I have the envelope.
Tim: The soap is mine.
Nina: The towel is mine.
Nina: The sock is mine.
Tim: Nina, my soap is yours now.
Erin: Who has the soap now?
Nina: I think it's with
```

**More item 1** — candidates: me, you, Ross, Luke · gold: **me**

```text
A chat between Laura, Ross, Luke and Hal.
Ross: I have the knife.
Luke: I have the wire.
Hal: The shoe is mine.
Luke: Hal, my wire is yours now.
Ross: Laura, I'm giving you my knife.
Laura: Who has the wire now?
Hal: I think it's with
```

### `an-you-voc` — Answer deixis · answer: you (named)

The asker holds it, answer opens with the asker's name -> ' you'. Calibrated read (vs body-hidden null).

**0-shot example** (`answer-v2_1-voc_001_qa`) — candidates: me, you, Oscar, Emily · gold: **you**

```text
A chat between Seth, Tim, Oscar and Emily.
Seth: I have the wallet.
Tim: I have the hammer.
Emily: I have the banana.
Tim: Seth, I'm giving you my hammer.
Seth: Who has the hammer now?
Tim: Seth, I think it's with
```

**More item 1** — candidates: me, you, Henry, Maya · gold: **you**

```text
A chat between Henry, Maya, Tony and John.
Henry: The orange is mine.
Maya: The envelope is mine.
Tony: I have the photo.
John: I have the rope.
Maya: Here, Henry, take my envelope.
John: Who has the rope now?
Tony: John, I think it's with
```

### `an-you-novoc` — Answer deixis · answer: you (no name)

The asker holds it, no name in the answer -> ' you' (reply structure only). Calibrated read (vs body-hidden null).

**0-shot example** (`answer-v2_1-novoc_001_qa`) — candidates: me, you, Julia, Amy · gold: **you**

```text
A chat between Julia, Wendy, Amy and Jim.
Julia: I have the bottle.
Amy: The dice is mine.
Jim: I have the rope.
Amy: Wendy, I'm giving you my dice.
Wendy: Who has the dice now?
Jim: I think it's with
```

**More item 1** — candidates: me, you, Kevin, Lily · gold: **you**

```text
A chat between Kevin, Eric, Lily and Helen.
Kevin: I have the stamp.
Helen: I have the jar.
Helen: I have the orange.
Helen: The ticket is mine.
Kevin: Lily, my stamp is yours now.
Helen: Who has the ticket now?
Eric: I think it's with
```

> **Adjacency:** Critical items (the scored ones) end with the object last moved by a grant; exemplars may be non-critical.

### `adj-adj` — Adjacency · 'yours' by reply

'Ben: Can I have the cup?' / 'Ana: Sure, the cup is yours.' -> Ben. Critical items (object last moved by a grant).

**0-shot example** (`adjacency-v1_1-1_000_adj`) — candidates: Kurt, Kevin, John · gold: **John**

```text
A chat between Kurt, Kevin and John.
Kurt: I have the nail.
Kevin: The potato is mine.
John: I have the fork.
John: Can I have the nail?
Kurt: Sure, the nail is yours.
Now the nail is with
```

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Adjacency column uses this layout)

```text
A chat between Leo, Adam and Ivan.
Leo: The thread is mine.
Leo: I have the brush.
Leo: The bell is mine.
Ivan: The map is mine.
Leo: Can I have the map?
Ivan: Sure, the map is yours.
Now the map is with Leo.

* * *

A chat between Anna, Lucy and Jane.
Lucy: The coin is mine.
Lucy: The glass is mine.
Lucy: I have the ring.
Anna: Can I have the coin?
Lucy: Sure, the coin is yours.
Now the coin is with Anna.

* * *

A chat between Mary, Kate and Sarah.
Kate: The shell is mine.
Kate: I have the cup.
Sarah: Can I have the cup?
Kate: Sure, take the cup.
Now the shell is with Kate.

* * *

A chat between Kurt, Kevin and John.
Kurt: I have the nail.
Kevin: The potato is mine.
John: I have the fork.
John: Can I have the nail?
Kurt: Sure, the nail is yours.
Now the nail is with
```

**More item 1** — candidates: Jill, John, Mike · gold: **Jill**

```text
A chat between Jill, John and Mike.
John: The letter is mine.
Mike: I have the lemon.
Jill: Can I have the lemon?
Mike: Sure, the lemon is yours.
Now the lemon is with
```

### `adj-voc` — Adjacency · 'yours' with the name

Same, grant names him: 'Sure, Ben, the cup is yours.' Critical items (object last moved by a grant).

**0-shot example** (`adjacency-v1_1-1_000_voc`) — candidates: Kurt, Kevin, John · gold: **John**

```text
A chat between Kurt, Kevin and John.
Kurt: I have the nail.
Kevin: The potato is mine.
John: I have the fork.
John: Can I have the nail?
Kurt: Sure, John, the nail is yours.
Now the nail is with
```

**More item 1** — candidates: Linda, Helen, Joe · gold: **Joe**

```text
A chat between Linda, Helen and Joe.
Helen: The soap is mine.
Helen: I have the glue.
Joe: Can I have the glue?
Helen: Sure, Joe, the glue is yours.
Now the glue is with
```

### `adj-third` — Adjacency · 3rd-person twin

'Ben asks Ana for the cup. Ana gives the cup to Ben.' Critical items (object last moved by a grant).

**0-shot example** (`adjacency-v1_1-1_000_third`) — candidates: Kurt, Kevin, John · gold: **John**

```text
Kurt, Kevin and John are sitting at a table. Kurt has the nail. Kevin has the potato. John has the fork. John asks Kurt for the nail. Kurt gives the nail to John. Now the nail is with
```

**More item 1** — candidates: David, Walt, Simon · gold: **Simon**

```text
David, Walt and Simon are sitting at a table. David has the envelope and the marble. Simon has the screw. Simon asks David for the marble. David gives the marble to Simon. Now the marble is with
```

### `cm-mutate` — Aliased lists · mutate via alias

b = a; a.append('x'); print(b[-1]) -> the appended value.

**0-shot example** (`codevarmut-v1_1-mutate_000`) — candidates: bolt, spoon, straw, mirror, lantern · gold: **lantern**

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

**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — every Aliased lists column uses this layout)

```text
# python
y = ["ring"]
z = ["apple"]
w = ["brush"]
t = y
y.append("thread")
z = ["key"]
print(t[-1])  # output: thread

# ---

# python
v = ["book"]
u = ["cup"]
r = ["shell"]
u = ["button"]
p = r
r.append("map")
print(p[-1])  # output: map

# ---

# python
y = ["pen"]
q = ["coin"]
t = ["bell"]
q.append("candle")
v = y
y.append("stone")
print(v[-1])  # output: stone

# ---

# python
k = ["bolt"]
f = ["spoon"]
n = ["straw"]
d = k
n.append("mirror")
k.append("lantern")
print(d[-1])  # output:
```

**More item 1** — candidates: notebook, knife, jar, bolt, hat · gold: **hat**

```text
# python
a = ["notebook"]
h = ["knife"]
k = ["jar"]
h.append("bolt")
n = a
a.append("hat")
print(n[-1])  # output:
```

### `cm-rebind` — Aliased lists · rebind after alias

b = a; a = ['new']; print(b[-1]) -> the OLD list's value.

**0-shot example** (`codevarmut-v1_1-rebind_000`) — candidates: glue, bottle, magnet, shoe, chain · gold: **glue**

```text
# python
m = ["glue"]
f = ["bottle"]
n = ["magnet"]
f.append("shoe")
k = m
m = ["chain"]
print(k[-1])  # output:
```

**More item 1** — candidates: screw, letter, ticket, boot, clock · gold: **screw**

```text
# python
c = ["screw"]
b = ["letter"]
f = ["ticket"]
h = c
c = ["boot"]
f.append("clock")
print(h[-1])  # output:
```

### `cm-copy` — Aliased lists · copy then mutate

b = list(a); a.append('x'); print(b[-1]) -> unaffected.

**0-shot example** (`codevarmut-v1_1-copy_000`) — candidates: wire, ribbon, funnel, spoon, tape · gold: **funnel**

```text
# python
b = ["wire"]
h = ["ribbon"]
e = ["funnel"]
k = list(e)
b.append("spoon")
e.append("tape")
print(k[-1])  # output:
```

**More item 1** — candidates: sock, mirror, dice, lid, magnet · gold: **sock**

```text
# python
d = ["sock"]
k = ["mirror"]
g = ["dice"]
a = list(d)
d.append("lid")
g.append("magnet")
print(a[-1])  # output:
```

## Rows

API rows (`or-…`) are all generation-scored; `-A` = reflex frame (single next word, reasoning off), `-B` = deliberate frame (think, then the word). Notes below list only what differs from that.

| id | label | group | size (B params) | note |
|---|---|---|---|---|
| `human` | Human reader (gauntlet) | Human | — | reflex, read cold: round 1 (31 items, 30 right) + round 2 (24 items, 22 right; both misses self-reported reading slips) |
| `or-amazon/nova-pro-v1-A` | Amazon Nova Pro · reflex | API · closed | — |  |
| `or-anthropic/claude-fable-5.1-A` | Claude Fable 5.1 · reflex | API · closed | — |  |
| `or-anthropic/claude-haiku-4.5-B` | Claude Haiku 4.5 · deliberate | API · closed | — |  |
| `or-anthropic/claude-haiku-4.5-A` | Claude Haiku 4.5 · reflex | API · closed | — |  |
| `or-anthropic/claude-haiku-5.5-B` | Claude Haiku 5.5 · deliberate | API · closed | — |  |
| `or-anthropic/claude-haiku-5.5-A` | Claude Haiku 5.5 · reflex | API · closed | — |  |
| `or-anthropic/claude-opus-4.5-A` | Claude Opus 4.5 · reflex | API · closed | — |  |
| `or-anthropic/claude-opus-5.5-B` | Claude Opus 5.5 · deliberate | API · closed | — |  |
| `or-anthropic/claude-opus-5.5-A` | Claude Opus 5.5 · reflex | API · closed | — | 27% of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls) |
| `or-anthropic/claude-sonnet-4.5-A` | Claude Sonnet 4.5 · reflex | API · closed | — |  |
| `or-anthropic/claude-sonnet-5.5-B` | Claude Sonnet 5.5 · deliberate | API · closed | — |  |
| `or-anthropic/claude-sonnet-5.5-A` | Claude Sonnet 5.5 · reflex | API · closed | — |  |
| `or-google/gemini-2.5-flash-B` | Gemini 2.5 Flash · deliberate | API · closed | — |  |
| `or-google/gemini-2.5-flash-A` | Gemini 2.5 Flash · reflex | API · closed | — |  |
| `or-google/gemini-2.5-flash-lite-A` | Gemini 2.5 Flash-Lite · reflex | API · closed | — |  |
| `or-google/gemini-3.1-pro-preview-B` | Gemini 3.1 Pro (preview) · deliberate | API · closed | — |  |
| `or-openai/gpt-4.1-mini-A` | GPT-4.1 mini · reflex | API · closed | — |  |
| `or-openai/gpt-4.1-nano-A` | GPT-4.1 nano · reflex | API · closed | — |  |
| `or-openai/gpt-4.1-A` | GPT-4.1 · reflex | API · closed | — |  |
| `or-openai/gpt-4o-mini-A` | GPT-4o mini · reflex | API · closed | — |  |
| `or-openai/gpt-5-mini-B` | GPT-5 mini · deliberate | API · closed | — |  |
| `or-openai/gpt-5-mini-A` | GPT-5 mini · reflex | API · closed | — |  |
| `or-openai/gpt-5-nano-B` | GPT-5 nano · deliberate | API · closed | — |  |
| `or-openai/gpt-5-nano-A` | GPT-5 nano · reflex | API · closed | — |  |
| `or-openai/gpt-5.5-B` | GPT-5.5 · deliberate | API · closed | — |  |
| `or-openai/gpt-5.5-A` | GPT-5.5 · reflex | API · closed | — |  |
| `or-x-ai/grok-4.3-B` | Grok 4.3 · deliberate | API · closed | — |  |
| `or-x-ai/grok-4.3-A` | Grok 4.3 · reflex | API · closed | — |  |
| `or-deepseek/deepseek-v3.2-B` | DeepSeek V3.2 (671B MoE) · deliberate | API · open weights | 671 |  |
| `or-deepseek/deepseek-v3.2-A` | DeepSeek V3.2 (671B MoE) · reflex | API · open weights | 671 | 28% of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls) |
| `or-google/gemma-3-12b-it-A` | Gemma-3 12B IT · reflex | API · open weights | 12 |  |
| `or-google/gemma-3-27b-it-A` | Gemma-3 27B IT · reflex | API · open weights | 27 |  |
| `or-google/gemma-3-4b-it-A` | Gemma-3 4B IT · reflex | API · open weights | 4 |  |
| `or-z-ai/glm-4.6-B` | GLM-4.6 (355B MoE) · deliberate | API · open weights | 355 |  |
| `or-z-ai/glm-4.6-A` | GLM-4.6 (355B MoE) · reflex | API · open weights | 355 |  |
| `or-openai/gpt-oss-120b-B` | gpt-oss 120B · deliberate | API · open weights | 120 |  |
| `or-openai/gpt-oss-120b-A` | gpt-oss 120B · reflex | API · open weights | 120 | 58% of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls) |
| `or-nousresearch/hermes-3-llama-3.1-405b-A` | Hermes-3 405B · reflex | API · open weights | 405 |  |
| `or-moonshotai/kimi-k2-0905-A` | Kimi K2 (1T MoE) · reflex | API · open weights | 1000 |  |
| `or-meta-llama/llama-3.1-8b-instruct-A` | Llama-3.1 8B Instruct · reflex | API · open weights | 8 |  |
| `or-meta-llama/llama-3.2-1b-instruct-A` | Llama-3.2 1B Instruct · reflex | API · open weights | 1 |  |
| `or-meta-llama/llama-3.2-3b-instruct-A` | Llama-3.2 3B Instruct · reflex | API · open weights | 3 |  |
| `or-meta-llama/llama-3.3-70b-instruct-A` | Llama-3.3 70B Instruct · reflex | API · open weights | 70 |  |
| `or-meta-llama/llama-4-maverick-A` | Llama-4 Maverick (400B MoE) · reflex | API · open weights | 400 |  |
| `or-mistralai/ministral-3b-2512-A` | Ministral 3B · reflex | API · open weights | 3 |  |
| `or-mistralai/ministral-8b-2512-A` | Ministral 8B · reflex | API · open weights | 8 |  |
| `or-mistralai/mistral-large-2512-A` | Mistral Large (2512) · reflex | API · open weights | — |  |
| `or-mistralai/mistral-small-3.2-24b-instruct-A` | Mistral Small 3.2 24B · reflex | API · open weights | 24 |  |
| `or-microsoft/phi-4-A` | Phi-4 14B · reflex | API · open weights | 14 |  |
| `or-qwen/qwen-2.5-72b-instruct-A` | Qwen2.5 72B Instruct · reflex | API · open weights | 72 |  |
| `or-qwen/qwen-2.5-7b-instruct-A` | Qwen2.5 7B Instruct · reflex | API · open weights | 7 |  |
| `or-qwen/qwen3-235b-a22b-2507-A` | Qwen3 235B-A22B · reflex | API · open weights | 235 |  |
| `or-qwen/qwen3-32b-A` | Qwen3 32B · reflex | API · open weights | 32 | 37% of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls) |
| `or-qwen/qwen3.5-397b-a17b-B` | Qwen3.5 397B-A17B · deliberate | API · open weights | 397 |  |
| `or-qwen/qwen3.5-397b-a17b-A` | Qwen3.5 397B-A17B · reflex | API · open weights | 397 | 2% of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls) |
| `smollm2-135m` | SmolLM2 135M | Pretrained bases | 0.135 |  |
| `pythia-160m` | Pythia 160M | Pretrained bases | 0.16 |  |
| `smollm2-360m` | SmolLM2 360M | Pretrained bases | 0.36 |  |
| `pythia-410m` | Pythia 410M | Pretrained bases | 0.41 |  |
| `qwen2.5-0.5b` | Qwen2.5 0.5B | Pretrained bases | 0.5 |  |
| `qwen3-0.6b-base` | Qwen3 0.6B | Pretrained bases | 0.6 |  |
| `llama3.2-1b` | Llama-3.2 1B | Pretrained bases | 1 |  |
| `olmo1-1b-0724` | OLMo-1 1B (0724) | Pretrained bases | 1 |  |
| `pythia-1b` | Pythia 1B | Pretrained bases | 1 |  |
| `tsumugi-release` | kotodama-tsumugi 1.4B (release) | Pretrained bases | 1.4 |  |
| `pythia-1.4b` | Pythia 1.4B | Pretrained bases | 1.4 |  |
| `olmo2-1b` | OLMo-2 1B | Pretrained bases | 1.5 |  |
| `qwen2.5-1.5b` | Qwen2.5 1.5B | Pretrained bases | 1.5 |  |
| `qwen3-1.7b-base` | Qwen3 1.7B | Pretrained bases | 1.7 |  |
| `smollm2-1.7b` | SmolLM2 1.7B | Pretrained bases | 1.7 |  |
| `pythia-2.8b` | Pythia 2.8B | Pretrained bases | 2.8 |  |
| `falcon3-3b` | Falcon3 3B | Pretrained bases | 3 |  |
| `qwen2.5-3b` | Qwen2.5 3B | Pretrained bases | 3 |  |
| `qwen3-4b-base` | Qwen3 4B | Pretrained bases | 4 |  |
| `pythia-6.9b` | Pythia 6.9B | Pretrained bases | 6.9 |  |
| `falcon3-7b` | Falcon3 7B | Pretrained bases | 7 |  |
| `olmo2-7b-base` | OLMo-2 7B | Pretrained bases | 7 |  |
| `qwen2.5-7b` | Qwen2.5 7B | Pretrained bases | 7 |  |
| `llama31-8b-base` | Llama-3.1 8B | Pretrained bases | 8 |  |
| `gemma2-9b` | Gemma-2 9B | Pretrained bases | 9 | scored with vLLM (parity-checked against HF) |
| `olmo2-13b` | OLMo-2 13B | Pretrained bases | 13 |  |
| `qwen3-30b-a3b-base` | Qwen3 30B-A3B | Pretrained bases | 30 | scored with vLLM (parity-checked against HF) |
| `gemma4-31b-base` | Gemma-4 31B | Pretrained bases | 31 | scored with vLLM (parity-checked against HF) |
| `glm4-32b-base` | GLM-4 32B | Pretrained bases | 32 | scored with vLLM (parity-checked against HF) |
| `olmo3-32b` | OLMo-3 32B | Pretrained bases | 32 | scored with vLLM (parity-checked against HF) |
| `olmo3-32b-stage1` | OLMo-3 32B @ stage 1 (pre-midtrain) | Pretrained bases | 32 | pre-midtraining checkpoint; scored with vLLM (parity-checked against HF) |
| `llama31-70b` | Llama-3.1 70B | Pretrained bases | 70 | scored with vLLM (parity-checked against HF) |
| `qwen25-72b` | Qwen2.5 72B | Pretrained bases | 72 | scored with vLLM (parity-checked against HF) |
| `dsv4-flash-base` | DeepSeek-V4 Flash Base | Pretrained bases | 290 | ≈290B total params (MoE); scored with vLLM (parity-checked against HF) |
| `mimo-v25-base` | MiMo-V2.5 Base | Pretrained bases | 310 | ≈310B total params (MoE); scored with vLLM (parity-checked against HF) |
| `trinity-base` | Trinity-Large Base | Pretrained bases | 400 | ≈400B MoE; scored with vLLM (parity-checked against HF) |
| `trinity-truebase` | Trinity-Large TrueBase | Pretrained bases | 400 | ≈400B MoE; scored with vLLM (parity-checked against HF) |
| `llama31-405b` | Llama-3.1 405B | Pretrained bases | 405 | scored with vLLM (parity-checked against HF) |
| `dsv3-base` | DeepSeek-V3 Base (671B MoE) | Pretrained bases | 671 | scored with vLLM (parity-checked against HF) |
| `dsv4-pro-base` | DeepSeek-V4 Pro Base | Pretrained bases | 860 | ≈860B total params (MoE); scored from an fp4 copy; scored with vLLM (parity-checked against HF) |
| `kimi-k2-base` | Kimi-K2 Base (1T MoE) | Pretrained bases | 1000 | scored with vLLM (parity-checked against HF) |
| `mimo-v25-pro-base` | MiMo-V2.5-Pro Base | Pretrained bases | 1000 | ≈1T total params (MoE); scored with vLLM (parity-checked against HF) |
| `olmo2-7b-pt2928B` | OLMo-2 7B @ 2.9T (stage 1) | OLMo-2 7B trajectory | 7 |  |
| `olmo2-7b-pt3896B` | OLMo-2 7B @ 3.9T (end stage 1) | OLMo-2 7B trajectory | 7 |  |
| `olmo2-7b-mid5B` | OLMo-2 7B mid-train +5B | OLMo-2 7B trajectory | 7 |  |
| `olmo2-7b-mid17B` | OLMo-2 7B mid-train +17B | OLMo-2 7B trajectory | 7 |  |
| `olmo2-7b-mid50B` | OLMo-2 7B mid-train +50B | OLMo-2 7B trajectory | 7 |  |
| `olmo2-7b-sft` | OLMo-2 7B SFT | Post-trained | 7 |  |
| `olmo2-7b-dpo` | OLMo-2 7B DPO | Post-trained | 7 |  |
| `olmo2-7b-instruct` | OLMo-2 7B Instruct | Post-trained | 7 |  |
| `llama31-8b-instruct` | Llama-3.1 8B Instruct | Post-trained | 8 |  |
| `falcon-h1-0.5b` | Falcon-H1 0.5B | Hybrid & SSM bases | 0.5 |  |
| `falcon-h1-1.5b` | Falcon-H1 1.5B (24L) | Hybrid & SSM bases | 1.5 |  |
| `falcon-h1-1.5b-deep` | Falcon-H1 1.5B-Deep (66L) | Hybrid & SSM bases | 1.5 |  |
| `falcon-h1-3b` | Falcon-H1 3B | Hybrid & SSM bases | 3 |  |
| `falcon-h1-7b` | Falcon-H1 7B | Hybrid & SSM bases | 7 |  |
| `falcon-mamba-7b` | Falcon-Mamba 7B (pure SSM) | Hybrid & SSM bases | 7 |  |
| `bamba-9b` | Bamba 9B (interleaved) | Hybrid & SSM bases | 9 |  |
| `recurrentgemma-9b` | RecurrentGemma 9B | Hybrid & SSM bases | 9 |  |
| `falcon-h1-34b-base` | Falcon-H1 34B | Hybrid & SSM bases | 34 |  |
| `nemotron3-ultra-550b-base` | Nemotron-3 Ultra 550B Base (hybrid) | Hybrid & SSM bases | 550 | Mamba–attention hybrid; scored with vLLM (parity-checked against HF) |

## 0-shot tables by family

Accuracy × 100, 0-shot; deixis columns are the **shortcut-free** subset. `—` = not measured; `†` = fewer than 20 items in that cell (all human cells; many API cells). Rows are grouped (bold separator rows); API rows are generation-scored, open rows logprob-scored. Full numbers (n, chance, se, every condition and mode) are in the CSV below.

### Core binding

| model | Static lookup | Code variables | 3rd-person updates |
|---|---|---|---|
| **Human** | | | |
| Human reader (gauntlet) | 100† | 80† | 100† |
| **API · closed** | | | |
| Amazon Nova Pro · reflex | 100 | 80 | 93 |
| Claude Fable 5.1 · reflex | 100 | 53 | 100 |
| Claude Haiku 4.5 · reflex | 100 | 90 | 87 |
| Claude Haiku 5.5 · reflex | 100 | 70 | 93 |
| Claude Opus 4.5 · reflex | 100 | 100 | 100 |
| Claude Opus 5.5 · reflex | 100 | 70 | 100 |
| Claude Sonnet 4.5 · reflex | 100 | 100 | 100 |
| Claude Sonnet 5.5 · reflex | 100 | 63 | 93 |
| Gemini 2.5 Flash · reflex | 100 | 93 | 93 |
| Gemini 2.5 Flash-Lite · reflex | 90 | 70 | 87 |
| GPT-4.1 mini · reflex | 83 | 90 | 97 |
| GPT-4.1 nano · reflex | 57 | 73 | 77 |
| GPT-4.1 · reflex | 93 | 93 | 100 |
| GPT-4o mini · reflex | 63 | 83 | 93 |
| GPT-5 mini · reflex | 70 | 80 | 83 |
| GPT-5 nano · reflex | 17 | 47 | 67 |
| GPT-5.5 · reflex | 100 | 97 | 100 |
| Grok 4.3 · reflex | 100 | 77 | 97 |
| **API · open weights** | | | |
| DeepSeek V3.2 (671B MoE) · reflex | 67 | 91 | 100 |
| Gemma-3 12B IT · reflex | 67 | 77 | 80 |
| Gemma-3 27B IT · reflex | 57 | 70 | 93 |
| Gemma-3 4B IT · reflex | 87 | 53 | 33 |
| GLM-4.6 (355B MoE) · reflex | 97 | 90 | 100 |
| gpt-oss 120B · reflex | 25† | 0† | 44† |
| Hermes-3 405B · reflex | 93 | 70 | 100 |
| Kimi K2 (1T MoE) · reflex | 100 | 93 | 100 |
| Llama-3.1 8B Instruct · reflex | 73 | 37 | 90 |
| Llama-3.2 1B Instruct · reflex | 77 | 13 | 50 |
| Llama-3.2 3B Instruct · reflex | 50 | 47 | 80 |
| Llama-3.3 70B Instruct · reflex | 93 | 73 | 100 |
| Llama-4 Maverick (400B MoE) · reflex | 100 | 100 | 100 |
| Ministral 3B · reflex | 80 | 67 | 70 |
| Ministral 8B · reflex | 60 | 77 | 73 |
| Mistral Large (2512) · reflex | 81† | 92† | 100† |
| Mistral Small 3.2 24B · reflex | 87 | 93 | 93 |
| Phi-4 14B · reflex | 100 | 73 | 90 |
| Qwen2.5 72B Instruct · reflex | 100 | 83 | 97 |
| Qwen2.5 7B Instruct · reflex | 93 | 67 | 90 |
| Qwen3 235B-A22B · reflex | 80 | 73 | 87 |
| Qwen3 32B · reflex | 95† | 73 | 100† |
| Qwen3.5 397B-A17B · reflex | 100 | 100 | 100 |
| **Pretrained bases** | | | |
| SmolLM2 135M | 39 | 28 | 43 |
| Pythia 160M | 5 | 10 | 10 |
| SmolLM2 360M | 45 | 31 | 58 |
| Pythia 410M | 53 | 25 | 39 |
| Qwen2.5 0.5B | 39 | 33 | 45 |
| Qwen3 0.6B | 63 | 40 | 55 |
| Llama-3.2 1B | 57 | 31 | 53 |
| OLMo-1 1B (0724) | 43 | 32 | 49 |
| Pythia 1B | 46 | 27 | 52 |
| kotodama-tsumugi 1.4B (release) | 55 | 32 | 60 |
| Pythia 1.4B | 52 | 24 | 48 |
| OLMo-2 1B | 33 | 34 | 60 |
| Qwen2.5 1.5B | 71 | 39 | 64 |
| Qwen3 1.7B | 72 | 46 | 71 |
| SmolLM2 1.7B | 41 | 36 | 60 |
| Pythia 2.8B | 61 | 33 | 57 |
| Falcon3 3B | 55 | 43 | 72 |
| Qwen2.5 3B | 70 | 43 | 67 |
| Qwen3 4B | 97 | 60 | 87 |
| Pythia 6.9B | 53 | 37 | 57 |
| Falcon3 7B | 75 | 62 | 86 |
| OLMo-2 7B | 46 | 37 | 77 |
| Qwen2.5 7B | 79 | 61 | 82 |
| Llama-3.1 8B | 66 | 56 | 82 |
| Gemma-2 9B | 76 | 61 | 84 |
| Qwen3 30B-A3B | 84 | 75 | 81 |
| Gemma-4 31B | 70 | 87 | 93 |
| GLM-4 32B | 77 | 79 | 91 |
| OLMo-3 32B | 78 | 75 | 89 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 70 | 53 | 85 |
| Llama-3.1 70B | 53 | 79 | 91 |
| Qwen2.5 72B | 95 | 74 | 94 |
| DeepSeek-V4 Flash Base | 75 | 90 | 92 |
| MiMo-V2.5 Base | 72 | 79 | 94 |
| Trinity-Large Base | 74 | 83 | 90 |
| Trinity-Large TrueBase | 79 | 87 | 88 |
| Llama-3.1 405B | 79 | 84 | 95 |
| DeepSeek-V3 Base (671B MoE) | 68 | 88 | 96 |
| DeepSeek-V4 Pro Base | 68 | 87 | 96 |
| Kimi-K2 Base (1T MoE) | 97 | 89 | 98 |
| MiMo-V2.5-Pro Base | 69 | 77 | 92 |
| **Post-trained** | | | |
| OLMo-2 7B Instruct | 65 | 42 | 79 |
| **Hybrid & SSM bases** | | | |
| Falcon-H1 1.5B (24L) | 84 | 44 | 50 |
| Falcon-H1 1.5B-Deep (66L) | 79 | 51 | 71 |
| Falcon-H1 3B | 94 | 55 | 73 |
| Falcon-H1 7B | 87 | 54 | 90 |
| Falcon-Mamba 7B (pure SSM) | 40 | 24 | 62 |
| Bamba 9B (interleaved) | 30 | 31 | 54 |
| Nemotron-3 Ultra 550B Base (hybrid) | 88 | 84 | 96 |
| *chance* | 17 | 27 | 31 |

### Deixis

Key: owner = 'Now the X is with' (P3); self = 'I'm holding the'; you = 'you're holding the'; 2/3 = people in the chat (you3 has a vocative, you2 is structural); I = the I/you/my version, 3rd = its 3rd-person twin.

| model | owner I | owner 3rd | self3 I | self3 3rd | self2 I | self2 3rd | you3 I | you3 3rd | you2 I | you2 3rd |
|---|---|---|---|---|---|---|---|---|---|---|
| **Human** | | | | | | | | | | |
| Human reader (gauntlet) | 100† | — | — | — | 100† | — | 100† | — | 100† | 100† |
| **API · closed** | | | | | | | | | | |
| Amazon Nova Pro · reflex | 75† | 100† | 74† | 95 | 48 | 89† | 50 | 84† | 44† | 87 |
| Claude Fable 5.1 · reflex | 100† | 100† | 100† | 100 | 90 | 100† | 95 | 100† | 100† | 100 |
| Claude Haiku 4.5 · deliberate | 100† | 100† | 100† | 100 | 96 | 100† | 100 | 100† | 100† | 100 |
| Claude Haiku 4.5 · reflex | 88† | 100† | 53† | 91 | 57 | 89† | 64 | 74† | 67† | 78 |
| Claude Haiku 5.5 · deliberate | 100† | 100† | 68† | 100 | 91 | 100† | 82 | 95† | 78† | 100 |
| Claude Haiku 5.5 · reflex | 88† | 100† | 58† | 100 | 74 | 100† | 73 | 95† | 61† | 100 |
| Claude Opus 4.5 · reflex | 100† | 100† | 100† | 100 | 87 | 100† | 91 | 100† | 56† | 100 |
| Claude Opus 5.5 · deliberate | 100† | 100† | 100† | 100 | 100 | 100† | 100 | 100† | 100† | 100 |
| Claude Opus 5.5 · reflex | — | 100† | 88† | 95 | — | 100† | — | 100† | — | 100† |
| Claude Sonnet 4.5 · reflex | 75† | 100† | 95† | 100 | 39 | 100† | 91 | 95† | 72† | 96 |
| Claude Sonnet 5.5 · deliberate | 100† | 100† | 100† | 100 | 100 | 100† | 100 | 100† | 100† | 100 |
| Claude Sonnet 5.5 · reflex | 100† | 100† | 58† | 95 | 57 | 95† | 82 | 95† | 61† | 91 |
| Gemini 2.5 Flash · deliberate | 100† | 100† | 100† | 100 | 91 | 100† | 95 | 100† | 83† | 100 |
| Gemini 2.5 Flash · reflex | 62† | 100† | 58† | 100 | 57 | 89† | 45 | 84† | 56† | 87 |
| Gemini 2.5 Flash-Lite · reflex | 100† | 100† | 26† | 95 | 52 | 79† | 14 | 84† | 39† | 70 |
| Gemini 3.1 Pro (preview) · deliberate | 100† | 100† | 100† | 100 | 100 | 100† | 100 | 100† | 100† | 100 |
| GPT-4.1 mini · reflex | 62† | 100† | 47† | 100 | 43 | 100† | 50 | 95† | 72† | 96 |
| GPT-4.1 nano · reflex | 75† | 100† | 37† | 91 | 30 | 84† | 55 | 79† | 22† | 78 |
| GPT-4.1 · reflex | 100† | 100† | 84† | 100 | 57 | 100† | 91 | 95† | 67† | 96 |
| GPT-4o mini · reflex | 62† | 75† | 37† | 91 | 52 | 84† | 64 | 84† | 61† | 78 |
| GPT-5 mini · deliberate | 100† | 100† | 100† | 100 | 87 | 100† | 100 | 100† | 94† | 100 |
| GPT-5 mini · reflex | 75† | 100† | 21† | 95 | 35 | 79† | 41 | 79† | 33† | 83 |
| GPT-5 nano · deliberate | 100† | 100† | 95† | 100 | 65 | 100† | 77 | 100† | 72† | 100 |
| GPT-5 nano · reflex | 38† | 88† | 11† | 64 | 35 | 63† | 18 | 53† | 6† | 57 |
| GPT-5.5 · deliberate | 100† | 100† | 100† | 100 | 96 | 100† | 100 | 100† | 100† | 100 |
| GPT-5.5 · reflex | 100† | 100† | 100† | 100 | 78 | 100† | 82 | 95† | 33† | 100 |
| Grok 4.3 · deliberate | 75† | 88† | 89† | 100 | 91 | 95† | 91 | 100† | 83† | 100 |
| Grok 4.3 · reflex | 62† | 88† | 21† | 95 | 52 | 100† | 45 | 89† | 33† | 87 |
| **API · open weights** | | | | | | | | | | |
| DeepSeek V3.2 (671B MoE) · deliberate | 75† | 100† | 100† | 100 | 87 | 100† | 77 | 100† | 56† | 100 |
| DeepSeek V3.2 (671B MoE) · reflex | — | — | 71† | 100† | 47† | 100† | 67† | 100† | 18† | 93† |
| Gemma-3 12B IT · reflex | 38† | 100† | 21† | 91 | 35 | 79† | 64 | 79† | 44† | 65 |
| Gemma-3 27B IT · reflex | 62† | 88† | 47† | 86 | 35 | 89† | 55 | 79† | 39† | 78 |
| Gemma-3 4B IT · reflex | 88† | 75† | 37† | 68 | 43 | 68† | 41 | 53† | 39† | 70 |
| GLM-4.6 (355B MoE) · deliberate | 88† | 100† | 95† | 100 | 96 | 100† | 95 | 100† | 89† | 100 |
| GLM-4.6 (355B MoE) · reflex | 100† | 100† | 74† | 100 | 52 | 89† | 55 | 95† | 61† | 91 |
| gpt-oss 120B · deliberate | 88† | 100† | 95† | 100 | 87 | 100† | 86 | 100† | 39† | 100 |
| gpt-oss 120B · reflex | — | — | — | 38† | 0† | 55† | 18† | 60† | — | 27† |
| Hermes-3 405B · reflex | 75† | 62† | 74† | 100 | 52 | 100† | 82 | 95† | 33† | 96 |
| Kimi K2 (1T MoE) · reflex | 100† | 100† | 58† | 100 | 50 | 95† | 73 | 100† | 59† | 100 |
| Llama-3.1 8B Instruct · reflex | 75† | 88† | 53† | 73 | 43 | 74† | 32 | 68† | 33† | 70 |
| Llama-3.2 1B Instruct · reflex | 12† | 38† | 21† | 55 | 17 | 58† | 14 | 37† | 17† | 43 |
| Llama-3.2 3B Instruct · reflex | 75† | 38† | 53† | 64 | 35 | 63† | 41 | 58† | 39† | 52 |
| Llama-3.3 70B Instruct · reflex | 62† | 75† | 79† | 100 | 52 | 95† | 64 | 95† | 44† | 96 |
| Llama-4 Maverick (400B MoE) · reflex | 88† | 100† | 89† | 100 | 65 | 95† | 73 | 100† | 83† | 100 |
| Ministral 3B · reflex | 75† | 88† | 32† | 82 | 43 | 84† | 41 | 58† | 50† | 74 |
| Ministral 8B · reflex | 88† | 75† | 26† | 59 | 39 | 74† | 45 | 53† | 39† | 52 |
| Mistral Large (2512) · reflex | — | — | 50† | 100† | 36† | — | 64† | 100† | 73† | 91† |
| Mistral Small 3.2 24B · reflex | 75† | 62† | 42† | 91 | 39 | 79† | 55 | 74† | 39† | 91 |
| Phi-4 14B · reflex | 62† | 100† | 63† | 100 | 52 | 74† | 59 | 84† | 39† | 74 |
| Qwen2.5 72B Instruct · reflex | 62† | 100† | 68† | 100 | 57 | 100† | 77 | 89† | 61† | 96 |
| Qwen2.5 7B Instruct · reflex | 50† | 100† | 47† | 82 | 61 | 84† | 64 | 84† | 61† | 74 |
| Qwen3 235B-A22B · reflex | 62† | 75† | 26† | 91 | 22 | 79† | 32 | 89† | 28† | 91 |
| Qwen3 32B · reflex | — | — | 36† | 82† | 45† | 83† | 40† | 86† | 21† | 86† |
| Qwen3.5 397B-A17B · deliberate | 100† | 88† | 89† | 100 | 87 | 100† | 91 | 95† | 83† | 100 |
| Qwen3.5 397B-A17B · reflex | 100† | 100† | 89† | 100 | 70 | 95† | 67 | 100† | 61† | 100 |
| **Pretrained bases** | | | | | | | | | | |
| SmolLM2 135M | 42 | 42 | 25 | 21 | 21 | 25 | 27 | 29 | 17 | 14 |
| Pythia 160M | 2 | 18 | 3 | 9 | 4 | 6 | 7 | 6 | 6 | 9 |
| SmolLM2 360M | 25 | 60 | 30 | 21 | 21 | 25 | 20 | 22 | 15 | 25 |
| Pythia 410M | 40 | 45 | 32 | 37 | 19 | 28 | 31 | 32 | 15 | 30 |
| Qwen2.5 0.5B | 48 | 50 | 37 | 10 | 32 | 21 | 35 | 16 | 21 | 16 |
| Qwen3 0.6B | 40 | 60 | 40 | 31 | 44 | 44 | 28 | 37 | 30 | 53 |
| Llama-3.2 1B | 30 | 22 | 37 | 25 | 27 | 34 | 30 | 26 | 19 | 36 |
| OLMo-1 1B (0724) | 22 | 50 | 35 | 34 | 40 | 36 | 27 | 30 | 19 | 36 |
| Pythia 1B | 50 | 38 | 39 | 37 | 33 | 35 | 31 | 33 | 15 | 33 |
| kotodama-tsumugi 1.4B (release) | 62 | 62 | 31 | 37 | 28 | 40 | 27 | 40 | 21 | 42 |
| Pythia 1.4B | 42 | 68 | 46 | 50 | 42 | 49 | 41 | 51 | 25 | 48 |
| OLMo-2 1B | 48 | 57 | 39 | 25 | 37 | 43 | 25 | 33 | 16 | 44 |
| Qwen2.5 1.5B | 32 | 55 | 45 | 30 | 48 | 49 | 33 | 33 | 32 | 56 |
| Qwen3 1.7B | 48 | 60 | 50 | 44 | 52 | 67 | 32 | 48 | 25 | 67 |
| SmolLM2 1.7B | 32 | 45 | 42 | 35 | 44 | 50 | 34 | 32 | 23 | 55 |
| Pythia 2.8B | 48 | 72 | 27 | 34 | 25 | 38 | 33 | 44 | 16 | 37 |
| Falcon3 3B | 35 | 52 | 32 | 39 | 22 | 56 | 29 | 46 | 21 | 48 |
| Qwen2.5 3B | 42 | 52 | 39 | 64 | 40 | 67 | 29 | 65 | 26 | 76 |
| Qwen3 4B | 78 | 95 | 47 | 79 | 57 | 79 | 48 | 79 | 41 | 79 |
| Pythia 6.9B | 52 | 78 | 33 | 31 | 35 | 40 | 31 | 40 | 18 | 44 |
| Falcon3 7B | 35 | 68 | 48 | 71 | 33 | 73 | 32 | 69 | 19 | 68 |
| OLMo-2 7B | 40 | 48 | 37 | 36 | 31 | 48 | 34 | 39 | 22 | 44 |
| Qwen2.5 7B | 57 | 85 | 56 | 53 | 56 | 68 | 52 | 56 | 44 | 71 |
| Llama-3.1 8B | 57 | 65 | 47 | 71 | 38 | 79 | 39 | 69 | 16 | 75 |
| Gemma-2 9B | 60 | 68 | 41 | 71 | 41 | 71 | 28 | 64 | 21 | 75 |
| OLMo-2 13B | 65 | 65 | 55 | 67 | 48 | 60 | 48 | 63 | 38 | 58 |
| Qwen3 30B-A3B | 60 | 68 | 42 | 71 | 41 | 78 | 33 | 70 | 29 | 81 |
| Gemma-4 31B | 62 | 80 | 74 | 94 | 60 | 91 | 43 | 93 | 33 | 91 |
| GLM-4 32B | 65 | 82 | 45 | 86 | 47 | 87 | 30 | 84 | 15 | 85 |
| OLMo-3 32B | 65 | 82 | 57 | 84 | 57 | 84 | 51 | 82 | 29 | 80 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 40 | 60 | 44 | 75 | 45 | 78 | 35 | 74 | 23 | 76 |
| Llama-3.1 70B | 55 | 50 | 58 | 80 | 37 | 83 | 50 | 73 | 25 | 83 |
| Qwen2.5 72B | 60 | 72 | 60 | 92 | 37 | 87 | 52 | 92 | 27 | 88 |
| DeepSeek-V4 Flash Base | 72 | 78 | 60 | 86 | 52 | 89 | 51 | 90 | 29 | 90 |
| MiMo-V2.5 Base | 48 | 57 | 44 | 79 | 48 | 96 | 40 | 83 | 25 | 91 |
| Trinity-Large Base | 40 | 82 | 35 | 87 | 44 | 83 | 38 | 91 | 17 | 90 |
| Trinity-Large TrueBase | 65 | 75 | 44 | 76 | 43 | 72 | 42 | 78 | 21 | 72 |
| Llama-3.1 405B | 72 | 85 | 55 | 99 | 40 | 99 | 38 | 98 | 21 | 99 |
| DeepSeek-V3 Base (671B MoE) | 65 | 75 | 40 | 96 | 27 | 97 | 48 | 94 | 21 | 95 |
| DeepSeek-V4 Pro Base | 50 | 65 | 40 | 89 | 48 | 94 | 29 | 90 | 31 | 96 |
| Kimi-K2 Base (1T MoE) | 68 | 100 | 39 | 95 | 33 | 94 | 39 | 97 | 25 | 93 |
| MiMo-V2.5-Pro Base | 45 | 68 | 25 | 88 | 21 | 88 | 26 | 88 | 17 | 90 |
| **OLMo-2 7B trajectory** | | | | | | | | | | |
| OLMo-2 7B @ 2.9T (stage 1) | 28 | 30 | 37 | 45 | 31 | 52 | 31 | 44 | 27 | 47 |
| OLMo-2 7B @ 3.9T (end stage 1) | 45 | 55 | 31 | 40 | 40 | 53 | 32 | 48 | 17 | 50 |
| OLMo-2 7B mid-train +5B | 32 | 45 | 40 | 39 | 31 | 39 | 36 | 35 | 28 | 46 |
| OLMo-2 7B mid-train +17B | 32 | 52 | 35 | 35 | 29 | 39 | 31 | 40 | 19 | 42 |
| OLMo-2 7B mid-train +50B | 35 | 60 | 40 | 37 | 29 | 47 | 33 | 37 | 19 | 44 |
| **Post-trained** | | | | | | | | | | |
| OLMo-2 7B SFT | 42 | 65 | 43 | 43 | 40 | 49 | 37 | 32 | 22 | 40 |
| OLMo-2 7B DPO | 62 | 65 | 49 | 48 | 46 | 54 | 37 | 41 | 28 | 43 |
| OLMo-2 7B Instruct | 60 | 68 | 50 | 49 | 46 | 56 | 37 | 42 | 29 | 42 |
| Llama-3.1 8B Instruct | 52 | 72 | 56 | 83 | 45 | 84 | 44 | 78 | 18 | 81 |
| **Hybrid & SSM bases** | | | | | | | | | | |
| Falcon-H1 0.5B | — | — | 29† | 24† | 27 | 19 | 54 | 35 | 26 | 45 |
| Falcon-H1 1.5B (24L) | — | — | 47 | 42 | 50 | 56 | 42 | 45 | 25 | 53 |
| Falcon-H1 1.5B-Deep (66L) | — | — | 35 | 58 | 42 | 83 | 32 | 64 | 12 | 78 |
| Falcon-H1 3B | — | — | 26 | 60 | 35 | 81 | 34 | 65 | 25 | 75 |
| Falcon-H1 7B | 64 | 91 | 41 | 86 | 40 | 78 | 38 | 84 | 29 | 82 |
| Falcon-Mamba 7B (pure SSM) | 5 | 38 | 19 | 17 | 19 | 32 | 19 | 21 | 17 | 37 |
| Bamba 9B (interleaved) | 25 | 25 | 21 | 11 | 13 | 19 | 23 | 15 | 17 | 23 |
| RecurrentGemma 9B | 42 | 45 | 10 | 10 | 15 | 10 | 11 | 14 | 5 | 11 |
| Falcon-H1 34B | 75 | 65 | 61 | 94 | 52 | 83 | 55 | 91 | 37 | 91 |
| Nemotron-3 Ultra 550B Base (hybrid) | 57 | 88 | 61 | 92 | 44 | 85 | 43 | 91 | 19 | 93 |
| *chance* | 33 | 33 | 29 | 29 | 28 | 29 | 28 | 29 | 29 | 28 |

### Set binding

| model | 1 object | 2-3 objects, exact set | split across sentences | after transfers |
|---|---|---|---|---|
| **Human** | | | | |
| Human reader (gauntlet) | 100† | 100† | 100† | 100† |
| **API · closed** | | | | |
| Amazon Nova Pro · reflex | 100 | — | — | — |
| Claude Fable 5.1 · reflex | 100 | — | — | — |
| Claude Haiku 4.5 · reflex | 100 | — | — | — |
| Claude Haiku 5.5 · reflex | 100 | — | — | — |
| Claude Opus 4.5 · reflex | 100 | — | — | — |
| Claude Opus 5.5 · reflex | 100 | — | — | — |
| Claude Sonnet 4.5 · reflex | 100 | — | — | — |
| Claude Sonnet 5.5 · reflex | 100 | — | — | — |
| Gemini 2.5 Flash · reflex | 100 | — | — | — |
| Gemini 2.5 Flash-Lite · reflex | 100 | — | — | — |
| GPT-4.1 mini · reflex | 96 | — | — | — |
| GPT-4.1 nano · reflex | 92 | — | — | — |
| GPT-4.1 · reflex | 88 | — | — | — |
| GPT-4o mini · reflex | 79 | — | — | — |
| GPT-5 mini · reflex | 96 | — | — | — |
| GPT-5 nano · reflex | 21 | — | — | — |
| GPT-5.5 · reflex | 100 | — | — | — |
| Grok 4.3 · reflex | 100 | — | — | — |
| **API · open weights** | | | | |
| DeepSeek V3.2 (671B MoE) · reflex | 82† | — | — | — |
| Gemma-3 12B IT · reflex | 96 | — | — | — |
| Gemma-3 27B IT · reflex | 58 | — | — | — |
| Gemma-3 4B IT · reflex | 88 | — | — | — |
| GLM-4.6 (355B MoE) · reflex | 100 | — | — | — |
| gpt-oss 120B · reflex | 33† | — | — | — |
| Hermes-3 405B · reflex | 100 | — | — | — |
| Kimi K2 (1T MoE) · reflex | 100 | — | — | — |
| Llama-3.1 8B Instruct · reflex | 92 | — | — | — |
| Llama-3.2 1B Instruct · reflex | 54 | — | — | — |
| Llama-3.2 3B Instruct · reflex | 58 | — | — | — |
| Llama-3.3 70B Instruct · reflex | 100 | — | — | — |
| Llama-4 Maverick (400B MoE) · reflex | 100 | — | — | — |
| Ministral 3B · reflex | 88 | — | — | — |
| Ministral 8B · reflex | 88 | — | — | — |
| Mistral Large (2512) · reflex | 82† | — | — | — |
| Mistral Small 3.2 24B · reflex | 100 | — | — | — |
| Phi-4 14B · reflex | 100 | — | — | — |
| Qwen2.5 72B Instruct · reflex | 100 | — | — | — |
| Qwen2.5 7B Instruct · reflex | 100 | — | — | — |
| Qwen3 235B-A22B · reflex | 75 | — | — | — |
| Qwen3 32B · reflex | 100† | — | — | — |
| Qwen3.5 397B-A17B · reflex | 100 | — | — | — |
| **Pretrained bases** | | | | |
| SmolLM2 135M | 33 | 8 | 7 | 5 |
| Pythia 160M | 22 | 9 | 5 | 7 |
| SmolLM2 360M | 39 | 16 | 10 | 9 |
| Pythia 410M | 58 | 10 | 12 | 11 |
| Qwen2.5 0.5B | 47 | 12 | 16 | 14 |
| Qwen3 0.6B | 58 | 41 | 27 | 12 |
| Llama-3.2 1B | 56 | 16 | 28 | 12 |
| OLMo-1 1B (0724) | 39 | 16 | 12 | 6 |
| Pythia 1B | 28 | 20 | 16 | 12 |
| kotodama-tsumugi 1.4B (release) | 50 | 9 | 16 | 12 |
| Pythia 1.4B | 42 | 12 | 18 | 13 |
| OLMo-2 1B | 33 | 24 | 5 | 11 |
| Qwen2.5 1.5B | 78 | 35 | 29 | 21 |
| Qwen3 1.7B | 77 | 38 | 41 | 34 |
| SmolLM2 1.7B | 42 | 20 | 21 | 10 |
| Pythia 2.8B | 64 | 35 | 27 | 14 |
| Qwen2.5 3B | 81 | 61 | 27 | 38 |
| Qwen3 4B | 100 | 74 | 69 | 55 |
| Pythia 6.9B | 58 | 24 | 10 | 11 |
| Falcon3 7B | 70 | 41 | 49 | 41 |
| OLMo-2 7B | 52 | 27 | 10 | 10 |
| Qwen2.5 7B | 100 | 75 | 64 | 51 |
| Llama-3.1 8B | 75 | 55 | 44 | 44 |
| Gemma-2 9B | 95 | 58 | 52 | 42 |
| Qwen3 30B-A3B | 89 | 59 | 55 | 49 |
| Gemma-4 31B | 91 | 44 | 72 | 66 |
| GLM-4 32B | 86 | 71 | 72 | 71 |
| OLMo-3 32B | 100 | 76 | 58 | 62 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 86 | 45 | 40 | 51 |
| Llama-3.1 70B | 31 | 18 | 64 | 59 |
| Qwen2.5 72B | 100 | 77 | 88 | 70 |
| DeepSeek-V4 Flash Base | 88 | 58 | 82 | 81 |
| MiMo-V2.5 Base | 78 | 41 | 44 | 40 |
| Trinity-Large Base | 92 | 77 | 77 | 54 |
| Trinity-Large TrueBase | 94 | 65 | 59 | 48 |
| Llama-3.1 405B | 88 | 76 | 81 | 84 |
| DeepSeek-V3 Base (671B MoE) | 88 | 70 | 68 | 87 |
| DeepSeek-V4 Pro Base | 94 | 71 | 90 | 88 |
| Kimi-K2 Base (1T MoE) | 100 | 93 | 83 | 74 |
| MiMo-V2.5-Pro Base | 97 | 54 | 51 | 55 |
| **Post-trained** | | | | |
| OLMo-2 7B SFT | 73 | 47 | 21 | 18 |
| OLMo-2 7B DPO | 77 | 57 | 27 | 22 |
| OLMo-2 7B Instruct | 78 | 59 | 30 | 23 |
| Llama-3.1 8B Instruct | 97 | 72 | 85 | 63 |
| **Hybrid & SSM bases** | | | | |
| Falcon-H1 1.5B (24L) | 88† | 68 | 34 | 13 |
| Falcon-H1 1.5B-Deep (66L) | 94† | 68 | 66 | 34 |
| Falcon-H1 7B | 98 | 70 | 73 | 61 |
| Falcon-H1 34B | 98 | 84 | 81 | 70 |
| Nemotron-3 Ultra 550B Base (hybrid) | 100 | 95 | 97 | 82 |
| *chance* | 12 | 3 | 3 | 3 |

### Answer deixis

| model | narrator query | answer: a name | answer: me | answer: you (named) | answer: you (no name) |
|---|---|---|---|---|---|
| **Human** | | | | | |
| Human reader (gauntlet) | 100† | 100† | 100† | 100† | 100† |
| **API · closed** | | | | | |
| Amazon Nova Pro · reflex | 67 | 71 | 33 | 67 | 50 |
| Claude Fable 5.1 · reflex | 100 | 100 | 100 | 100 | 100 |
| Claude Haiku 4.5 · deliberate | 100 | 100 | 96 | 100 | 100 |
| Claude Haiku 4.5 · reflex | 79 | 88 | 33 | 58 | 33 |
| Claude Haiku 5.5 · deliberate | 96 | 100 | 100 | 100 | 100 |
| Claude Haiku 5.5 · reflex | 83 | 100 | 75 | 71 | 83 |
| Claude Opus 4.5 · reflex | 100 | 100 | 100 | 96 | 96 |
| Claude Opus 5.5 · deliberate | 100 | 100 | 100 | 100 | 100 |
| Claude Opus 5.5 · reflex | 100 | 95 | — | 56† | 93† |
| Claude Sonnet 4.5 · reflex | 88 | 100 | 67 | 100 | 96 |
| Claude Sonnet 5.5 · deliberate | 100 | 100 | 100 | 100 | 100 |
| Claude Sonnet 5.5 · reflex | 100 | 100 | 92 | 100 | 100 |
| Gemini 2.5 Flash · deliberate | 100 | 100 | 100 | 100 | 100 |
| Gemini 2.5 Flash · reflex | 58 | 79 | 21 | 42 | 21 |
| Gemini 2.5 Flash-Lite · reflex | 62 | 83 | 0 | 71 | 8 |
| Gemini 3.1 Pro (preview) · deliberate | 100 | 100 | 100 | 100 | 100 |
| GPT-4.1 mini · reflex | 58 | 75 | 50 | 83 | 75 |
| GPT-4.1 nano · reflex | 54 | 75 | 4 | 12 | 12 |
| GPT-4.1 · reflex | 100 | 92 | 67 | 83 | 88 |
| GPT-4o mini · reflex | 33 | 71 | 8 | 50 | 17 |
| GPT-5 mini · deliberate | 83 | 100 | 79 | 92 | 96 |
| GPT-5 mini · reflex | 46 | 83 | 25 | 71 | 71 |
| GPT-5 nano · deliberate | 100 | 100 | 96 | 100 | 100 |
| GPT-5 nano · reflex | 25 | 62 | 12 | 29 | 17 |
| GPT-5.5 · deliberate | 100 | 100 | 100 | 100 | 100 |
| GPT-5.5 · reflex | 100 | 100 | 100 | 96 | 100 |
| Grok 4.3 · deliberate | 100 | 88 | 92 | 83 | 88 |
| Grok 4.3 · reflex | 58 | 75 | 0 | 67 | 38 |
| **API · open weights** | | | | | |
| DeepSeek V3.2 (671B MoE) · deliberate | 96 | 96 | 96 | 75 | 75 |
| DeepSeek V3.2 (671B MoE) · reflex | 52 | 93† | 6† | 59† | 33† |
| Gemma-3 12B IT · reflex | 33 | 50 | 0 | 25 | 0 |
| Gemma-3 27B IT · reflex | 50 | 79 | 4 | 42 | 33 |
| Gemma-3 4B IT · reflex | 50 | 25 | 8 | 83 | 0 |
| GLM-4.6 (355B MoE) · deliberate | 100 | 96 | 100 | 92 | 100 |
| GLM-4.6 (355B MoE) · reflex | 79 | 88 | 54 | 100 | 54 |
| gpt-oss 120B · deliberate | 83 | 88 | 54 | 88 | 83 |
| gpt-oss 120B · reflex | 17† | 0† | 0† | 38† | 12† |
| Hermes-3 405B · reflex | 75 | 92 | 46 | 100 | 38 |
| Kimi K2 (1T MoE) · reflex | 88 | 96 | 25 | 96 | 79 |
| Llama-3.1 8B Instruct · reflex | 46 | 67 | 0 | 17 | 12 |
| Llama-3.2 1B Instruct · reflex | 38 | 29 | 8 | 25 | 25 |
| Llama-3.2 3B Instruct · reflex | 46 | 46 | 50 | 0 | 12 |
| Llama-3.3 70B Instruct · reflex | 88 | 92 | 25 | 54 | 42 |
| Llama-4 Maverick (400B MoE) · reflex | 83 | 100 | 54 | 79 | 83 |
| Ministral 3B · reflex | 50 | 42 | 4 | 4 | 12 |
| Ministral 8B · reflex | 50 | 62 | 4 | 46 | 0 |
| Mistral Small 3.2 24B · reflex | 54 | 79 | 38 | 75 | 8 |
| Phi-4 14B · reflex | 71 | 71 | 29 | 71 | 38 |
| Qwen2.5 72B Instruct · reflex | 71 | 92 | 25 | 75 | 42 |
| Qwen2.5 7B Instruct · reflex | 50 | 58 | 12 | 92 | 33 |
| Qwen3 235B-A22B · reflex | 46 | 71 | 12 | 83 | 33 |
| Qwen3 32B · reflex | 87† | 79† | 12† | 91† | 69† |
| Qwen3.5 397B-A17B · deliberate | 100 | 92 | 75 | 88 | 88 |
| Qwen3.5 397B-A17B · reflex | 100 | 96 | 50 | 100 | 96 |
| **Pretrained bases** | | | | | |
| SmolLM2 135M | 42 | 37 | 42 | 26 | 28 |
| Pythia 160M | 6 | 21 | 57 | 1 | 18 |
| SmolLM2 360M | 32 | 18 | 44 | 60 | 40 |
| Pythia 410M | 37 | 24 | 22 | 30 | 79 |
| Qwen2.5 0.5B | 34 | 40 | 47 | 11 | 15 |
| Qwen3 0.6B | 39 | 37 | 19 | 35 | 50 |
| Llama-3.2 1B | 40 | 43 | 9 | 34 | 14 |
| OLMo-1 1B (0724) | 31 | 44 | 19 | 20 | 11 |
| Pythia 1B | 36 | 28 | 28 | 51 | 44 |
| kotodama-tsumugi 1.4B (release) | 44 | 20 | 21 | 50 | 82 |
| Pythia 1.4B | 32 | 22 | 7 | 89 | 51 |
| OLMo-2 1B | 48 | 43 | 9 | 28 | 32 |
| Qwen2.5 1.5B | 39 | 30 | 22 | 38 | 41 |
| Qwen3 1.7B | 45 | 52 | 6 | 28 | 35 |
| SmolLM2 1.7B | 28 | 39 | 14 | 42 | 11 |
| Pythia 2.8B | 40 | 14 | 68 | 20 | 12 |
| Qwen2.5 3B | 37 | 40 | 0 | 81 | 40 |
| Qwen3 4B | 65 | 51 | 13 | 41 | 78 |
| Pythia 6.9B | 40 | 65 | 15 | 18 | 5 |
| Falcon3 7B | 50 | 43 | 9 | 40 | 44 |
| OLMo-2 7B | 49 | 7 | 66 | 72 | 30 |
| Qwen2.5 7B | 45 | 14 | 1 | 89 | 99 |
| Llama-3.1 8B | 57 | 57 | 1 | 20 | 79 |
| Gemma-2 9B | 62 | 22 | 36 | 32 | 40 |
| Qwen3 30B-A3B | 49 | 42 | 12 | 94 | 64 |
| Gemma-4 31B | 65 | 71 | 68 | 57 | 51 |
| GLM-4 32B | 65 | 52 | 34 | 54 | 45 |
| OLMo-3 32B | 67 | 6 | 71 | 49 | 8 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 43 | 28 | 50 | 46 | 9 |
| Llama-3.1 70B | 59 | 45 | 65 | 38 | 42 |
| Qwen2.5 72B | 63 | 30 | 72 | 74 | 54 |
| DeepSeek-V4 Flash Base | 69 | 55 | 68 | 26 | 20 |
| MiMo-V2.5 Base | 54 | 30 | 69 | 52 | 35 |
| Trinity-Large Base | 60 | 43 | 38 | 52 | 24 |
| Trinity-Large TrueBase | 61 | 14 | 60 | 20 | 49 |
| Llama-3.1 405B | 74 | 57 | 65 | 14 | 20 |
| DeepSeek-V3 Base (671B MoE) | 70 | 59 | 58 | 60 | 39 |
| DeepSeek-V4 Pro Base | 49 | 18 | 40 | 38 | 84 |
| Kimi-K2 Base (1T MoE) | 73 | 25 | 52 | 84 | 92 |
| MiMo-V2.5-Pro Base | 44 | 18 | 66 | 30 | 45 |
| **Post-trained** | | | | | |
| OLMo-2 7B SFT | 52 | 2 | 47 | 86 | 38 |
| OLMo-2 7B DPO | 53 | 5 | 38 | 91 | 61 |
| OLMo-2 7B Instruct | 55 | 6 | 39 | 92 | 60 |
| Llama-3.1 8B Instruct | 65 | 80 | 9 | 45 | 76 |
| **Hybrid & SSM bases** | | | | | |
| Falcon-H1 1.5B (24L) | 39 | 30 | 75 | 9 | 2 |
| Falcon-H1 1.5B-Deep (66L) | 45 | 43 | 16 | 47 | 55 |
| Falcon-H1 7B | 59 | 49 | 26 | 64 | 42 |
| Falcon-H1 34B | 71 | 36 | 80 | 65 | 19 |
| Nemotron-3 Ultra 550B Base (hybrid) | 56 | 46 | 84 | 90 | 40 |
| *chance* | 25 | 25 | 25 | 25 | 25 |

### Adjacency

| model | 'yours' by reply | 'yours' with the name | 3rd-person twin |
|---|---|---|---|
| **Human** | | | |
| Human reader (gauntlet) | 50† | 100† | 100† |
| **API · closed** | | | |
| Amazon Nova Pro · reflex | 96 | 100 | 100 |
| Claude Fable 5.1 · reflex | 100 | 100 | 100 |
| Claude Haiku 4.5 · deliberate | 100 | 100 | 100 |
| Claude Haiku 4.5 · reflex | 96 | 100 | 100 |
| Claude Haiku 5.5 · deliberate | 96 | 100 | 100 |
| Claude Haiku 5.5 · reflex | 92 | 92 | 100 |
| Claude Opus 4.5 · reflex | 100 | 100 | 100 |
| Claude Opus 5.5 · deliberate | 100 | 100 | 100 |
| Claude Opus 5.5 · reflex | 100 | 100 | 100 |
| Claude Sonnet 4.5 · reflex | 100 | 100 | 100 |
| Claude Sonnet 5.5 · deliberate | 100 | 100 | 100 |
| Claude Sonnet 5.5 · reflex | 100 | 100 | 100 |
| Gemini 2.5 Flash · deliberate | 100 | 100 | 100 |
| Gemini 2.5 Flash · reflex | 100 | 100 | 100 |
| Gemini 2.5 Flash-Lite · reflex | 56 | 84 | 100 |
| Gemini 3.1 Pro (preview) · deliberate | 100 | 100 | 100 |
| GPT-4.1 mini · reflex | 100 | 100 | 100 |
| GPT-4.1 nano · reflex | 56 | 80 | 100 |
| GPT-4.1 · reflex | 100 | 100 | 100 |
| GPT-4o mini · reflex | 72 | 92 | 100 |
| GPT-5 mini · deliberate | 100 | 100 | 100 |
| GPT-5 mini · reflex | 80 | 92 | 100 |
| GPT-5 nano · deliberate | 100 | 100 | 100 |
| GPT-5 nano · reflex | 44 | 36 | 60 |
| GPT-5.5 · deliberate | 100 | 100 | 100 |
| GPT-5.5 · reflex | 100 | 100 | 100 |
| Grok 4.3 · deliberate | 100 | 100 | 100 |
| Grok 4.3 · reflex | 84 | 100 | 100 |
| **API · open weights** | | | |
| DeepSeek V3.2 (671B MoE) · deliberate | 100 | 100 | 100 |
| DeepSeek V3.2 (671B MoE) · reflex | 100† | 100† | 100† |
| Gemma-3 12B IT · reflex | 84 | 88 | 88 |
| Gemma-3 27B IT · reflex | 92 | 100 | 100 |
| Gemma-3 4B IT · reflex | 68 | 80 | 52 |
| GLM-4.6 (355B MoE) · deliberate | 96 | 100 | 100 |
| GLM-4.6 (355B MoE) · reflex | 92 | 100 | 100 |
| gpt-oss 120B · deliberate | 100 | 100 | 100 |
| gpt-oss 120B · reflex | 10† | 11† | 45† |
| Hermes-3 405B · reflex | 100 | 100 | 100 |
| Kimi K2 (1T MoE) · reflex | 100 | 100 | 100 |
| Llama-3.1 8B Instruct · reflex | 84 | 80 | 92 |
| Llama-3.2 1B Instruct · reflex | 36 | 52 | 64 |
| Llama-3.2 3B Instruct · reflex | 96 | 96 | 88 |
| Llama-3.3 70B Instruct · reflex | 100 | 100 | 100 |
| Llama-4 Maverick (400B MoE) · reflex | 100 | 100 | 100 |
| Ministral 3B · reflex | 56 | 68 | 92 |
| Ministral 8B · reflex | 92 | 100 | 100 |
| Mistral Small 3.2 24B · reflex | 100 | 100 | 96 |
| Phi-4 14B · reflex | 100 | 100 | 100 |
| Qwen2.5 72B Instruct · reflex | 96 | 100 | 100 |
| Qwen2.5 7B Instruct · reflex | 72 | 84 | 96 |
| Qwen3 235B-A22B · reflex | 84 | 92 | 88 |
| Qwen3 32B · reflex | 100† | 100† | 100† |
| Qwen3.5 397B-A17B · deliberate | 100 | 96 | 100 |
| Qwen3.5 397B-A17B · reflex | 100 | 100 | 100 |
| **Pretrained bases** | | | |
| SmolLM2 135M | 38 | 55 | 52 |
| Pythia 160M | 11 | 14 | 21 |
| SmolLM2 360M | 40 | 42 | 58 |
| Pythia 410M | 35 | 42 | 49 |
| Qwen2.5 0.5B | 37 | 42 | 52 |
| Qwen3 0.6B | 44 | 57 | 76 |
| Llama-3.2 1B | 55 | 75 | 55 |
| OLMo-1 1B (0724) | 43 | 55 | 55 |
| Pythia 1B | 33 | 41 | 52 |
| kotodama-tsumugi 1.4B (release) | 59 | 83 | 72 |
| Pythia 1.4B | 40 | 40 | 63 |
| OLMo-2 1B | 64 | 78 | 63 |
| Qwen2.5 1.5B | 55 | 65 | 67 |
| Qwen3 1.7B | 76 | 93 | 55 |
| SmolLM2 1.7B | 53 | 54 | 41 |
| Pythia 2.8B | 53 | 68 | 82 |
| Qwen2.5 3B | 59 | 68 | 59 |
| Qwen3 4B | 89 | 95 | 95 |
| Pythia 6.9B | 42 | 56 | 67 |
| Falcon3 7B | 86 | 90 | 99 |
| OLMo-2 7B | 87 | 90 | 73 |
| Qwen2.5 7B | 82 | 90 | 98 |
| Llama-3.1 8B | 87 | 89 | 85 |
| Gemma-2 9B | 95 | 97 | 98 |
| Qwen3 30B-A3B | 85 | 95 | 97 |
| Gemma-4 31B | 100 | 100 | 99 |
| GLM-4 32B | 99 | 100 | 99 |
| OLMo-3 32B | 98 | 100 | 100 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 90 | 95 | 98 |
| Llama-3.1 70B | 99 | 99 | 99 |
| Qwen2.5 72B | 99 | 100 | 100 |
| DeepSeek-V4 Flash Base | 98 | 99 | 100 |
| MiMo-V2.5 Base | 92 | 97 | 100 |
| Trinity-Large Base | 99 | 99 | 100 |
| Trinity-Large TrueBase | 93 | 95 | 98 |
| Llama-3.1 405B | 100 | 100 | 100 |
| DeepSeek-V3 Base (671B MoE) | 96 | 97 | 99 |
| DeepSeek-V4 Pro Base | 98 | 100 | 100 |
| Kimi-K2 Base (1T MoE) | 94 | 98 | 100 |
| MiMo-V2.5-Pro Base | 88 | 97 | 99 |
| **Post-trained** | | | |
| OLMo-2 7B SFT | 87 | 88 | 83 |
| OLMo-2 7B DPO | 88 | 92 | 91 |
| OLMo-2 7B Instruct | 89 | 93 | 92 |
| Llama-3.1 8B Instruct | 97 | 99 | 97 |
| **Hybrid & SSM bases** | | | |
| Falcon-H1 1.5B (24L) | 66 | 73 | 69 |
| Falcon-H1 1.5B-Deep (66L) | 76 | 77 | 81 |
| Falcon-H1 7B | 95 | 97 | 100 |
| Falcon-H1 34B | 99 | 100 | 100 |
| Nemotron-3 Ultra 550B Base (hybrid) | 99 | 99 | 100 |
| *chance* | 33 | 33 | 33 |

### Aliased lists

| model | mutate via alias | rebind after alias | copy then mutate |
|---|---|---|---|
| **Human** | | | |
| Human reader (gauntlet) | 50† | 100† | 100† |
| **API · closed** | | | |
| Amazon Nova Pro · reflex | 100 | 70 | 45 |
| Claude Fable 5.1 · reflex | 85 | 80 | 100 |
| Claude Haiku 4.5 · reflex | 95 | 85 | 95 |
| Claude Haiku 5.5 · reflex | 100 | 100 | 100 |
| Claude Opus 4.5 · reflex | 100 | 100 | 90 |
| Claude Opus 5.5 · reflex | 94† | 89† | 89† |
| Claude Sonnet 4.5 · reflex | 100 | 100 | 95 |
| Claude Sonnet 5.5 · reflex | 100 | 95 | 70 |
| Gemini 2.5 Flash · reflex | 100 | 100 | 100 |
| Gemini 2.5 Flash-Lite · reflex | 100 | 60 | 40 |
| GPT-4.1 mini · reflex | 100 | 80 | 60 |
| GPT-4.1 nano · reflex | 100 | 85 | 65 |
| GPT-4.1 · reflex | 100 | 90 | 85 |
| GPT-4o mini · reflex | 100 | 95 | 30 |
| GPT-5 mini · reflex | 100 | 95 | 85 |
| GPT-5 nano · reflex | 100 | 55 | 30 |
| GPT-5.5 · reflex | 100 | 100 | 100 |
| Grok 4.3 · reflex | 100 | 90 | 90 |
| **API · open weights** | | | |
| DeepSeek V3.2 (671B MoE) · reflex | 100† | 100† | 100† |
| Gemma-3 12B IT · reflex | 20 | 100 | 90 |
| Gemma-3 27B IT · reflex | 100 | 90 | 45 |
| Gemma-3 4B IT · reflex | 90 | 0 | 0 |
| GLM-4.6 (355B MoE) · reflex | 100 | 85 | 95 |
| gpt-oss 120B · reflex | 50† | 38† | 10† |
| Hermes-3 405B · reflex | 90 | 95 | 100 |
| Kimi K2 (1T MoE) · reflex | 100 | 100 | 95 |
| Llama-3.1 8B Instruct · reflex | 65 | 55 | 15 |
| Llama-3.2 1B Instruct · reflex | 55 | 0 | 5 |
| Llama-3.2 3B Instruct · reflex | 85 | 20 | 10 |
| Llama-3.3 70B Instruct · reflex | 100 | 95 | 90 |
| Llama-4 Maverick (400B MoE) · reflex | 100 | 95 | 90 |
| Ministral 3B · reflex | 90 | 45 | 35 |
| Ministral 8B · reflex | 100 | 80 | 20 |
| Mistral Small 3.2 24B · reflex | 95 | 60 | 70 |
| Phi-4 14B · reflex | 60 | 95 | 85 |
| Qwen2.5 72B Instruct · reflex | 100 | 75 | 90 |
| Qwen2.5 7B Instruct · reflex | 70 | 50 | 55 |
| Qwen3 235B-A22B · reflex | 90 | 75 | 80 |
| Qwen3 32B · reflex | 100† | 89† | 0† |
| Qwen3.5 397B-A17B · reflex | 100 | 90 | 90 |
| **Pretrained bases** | | | |
| SmolLM2 135M | 42 | 2 | 4 |
| Pythia 160M | 24 | 10 | 4 |
| SmolLM2 360M | 56 | 4 | 1 |
| Pythia 410M | 32 | 13 | 12 |
| Qwen2.5 0.5B | 58 | 5 | 4 |
| Qwen3 0.6B | 80 | 7 | 0 |
| Llama-3.2 1B | 87 | 11 | 2 |
| OLMo-1 1B (0724) | 64 | 3 | 1 |
| Pythia 1B | 52 | 10 | 4 |
| kotodama-tsumugi 1.4B (release) | 58 | 9 | 7 |
| Pythia 1.4B | 26 | 22 | 14 |
| OLMo-2 1B | 60 | 1 | 4 |
| Qwen2.5 1.5B | 79 | 2 | 0 |
| Qwen3 1.7B | 75 | 22 | 11 |
| SmolLM2 1.7B | 76 | 6 | 2 |
| Pythia 2.8B | 76 | 10 | 4 |
| Qwen2.5 3B | 89 | 11 | 9 |
| Qwen3 4B | 94 | 31 | 30 |
| Pythia 6.9B | 54 | 18 | 10 |
| Falcon3 7B | 84 | 22 | 12 |
| OLMo-2 7B | 83 | 10 | 3 |
| Qwen2.5 7B | 90 | 50 | 25 |
| Llama-3.1 8B | 86 | 28 | 12 |
| Gemma-2 9B | 86 | 67 | 27 |
| Qwen3 30B-A3B | 92 | 72 | 64 |
| Gemma-4 31B | 99 | 81 | 56 |
| GLM-4 32B | 96 | 57 | 9 |
| OLMo-3 32B | 98 | 69 | 35 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | 86 | 14 | 3 |
| Llama-3.1 70B | 97 | 97 | 55 |
| Qwen2.5 72B | 99 | 82 | 69 |
| DeepSeek-V4 Flash Base | 86 | 92 | 89 |
| MiMo-V2.5 Base | 88 | 79 | 70 |
| Trinity-Large Base | 98 | 93 | 98 |
| Trinity-Large TrueBase | 98 | 78 | 36 |
| Llama-3.1 405B | 99 | 80 | 83 |
| DeepSeek-V3 Base (671B MoE) | 96 | 94 | 90 |
| DeepSeek-V4 Pro Base | 100 | 88 | 98 |
| Kimi-K2 Base (1T MoE) | 98 | 97 | 90 |
| MiMo-V2.5-Pro Base | 98 | 90 | 88 |
| **Post-trained** | | | |
| OLMo-2 7B SFT | 86 | 10 | 4 |
| OLMo-2 7B DPO | 94 | 6 | 0 |
| OLMo-2 7B Instruct | 94 | 6 | 0 |
| Llama-3.1 8B Instruct | 77 | 44 | 54 |
| **Hybrid & SSM bases** | | | |
| Falcon-H1 1.5B (24L) | 95 | 11 | 1 |
| Falcon-H1 1.5B-Deep (66L) | 84 | 39 | 2 |
| Falcon-H1 7B | 98 | 9 | 6 |
| Falcon-H1 34B | 84 | 78 | 78 |
| Nemotron-3 Ultra 550B Base (hybrid) | 100 | 84 | 83 |
| *chance* | 20 | 20 | 20 |

## Speaker-swap test (open bases)

Two-person chats; the prompt is repeated with ONLY the name on the final query line changed, which flips the right answer. `shift` = mean P(original gold | original asker) − P(original gold | swapped asker), candidates renormalised; 95% CI by paired bootstrap over 180 items. `> 0` = the answer follows the speaker (binds); `≈ 0` = ignores who is speaking; `< 0` = moves the wrong way (inverts).

**Reading rule:** `binds` if the CI's lower bound is above +.02; `inverts` if its upper bound is below −.02; otherwise `ignores speaker` (no reliable movement — a CI can exclude 0 and still be read this way).

| model | I 0-shot | I 3-shot | you 0-shot | you 0-shot 95% CI | you 0-shot reading | you 3-shot |
|---|---|---|---|---|---|---|
| DeepSeek-V3 Base (671B MoE) | +0.062 | +0.395 | +0.067 | [+0.051, +0.083] | binds | +0.065 |
| DeepSeek-V4 Flash Base | +0.166 | +0.499 | +0.102 | [+0.083, +0.123] | binds | +0.083 |
| DeepSeek-V4 Pro Base | +0.088 | +0.520 | +0.004 | [-0.011, +0.018] | ignores speaker | +0.032 |
| Gemma-2 9B | +0.071 | +0.161 | -0.040 | [-0.052, -0.030] | INVERTS | +0.010 |
| Gemma-4 31B | +0.153 | +0.528 | -0.048 | [-0.070, -0.029] | INVERTS | +0.174 |
| GLM-4 32B | +0.102 | +0.414 | +0.027 | [+0.013, +0.041] | ignores speaker | +0.011 |
| Kimi-K2 Base (1T MoE) | +0.033 | +0.418 | +0.045 | [+0.027, +0.064] | binds | +0.070 |
| Llama-3.1 405B | +0.083 | +0.523 | +0.054 | [+0.039, +0.068] | binds | +0.145 |
| Llama-3.1 70B | +0.060 | +0.416 | +0.022 | [+0.013, +0.032] | ignores speaker | +0.039 |
| MiMo-V2.5 Base | +0.092 | +0.391 | -0.012 | [-0.031, +0.005] | ignores speaker | +0.059 |
| MiMo-V2.5-Pro Base | +0.056 | +0.443 | +0.036 | [+0.020, +0.054] | ignores speaker | +0.083 |
| Nemotron-3 Ultra 550B Base (hybrid) | +0.081 | +0.581 | +0.014 | [+0.001, +0.025] | ignores speaker | +0.068 |
| OLMo-3 32B | +0.150 | +0.266 | -0.048 | [-0.070, -0.027] | INVERTS | +0.027 |
| OLMo-3 32B @ stage 1 (pre-midtrain) | +0.073 | +0.143 | -0.023 | [-0.042, -0.007] | ignores speaker | +0.009 |
| Qwen2.5 72B | +0.090 | +0.409 | +0.034 | [+0.014, +0.054] | ignores speaker | +0.040 |
| Qwen3 30B-A3B | +0.062 | +0.215 | +0.025 | [+0.004, +0.046] | ignores speaker | +0.049 |
| Trinity-Large Base | +0.065 | +0.273 | +0.000 | [-0.014, +0.013] | ignores speaker | +0.110 |
| Trinity-Large TrueBase | +0.060 | +0.204 | -0.012 | [-0.023, -0.001] | ignores speaker | +0.044 |

## Rows (CSV)

```csv
row_id,label,group,size_b,scoring
human,"Human reader (gauntlet)","Human",,generation
or-amazon/nova-pro-v1-A,"Amazon Nova Pro · reflex","API · closed",,generation
or-anthropic/claude-fable-5.1-A,"Claude Fable 5.1 · reflex","API · closed",,generation
or-anthropic/claude-haiku-4.5-B,"Claude Haiku 4.5 · deliberate","API · closed",,generation
or-anthropic/claude-haiku-4.5-A,"Claude Haiku 4.5 · reflex","API · closed",,generation
or-anthropic/claude-haiku-5.5-B,"Claude Haiku 5.5 · deliberate","API · closed",,generation
or-anthropic/claude-haiku-5.5-A,"Claude Haiku 5.5 · reflex","API · closed",,generation
or-anthropic/claude-opus-4.5-A,"Claude Opus 4.5 · reflex","API · closed",,generation
or-anthropic/claude-opus-5.5-B,"Claude Opus 5.5 · deliberate","API · closed",,generation
or-anthropic/claude-opus-5.5-A,"Claude Opus 5.5 · reflex","API · closed",,generation
or-anthropic/claude-sonnet-4.5-A,"Claude Sonnet 4.5 · reflex","API · closed",,generation
or-anthropic/claude-sonnet-5.5-B,"Claude Sonnet 5.5 · deliberate","API · closed",,generation
or-anthropic/claude-sonnet-5.5-A,"Claude Sonnet 5.5 · reflex","API · closed",,generation
or-google/gemini-2.5-flash-B,"Gemini 2.5 Flash · deliberate","API · closed",,generation
or-google/gemini-2.5-flash-A,"Gemini 2.5 Flash · reflex","API · closed",,generation
or-google/gemini-2.5-flash-lite-A,"Gemini 2.5 Flash-Lite · reflex","API · closed",,generation
or-google/gemini-3.1-pro-preview-B,"Gemini 3.1 Pro (preview) · deliberate","API · closed",,generation
or-openai/gpt-4.1-mini-A,"GPT-4.1 mini · reflex","API · closed",,generation
or-openai/gpt-4.1-nano-A,"GPT-4.1 nano · reflex","API · closed",,generation
or-openai/gpt-4.1-A,"GPT-4.1 · reflex","API · closed",,generation
or-openai/gpt-4o-mini-A,"GPT-4o mini · reflex","API · closed",,generation
or-openai/gpt-5-mini-B,"GPT-5 mini · deliberate","API · closed",,generation
or-openai/gpt-5-mini-A,"GPT-5 mini · reflex","API · closed",,generation
or-openai/gpt-5-nano-B,"GPT-5 nano · deliberate","API · closed",,generation
or-openai/gpt-5-nano-A,"GPT-5 nano · reflex","API · closed",,generation
or-openai/gpt-5.5-B,"GPT-5.5 · deliberate","API · closed",,generation
or-openai/gpt-5.5-A,"GPT-5.5 · reflex","API · closed",,generation
or-x-ai/grok-4.3-B,"Grok 4.3 · deliberate","API · closed",,generation
or-x-ai/grok-4.3-A,"Grok 4.3 · reflex","API · closed",,generation
or-deepseek/deepseek-v3.2-B,"DeepSeek V3.2 (671B MoE) · deliberate","API · open weights",671,generation
or-deepseek/deepseek-v3.2-A,"DeepSeek V3.2 (671B MoE) · reflex","API · open weights",671,generation
or-google/gemma-3-12b-it-A,"Gemma-3 12B IT · reflex","API · open weights",12,generation
or-google/gemma-3-27b-it-A,"Gemma-3 27B IT · reflex","API · open weights",27,generation
or-google/gemma-3-4b-it-A,"Gemma-3 4B IT · reflex","API · open weights",4,generation
or-z-ai/glm-4.6-B,"GLM-4.6 (355B MoE) · deliberate","API · open weights",355,generation
or-z-ai/glm-4.6-A,"GLM-4.6 (355B MoE) · reflex","API · open weights",355,generation
or-openai/gpt-oss-120b-B,"gpt-oss 120B · deliberate","API · open weights",120,generation
or-openai/gpt-oss-120b-A,"gpt-oss 120B · reflex","API · open weights",120,generation
or-nousresearch/hermes-3-llama-3.1-405b-A,"Hermes-3 405B · reflex","API · open weights",405,generation
or-moonshotai/kimi-k2-0905-A,"Kimi K2 (1T MoE) · reflex","API · open weights",1000,generation
or-meta-llama/llama-3.1-8b-instruct-A,"Llama-3.1 8B Instruct · reflex","API · open weights",8,generation
or-meta-llama/llama-3.2-1b-instruct-A,"Llama-3.2 1B Instruct · reflex","API · open weights",1,generation
or-meta-llama/llama-3.2-3b-instruct-A,"Llama-3.2 3B Instruct · reflex","API · open weights",3,generation
or-meta-llama/llama-3.3-70b-instruct-A,"Llama-3.3 70B Instruct · reflex","API · open weights",70,generation
or-meta-llama/llama-4-maverick-A,"Llama-4 Maverick (400B MoE) · reflex","API · open weights",400,generation
or-mistralai/ministral-3b-2512-A,"Ministral 3B · reflex","API · open weights",3,generation
or-mistralai/ministral-8b-2512-A,"Ministral 8B · reflex","API · open weights",8,generation
or-mistralai/mistral-large-2512-A,"Mistral Large (2512) · reflex","API · open weights",,generation
or-mistralai/mistral-small-3.2-24b-instruct-A,"Mistral Small 3.2 24B · reflex","API · open weights",24,generation
or-microsoft/phi-4-A,"Phi-4 14B · reflex","API · open weights",14,generation
or-qwen/qwen-2.5-72b-instruct-A,"Qwen2.5 72B Instruct · reflex","API · open weights",72,generation
or-qwen/qwen-2.5-7b-instruct-A,"Qwen2.5 7B Instruct · reflex","API · open weights",7,generation
or-qwen/qwen3-235b-a22b-2507-A,"Qwen3 235B-A22B · reflex","API · open weights",235,generation
or-qwen/qwen3-32b-A,"Qwen3 32B · reflex","API · open weights",32,generation
or-qwen/qwen3.5-397b-a17b-B,"Qwen3.5 397B-A17B · deliberate","API · open weights",397,generation
or-qwen/qwen3.5-397b-a17b-A,"Qwen3.5 397B-A17B · reflex","API · open weights",397,generation
smollm2-135m,"SmolLM2 135M","Pretrained bases",0.135,logprob
pythia-160m,"Pythia 160M","Pretrained bases",0.16,logprob
smollm2-360m,"SmolLM2 360M","Pretrained bases",0.36,logprob
pythia-410m,"Pythia 410M","Pretrained bases",0.41,logprob
qwen2.5-0.5b,"Qwen2.5 0.5B","Pretrained bases",0.5,logprob
qwen3-0.6b-base,"Qwen3 0.6B","Pretrained bases",0.6,logprob
llama3.2-1b,"Llama-3.2 1B","Pretrained bases",1,logprob
olmo1-1b-0724,"OLMo-1 1B (0724)","Pretrained bases",1,logprob
pythia-1b,"Pythia 1B","Pretrained bases",1,logprob
tsumugi-release,"kotodama-tsumugi 1.4B (release)","Pretrained bases",1.4,logprob
pythia-1.4b,"Pythia 1.4B","Pretrained bases",1.4,logprob
olmo2-1b,"OLMo-2 1B","Pretrained bases",1.5,logprob
qwen2.5-1.5b,"Qwen2.5 1.5B","Pretrained bases",1.5,logprob
qwen3-1.7b-base,"Qwen3 1.7B","Pretrained bases",1.7,logprob
smollm2-1.7b,"SmolLM2 1.7B","Pretrained bases",1.7,logprob
pythia-2.8b,"Pythia 2.8B","Pretrained bases",2.8,logprob
falcon3-3b,"Falcon3 3B","Pretrained bases",3,logprob
qwen2.5-3b,"Qwen2.5 3B","Pretrained bases",3,logprob
qwen3-4b-base,"Qwen3 4B","Pretrained bases",4,logprob
pythia-6.9b,"Pythia 6.9B","Pretrained bases",6.9,logprob
falcon3-7b,"Falcon3 7B","Pretrained bases",7,logprob
olmo2-7b-base,"OLMo-2 7B","Pretrained bases",7,logprob
qwen2.5-7b,"Qwen2.5 7B","Pretrained bases",7,logprob
llama31-8b-base,"Llama-3.1 8B","Pretrained bases",8,logprob
gemma2-9b,"Gemma-2 9B","Pretrained bases",9,logprob
olmo2-13b,"OLMo-2 13B","Pretrained bases",13,logprob
qwen3-30b-a3b-base,"Qwen3 30B-A3B","Pretrained bases",30,logprob
gemma4-31b-base,"Gemma-4 31B","Pretrained bases",31,logprob
glm4-32b-base,"GLM-4 32B","Pretrained bases",32,logprob
olmo3-32b,"OLMo-3 32B","Pretrained bases",32,logprob
olmo3-32b-stage1,"OLMo-3 32B @ stage 1 (pre-midtrain)","Pretrained bases",32,logprob
llama31-70b,"Llama-3.1 70B","Pretrained bases",70,logprob
qwen25-72b,"Qwen2.5 72B","Pretrained bases",72,logprob
dsv4-flash-base,"DeepSeek-V4 Flash Base","Pretrained bases",290,logprob
mimo-v25-base,"MiMo-V2.5 Base","Pretrained bases",310,logprob
trinity-base,"Trinity-Large Base","Pretrained bases",400,logprob
trinity-truebase,"Trinity-Large TrueBase","Pretrained bases",400,logprob
llama31-405b,"Llama-3.1 405B","Pretrained bases",405,logprob
dsv3-base,"DeepSeek-V3 Base (671B MoE)","Pretrained bases",671,logprob
dsv4-pro-base,"DeepSeek-V4 Pro Base","Pretrained bases",860,logprob
kimi-k2-base,"Kimi-K2 Base (1T MoE)","Pretrained bases",1000,logprob
mimo-v25-pro-base,"MiMo-V2.5-Pro Base","Pretrained bases",1000,logprob
olmo2-7b-pt2928B,"OLMo-2 7B @ 2.9T (stage 1)","OLMo-2 7B trajectory",7,logprob
olmo2-7b-pt3896B,"OLMo-2 7B @ 3.9T (end stage 1)","OLMo-2 7B trajectory",7,logprob
olmo2-7b-mid5B,"OLMo-2 7B mid-train +5B","OLMo-2 7B trajectory",7,logprob
olmo2-7b-mid17B,"OLMo-2 7B mid-train +17B","OLMo-2 7B trajectory",7,logprob
olmo2-7b-mid50B,"OLMo-2 7B mid-train +50B","OLMo-2 7B trajectory",7,logprob
olmo2-7b-sft,"OLMo-2 7B SFT","Post-trained",7,logprob
olmo2-7b-dpo,"OLMo-2 7B DPO","Post-trained",7,logprob
olmo2-7b-instruct,"OLMo-2 7B Instruct","Post-trained",7,logprob
llama31-8b-instruct,"Llama-3.1 8B Instruct","Post-trained",8,logprob
falcon-h1-0.5b,"Falcon-H1 0.5B","Hybrid & SSM bases",0.5,logprob
falcon-h1-1.5b,"Falcon-H1 1.5B (24L)","Hybrid & SSM bases",1.5,logprob
falcon-h1-1.5b-deep,"Falcon-H1 1.5B-Deep (66L)","Hybrid & SSM bases",1.5,logprob
falcon-h1-3b,"Falcon-H1 3B","Hybrid & SSM bases",3,logprob
falcon-h1-7b,"Falcon-H1 7B","Hybrid & SSM bases",7,logprob
falcon-mamba-7b,"Falcon-Mamba 7B (pure SSM)","Hybrid & SSM bases",7,logprob
bamba-9b,"Bamba 9B (interleaved)","Hybrid & SSM bases",9,logprob
recurrentgemma-9b,"RecurrentGemma 9B","Hybrid & SSM bases",9,logprob
falcon-h1-34b-base,"Falcon-H1 34B","Hybrid & SSM bases",34,logprob
nemotron3-ultra-550b-base,"Nemotron-3 Ultra 550B Base (hybrid)","Hybrid & SSM bases",550,logprob
```

## All cells (CSV)

`row_id,col_id,condition,mode,acc,n,chance,se,seat_form,errs` — `score` (= (acc − chance)/(1 − chance)) is left out as derivable; `errs` = error-type shares among items (deixis I/you columns; `;`-separated `type=share`, types in the glossary); `seat_form` = share of referent-correct API answers that used me/you.

```csv
row_id,col_id,condition,mode,acc,n,chance,se,seat_form,errs
human,attr,0shot,all,1.000,5,0.185,0.000,,
human,codevar,0shot,all,0.800,5,0.257,0.179,,
human,dialogue,0shot,all,1.000,6,0.319,0.000,,
human,dialogue,3shot_collide,all,1.000,1,0.250,0.000,,
human,dialogue,3shot,all,1.000,1,0.250,0.000,,
human,dx-owner-3-deictic,0shot,all,1.000,5,0.333,0.000,,
human,dx-owner-3-deictic,0shot,free,1.000,1,0.333,0.000,,
human,dx-self-3-deictic,0shot,all,1.000,2,0.333,0.000,,
human,dx-self-2-deictic,0shot,all,1.000,3,0.306,0.000,,
human,dx-self-2-deictic,0shot,free,1.000,1,0.250,0.000,,
human,dx-addressee-3-deictic,0shot,all,1.000,3,0.306,0.000,,
human,dx-addressee-3-deictic,0shot,free,1.000,1,0.250,0.000,,
human,dx-addressee-3-third,0shot,all,1.000,1,0.333,0.000,,
human,dx-addressee-2-deictic,0shot,all,1.000,4,0.312,0.000,,
human,dx-addressee-2-deictic,0shot,free,1.000,2,0.292,0.000,,
human,dx-addressee-2-third,0shot,all,1.000,1,0.250,0.000,,
human,dx-addressee-2-third,0shot,free,1.000,1,0.250,0.000,,
human,sb-k1,0shot,all,1.000,1,0.125,0.000,,
human,sb-k23,0shot,all,1.000,1,0.018,0.000,,
human,sb-split,0shot,all,1.000,1,0.018,0.000,,
human,sb-dyn,0shot,all,1.000,1,0.036,0.000,,
human,an-narr,0shot,all,1.000,1,0.250,0.000,,
human,an-other-any,0shot,all,1.000,1,0.250,0.000,,
human,an-me-any,0shot,all,1.000,1,0.250,0.000,,
human,an-you-voc,0shot,all,1.000,1,0.250,0.000,,
human,an-you-novoc,0shot,all,1.000,1,0.250,0.000,,
human,adj-adj,0shot,all,0.500,2,0.333,0.354,,
human,adj-voc,0shot,all,1.000,1,0.333,0.000,,
human,adj-third,0shot,all,1.000,1,0.333,0.000,,
human,cm-mutate,0shot,all,0.500,2,0.200,0.354,,
human,cm-rebind,0shot,all,1.000,1,0.200,0.000,,
human,cm-copy,0shot,all,1.000,1,0.200,0.000,,
or-amazon/nova-pro-v1-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-amazon/nova-pro-v1-A,codevar,0shot,all,0.800,30,0.274,0.073,,
or-amazon/nova-pro-v1-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-amazon/nova-pro-v1-A,dx-owner-3-deictic,0shot,all,0.750,28,0.333,0.082,,other=0.25
or-amazon/nova-pro-v1-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-amazon/nova-pro-v1-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-amazon/nova-pro-v1-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-amazon/nova-pro-v1-A,dx-self-3-deictic,0shot,all,0.630,27,0.284,0.093,,other_participant=0.37
or-amazon/nova-pro-v1-A,dx-self-3-deictic,0shot,free,0.737,19,0.281,0.101,,other_participant=0.263
or-amazon/nova-pro-v1-A,dx-self-3-third,0shot,all,0.875,32,0.286,0.059,,
or-amazon/nova-pro-v1-A,dx-self-3-third,0shot,free,0.955,22,0.284,0.044,,
or-amazon/nova-pro-v1-A,dx-self-2-deictic,0shot,all,0.485,33,0.280,0.087,,other=0.303;other_object=0.091;other_participant=0.121
or-amazon/nova-pro-v1-A,dx-self-2-deictic,0shot,free,0.478,23,0.275,0.104,,other=0.304;other_object=0.13;other_participant=0.087
or-amazon/nova-pro-v1-A,dx-self-2-third,0shot,all,0.857,28,0.295,0.066,,
or-amazon/nova-pro-v1-A,dx-self-2-third,0shot,free,0.895,19,0.289,0.070,,
or-amazon/nova-pro-v1-A,dx-addressee-3-deictic,0shot,all,0.424,33,0.283,0.086,,other_participant=0.303;speaker_confusion=0.273
or-amazon/nova-pro-v1-A,dx-addressee-3-deictic,0shot,free,0.500,22,0.280,0.107,,other_participant=0.273;speaker_confusion=0.227
or-amazon/nova-pro-v1-A,dx-addressee-3-third,0shot,all,0.889,27,0.287,0.060,,
or-amazon/nova-pro-v1-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-amazon/nova-pro-v1-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,other_participant=0.111;speaker_confusion=0.556
or-amazon/nova-pro-v1-A,dx-addressee-2-deictic,0shot,free,0.444,18,0.292,0.117,,other_participant=0.167;speaker_confusion=0.389
or-amazon/nova-pro-v1-A,dx-addressee-2-third,0shot,all,0.758,33,0.285,0.075,,
or-amazon/nova-pro-v1-A,dx-addressee-2-third,0shot,free,0.870,23,0.279,0.070,,
or-amazon/nova-pro-v1-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-amazon/nova-pro-v1-A,an-narr,0shot,all,0.667,24,0.250,0.096,,
or-amazon/nova-pro-v1-A,an-other-any,0shot,all,0.708,24,0.250,0.093,,
or-amazon/nova-pro-v1-A,an-me-any,0shot,all,0.333,24,0.250,0.096,1.0,
or-amazon/nova-pro-v1-A,an-you-voc,0shot,all,0.667,24,0.250,0.096,0.938,
or-amazon/nova-pro-v1-A,an-you-novoc,0shot,all,0.500,24,0.250,0.102,0.0,
or-amazon/nova-pro-v1-A,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-amazon/nova-pro-v1-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-amazon/nova-pro-v1-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-amazon/nova-pro-v1-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-amazon/nova-pro-v1-A,cm-rebind,0shot,all,0.700,20,0.200,0.102,,
or-amazon/nova-pro-v1-A,cm-copy,0shot,all,0.450,20,0.200,0.111,,
or-anthropic/claude-fable-5.1-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-fable-5.1-A,codevar,0shot,all,0.533,30,0.274,0.091,,
or-anthropic/claude-fable-5.1-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-anthropic/claude-fable-5.1-A,dx-owner-3-deictic,0shot,all,0.964,28,0.333,0.035,,other=0.036
or-anthropic/claude-fable-5.1-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-2-deictic,0shot,all,0.839,31,0.280,0.066,,format=0.129;other=0.032
or-anthropic/claude-fable-5.1-A,dx-self-2-deictic,0shot,free,0.905,21,0.274,0.064,,format=0.095
or-anthropic/claude-fable-5.1-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-fable-5.1-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-fable-5.1-A,dx-addressee-3-deictic,0shot,all,0.939,33,0.283,0.042,,format=0.03;other_participant=0.03
or-anthropic/claude-fable-5.1-A,dx-addressee-3-deictic,0shot,free,0.955,22,0.280,0.044,,format=0.045
or-anthropic/claude-fable-5.1-A,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-anthropic/claude-fable-5.1-A,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-anthropic/claude-fable-5.1-A,dx-addressee-2-deictic,0shot,all,0.958,24,0.292,0.041,,speaker_confusion=0.042
or-anthropic/claude-fable-5.1-A,dx-addressee-2-deictic,0shot,free,1.000,17,0.289,0.000,,
or-anthropic/claude-fable-5.1-A,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-fable-5.1-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-fable-5.1-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-fable-5.1-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-fable-5.1-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-fable-5.1-A,an-me-any,0shot,all,1.000,24,0.250,0.000,1.0,
or-anthropic/claude-fable-5.1-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-fable-5.1-A,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.042,
or-anthropic/claude-fable-5.1-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-fable-5.1-A,cm-mutate,0shot,all,0.850,20,0.200,0.080,,
or-anthropic/claude-fable-5.1-A,cm-rebind,0shot,all,0.800,20,0.200,0.089,,
or-anthropic/claude-fable-5.1-A,cm-copy,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-2-deictic,0shot,all,0.970,33,0.280,0.030,,other_object=0.03
or-anthropic/claude-haiku-4.5-B,dx-self-2-deictic,0shot,free,0.957,23,0.275,0.043,,other_object=0.043
or-anthropic/claude-haiku-4.5-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-2-deictic,0shot,all,1.000,27,0.293,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-2-deictic,0shot,free,1.000,18,0.292,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-haiku-4.5-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-haiku-4.5-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-haiku-4.5-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-haiku-4.5-B,an-me-any,0shot,all,0.958,24,0.250,0.041,1.0,
or-anthropic/claude-haiku-4.5-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.875,
or-anthropic/claude-haiku-4.5-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.042,
or-anthropic/claude-haiku-4.5-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-4.5-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-4.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-haiku-4.5-A,codevar,0shot,all,0.900,30,0.274,0.055,,
or-anthropic/claude-haiku-4.5-A,dialogue,0shot,all,0.867,30,0.311,0.062,,
or-anthropic/claude-haiku-4.5-A,dx-owner-3-deictic,0shot,all,0.893,28,0.333,0.059,,other=0.107
or-anthropic/claude-haiku-4.5-A,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-anthropic/claude-haiku-4.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-haiku-4.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-4.5-A,dx-self-3-deictic,0shot,all,0.444,27,0.284,0.096,,other_participant=0.556
or-anthropic/claude-haiku-4.5-A,dx-self-3-deictic,0shot,free,0.526,19,0.281,0.115,,other_participant=0.474
or-anthropic/claude-haiku-4.5-A,dx-self-3-third,0shot,all,0.688,32,0.286,0.082,,
or-anthropic/claude-haiku-4.5-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-anthropic/claude-haiku-4.5-A,dx-self-2-deictic,0shot,all,0.424,33,0.280,0.086,,other=0.545;other_participant=0.03
or-anthropic/claude-haiku-4.5-A,dx-self-2-deictic,0shot,free,0.565,23,0.275,0.103,,other=0.435
or-anthropic/claude-haiku-4.5-A,dx-self-2-third,0shot,all,0.643,28,0.295,0.091,,
or-anthropic/claude-haiku-4.5-A,dx-self-2-third,0shot,free,0.895,19,0.289,0.070,,
or-anthropic/claude-haiku-4.5-A,dx-addressee-3-deictic,0shot,all,0.485,33,0.283,0.087,,other_participant=0.182;speaker_confusion=0.333
or-anthropic/claude-haiku-4.5-A,dx-addressee-3-deictic,0shot,free,0.636,22,0.280,0.103,,other_participant=0.136;speaker_confusion=0.227
or-anthropic/claude-haiku-4.5-A,dx-addressee-3-third,0shot,all,0.556,27,0.287,0.096,,
or-anthropic/claude-haiku-4.5-A,dx-addressee-3-third,0shot,free,0.737,19,0.285,0.101,,
or-anthropic/claude-haiku-4.5-A,dx-addressee-2-deictic,0shot,all,0.444,27,0.293,0.096,,other_participant=0.037;speaker_confusion=0.519
or-anthropic/claude-haiku-4.5-A,dx-addressee-2-deictic,0shot,free,0.667,18,0.292,0.111,,speaker_confusion=0.333
or-anthropic/claude-haiku-4.5-A,dx-addressee-2-third,0shot,all,0.576,33,0.285,0.086,,
or-anthropic/claude-haiku-4.5-A,dx-addressee-2-third,0shot,free,0.783,23,0.279,0.086,,
or-anthropic/claude-haiku-4.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-haiku-4.5-A,an-narr,0shot,all,0.792,24,0.250,0.083,,
or-anthropic/claude-haiku-4.5-A,an-other-any,0shot,all,0.875,24,0.250,0.068,,
or-anthropic/claude-haiku-4.5-A,an-me-any,0shot,all,0.333,24,0.250,0.096,1.0,
or-anthropic/claude-haiku-4.5-A,an-you-voc,0shot,all,0.583,24,0.250,0.101,1.0,
or-anthropic/claude-haiku-4.5-A,an-you-novoc,0shot,all,0.333,24,0.250,0.096,0.0,
or-anthropic/claude-haiku-4.5-A,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-anthropic/claude-haiku-4.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-4.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-4.5-A,cm-mutate,0shot,all,0.950,20,0.200,0.049,,
or-anthropic/claude-haiku-4.5-A,cm-rebind,0shot,all,0.850,20,0.200,0.080,,
or-anthropic/claude-haiku-4.5-A,cm-copy,0shot,all,0.950,20,0.200,0.049,,
or-anthropic/claude-haiku-5.5-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-self-3-deictic,0shot,all,0.778,27,0.284,0.080,,format=0.222
or-anthropic/claude-haiku-5.5-B,dx-self-3-deictic,0shot,free,0.684,19,0.281,0.107,,format=0.316
or-anthropic/claude-haiku-5.5-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-self-2-deictic,0shot,all,0.879,33,0.280,0.057,,format=0.091;other_object=0.03
or-anthropic/claude-haiku-5.5-B,dx-self-2-deictic,0shot,free,0.913,23,0.275,0.059,,format=0.043;other_object=0.043
or-anthropic/claude-haiku-5.5-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-addressee-3-deictic,0shot,all,0.879,33,0.283,0.057,,format=0.121
or-anthropic/claude-haiku-5.5-B,dx-addressee-3-deictic,0shot,free,0.818,22,0.280,0.082,,format=0.182
or-anthropic/claude-haiku-5.5-B,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-anthropic/claude-haiku-5.5-B,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-anthropic/claude-haiku-5.5-B,dx-addressee-2-deictic,0shot,all,0.852,27,0.293,0.068,,format=0.074;other_participant=0.074
or-anthropic/claude-haiku-5.5-B,dx-addressee-2-deictic,0shot,free,0.778,18,0.292,0.098,,format=0.111;other_participant=0.111
or-anthropic/claude-haiku-5.5-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-haiku-5.5-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-haiku-5.5-B,an-narr,0shot,all,0.958,24,0.250,0.041,,
or-anthropic/claude-haiku-5.5-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-haiku-5.5-B,an-me-any,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-haiku-5.5-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-haiku-5.5-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-anthropic/claude-haiku-5.5-B,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-anthropic/claude-haiku-5.5-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-5.5-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-5.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-haiku-5.5-A,codevar,0shot,all,0.700,30,0.274,0.084,,
or-anthropic/claude-haiku-5.5-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-anthropic/claude-haiku-5.5-A,dx-owner-3-deictic,0shot,all,0.714,28,0.333,0.085,,format=0.036;other=0.25
or-anthropic/claude-haiku-5.5-A,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-anthropic/claude-haiku-5.5-A,dx-owner-3-third,0shot,all,0.964,28,0.333,0.035,,
or-anthropic/claude-haiku-5.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-haiku-5.5-A,dx-self-3-deictic,0shot,all,0.630,27,0.284,0.093,,format=0.037;other_participant=0.333
or-anthropic/claude-haiku-5.5-A,dx-self-3-deictic,0shot,free,0.579,19,0.281,0.113,,format=0.053;other_participant=0.368
or-anthropic/claude-haiku-5.5-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-haiku-5.5-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-haiku-5.5-A,dx-self-2-deictic,0shot,all,0.545,33,0.280,0.087,,other=0.424;other_object=0.03
or-anthropic/claude-haiku-5.5-A,dx-self-2-deictic,0shot,free,0.739,23,0.275,0.092,,other=0.217;other_object=0.043
or-anthropic/claude-haiku-5.5-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-haiku-5.5-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-haiku-5.5-A,dx-addressee-3-deictic,0shot,all,0.667,33,0.283,0.082,,other_participant=0.121;speaker_confusion=0.212
or-anthropic/claude-haiku-5.5-A,dx-addressee-3-deictic,0shot,free,0.727,22,0.280,0.095,,other_participant=0.045;speaker_confusion=0.227
or-anthropic/claude-haiku-5.5-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-anthropic/claude-haiku-5.5-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-anthropic/claude-haiku-5.5-A,dx-addressee-2-deictic,0shot,all,0.407,27,0.293,0.095,,other_participant=0.074;speaker_confusion=0.519
or-anthropic/claude-haiku-5.5-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.333
or-anthropic/claude-haiku-5.5-A,dx-addressee-2-third,0shot,all,0.970,33,0.285,0.030,,
or-anthropic/claude-haiku-5.5-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-haiku-5.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-haiku-5.5-A,an-narr,0shot,all,0.833,24,0.250,0.076,,
or-anthropic/claude-haiku-5.5-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-haiku-5.5-A,an-me-any,0shot,all,0.750,24,0.250,0.088,0.0,
or-anthropic/claude-haiku-5.5-A,an-you-voc,0shot,all,0.708,24,0.250,0.093,0.0,
or-anthropic/claude-haiku-5.5-A,an-you-novoc,0shot,all,0.833,24,0.250,0.076,0.0,
or-anthropic/claude-haiku-5.5-A,adj-adj,0shot,all,0.920,25,0.333,0.054,,
or-anthropic/claude-haiku-5.5-A,adj-voc,0shot,all,0.920,25,0.333,0.054,,
or-anthropic/claude-haiku-5.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-haiku-5.5-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-haiku-5.5-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-haiku-5.5-A,cm-copy,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-opus-4.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-opus-4.5-A,codevar,0shot,all,1.000,30,0.274,0.000,,
or-anthropic/claude-opus-4.5-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-anthropic/claude-opus-4.5-A,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,dx-self-3-deictic,0shot,all,0.926,27,0.284,0.050,,other_participant=0.074
or-anthropic/claude-opus-4.5-A,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-anthropic/claude-opus-4.5-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-opus-4.5-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-opus-4.5-A,dx-self-2-deictic,0shot,all,0.758,33,0.280,0.075,,other=0.212;other_object=0.03
or-anthropic/claude-opus-4.5-A,dx-self-2-deictic,0shot,free,0.870,23,0.275,0.070,,other=0.087;other_object=0.043
or-anthropic/claude-opus-4.5-A,dx-self-2-third,0shot,all,0.964,28,0.295,0.035,,
or-anthropic/claude-opus-4.5-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-opus-4.5-A,dx-addressee-3-deictic,0shot,all,0.909,33,0.283,0.050,,other_participant=0.03;speaker_confusion=0.061
or-anthropic/claude-opus-4.5-A,dx-addressee-3-deictic,0shot,free,0.909,22,0.280,0.061,,other_participant=0.045;speaker_confusion=0.045
or-anthropic/claude-opus-4.5-A,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-anthropic/claude-opus-4.5-A,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-anthropic/claude-opus-4.5-A,dx-addressee-2-deictic,0shot,all,0.481,27,0.293,0.096,,other_participant=0.185;speaker_confusion=0.333
or-anthropic/claude-opus-4.5-A,dx-addressee-2-deictic,0shot,free,0.556,18,0.292,0.117,,other_participant=0.056;speaker_confusion=0.389
or-anthropic/claude-opus-4.5-A,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-opus-4.5-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-opus-4.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-opus-4.5-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-opus-4.5-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-opus-4.5-A,an-me-any,0shot,all,1.000,24,0.250,0.000,0.75,
or-anthropic/claude-opus-4.5-A,an-you-voc,0shot,all,0.958,24,0.250,0.041,0.826,
or-anthropic/claude-opus-4.5-A,an-you-novoc,0shot,all,0.958,24,0.250,0.041,0.0,
or-anthropic/claude-opus-4.5-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-4.5-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-opus-4.5-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-opus-4.5-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-anthropic/claude-opus-5.5-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-2-deictic,0shot,all,1.000,33,0.280,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-2-deictic,0shot,free,1.000,23,0.275,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-opus-5.5-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-2-deictic,0shot,all,1.000,27,0.293,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-2-deictic,0shot,free,1.000,18,0.292,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-opus-5.5-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-opus-5.5-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-opus-5.5-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-opus-5.5-B,an-me-any,0shot,all,1.000,24,0.250,0.000,1.0,
or-anthropic/claude-opus-5.5-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,1.0,
or-anthropic/claude-opus-5.5-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.5,
or-anthropic/claude-opus-5.5-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-opus-5.5-A,codevar,0shot,all,0.696,23,0.269,0.096,,
or-anthropic/claude-opus-5.5-A,dialogue,0shot,all,1.000,25,0.310,0.000,,
or-anthropic/claude-opus-5.5-A,dx-owner-3-deictic,0shot,all,1.000,27,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,dx-self-3-deictic,0shot,all,0.889,9,0.287,0.105,,format=0.111
or-anthropic/claude-opus-5.5-A,dx-self-3-deictic,0shot,free,0.875,8,0.292,0.117,,format=0.125
or-anthropic/claude-opus-5.5-A,dx-self-3-third,0shot,all,0.968,31,0.288,0.032,,
or-anthropic/claude-opus-5.5-A,dx-self-3-third,0shot,free,0.952,21,0.286,0.046,,
or-anthropic/claude-opus-5.5-A,dx-self-2-third,0shot,all,1.000,23,0.293,0.000,,
or-anthropic/claude-opus-5.5-A,dx-self-2-third,0shot,free,1.000,14,0.286,0.000,,
or-anthropic/claude-opus-5.5-A,dx-addressee-3-deictic,0shot,all,0.636,11,0.295,0.145,,format=0.364
or-anthropic/claude-opus-5.5-A,dx-addressee-3-third,0shot,all,1.000,22,0.288,0.000,,
or-anthropic/claude-opus-5.5-A,dx-addressee-3-third,0shot,free,1.000,14,0.286,0.000,,
or-anthropic/claude-opus-5.5-A,dx-addressee-2-third,0shot,all,1.000,28,0.286,0.000,,
or-anthropic/claude-opus-5.5-A,dx-addressee-2-third,0shot,free,1.000,19,0.276,0.000,,
or-anthropic/claude-opus-5.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-opus-5.5-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-opus-5.5-A,an-other-any,0shot,all,0.955,22,0.250,0.044,,
or-anthropic/claude-opus-5.5-A,an-you-voc,0shot,all,0.556,9,0.250,0.166,0.8,
or-anthropic/claude-opus-5.5-A,an-you-novoc,0shot,all,0.929,14,0.250,0.069,0.0,
or-anthropic/claude-opus-5.5-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-opus-5.5-A,cm-mutate,0shot,all,0.944,18,0.200,0.054,,
or-anthropic/claude-opus-5.5-A,cm-rebind,0shot,all,0.889,18,0.200,0.074,,
or-anthropic/claude-opus-5.5-A,cm-copy,0shot,all,0.889,9,0.200,0.105,,
or-anthropic/claude-sonnet-4.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-sonnet-4.5-A,codevar,0shot,all,1.000,30,0.274,0.000,,
or-anthropic/claude-sonnet-4.5-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-anthropic/claude-sonnet-4.5-A,dx-owner-3-deictic,0shot,all,0.929,28,0.333,0.049,,other=0.071
or-anthropic/claude-sonnet-4.5-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-anthropic/claude-sonnet-4.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-sonnet-4.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-sonnet-4.5-A,dx-self-3-deictic,0shot,all,0.815,27,0.284,0.075,,other_participant=0.185
or-anthropic/claude-sonnet-4.5-A,dx-self-3-deictic,0shot,free,0.947,19,0.281,0.051,,other_participant=0.053
or-anthropic/claude-sonnet-4.5-A,dx-self-3-third,0shot,all,0.969,32,0.286,0.031,,
or-anthropic/claude-sonnet-4.5-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-sonnet-4.5-A,dx-self-2-deictic,0shot,all,0.424,33,0.280,0.086,,other=0.485;other_object=0.061;other_participant=0.03
or-anthropic/claude-sonnet-4.5-A,dx-self-2-deictic,0shot,free,0.391,23,0.275,0.102,,other=0.478;other_object=0.087;other_participant=0.043
or-anthropic/claude-sonnet-4.5-A,dx-self-2-third,0shot,all,0.893,28,0.295,0.059,,
or-anthropic/claude-sonnet-4.5-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-sonnet-4.5-A,dx-addressee-3-deictic,0shot,all,0.818,33,0.283,0.067,,speaker_confusion=0.182
or-anthropic/claude-sonnet-4.5-A,dx-addressee-3-deictic,0shot,free,0.909,22,0.280,0.061,,speaker_confusion=0.091
or-anthropic/claude-sonnet-4.5-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-anthropic/claude-sonnet-4.5-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-anthropic/claude-sonnet-4.5-A,dx-addressee-2-deictic,0shot,all,0.556,27,0.293,0.096,,other_participant=0.148;speaker_confusion=0.296
or-anthropic/claude-sonnet-4.5-A,dx-addressee-2-deictic,0shot,free,0.722,18,0.292,0.106,,other_participant=0.167;speaker_confusion=0.111
or-anthropic/claude-sonnet-4.5-A,dx-addressee-2-third,0shot,all,0.879,33,0.285,0.057,,
or-anthropic/claude-sonnet-4.5-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-anthropic/claude-sonnet-4.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-sonnet-4.5-A,an-narr,0shot,all,0.875,24,0.250,0.068,,
or-anthropic/claude-sonnet-4.5-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-sonnet-4.5-A,an-me-any,0shot,all,0.667,24,0.250,0.096,1.0,
or-anthropic/claude-sonnet-4.5-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-sonnet-4.5-A,an-you-novoc,0shot,all,0.958,24,0.250,0.041,0.0,
or-anthropic/claude-sonnet-4.5-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-4.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-4.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-4.5-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-sonnet-4.5-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-sonnet-4.5-A,cm-copy,0shot,all,0.950,20,0.200,0.049,,
or-anthropic/claude-sonnet-5.5-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-2-deictic,0shot,all,1.000,33,0.280,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-2-deictic,0shot,free,1.000,23,0.275,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-2-deictic,0shot,all,1.000,27,0.293,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-2-deictic,0shot,free,1.000,18,0.292,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-anthropic/claude-sonnet-5.5-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-anthropic/claude-sonnet-5.5-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-sonnet-5.5-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-sonnet-5.5-B,an-me-any,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-sonnet-5.5-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-sonnet-5.5-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-anthropic/claude-sonnet-5.5-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-anthropic/claude-sonnet-5.5-A,codevar,0shot,all,0.633,30,0.274,0.088,,
or-anthropic/claude-sonnet-5.5-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-anthropic/claude-sonnet-5.5-A,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,dx-self-3-deictic,0shot,all,0.667,27,0.284,0.091,,format=0.333
or-anthropic/claude-sonnet-5.5-A,dx-self-3-deictic,0shot,free,0.579,19,0.281,0.113,,format=0.421
or-anthropic/claude-sonnet-5.5-A,dx-self-3-third,0shot,all,0.938,32,0.286,0.043,,
or-anthropic/claude-sonnet-5.5-A,dx-self-3-third,0shot,free,0.955,22,0.284,0.044,,
or-anthropic/claude-sonnet-5.5-A,dx-self-2-deictic,0shot,all,0.576,33,0.280,0.086,,format=0.424
or-anthropic/claude-sonnet-5.5-A,dx-self-2-deictic,0shot,free,0.565,23,0.275,0.103,,format=0.435
or-anthropic/claude-sonnet-5.5-A,dx-self-2-third,0shot,all,0.929,28,0.295,0.049,,
or-anthropic/claude-sonnet-5.5-A,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-anthropic/claude-sonnet-5.5-A,dx-addressee-3-deictic,0shot,all,0.758,33,0.283,0.075,,format=0.212;speaker_confusion=0.03
or-anthropic/claude-sonnet-5.5-A,dx-addressee-3-deictic,0shot,free,0.818,22,0.280,0.082,,format=0.136;speaker_confusion=0.045
or-anthropic/claude-sonnet-5.5-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-anthropic/claude-sonnet-5.5-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-anthropic/claude-sonnet-5.5-A,dx-addressee-2-deictic,0shot,all,0.630,27,0.293,0.093,,format=0.37
or-anthropic/claude-sonnet-5.5-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,format=0.389
or-anthropic/claude-sonnet-5.5-A,dx-addressee-2-third,0shot,all,0.849,33,0.285,0.062,,
or-anthropic/claude-sonnet-5.5-A,dx-addressee-2-third,0shot,free,0.913,23,0.279,0.059,,
or-anthropic/claude-sonnet-5.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-anthropic/claude-sonnet-5.5-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-sonnet-5.5-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-anthropic/claude-sonnet-5.5-A,an-me-any,0shot,all,0.917,24,0.250,0.056,0.636,
or-anthropic/claude-sonnet-5.5-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-anthropic/claude-sonnet-5.5-A,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-anthropic/claude-sonnet-5.5-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-anthropic/claude-sonnet-5.5-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-anthropic/claude-sonnet-5.5-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-anthropic/claude-sonnet-5.5-A,cm-copy,0shot,all,0.700,20,0.200,0.102,,
or-google/gemini-2.5-flash-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-2.5-flash-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-2.5-flash-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-2.5-flash-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-2.5-flash-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-google/gemini-2.5-flash-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-google/gemini-2.5-flash-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-google/gemini-2.5-flash-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-google/gemini-2.5-flash-B,dx-self-2-deictic,0shot,all,0.879,33,0.280,0.057,,other=0.091;other_object=0.03
or-google/gemini-2.5-flash-B,dx-self-2-deictic,0shot,free,0.913,23,0.275,0.059,,other=0.043;other_object=0.043
or-google/gemini-2.5-flash-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-google/gemini-2.5-flash-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-google/gemini-2.5-flash-B,dx-addressee-3-deictic,0shot,all,0.970,33,0.283,0.030,,other_participant=0.03
or-google/gemini-2.5-flash-B,dx-addressee-3-deictic,0shot,free,0.955,22,0.280,0.044,,other_participant=0.045
or-google/gemini-2.5-flash-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-google/gemini-2.5-flash-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-google/gemini-2.5-flash-B,dx-addressee-2-deictic,0shot,all,0.889,27,0.293,0.060,,other_participant=0.037;speaker_confusion=0.074
or-google/gemini-2.5-flash-B,dx-addressee-2-deictic,0shot,free,0.833,18,0.292,0.088,,other_participant=0.056;speaker_confusion=0.111
or-google/gemini-2.5-flash-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-google/gemini-2.5-flash-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-google/gemini-2.5-flash-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-google/gemini-2.5-flash-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-google/gemini-2.5-flash-B,an-me-any,0shot,all,1.000,24,0.250,0.000,0.958,
or-google/gemini-2.5-flash-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,1.0,
or-google/gemini-2.5-flash-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.042,
or-google/gemini-2.5-flash-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-google/gemini-2.5-flash-A,codevar,0shot,all,0.933,30,0.274,0.045,,
or-google/gemini-2.5-flash-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-google/gemini-2.5-flash-A,dx-owner-3-deictic,0shot,all,0.750,28,0.333,0.082,,other=0.25
or-google/gemini-2.5-flash-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-google/gemini-2.5-flash-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-2.5-flash-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-2.5-flash-A,dx-self-3-deictic,0shot,all,0.593,27,0.284,0.095,,other_participant=0.407
or-google/gemini-2.5-flash-A,dx-self-3-deictic,0shot,free,0.579,19,0.281,0.113,,other_participant=0.421
or-google/gemini-2.5-flash-A,dx-self-3-third,0shot,all,0.969,32,0.286,0.031,,
or-google/gemini-2.5-flash-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-google/gemini-2.5-flash-A,dx-self-2-deictic,0shot,all,0.455,33,0.280,0.087,,format=0.03;other=0.364;other_object=0.152
or-google/gemini-2.5-flash-A,dx-self-2-deictic,0shot,free,0.565,23,0.275,0.103,,other=0.261;other_object=0.174
or-google/gemini-2.5-flash-A,dx-self-2-third,0shot,all,0.893,28,0.295,0.059,,
or-google/gemini-2.5-flash-A,dx-self-2-third,0shot,free,0.895,19,0.289,0.070,,
or-google/gemini-2.5-flash-A,dx-addressee-3-deictic,0shot,all,0.455,33,0.283,0.087,,other_participant=0.182;speaker_confusion=0.364
or-google/gemini-2.5-flash-A,dx-addressee-3-deictic,0shot,free,0.455,22,0.280,0.106,,other_participant=0.182;speaker_confusion=0.364
or-google/gemini-2.5-flash-A,dx-addressee-3-third,0shot,all,0.889,27,0.287,0.060,,
or-google/gemini-2.5-flash-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-google/gemini-2.5-flash-A,dx-addressee-2-deictic,0shot,all,0.481,27,0.293,0.096,,other_participant=0.074;speaker_confusion=0.444
or-google/gemini-2.5-flash-A,dx-addressee-2-deictic,0shot,free,0.556,18,0.292,0.117,,other_participant=0.056;speaker_confusion=0.389
or-google/gemini-2.5-flash-A,dx-addressee-2-third,0shot,all,0.909,33,0.285,0.050,,
or-google/gemini-2.5-flash-A,dx-addressee-2-third,0shot,free,0.870,23,0.279,0.070,,
or-google/gemini-2.5-flash-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-google/gemini-2.5-flash-A,an-narr,0shot,all,0.583,24,0.250,0.101,,
or-google/gemini-2.5-flash-A,an-other-any,0shot,all,0.792,24,0.250,0.083,,
or-google/gemini-2.5-flash-A,an-me-any,0shot,all,0.208,24,0.250,0.083,1.0,
or-google/gemini-2.5-flash-A,an-you-voc,0shot,all,0.417,24,0.250,0.101,1.0,
or-google/gemini-2.5-flash-A,an-you-novoc,0shot,all,0.208,24,0.250,0.083,0.2,
or-google/gemini-2.5-flash-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-google/gemini-2.5-flash-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-google/gemini-2.5-flash-A,cm-copy,0shot,all,1.000,20,0.200,0.000,,
or-google/gemini-2.5-flash-lite-A,attr,0shot,all,0.900,30,0.161,0.055,,
or-google/gemini-2.5-flash-lite-A,codevar,0shot,all,0.700,30,0.274,0.084,,
or-google/gemini-2.5-flash-lite-A,dialogue,0shot,all,0.867,30,0.311,0.062,,
or-google/gemini-2.5-flash-lite-A,dx-owner-3-deictic,0shot,all,0.750,28,0.333,0.082,,other=0.25
or-google/gemini-2.5-flash-lite-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-2.5-flash-lite-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-2.5-flash-lite-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-2.5-flash-lite-A,dx-self-3-deictic,0shot,all,0.296,27,0.284,0.088,,other_participant=0.704
or-google/gemini-2.5-flash-lite-A,dx-self-3-deictic,0shot,free,0.263,19,0.281,0.101,,other_participant=0.737
or-google/gemini-2.5-flash-lite-A,dx-self-3-third,0shot,all,0.844,32,0.286,0.064,,
or-google/gemini-2.5-flash-lite-A,dx-self-3-third,0shot,free,0.955,22,0.284,0.044,,
or-google/gemini-2.5-flash-lite-A,dx-self-2-deictic,0shot,all,0.394,33,0.280,0.085,,format=0.03;other=0.424;other_object=0.121;other_participant=0.03
or-google/gemini-2.5-flash-lite-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,format=0.043;other=0.304;other_object=0.13
or-google/gemini-2.5-flash-lite-A,dx-self-2-third,0shot,all,0.679,28,0.295,0.088,,
or-google/gemini-2.5-flash-lite-A,dx-self-2-third,0shot,free,0.789,19,0.289,0.093,,
or-google/gemini-2.5-flash-lite-A,dx-addressee-3-deictic,0shot,all,0.091,33,0.283,0.050,,other_participant=0.364;speaker_confusion=0.545
or-google/gemini-2.5-flash-lite-A,dx-addressee-3-deictic,0shot,free,0.136,22,0.280,0.073,,other_participant=0.364;speaker_confusion=0.5
or-google/gemini-2.5-flash-lite-A,dx-addressee-3-third,0shot,all,0.815,27,0.287,0.075,,
or-google/gemini-2.5-flash-lite-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-google/gemini-2.5-flash-lite-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,other_participant=0.074;speaker_confusion=0.593
or-google/gemini-2.5-flash-lite-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,other_participant=0.111;speaker_confusion=0.5
or-google/gemini-2.5-flash-lite-A,dx-addressee-2-third,0shot,all,0.576,33,0.285,0.086,,
or-google/gemini-2.5-flash-lite-A,dx-addressee-2-third,0shot,free,0.696,23,0.279,0.096,,
or-google/gemini-2.5-flash-lite-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-google/gemini-2.5-flash-lite-A,an-narr,0shot,all,0.625,24,0.250,0.099,,
or-google/gemini-2.5-flash-lite-A,an-other-any,0shot,all,0.833,24,0.250,0.076,,
or-google/gemini-2.5-flash-lite-A,an-me-any,0shot,all,0.000,24,0.250,0.000,,
or-google/gemini-2.5-flash-lite-A,an-you-voc,0shot,all,0.708,24,0.250,0.093,1.0,
or-google/gemini-2.5-flash-lite-A,an-you-novoc,0shot,all,0.083,24,0.250,0.056,0.0,
or-google/gemini-2.5-flash-lite-A,adj-adj,0shot,all,0.560,25,0.333,0.099,,
or-google/gemini-2.5-flash-lite-A,adj-voc,0shot,all,0.840,25,0.333,0.073,,
or-google/gemini-2.5-flash-lite-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-2.5-flash-lite-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-google/gemini-2.5-flash-lite-A,cm-rebind,0shot,all,0.600,20,0.200,0.110,,
or-google/gemini-2.5-flash-lite-A,cm-copy,0shot,all,0.400,20,0.200,0.110,,
or-google/gemini-3.1-pro-preview-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-2-deictic,0shot,all,1.000,33,0.280,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-2-deictic,0shot,free,1.000,23,0.275,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-2-deictic,0shot,all,1.000,27,0.293,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-2-deictic,0shot,free,1.000,18,0.292,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-google/gemini-3.1-pro-preview-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-google/gemini-3.1-pro-preview-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-google/gemini-3.1-pro-preview-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-google/gemini-3.1-pro-preview-B,an-me-any,0shot,all,1.000,24,0.250,0.000,1.0,
or-google/gemini-3.1-pro-preview-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-google/gemini-3.1-pro-preview-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.167,
or-google/gemini-3.1-pro-preview-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-google/gemini-3.1-pro-preview-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-mini-A,attr,0shot,all,0.833,30,0.161,0.068,,
or-openai/gpt-4.1-mini-A,codevar,0shot,all,0.900,30,0.274,0.055,,
or-openai/gpt-4.1-mini-A,dialogue,0shot,all,0.967,30,0.311,0.033,,
or-openai/gpt-4.1-mini-A,dx-owner-3-deictic,0shot,all,0.714,28,0.333,0.085,,other=0.286
or-openai/gpt-4.1-mini-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-openai/gpt-4.1-mini-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-4.1-mini-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-4.1-mini-A,dx-self-3-deictic,0shot,all,0.444,27,0.284,0.096,,other_participant=0.556
or-openai/gpt-4.1-mini-A,dx-self-3-deictic,0shot,free,0.474,19,0.281,0.115,,other_participant=0.526
or-openai/gpt-4.1-mini-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-4.1-mini-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-4.1-mini-A,dx-self-2-deictic,0shot,all,0.394,33,0.280,0.085,,other=0.545;other_object=0.03;other_participant=0.03
or-openai/gpt-4.1-mini-A,dx-self-2-deictic,0shot,free,0.435,23,0.275,0.103,,other=0.522;other_participant=0.043
or-openai/gpt-4.1-mini-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-4.1-mini-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-4.1-mini-A,dx-addressee-3-deictic,0shot,all,0.515,33,0.283,0.087,,other_participant=0.182;speaker_confusion=0.303
or-openai/gpt-4.1-mini-A,dx-addressee-3-deictic,0shot,free,0.500,22,0.280,0.107,,other_participant=0.227;speaker_confusion=0.273
or-openai/gpt-4.1-mini-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-openai/gpt-4.1-mini-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-openai/gpt-4.1-mini-A,dx-addressee-2-deictic,0shot,all,0.630,27,0.293,0.093,,other_participant=0.074;speaker_confusion=0.296
or-openai/gpt-4.1-mini-A,dx-addressee-2-deictic,0shot,free,0.722,18,0.292,0.106,,other_participant=0.056;speaker_confusion=0.222
or-openai/gpt-4.1-mini-A,dx-addressee-2-third,0shot,all,0.970,33,0.285,0.030,,
or-openai/gpt-4.1-mini-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-openai/gpt-4.1-mini-A,sb-k1,0shot,all,0.958,24,0.125,0.041,,
or-openai/gpt-4.1-mini-A,an-narr,0shot,all,0.583,24,0.250,0.101,,
or-openai/gpt-4.1-mini-A,an-other-any,0shot,all,0.750,24,0.250,0.088,,
or-openai/gpt-4.1-mini-A,an-me-any,0shot,all,0.500,24,0.250,0.102,0.917,
or-openai/gpt-4.1-mini-A,an-you-voc,0shot,all,0.833,24,0.250,0.076,1.0,
or-openai/gpt-4.1-mini-A,an-you-novoc,0shot,all,0.750,24,0.250,0.088,0.0,
or-openai/gpt-4.1-mini-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-mini-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-mini-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-mini-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-4.1-mini-A,cm-rebind,0shot,all,0.800,20,0.200,0.089,,
or-openai/gpt-4.1-mini-A,cm-copy,0shot,all,0.600,20,0.200,0.110,,
or-openai/gpt-4.1-nano-A,attr,0shot,all,0.567,30,0.161,0.090,,
or-openai/gpt-4.1-nano-A,codevar,0shot,all,0.733,30,0.274,0.081,,
or-openai/gpt-4.1-nano-A,dialogue,0shot,all,0.767,30,0.311,0.077,,
or-openai/gpt-4.1-nano-A,dx-owner-3-deictic,0shot,all,0.393,28,0.333,0.092,,format=0.25;other=0.357
or-openai/gpt-4.1-nano-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,format=0.25
or-openai/gpt-4.1-nano-A,dx-owner-3-third,0shot,all,0.714,28,0.333,0.085,,
or-openai/gpt-4.1-nano-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-4.1-nano-A,dx-self-3-deictic,0shot,all,0.259,27,0.284,0.084,,format=0.148;other_participant=0.593
or-openai/gpt-4.1-nano-A,dx-self-3-deictic,0shot,free,0.368,19,0.281,0.111,,format=0.158;other_participant=0.474
or-openai/gpt-4.1-nano-A,dx-self-3-third,0shot,all,0.812,32,0.286,0.069,,
or-openai/gpt-4.1-nano-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-openai/gpt-4.1-nano-A,dx-self-2-deictic,0shot,all,0.212,33,0.280,0.071,,format=0.394;other=0.394
or-openai/gpt-4.1-nano-A,dx-self-2-deictic,0shot,free,0.304,23,0.275,0.096,,format=0.391;other=0.304
or-openai/gpt-4.1-nano-A,dx-self-2-third,0shot,all,0.679,28,0.295,0.088,,
or-openai/gpt-4.1-nano-A,dx-self-2-third,0shot,free,0.842,19,0.289,0.084,,
or-openai/gpt-4.1-nano-A,dx-addressee-3-deictic,0shot,all,0.364,33,0.283,0.084,,format=0.091;other_participant=0.152;speaker_confusion=0.394
or-openai/gpt-4.1-nano-A,dx-addressee-3-deictic,0shot,free,0.545,22,0.280,0.106,,format=0.091;other_participant=0.091;speaker_confusion=0.273
or-openai/gpt-4.1-nano-A,dx-addressee-3-third,0shot,all,0.778,27,0.287,0.080,,
or-openai/gpt-4.1-nano-A,dx-addressee-3-third,0shot,free,0.789,19,0.285,0.093,,
or-openai/gpt-4.1-nano-A,dx-addressee-2-deictic,0shot,all,0.148,27,0.293,0.068,,format=0.185;speaker_confusion=0.667
or-openai/gpt-4.1-nano-A,dx-addressee-2-deictic,0shot,free,0.222,18,0.292,0.098,,format=0.222;speaker_confusion=0.556
or-openai/gpt-4.1-nano-A,dx-addressee-2-third,0shot,all,0.667,33,0.285,0.082,,
or-openai/gpt-4.1-nano-A,dx-addressee-2-third,0shot,free,0.783,23,0.279,0.086,,
or-openai/gpt-4.1-nano-A,sb-k1,0shot,all,0.917,24,0.125,0.056,,
or-openai/gpt-4.1-nano-A,an-narr,0shot,all,0.542,24,0.250,0.102,,
or-openai/gpt-4.1-nano-A,an-other-any,0shot,all,0.750,24,0.250,0.088,,
or-openai/gpt-4.1-nano-A,an-me-any,0shot,all,0.042,24,0.250,0.041,1.0,
or-openai/gpt-4.1-nano-A,an-you-voc,0shot,all,0.125,24,0.250,0.068,1.0,
or-openai/gpt-4.1-nano-A,an-you-novoc,0shot,all,0.125,24,0.250,0.068,0.0,
or-openai/gpt-4.1-nano-A,adj-adj,0shot,all,0.560,25,0.333,0.099,,
or-openai/gpt-4.1-nano-A,adj-voc,0shot,all,0.800,25,0.333,0.080,,
or-openai/gpt-4.1-nano-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-nano-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-4.1-nano-A,cm-rebind,0shot,all,0.850,20,0.200,0.080,,
or-openai/gpt-4.1-nano-A,cm-copy,0shot,all,0.650,20,0.200,0.107,,
or-openai/gpt-4.1-A,attr,0shot,all,0.933,30,0.161,0.045,,
or-openai/gpt-4.1-A,codevar,0shot,all,0.933,30,0.274,0.045,,
or-openai/gpt-4.1-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-openai/gpt-4.1-A,dx-owner-3-deictic,0shot,all,0.929,28,0.333,0.049,,other=0.071
or-openai/gpt-4.1-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-4.1-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-4.1-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-4.1-A,dx-self-3-deictic,0shot,all,0.889,27,0.284,0.060,,other_participant=0.111
or-openai/gpt-4.1-A,dx-self-3-deictic,0shot,free,0.842,19,0.281,0.084,,other_participant=0.158
or-openai/gpt-4.1-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-4.1-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-4.1-A,dx-self-2-deictic,0shot,all,0.515,33,0.280,0.087,,other=0.424;other_object=0.03;other_participant=0.03
or-openai/gpt-4.1-A,dx-self-2-deictic,0shot,free,0.565,23,0.275,0.103,,other=0.435
or-openai/gpt-4.1-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-4.1-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-4.1-A,dx-addressee-3-deictic,0shot,all,0.939,33,0.283,0.042,,speaker_confusion=0.061
or-openai/gpt-4.1-A,dx-addressee-3-deictic,0shot,free,0.909,22,0.280,0.061,,speaker_confusion=0.091
or-openai/gpt-4.1-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-openai/gpt-4.1-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-openai/gpt-4.1-A,dx-addressee-2-deictic,0shot,all,0.593,27,0.293,0.095,,other_participant=0.074;speaker_confusion=0.333
or-openai/gpt-4.1-A,dx-addressee-2-deictic,0shot,free,0.667,18,0.292,0.111,,speaker_confusion=0.333
or-openai/gpt-4.1-A,dx-addressee-2-third,0shot,all,0.939,33,0.285,0.042,,
or-openai/gpt-4.1-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-openai/gpt-4.1-A,sb-k1,0shot,all,0.875,24,0.125,0.068,,
or-openai/gpt-4.1-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-4.1-A,an-other-any,0shot,all,0.917,24,0.250,0.056,,
or-openai/gpt-4.1-A,an-me-any,0shot,all,0.667,24,0.250,0.096,1.0,
or-openai/gpt-4.1-A,an-you-voc,0shot,all,0.833,24,0.250,0.076,1.0,
or-openai/gpt-4.1-A,an-you-novoc,0shot,all,0.875,24,0.250,0.068,0.238,
or-openai/gpt-4.1-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4.1-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-4.1-A,cm-rebind,0shot,all,0.900,20,0.200,0.067,,
or-openai/gpt-4.1-A,cm-copy,0shot,all,0.850,20,0.200,0.080,,
or-openai/gpt-4o-mini-A,attr,0shot,all,0.633,30,0.161,0.088,,
or-openai/gpt-4o-mini-A,codevar,0shot,all,0.833,30,0.274,0.068,,
or-openai/gpt-4o-mini-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-openai/gpt-4o-mini-A,dx-owner-3-deictic,0shot,all,0.607,28,0.333,0.092,,format=0.107;other=0.286
or-openai/gpt-4o-mini-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-openai/gpt-4o-mini-A,dx-owner-3-third,0shot,all,0.893,28,0.333,0.059,,
or-openai/gpt-4o-mini-A,dx-owner-3-third,0shot,free,0.750,8,0.333,0.153,,
or-openai/gpt-4o-mini-A,dx-self-3-deictic,0shot,all,0.259,27,0.284,0.084,,other_participant=0.741
or-openai/gpt-4o-mini-A,dx-self-3-deictic,0shot,free,0.368,19,0.281,0.111,,other_participant=0.632
or-openai/gpt-4o-mini-A,dx-self-3-third,0shot,all,0.938,32,0.286,0.043,,
or-openai/gpt-4o-mini-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-openai/gpt-4o-mini-A,dx-self-2-deictic,0shot,all,0.364,33,0.280,0.084,,format=0.091;other=0.515;other_object=0.03
or-openai/gpt-4o-mini-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,format=0.087;other=0.391
or-openai/gpt-4o-mini-A,dx-self-2-third,0shot,all,0.857,28,0.295,0.066,,
or-openai/gpt-4o-mini-A,dx-self-2-third,0shot,free,0.842,19,0.289,0.084,,
or-openai/gpt-4o-mini-A,dx-addressee-3-deictic,0shot,all,0.455,33,0.283,0.087,,other_participant=0.121;speaker_confusion=0.424
or-openai/gpt-4o-mini-A,dx-addressee-3-deictic,0shot,free,0.636,22,0.280,0.103,,other_participant=0.091;speaker_confusion=0.273
or-openai/gpt-4o-mini-A,dx-addressee-3-third,0shot,all,0.889,27,0.287,0.060,,
or-openai/gpt-4o-mini-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-openai/gpt-4o-mini-A,dx-addressee-2-deictic,0shot,all,0.444,27,0.293,0.096,,speaker_confusion=0.556
or-openai/gpt-4o-mini-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,speaker_confusion=0.389
or-openai/gpt-4o-mini-A,dx-addressee-2-third,0shot,all,0.818,33,0.285,0.067,,
or-openai/gpt-4o-mini-A,dx-addressee-2-third,0shot,free,0.783,23,0.279,0.086,,
or-openai/gpt-4o-mini-A,sb-k1,0shot,all,0.792,24,0.125,0.083,,
or-openai/gpt-4o-mini-A,an-narr,0shot,all,0.333,24,0.250,0.096,,
or-openai/gpt-4o-mini-A,an-other-any,0shot,all,0.708,24,0.250,0.093,,
or-openai/gpt-4o-mini-A,an-me-any,0shot,all,0.083,24,0.250,0.056,1.0,
or-openai/gpt-4o-mini-A,an-you-voc,0shot,all,0.500,24,0.250,0.102,1.0,
or-openai/gpt-4o-mini-A,an-you-novoc,0shot,all,0.167,24,0.250,0.076,0.25,
or-openai/gpt-4o-mini-A,adj-adj,0shot,all,0.720,25,0.333,0.090,,
or-openai/gpt-4o-mini-A,adj-voc,0shot,all,0.920,25,0.333,0.054,,
or-openai/gpt-4o-mini-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-4o-mini-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-4o-mini-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-openai/gpt-4o-mini-A,cm-copy,0shot,all,0.300,20,0.200,0.102,,
or-openai/gpt-5-mini-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5-mini-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5-mini-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5-mini-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5-mini-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-openai/gpt-5-mini-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-openai/gpt-5-mini-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-5-mini-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-5-mini-B,dx-self-2-deictic,0shot,all,0.909,33,0.280,0.050,,other_object=0.091
or-openai/gpt-5-mini-B,dx-self-2-deictic,0shot,free,0.870,23,0.275,0.070,,other_object=0.13
or-openai/gpt-5-mini-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-5-mini-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-2-deictic,0shot,all,0.963,27,0.293,0.036,,other_participant=0.037
or-openai/gpt-5-mini-B,dx-addressee-2-deictic,0shot,free,0.944,18,0.292,0.054,,other_participant=0.056
or-openai/gpt-5-mini-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-openai/gpt-5-mini-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-openai/gpt-5-mini-B,an-narr,0shot,all,0.833,24,0.250,0.076,,
or-openai/gpt-5-mini-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5-mini-B,an-me-any,0shot,all,0.792,24,0.250,0.083,0.947,
or-openai/gpt-5-mini-B,an-you-voc,0shot,all,0.917,24,0.250,0.056,0.955,
or-openai/gpt-5-mini-B,an-you-novoc,0shot,all,0.958,24,0.250,0.041,0.0,
or-openai/gpt-5-mini-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-mini-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-mini-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-mini-A,attr,0shot,all,0.700,30,0.161,0.084,,
or-openai/gpt-5-mini-A,codevar,0shot,all,0.800,30,0.274,0.073,,
or-openai/gpt-5-mini-A,dialogue,0shot,all,0.833,30,0.311,0.068,,
or-openai/gpt-5-mini-A,dx-owner-3-deictic,0shot,all,0.571,28,0.333,0.093,,other=0.429
or-openai/gpt-5-mini-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-openai/gpt-5-mini-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5-mini-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5-mini-A,dx-self-3-deictic,0shot,all,0.222,27,0.284,0.080,,format=0.074;other_participant=0.704
or-openai/gpt-5-mini-A,dx-self-3-deictic,0shot,free,0.210,19,0.281,0.093,,other_participant=0.789
or-openai/gpt-5-mini-A,dx-self-3-third,0shot,all,0.875,32,0.286,0.059,,
or-openai/gpt-5-mini-A,dx-self-3-third,0shot,free,0.955,22,0.284,0.044,,
or-openai/gpt-5-mini-A,dx-self-2-deictic,0shot,all,0.303,33,0.280,0.080,,format=0.061;other=0.576;other_object=0.061
or-openai/gpt-5-mini-A,dx-self-2-deictic,0shot,free,0.348,23,0.275,0.099,,format=0.087;other=0.522;other_object=0.043
or-openai/gpt-5-mini-A,dx-self-2-third,0shot,all,0.643,28,0.295,0.091,,
or-openai/gpt-5-mini-A,dx-self-2-third,0shot,free,0.789,19,0.289,0.093,,
or-openai/gpt-5-mini-A,dx-addressee-3-deictic,0shot,all,0.333,33,0.283,0.082,,other_participant=0.212;speaker_confusion=0.455
or-openai/gpt-5-mini-A,dx-addressee-3-deictic,0shot,free,0.409,22,0.280,0.105,,other_participant=0.227;speaker_confusion=0.364
or-openai/gpt-5-mini-A,dx-addressee-3-third,0shot,all,0.704,27,0.287,0.088,,
or-openai/gpt-5-mini-A,dx-addressee-3-third,0shot,free,0.789,19,0.285,0.093,,
or-openai/gpt-5-mini-A,dx-addressee-2-deictic,0shot,all,0.296,27,0.293,0.088,,format=0.037;other_participant=0.074;speaker_confusion=0.593
or-openai/gpt-5-mini-A,dx-addressee-2-deictic,0shot,free,0.333,18,0.292,0.111,,format=0.056;speaker_confusion=0.611
or-openai/gpt-5-mini-A,dx-addressee-2-third,0shot,all,0.667,33,0.285,0.082,,
or-openai/gpt-5-mini-A,dx-addressee-2-third,0shot,free,0.826,23,0.279,0.079,,
or-openai/gpt-5-mini-A,sb-k1,0shot,all,0.958,24,0.125,0.041,,
or-openai/gpt-5-mini-A,an-narr,0shot,all,0.458,24,0.250,0.102,,
or-openai/gpt-5-mini-A,an-other-any,0shot,all,0.833,24,0.250,0.076,,
or-openai/gpt-5-mini-A,an-me-any,0shot,all,0.250,24,0.250,0.088,0.333,
or-openai/gpt-5-mini-A,an-you-voc,0shot,all,0.708,24,0.250,0.093,0.294,
or-openai/gpt-5-mini-A,an-you-novoc,0shot,all,0.708,24,0.250,0.093,0.0,
or-openai/gpt-5-mini-A,adj-adj,0shot,all,0.800,25,0.333,0.080,,
or-openai/gpt-5-mini-A,adj-voc,0shot,all,0.920,25,0.333,0.054,,
or-openai/gpt-5-mini-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-mini-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-5-mini-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-openai/gpt-5-mini-A,cm-copy,0shot,all,0.850,20,0.200,0.080,,
or-openai/gpt-5-nano-B,dx-owner-3-deictic,0shot,all,0.964,28,0.333,0.035,,other=0.036
or-openai/gpt-5-nano-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5-nano-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5-nano-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5-nano-B,dx-self-3-deictic,0shot,all,0.926,27,0.284,0.050,,format=0.037;other_participant=0.037
or-openai/gpt-5-nano-B,dx-self-3-deictic,0shot,free,0.947,19,0.281,0.051,,format=0.053
or-openai/gpt-5-nano-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-5-nano-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-5-nano-B,dx-self-2-deictic,0shot,all,0.758,33,0.280,0.075,,format=0.121;other=0.03;other_object=0.061;other_participant=0.03
or-openai/gpt-5-nano-B,dx-self-2-deictic,0shot,free,0.652,23,0.275,0.099,,format=0.174;other=0.043;other_object=0.087;other_participant=0.043
or-openai/gpt-5-nano-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-5-nano-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-5-nano-B,dx-addressee-3-deictic,0shot,all,0.818,33,0.283,0.067,,other_participant=0.091;speaker_confusion=0.091
or-openai/gpt-5-nano-B,dx-addressee-3-deictic,0shot,free,0.773,22,0.280,0.089,,other_participant=0.136;speaker_confusion=0.091
or-openai/gpt-5-nano-B,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-openai/gpt-5-nano-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-openai/gpt-5-nano-B,dx-addressee-2-deictic,0shot,all,0.741,27,0.293,0.084,,format=0.037;other_participant=0.037;speaker_confusion=0.185
or-openai/gpt-5-nano-B,dx-addressee-2-deictic,0shot,free,0.722,18,0.292,0.106,,format=0.056;speaker_confusion=0.222
or-openai/gpt-5-nano-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-openai/gpt-5-nano-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-openai/gpt-5-nano-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5-nano-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5-nano-B,an-me-any,0shot,all,0.958,24,0.250,0.041,0.261,
or-openai/gpt-5-nano-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.417,
or-openai/gpt-5-nano-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-openai/gpt-5-nano-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-nano-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-nano-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5-nano-A,attr,0shot,all,0.167,30,0.161,0.068,,
or-openai/gpt-5-nano-A,codevar,0shot,all,0.467,30,0.274,0.091,,
or-openai/gpt-5-nano-A,dialogue,0shot,all,0.667,30,0.311,0.086,,
or-openai/gpt-5-nano-A,dx-owner-3-deictic,0shot,all,0.357,28,0.333,0.091,,format=0.321;other=0.321
or-openai/gpt-5-nano-A,dx-owner-3-deictic,0shot,free,0.375,8,0.333,0.171,,format=0.25;other=0.375
or-openai/gpt-5-nano-A,dx-owner-3-third,0shot,all,0.679,28,0.333,0.088,,
or-openai/gpt-5-nano-A,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-openai/gpt-5-nano-A,dx-self-3-deictic,0shot,all,0.111,27,0.284,0.060,,format=0.333;other_participant=0.556
or-openai/gpt-5-nano-A,dx-self-3-deictic,0shot,free,0.105,19,0.281,0.070,,format=0.474;other_participant=0.421
or-openai/gpt-5-nano-A,dx-self-3-third,0shot,all,0.500,32,0.286,0.088,,
or-openai/gpt-5-nano-A,dx-self-3-third,0shot,free,0.636,22,0.284,0.103,,
or-openai/gpt-5-nano-A,dx-self-2-deictic,0shot,all,0.242,33,0.280,0.075,,format=0.273;other=0.394;other_object=0.061;other_participant=0.03
or-openai/gpt-5-nano-A,dx-self-2-deictic,0shot,free,0.348,23,0.275,0.099,,format=0.261;other=0.304;other_object=0.043;other_participant=0.043
or-openai/gpt-5-nano-A,dx-self-2-third,0shot,all,0.464,28,0.295,0.094,,
or-openai/gpt-5-nano-A,dx-self-2-third,0shot,free,0.632,19,0.289,0.111,,
or-openai/gpt-5-nano-A,dx-addressee-3-deictic,0shot,all,0.151,33,0.283,0.062,,format=0.303;other_participant=0.152;speaker_confusion=0.394
or-openai/gpt-5-nano-A,dx-addressee-3-deictic,0shot,free,0.182,22,0.280,0.082,,format=0.273;other_participant=0.227;speaker_confusion=0.318
or-openai/gpt-5-nano-A,dx-addressee-3-third,0shot,all,0.370,27,0.287,0.093,,
or-openai/gpt-5-nano-A,dx-addressee-3-third,0shot,free,0.526,19,0.285,0.115,,
or-openai/gpt-5-nano-A,dx-addressee-2-deictic,0shot,all,0.037,27,0.293,0.036,,format=0.259;other_participant=0.111;speaker_confusion=0.593
or-openai/gpt-5-nano-A,dx-addressee-2-deictic,0shot,free,0.056,18,0.292,0.054,,format=0.167;other_participant=0.111;speaker_confusion=0.667
or-openai/gpt-5-nano-A,dx-addressee-2-third,0shot,all,0.394,33,0.285,0.085,,
or-openai/gpt-5-nano-A,dx-addressee-2-third,0shot,free,0.565,23,0.279,0.103,,
or-openai/gpt-5-nano-A,sb-k1,0shot,all,0.208,24,0.125,0.083,,
or-openai/gpt-5-nano-A,an-narr,0shot,all,0.250,24,0.250,0.088,,
or-openai/gpt-5-nano-A,an-other-any,0shot,all,0.625,24,0.250,0.099,,
or-openai/gpt-5-nano-A,an-me-any,0shot,all,0.125,24,0.250,0.068,0.0,
or-openai/gpt-5-nano-A,an-you-voc,0shot,all,0.292,24,0.250,0.093,0.286,
or-openai/gpt-5-nano-A,an-you-novoc,0shot,all,0.167,24,0.250,0.076,0.0,
or-openai/gpt-5-nano-A,adj-adj,0shot,all,0.440,25,0.333,0.099,,
or-openai/gpt-5-nano-A,adj-voc,0shot,all,0.360,25,0.333,0.096,,
or-openai/gpt-5-nano-A,adj-third,0shot,all,0.600,25,0.333,0.098,,
or-openai/gpt-5-nano-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-5-nano-A,cm-rebind,0shot,all,0.550,20,0.200,0.111,,
or-openai/gpt-5-nano-A,cm-copy,0shot,all,0.300,20,0.200,0.102,,
or-openai/gpt-5.5-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5.5-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5.5-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5.5-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5.5-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-openai/gpt-5.5-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-openai/gpt-5.5-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-5.5-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-5.5-B,dx-self-2-deictic,0shot,all,0.970,33,0.280,0.030,,other_object=0.03
or-openai/gpt-5.5-B,dx-self-2-deictic,0shot,free,0.957,23,0.275,0.043,,other_object=0.043
or-openai/gpt-5.5-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-5.5-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-5.5-B,dx-addressee-3-deictic,0shot,all,1.000,33,0.283,0.000,,
or-openai/gpt-5.5-B,dx-addressee-3-deictic,0shot,free,1.000,22,0.280,0.000,,
or-openai/gpt-5.5-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-openai/gpt-5.5-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-openai/gpt-5.5-B,dx-addressee-2-deictic,0shot,all,1.000,27,0.293,0.000,,
or-openai/gpt-5.5-B,dx-addressee-2-deictic,0shot,free,1.000,18,0.292,0.000,,
or-openai/gpt-5.5-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-openai/gpt-5.5-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-openai/gpt-5.5-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5.5-B,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5.5-B,an-me-any,0shot,all,1.000,24,0.250,0.000,1.0,
or-openai/gpt-5.5-B,an-you-voc,0shot,all,1.000,24,0.250,0.000,1.0,
or-openai/gpt-5.5-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-openai/gpt-5.5-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-openai/gpt-5.5-A,codevar,0shot,all,0.967,30,0.274,0.033,,
or-openai/gpt-5.5-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-openai/gpt-5.5-A,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5.5-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5.5-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-5.5-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-5.5-A,dx-self-3-deictic,0shot,all,0.963,27,0.284,0.036,,other_participant=0.037
or-openai/gpt-5.5-A,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-openai/gpt-5.5-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-5.5-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-5.5-A,dx-self-2-deictic,0shot,all,0.758,33,0.280,0.075,,other=0.152;other_object=0.091
or-openai/gpt-5.5-A,dx-self-2-deictic,0shot,free,0.783,23,0.275,0.086,,other=0.087;other_object=0.13
or-openai/gpt-5.5-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-5.5-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-5.5-A,dx-addressee-3-deictic,0shot,all,0.818,33,0.283,0.067,,other_participant=0.03;speaker_confusion=0.152
or-openai/gpt-5.5-A,dx-addressee-3-deictic,0shot,free,0.818,22,0.280,0.082,,speaker_confusion=0.182
or-openai/gpt-5.5-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-openai/gpt-5.5-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-openai/gpt-5.5-A,dx-addressee-2-deictic,0shot,all,0.481,27,0.293,0.096,,other_participant=0.222;speaker_confusion=0.296
or-openai/gpt-5.5-A,dx-addressee-2-deictic,0shot,free,0.333,18,0.292,0.111,,other_participant=0.222;speaker_confusion=0.444
or-openai/gpt-5.5-A,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-openai/gpt-5.5-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-openai/gpt-5.5-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-openai/gpt-5.5-A,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5.5-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-openai/gpt-5.5-A,an-me-any,0shot,all,1.000,24,0.250,0.000,0.958,
or-openai/gpt-5.5-A,an-you-voc,0shot,all,0.958,24,0.250,0.041,0.87,
or-openai/gpt-5.5-A,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-openai/gpt-5.5-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-5.5-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-5.5-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-openai/gpt-5.5-A,cm-copy,0shot,all,1.000,20,0.200,0.000,,
or-x-ai/grok-4.3-B,dx-owner-3-deictic,0shot,all,0.929,28,0.333,0.049,,format=0.071
or-x-ai/grok-4.3-B,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,format=0.25
or-x-ai/grok-4.3-B,dx-owner-3-third,0shot,all,0.964,28,0.333,0.035,,
or-x-ai/grok-4.3-B,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-x-ai/grok-4.3-B,dx-self-3-deictic,0shot,all,0.926,27,0.284,0.050,,format=0.074
or-x-ai/grok-4.3-B,dx-self-3-deictic,0shot,free,0.895,19,0.281,0.070,,format=0.105
or-x-ai/grok-4.3-B,dx-self-3-third,0shot,all,0.969,32,0.286,0.031,,
or-x-ai/grok-4.3-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-x-ai/grok-4.3-B,dx-self-2-deictic,0shot,all,0.939,33,0.280,0.042,,format=0.061
or-x-ai/grok-4.3-B,dx-self-2-deictic,0shot,free,0.913,23,0.275,0.059,,format=0.087
or-x-ai/grok-4.3-B,dx-self-2-third,0shot,all,0.964,28,0.295,0.035,,
or-x-ai/grok-4.3-B,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-x-ai/grok-4.3-B,dx-addressee-3-deictic,0shot,all,0.909,33,0.283,0.050,,format=0.091
or-x-ai/grok-4.3-B,dx-addressee-3-deictic,0shot,free,0.909,22,0.280,0.061,,format=0.091
or-x-ai/grok-4.3-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-x-ai/grok-4.3-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-x-ai/grok-4.3-B,dx-addressee-2-deictic,0shot,all,0.889,27,0.293,0.060,,format=0.111
or-x-ai/grok-4.3-B,dx-addressee-2-deictic,0shot,free,0.833,18,0.292,0.088,,format=0.167
or-x-ai/grok-4.3-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-x-ai/grok-4.3-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-x-ai/grok-4.3-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-x-ai/grok-4.3-B,an-other-any,0shot,all,0.875,24,0.250,0.068,,
or-x-ai/grok-4.3-B,an-me-any,0shot,all,0.917,24,0.250,0.056,1.0,
or-x-ai/grok-4.3-B,an-you-voc,0shot,all,0.833,24,0.250,0.076,1.0,
or-x-ai/grok-4.3-B,an-you-novoc,0shot,all,0.875,24,0.250,0.068,0.048,
or-x-ai/grok-4.3-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-x-ai/grok-4.3-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-x-ai/grok-4.3-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-x-ai/grok-4.3-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-x-ai/grok-4.3-A,codevar,0shot,all,0.767,30,0.274,0.077,,
or-x-ai/grok-4.3-A,dialogue,0shot,all,0.967,30,0.311,0.033,,
or-x-ai/grok-4.3-A,dx-owner-3-deictic,0shot,all,0.607,28,0.333,0.092,,other=0.393
or-x-ai/grok-4.3-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-x-ai/grok-4.3-A,dx-owner-3-third,0shot,all,0.929,28,0.333,0.049,,
or-x-ai/grok-4.3-A,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-x-ai/grok-4.3-A,dx-self-3-deictic,0shot,all,0.370,27,0.284,0.093,,other_participant=0.63
or-x-ai/grok-4.3-A,dx-self-3-deictic,0shot,free,0.210,19,0.281,0.093,,other_participant=0.789
or-x-ai/grok-4.3-A,dx-self-3-third,0shot,all,0.906,32,0.286,0.051,,
or-x-ai/grok-4.3-A,dx-self-3-third,0shot,free,0.955,22,0.284,0.044,,
or-x-ai/grok-4.3-A,dx-self-2-deictic,0shot,all,0.485,33,0.280,0.087,,other=0.364;other_object=0.152
or-x-ai/grok-4.3-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,other=0.304;other_object=0.174
or-x-ai/grok-4.3-A,dx-self-2-third,0shot,all,0.964,28,0.295,0.035,,
or-x-ai/grok-4.3-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-x-ai/grok-4.3-A,dx-addressee-3-deictic,0shot,all,0.515,33,0.283,0.087,,other_participant=0.152;speaker_confusion=0.333
or-x-ai/grok-4.3-A,dx-addressee-3-deictic,0shot,free,0.455,22,0.280,0.106,,other_participant=0.227;speaker_confusion=0.318
or-x-ai/grok-4.3-A,dx-addressee-3-third,0shot,all,0.926,27,0.287,0.050,,
or-x-ai/grok-4.3-A,dx-addressee-3-third,0shot,free,0.895,19,0.285,0.070,,
or-x-ai/grok-4.3-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,other_participant=0.111;speaker_confusion=0.556
or-x-ai/grok-4.3-A,dx-addressee-2-deictic,0shot,free,0.333,18,0.292,0.111,,other_participant=0.167;speaker_confusion=0.5
or-x-ai/grok-4.3-A,dx-addressee-2-third,0shot,all,0.818,33,0.285,0.067,,
or-x-ai/grok-4.3-A,dx-addressee-2-third,0shot,free,0.870,23,0.279,0.070,,
or-x-ai/grok-4.3-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-x-ai/grok-4.3-A,an-narr,0shot,all,0.583,24,0.250,0.101,,
or-x-ai/grok-4.3-A,an-other-any,0shot,all,0.750,24,0.250,0.088,,
or-x-ai/grok-4.3-A,an-me-any,0shot,all,0.000,24,0.250,0.000,,
or-x-ai/grok-4.3-A,an-you-voc,0shot,all,0.667,24,0.250,0.096,0.562,
or-x-ai/grok-4.3-A,an-you-novoc,0shot,all,0.375,24,0.250,0.099,0.0,
or-x-ai/grok-4.3-A,adj-adj,0shot,all,0.840,25,0.333,0.073,,
or-x-ai/grok-4.3-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-x-ai/grok-4.3-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-x-ai/grok-4.3-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-x-ai/grok-4.3-A,cm-rebind,0shot,all,0.900,20,0.200,0.067,,
or-x-ai/grok-4.3-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-deepseek/deepseek-v3.2-B,dx-owner-3-deictic,0shot,all,0.857,28,0.333,0.066,,format=0.107;other=0.036
or-deepseek/deepseek-v3.2-B,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,format=0.25
or-deepseek/deepseek-v3.2-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-deepseek/deepseek-v3.2-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-3-deictic,0shot,all,1.000,27,0.284,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-3-deictic,0shot,free,1.000,19,0.281,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-2-deictic,0shot,all,0.909,33,0.280,0.050,,format=0.03;other_object=0.03;other_participant=0.03
or-deepseek/deepseek-v3.2-B,dx-self-2-deictic,0shot,free,0.870,23,0.275,0.070,,format=0.043;other_object=0.043;other_participant=0.043
or-deepseek/deepseek-v3.2-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-deepseek/deepseek-v3.2-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-deepseek/deepseek-v3.2-B,dx-addressee-3-deictic,0shot,all,0.818,33,0.283,0.067,,format=0.121;speaker_confusion=0.061
or-deepseek/deepseek-v3.2-B,dx-addressee-3-deictic,0shot,free,0.773,22,0.280,0.089,,format=0.136;speaker_confusion=0.091
or-deepseek/deepseek-v3.2-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-deepseek/deepseek-v3.2-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-deepseek/deepseek-v3.2-B,dx-addressee-2-deictic,0shot,all,0.704,27,0.293,0.088,,format=0.222;other_participant=0.074
or-deepseek/deepseek-v3.2-B,dx-addressee-2-deictic,0shot,free,0.556,18,0.292,0.117,,format=0.333;other_participant=0.111
or-deepseek/deepseek-v3.2-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-deepseek/deepseek-v3.2-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-deepseek/deepseek-v3.2-B,an-narr,0shot,all,0.958,24,0.250,0.041,,
or-deepseek/deepseek-v3.2-B,an-other-any,0shot,all,0.958,24,0.250,0.041,,
or-deepseek/deepseek-v3.2-B,an-me-any,0shot,all,0.958,24,0.250,0.041,0.826,
or-deepseek/deepseek-v3.2-B,an-you-voc,0shot,all,0.750,24,0.250,0.088,0.444,
or-deepseek/deepseek-v3.2-B,an-you-novoc,0shot,all,0.750,24,0.250,0.088,0.167,
or-deepseek/deepseek-v3.2-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-deepseek/deepseek-v3.2-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-deepseek/deepseek-v3.2-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-deepseek/deepseek-v3.2-A,attr,0shot,all,0.667,27,0.162,0.091,,
or-deepseek/deepseek-v3.2-A,codevar,0shot,all,0.909,22,0.280,0.061,,
or-deepseek/deepseek-v3.2-A,dialogue,0shot,all,1.000,21,0.309,0.000,,
or-deepseek/deepseek-v3.2-A,dx-owner-3-deictic,0shot,all,0.700,20,0.333,0.102,,other=0.3
or-deepseek/deepseek-v3.2-A,dx-owner-3-third,0shot,all,1.000,20,0.333,0.000,,
or-deepseek/deepseek-v3.2-A,dx-self-3-deictic,0shot,all,0.522,23,0.283,0.104,,other_participant=0.478
or-deepseek/deepseek-v3.2-A,dx-self-3-deictic,0shot,free,0.706,17,0.279,0.111,,other_participant=0.294
or-deepseek/deepseek-v3.2-A,dx-self-3-third,0shot,all,0.818,22,0.280,0.082,,
or-deepseek/deepseek-v3.2-A,dx-self-3-third,0shot,free,1.000,15,0.278,0.000,,
or-deepseek/deepseek-v3.2-A,dx-self-2-deictic,0shot,all,0.393,28,0.283,0.092,,other=0.5;other_participant=0.107
or-deepseek/deepseek-v3.2-A,dx-self-2-deictic,0shot,free,0.474,19,0.276,0.115,,other=0.421;other_participant=0.105
or-deepseek/deepseek-v3.2-A,dx-self-2-third,0shot,all,0.889,18,0.292,0.074,,
or-deepseek/deepseek-v3.2-A,dx-self-2-third,0shot,free,1.000,13,0.295,0.000,,
or-deepseek/deepseek-v3.2-A,dx-addressee-3-deictic,0shot,all,0.522,23,0.279,0.104,,other_participant=0.217;speaker_confusion=0.261
or-deepseek/deepseek-v3.2-A,dx-addressee-3-deictic,0shot,free,0.667,15,0.278,0.122,,other_participant=0.133;speaker_confusion=0.2
or-deepseek/deepseek-v3.2-A,dx-addressee-3-third,0shot,all,0.938,16,0.292,0.060,,
or-deepseek/deepseek-v3.2-A,dx-addressee-3-third,0shot,free,1.000,10,0.283,0.000,,
or-deepseek/deepseek-v3.2-A,dx-addressee-2-deictic,0shot,all,0.210,19,0.289,0.093,,other_participant=0.105;speaker_confusion=0.684
or-deepseek/deepseek-v3.2-A,dx-addressee-2-deictic,0shot,free,0.182,11,0.288,0.116,,other_participant=0.182;speaker_confusion=0.636
or-deepseek/deepseek-v3.2-A,dx-addressee-2-third,0shot,all,0.783,23,0.290,0.086,,
or-deepseek/deepseek-v3.2-A,dx-addressee-2-third,0shot,free,0.929,14,0.280,0.069,,
or-deepseek/deepseek-v3.2-A,sb-k1,0shot,all,0.824,17,0.125,0.092,,
or-deepseek/deepseek-v3.2-A,an-narr,0shot,all,0.524,21,0.250,0.109,,
or-deepseek/deepseek-v3.2-A,an-other-any,0shot,all,0.933,15,0.250,0.064,,
or-deepseek/deepseek-v3.2-A,an-me-any,0shot,all,0.056,18,0.250,0.054,1.0,
or-deepseek/deepseek-v3.2-A,an-you-voc,0shot,all,0.588,17,0.250,0.119,1.0,
or-deepseek/deepseek-v3.2-A,an-you-novoc,0shot,all,0.333,15,0.250,0.122,0.0,
or-deepseek/deepseek-v3.2-A,adj-adj,0shot,all,1.000,17,0.333,0.000,,
or-deepseek/deepseek-v3.2-A,adj-voc,0shot,all,1.000,16,0.333,0.000,,
or-deepseek/deepseek-v3.2-A,adj-third,0shot,all,1.000,19,0.333,0.000,,
or-deepseek/deepseek-v3.2-A,cm-mutate,0shot,all,1.000,16,0.200,0.000,,
or-deepseek/deepseek-v3.2-A,cm-rebind,0shot,all,1.000,13,0.200,0.000,,
or-deepseek/deepseek-v3.2-A,cm-copy,0shot,all,1.000,11,0.200,0.000,,
or-google/gemma-3-12b-it-A,attr,0shot,all,0.667,30,0.161,0.086,,
or-google/gemma-3-12b-it-A,codevar,0shot,all,0.767,30,0.274,0.077,,
or-google/gemma-3-12b-it-A,dialogue,0shot,all,0.800,30,0.311,0.073,,
or-google/gemma-3-12b-it-A,dx-owner-3-deictic,0shot,all,0.464,28,0.333,0.094,,other=0.536
or-google/gemma-3-12b-it-A,dx-owner-3-deictic,0shot,free,0.375,8,0.333,0.171,,other=0.625
or-google/gemma-3-12b-it-A,dx-owner-3-third,0shot,all,0.893,28,0.333,0.059,,
or-google/gemma-3-12b-it-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-google/gemma-3-12b-it-A,dx-self-3-deictic,0shot,all,0.148,27,0.284,0.068,,format=0.074;other_participant=0.778
or-google/gemma-3-12b-it-A,dx-self-3-deictic,0shot,free,0.210,19,0.281,0.093,,format=0.105;other_participant=0.684
or-google/gemma-3-12b-it-A,dx-self-3-third,0shot,all,0.875,32,0.286,0.059,,
or-google/gemma-3-12b-it-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-google/gemma-3-12b-it-A,dx-self-2-deictic,0shot,all,0.242,33,0.280,0.075,,format=0.061;other=0.636;other_object=0.061
or-google/gemma-3-12b-it-A,dx-self-2-deictic,0shot,free,0.348,23,0.275,0.099,,format=0.043;other=0.565;other_object=0.043
or-google/gemma-3-12b-it-A,dx-self-2-third,0shot,all,0.679,28,0.295,0.088,,
or-google/gemma-3-12b-it-A,dx-self-2-third,0shot,free,0.789,19,0.289,0.093,,
or-google/gemma-3-12b-it-A,dx-addressee-3-deictic,0shot,all,0.424,33,0.283,0.086,,other_participant=0.121;speaker_confusion=0.455
or-google/gemma-3-12b-it-A,dx-addressee-3-deictic,0shot,free,0.636,22,0.280,0.103,,other_participant=0.091;speaker_confusion=0.273
or-google/gemma-3-12b-it-A,dx-addressee-3-third,0shot,all,0.852,27,0.287,0.068,,
or-google/gemma-3-12b-it-A,dx-addressee-3-third,0shot,free,0.789,19,0.285,0.093,,
or-google/gemma-3-12b-it-A,dx-addressee-2-deictic,0shot,all,0.296,27,0.293,0.088,,format=0.037;other_participant=0.037;speaker_confusion=0.63
or-google/gemma-3-12b-it-A,dx-addressee-2-deictic,0shot,free,0.444,18,0.292,0.117,,format=0.056;other_participant=0.056;speaker_confusion=0.444
or-google/gemma-3-12b-it-A,dx-addressee-2-third,0shot,all,0.576,33,0.285,0.086,,
or-google/gemma-3-12b-it-A,dx-addressee-2-third,0shot,free,0.652,23,0.279,0.099,,
or-google/gemma-3-12b-it-A,sb-k1,0shot,all,0.958,24,0.125,0.041,,
or-google/gemma-3-12b-it-A,an-narr,0shot,all,0.333,24,0.250,0.096,,
or-google/gemma-3-12b-it-A,an-other-any,0shot,all,0.500,24,0.250,0.102,,
or-google/gemma-3-12b-it-A,an-me-any,0shot,all,0.000,24,0.250,0.000,,
or-google/gemma-3-12b-it-A,an-you-voc,0shot,all,0.250,24,0.250,0.088,1.0,
or-google/gemma-3-12b-it-A,an-you-novoc,0shot,all,0.000,24,0.250,0.000,,
or-google/gemma-3-12b-it-A,adj-adj,0shot,all,0.840,25,0.333,0.073,,
or-google/gemma-3-12b-it-A,adj-voc,0shot,all,0.880,25,0.333,0.065,,
or-google/gemma-3-12b-it-A,adj-third,0shot,all,0.880,25,0.333,0.065,,
or-google/gemma-3-12b-it-A,cm-mutate,0shot,all,0.200,20,0.200,0.089,,
or-google/gemma-3-12b-it-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-google/gemma-3-12b-it-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-google/gemma-3-27b-it-A,attr,0shot,all,0.567,30,0.161,0.090,,
or-google/gemma-3-27b-it-A,codevar,0shot,all,0.700,30,0.274,0.084,,
or-google/gemma-3-27b-it-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-google/gemma-3-27b-it-A,dx-owner-3-deictic,0shot,all,0.607,28,0.333,0.092,,other=0.393
or-google/gemma-3-27b-it-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-google/gemma-3-27b-it-A,dx-owner-3-third,0shot,all,0.786,28,0.333,0.077,,
or-google/gemma-3-27b-it-A,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-google/gemma-3-27b-it-A,dx-self-3-deictic,0shot,all,0.370,27,0.284,0.093,,format=0.185;other_participant=0.444
or-google/gemma-3-27b-it-A,dx-self-3-deictic,0shot,free,0.474,19,0.281,0.115,,format=0.211;other_participant=0.316
or-google/gemma-3-27b-it-A,dx-self-3-third,0shot,all,0.844,32,0.286,0.064,,
or-google/gemma-3-27b-it-A,dx-self-3-third,0shot,free,0.864,22,0.284,0.073,,
or-google/gemma-3-27b-it-A,dx-self-2-deictic,0shot,all,0.242,33,0.280,0.075,,format=0.242;other=0.515
or-google/gemma-3-27b-it-A,dx-self-2-deictic,0shot,free,0.348,23,0.275,0.099,,format=0.217;other=0.435
or-google/gemma-3-27b-it-A,dx-self-2-third,0shot,all,0.714,28,0.295,0.085,,
or-google/gemma-3-27b-it-A,dx-self-2-third,0shot,free,0.895,19,0.289,0.070,,
or-google/gemma-3-27b-it-A,dx-addressee-3-deictic,0shot,all,0.394,33,0.283,0.085,,other_participant=0.303;speaker_confusion=0.303
or-google/gemma-3-27b-it-A,dx-addressee-3-deictic,0shot,free,0.545,22,0.280,0.106,,other_participant=0.182;speaker_confusion=0.273
or-google/gemma-3-27b-it-A,dx-addressee-3-third,0shot,all,0.741,27,0.287,0.084,,
or-google/gemma-3-27b-it-A,dx-addressee-3-third,0shot,free,0.789,19,0.285,0.093,,
or-google/gemma-3-27b-it-A,dx-addressee-2-deictic,0shot,all,0.259,27,0.293,0.084,,format=0.037;speaker_confusion=0.704
or-google/gemma-3-27b-it-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,format=0.056;speaker_confusion=0.556
or-google/gemma-3-27b-it-A,dx-addressee-2-third,0shot,all,0.636,33,0.285,0.084,,
or-google/gemma-3-27b-it-A,dx-addressee-2-third,0shot,free,0.783,23,0.279,0.086,,
or-google/gemma-3-27b-it-A,sb-k1,0shot,all,0.583,24,0.125,0.101,,
or-google/gemma-3-27b-it-A,an-narr,0shot,all,0.500,24,0.250,0.102,,
or-google/gemma-3-27b-it-A,an-other-any,0shot,all,0.792,24,0.250,0.083,,
or-google/gemma-3-27b-it-A,an-me-any,0shot,all,0.042,24,0.250,0.041,1.0,
or-google/gemma-3-27b-it-A,an-you-voc,0shot,all,0.417,24,0.250,0.101,1.0,
or-google/gemma-3-27b-it-A,an-you-novoc,0shot,all,0.333,24,0.250,0.096,0.5,
or-google/gemma-3-27b-it-A,adj-adj,0shot,all,0.920,25,0.333,0.054,,
or-google/gemma-3-27b-it-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-google/gemma-3-27b-it-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-google/gemma-3-27b-it-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-google/gemma-3-27b-it-A,cm-rebind,0shot,all,0.900,20,0.200,0.067,,
or-google/gemma-3-27b-it-A,cm-copy,0shot,all,0.450,20,0.200,0.111,,
or-google/gemma-3-4b-it-A,attr,0shot,all,0.867,30,0.161,0.062,,
or-google/gemma-3-4b-it-A,codevar,0shot,all,0.533,30,0.274,0.091,,
or-google/gemma-3-4b-it-A,dialogue,0shot,all,0.333,30,0.311,0.086,,
or-google/gemma-3-4b-it-A,dx-owner-3-deictic,0shot,all,0.500,28,0.333,0.095,,format=0.036;other=0.464
or-google/gemma-3-4b-it-A,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-google/gemma-3-4b-it-A,dx-owner-3-third,0shot,all,0.357,28,0.333,0.091,,
or-google/gemma-3-4b-it-A,dx-owner-3-third,0shot,free,0.750,8,0.333,0.153,,
or-google/gemma-3-4b-it-A,dx-self-3-deictic,0shot,all,0.259,27,0.284,0.084,,format=0.148;other_participant=0.593
or-google/gemma-3-4b-it-A,dx-self-3-deictic,0shot,free,0.368,19,0.281,0.111,,format=0.158;other_participant=0.474
or-google/gemma-3-4b-it-A,dx-self-3-third,0shot,all,0.562,32,0.286,0.088,,
or-google/gemma-3-4b-it-A,dx-self-3-third,0shot,free,0.682,22,0.284,0.099,,
or-google/gemma-3-4b-it-A,dx-self-2-deictic,0shot,all,0.303,33,0.280,0.080,,format=0.303;other=0.364;other_object=0.03
or-google/gemma-3-4b-it-A,dx-self-2-deictic,0shot,free,0.435,23,0.275,0.103,,format=0.217;other=0.304;other_object=0.043
or-google/gemma-3-4b-it-A,dx-self-2-third,0shot,all,0.464,28,0.295,0.094,,
or-google/gemma-3-4b-it-A,dx-self-2-third,0shot,free,0.684,19,0.289,0.107,,
or-google/gemma-3-4b-it-A,dx-addressee-3-deictic,0shot,all,0.303,33,0.283,0.080,,format=0.03;other_participant=0.182;speaker_confusion=0.485
or-google/gemma-3-4b-it-A,dx-addressee-3-deictic,0shot,free,0.409,22,0.280,0.105,,other_participant=0.091;speaker_confusion=0.5
or-google/gemma-3-4b-it-A,dx-addressee-3-third,0shot,all,0.444,27,0.287,0.096,,
or-google/gemma-3-4b-it-A,dx-addressee-3-third,0shot,free,0.526,19,0.285,0.115,,
or-google/gemma-3-4b-it-A,dx-addressee-2-deictic,0shot,all,0.259,27,0.293,0.084,,format=0.074;other_participant=0.037;speaker_confusion=0.63
or-google/gemma-3-4b-it-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,format=0.111;speaker_confusion=0.5
or-google/gemma-3-4b-it-A,dx-addressee-2-third,0shot,all,0.545,33,0.285,0.087,,
or-google/gemma-3-4b-it-A,dx-addressee-2-third,0shot,free,0.696,23,0.279,0.096,,
or-google/gemma-3-4b-it-A,sb-k1,0shot,all,0.875,24,0.125,0.068,,
or-google/gemma-3-4b-it-A,an-narr,0shot,all,0.500,24,0.250,0.102,,
or-google/gemma-3-4b-it-A,an-other-any,0shot,all,0.250,24,0.250,0.088,,
or-google/gemma-3-4b-it-A,an-me-any,0shot,all,0.083,24,0.250,0.056,1.0,
or-google/gemma-3-4b-it-A,an-you-voc,0shot,all,0.833,24,0.250,0.076,0.95,
or-google/gemma-3-4b-it-A,an-you-novoc,0shot,all,0.000,24,0.250,0.000,,
or-google/gemma-3-4b-it-A,adj-adj,0shot,all,0.680,25,0.333,0.093,,
or-google/gemma-3-4b-it-A,adj-voc,0shot,all,0.800,25,0.333,0.080,,
or-google/gemma-3-4b-it-A,adj-third,0shot,all,0.520,25,0.333,0.100,,
or-google/gemma-3-4b-it-A,cm-mutate,0shot,all,0.900,20,0.200,0.067,,
or-google/gemma-3-4b-it-A,cm-rebind,0shot,all,0.000,20,0.200,0.000,,
or-google/gemma-3-4b-it-A,cm-copy,0shot,all,0.000,20,0.200,0.000,,
or-z-ai/glm-4.6-B,dx-owner-3-deictic,0shot,all,0.964,28,0.333,0.035,,format=0.036
or-z-ai/glm-4.6-B,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,format=0.125
or-z-ai/glm-4.6-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-z-ai/glm-4.6-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-z-ai/glm-4.6-B,dx-self-3-deictic,0shot,all,0.963,27,0.284,0.036,,format=0.037
or-z-ai/glm-4.6-B,dx-self-3-deictic,0shot,free,0.947,19,0.281,0.051,,format=0.053
or-z-ai/glm-4.6-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-z-ai/glm-4.6-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-z-ai/glm-4.6-B,dx-self-2-deictic,0shot,all,0.939,33,0.280,0.042,,format=0.061
or-z-ai/glm-4.6-B,dx-self-2-deictic,0shot,free,0.957,23,0.275,0.043,,format=0.043
or-z-ai/glm-4.6-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-z-ai/glm-4.6-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-z-ai/glm-4.6-B,dx-addressee-3-deictic,0shot,all,0.970,33,0.283,0.030,,format=0.03
or-z-ai/glm-4.6-B,dx-addressee-3-deictic,0shot,free,0.955,22,0.280,0.044,,format=0.045
or-z-ai/glm-4.6-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-z-ai/glm-4.6-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-z-ai/glm-4.6-B,dx-addressee-2-deictic,0shot,all,0.926,27,0.293,0.050,,format=0.037;other_participant=0.037
or-z-ai/glm-4.6-B,dx-addressee-2-deictic,0shot,free,0.889,18,0.292,0.074,,format=0.056;other_participant=0.056
or-z-ai/glm-4.6-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-z-ai/glm-4.6-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-z-ai/glm-4.6-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-z-ai/glm-4.6-B,an-other-any,0shot,all,0.958,24,0.250,0.041,,
or-z-ai/glm-4.6-B,an-me-any,0shot,all,1.000,24,0.250,0.000,0.917,
or-z-ai/glm-4.6-B,an-you-voc,0shot,all,0.917,24,0.250,0.056,0.591,
or-z-ai/glm-4.6-B,an-you-novoc,0shot,all,1.000,24,0.250,0.000,0.0,
or-z-ai/glm-4.6-B,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-z-ai/glm-4.6-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-z-ai/glm-4.6-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-z-ai/glm-4.6-A,attr,0shot,all,0.967,30,0.161,0.033,,
or-z-ai/glm-4.6-A,codevar,0shot,all,0.900,30,0.274,0.055,,
or-z-ai/glm-4.6-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-z-ai/glm-4.6-A,dx-owner-3-deictic,0shot,all,0.857,28,0.333,0.066,,other=0.143
or-z-ai/glm-4.6-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-z-ai/glm-4.6-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-z-ai/glm-4.6-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-z-ai/glm-4.6-A,dx-self-3-deictic,0shot,all,0.704,27,0.284,0.088,,other_participant=0.296
or-z-ai/glm-4.6-A,dx-self-3-deictic,0shot,free,0.737,19,0.281,0.101,,other_participant=0.263
or-z-ai/glm-4.6-A,dx-self-3-third,0shot,all,0.875,32,0.286,0.059,,
or-z-ai/glm-4.6-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-z-ai/glm-4.6-A,dx-self-2-deictic,0shot,all,0.485,33,0.280,0.087,,other=0.333;other_object=0.091;other_participant=0.091
or-z-ai/glm-4.6-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,other=0.217;other_object=0.13;other_participant=0.13
or-z-ai/glm-4.6-A,dx-self-2-third,0shot,all,0.786,28,0.295,0.077,,
or-z-ai/glm-4.6-A,dx-self-2-third,0shot,free,0.895,19,0.289,0.070,,
or-z-ai/glm-4.6-A,dx-addressee-3-deictic,0shot,all,0.606,33,0.283,0.085,,other_participant=0.121;speaker_confusion=0.273
or-z-ai/glm-4.6-A,dx-addressee-3-deictic,0shot,free,0.545,22,0.280,0.106,,other_participant=0.136;speaker_confusion=0.318
or-z-ai/glm-4.6-A,dx-addressee-3-third,0shot,all,0.926,27,0.287,0.050,,
or-z-ai/glm-4.6-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-z-ai/glm-4.6-A,dx-addressee-2-deictic,0shot,all,0.556,27,0.293,0.096,,other_participant=0.148;speaker_confusion=0.296
or-z-ai/glm-4.6-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,other_participant=0.167;speaker_confusion=0.222
or-z-ai/glm-4.6-A,dx-addressee-2-third,0shot,all,0.818,33,0.285,0.067,,
or-z-ai/glm-4.6-A,dx-addressee-2-third,0shot,free,0.913,23,0.279,0.059,,
or-z-ai/glm-4.6-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-z-ai/glm-4.6-A,an-narr,0shot,all,0.792,24,0.250,0.083,,
or-z-ai/glm-4.6-A,an-other-any,0shot,all,0.875,24,0.250,0.068,,
or-z-ai/glm-4.6-A,an-me-any,0shot,all,0.542,24,0.250,0.102,1.0,
or-z-ai/glm-4.6-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,1.0,
or-z-ai/glm-4.6-A,an-you-novoc,0shot,all,0.542,24,0.250,0.102,0.231,
or-z-ai/glm-4.6-A,adj-adj,0shot,all,0.920,25,0.333,0.054,,
or-z-ai/glm-4.6-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-z-ai/glm-4.6-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-z-ai/glm-4.6-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-z-ai/glm-4.6-A,cm-rebind,0shot,all,0.850,20,0.200,0.080,,
or-z-ai/glm-4.6-A,cm-copy,0shot,all,0.950,20,0.200,0.049,,
or-openai/gpt-oss-120b-B,dx-owner-3-deictic,0shot,all,0.929,28,0.333,0.049,,other=0.071
or-openai/gpt-oss-120b-B,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-openai/gpt-oss-120b-B,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-openai/gpt-oss-120b-B,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-openai/gpt-oss-120b-B,dx-self-3-deictic,0shot,all,0.963,27,0.284,0.036,,other_participant=0.037
or-openai/gpt-oss-120b-B,dx-self-3-deictic,0shot,free,0.947,19,0.281,0.051,,other_participant=0.053
or-openai/gpt-oss-120b-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-openai/gpt-oss-120b-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-openai/gpt-oss-120b-B,dx-self-2-deictic,0shot,all,0.879,33,0.280,0.057,,format=0.03;other=0.061;other_participant=0.03
or-openai/gpt-oss-120b-B,dx-self-2-deictic,0shot,free,0.870,23,0.275,0.070,,format=0.043;other=0.087
or-openai/gpt-oss-120b-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-openai/gpt-oss-120b-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-openai/gpt-oss-120b-B,dx-addressee-3-deictic,0shot,all,0.879,33,0.283,0.057,,other_participant=0.091;speaker_confusion=0.03
or-openai/gpt-oss-120b-B,dx-addressee-3-deictic,0shot,free,0.864,22,0.280,0.073,,other_participant=0.091;speaker_confusion=0.045
or-openai/gpt-oss-120b-B,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-openai/gpt-oss-120b-B,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-openai/gpt-oss-120b-B,dx-addressee-2-deictic,0shot,all,0.556,27,0.293,0.096,,format=0.037;other_participant=0.222;speaker_confusion=0.185
or-openai/gpt-oss-120b-B,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,format=0.056;other_participant=0.278;speaker_confusion=0.278
or-openai/gpt-oss-120b-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-openai/gpt-oss-120b-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-openai/gpt-oss-120b-B,an-narr,0shot,all,0.833,24,0.250,0.076,,
or-openai/gpt-oss-120b-B,an-other-any,0shot,all,0.875,24,0.250,0.068,,
or-openai/gpt-oss-120b-B,an-me-any,0shot,all,0.542,24,0.250,0.102,0.462,
or-openai/gpt-oss-120b-B,an-you-voc,0shot,all,0.875,24,0.250,0.068,0.905,
or-openai/gpt-oss-120b-B,an-you-novoc,0shot,all,0.833,24,0.250,0.076,0.05,
or-openai/gpt-oss-120b-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-oss-120b-B,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-oss-120b-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-openai/gpt-oss-120b-A,attr,0shot,all,0.250,16,0.164,0.108,,
or-openai/gpt-oss-120b-A,codevar,0shot,all,0.000,13,0.258,0.000,,
or-openai/gpt-oss-120b-A,dialogue,0shot,all,0.444,9,0.306,0.166,,
or-openai/gpt-oss-120b-A,dx-owner-3-deictic,0shot,all,0.167,12,0.333,0.108,,format=0.5;other=0.333
or-openai/gpt-oss-120b-A,dx-owner-3-third,0shot,all,0.556,9,0.333,0.166,,
or-openai/gpt-oss-120b-A,dx-self-3-deictic,0shot,all,0.250,8,0.281,0.153,,format=0.625;other_participant=0.125
or-openai/gpt-oss-120b-A,dx-self-3-third,0shot,all,0.500,14,0.274,0.134,,
or-openai/gpt-oss-120b-A,dx-self-3-third,0shot,free,0.375,8,0.271,0.171,,
or-openai/gpt-oss-120b-A,dx-self-2-deictic,0shot,all,0.083,12,0.278,0.080,,format=0.667;other=0.25
or-openai/gpt-oss-120b-A,dx-self-2-deictic,0shot,free,0.000,9,0.278,0.000,,format=0.778;other=0.222
or-openai/gpt-oss-120b-A,dx-self-2-third,0shot,all,0.545,11,0.311,0.150,,
or-openai/gpt-oss-120b-A,dx-self-2-third,0shot,free,0.545,11,0.311,0.150,,
or-openai/gpt-oss-120b-A,dx-addressee-3-deictic,0shot,all,0.111,18,0.287,0.074,,format=0.611;other_participant=0.111;speaker_confusion=0.167
or-openai/gpt-oss-120b-A,dx-addressee-3-deictic,0shot,free,0.182,11,0.288,0.116,,format=0.545;other_participant=0.182;speaker_confusion=0.091
or-openai/gpt-oss-120b-A,dx-addressee-3-third,0shot,all,0.692,13,0.276,0.128,,
or-openai/gpt-oss-120b-A,dx-addressee-3-third,0shot,free,0.600,10,0.275,0.155,,
or-openai/gpt-oss-120b-A,dx-addressee-2-deictic,0shot,all,0.100,10,0.292,0.095,,format=0.8;speaker_confusion=0.1
or-openai/gpt-oss-120b-A,dx-addressee-2-third,0shot,all,0.450,20,0.279,0.111,,
or-openai/gpt-oss-120b-A,dx-addressee-2-third,0shot,free,0.267,15,0.272,0.114,,
or-openai/gpt-oss-120b-A,sb-k1,0shot,all,0.333,9,0.125,0.157,,
or-openai/gpt-oss-120b-A,an-narr,0shot,all,0.167,12,0.250,0.108,,
or-openai/gpt-oss-120b-A,an-other-any,0shot,all,0.000,11,0.250,0.000,,
or-openai/gpt-oss-120b-A,an-me-any,0shot,all,0.000,9,0.250,0.000,,
or-openai/gpt-oss-120b-A,an-you-voc,0shot,all,0.375,8,0.250,0.171,1.0,
or-openai/gpt-oss-120b-A,an-you-novoc,0shot,all,0.125,8,0.250,0.117,1.0,
or-openai/gpt-oss-120b-A,adj-adj,0shot,all,0.100,10,0.333,0.095,,
or-openai/gpt-oss-120b-A,adj-voc,0shot,all,0.111,9,0.333,0.105,,
or-openai/gpt-oss-120b-A,adj-third,0shot,all,0.455,11,0.333,0.150,,
or-openai/gpt-oss-120b-A,cm-mutate,0shot,all,0.500,8,0.200,0.177,,
or-openai/gpt-oss-120b-A,cm-rebind,0shot,all,0.375,8,0.200,0.171,,
or-openai/gpt-oss-120b-A,cm-copy,0shot,all,0.100,10,0.200,0.095,,
or-nousresearch/hermes-3-llama-3.1-405b-A,attr,0shot,all,0.933,30,0.161,0.045,,
or-nousresearch/hermes-3-llama-3.1-405b-A,codevar,0shot,all,0.700,30,0.274,0.084,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-owner-3-deictic,0shot,all,0.786,28,0.333,0.077,,other=0.214
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-owner-3-third,0shot,all,0.893,28,0.333,0.059,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-owner-3-third,0shot,free,0.625,8,0.333,0.171,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-3-deictic,0shot,all,0.667,27,0.284,0.091,,other_participant=0.333
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-3-deictic,0shot,free,0.737,19,0.281,0.101,,other_participant=0.263
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-2-deictic,0shot,all,0.455,33,0.280,0.087,,other=0.333;other_object=0.182;other_participant=0.03
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,other=0.174;other_object=0.261;other_participant=0.043
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-2-third,0shot,all,0.929,28,0.295,0.049,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-3-deictic,0shot,all,0.697,33,0.283,0.080,,other_participant=0.061;speaker_confusion=0.242
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-3-deictic,0shot,free,0.818,22,0.280,0.082,,other_participant=0.045;speaker_confusion=0.136
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-2-deictic,0shot,all,0.370,27,0.293,0.093,,other_participant=0.259;speaker_confusion=0.37
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-2-deictic,0shot,free,0.333,18,0.292,0.111,,other_participant=0.278;speaker_confusion=0.389
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-2-third,0shot,all,0.909,33,0.285,0.050,,
or-nousresearch/hermes-3-llama-3.1-405b-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-nousresearch/hermes-3-llama-3.1-405b-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,an-narr,0shot,all,0.750,24,0.250,0.088,,
or-nousresearch/hermes-3-llama-3.1-405b-A,an-other-any,0shot,all,0.917,24,0.250,0.056,,
or-nousresearch/hermes-3-llama-3.1-405b-A,an-me-any,0shot,all,0.458,24,0.250,0.102,0.909,
or-nousresearch/hermes-3-llama-3.1-405b-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,1.0,
or-nousresearch/hermes-3-llama-3.1-405b-A,an-you-novoc,0shot,all,0.375,24,0.250,0.099,0.0,
or-nousresearch/hermes-3-llama-3.1-405b-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-nousresearch/hermes-3-llama-3.1-405b-A,cm-mutate,0shot,all,0.900,20,0.200,0.067,,
or-nousresearch/hermes-3-llama-3.1-405b-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-nousresearch/hermes-3-llama-3.1-405b-A,cm-copy,0shot,all,1.000,20,0.200,0.000,,
or-moonshotai/kimi-k2-0905-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-moonshotai/kimi-k2-0905-A,codevar,0shot,all,0.933,30,0.274,0.045,,
or-moonshotai/kimi-k2-0905-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-owner-3-deictic,0shot,all,0.893,28,0.333,0.059,,other=0.107
or-moonshotai/kimi-k2-0905-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-self-3-deictic,0shot,all,0.615,26,0.285,0.095,,other_participant=0.385
or-moonshotai/kimi-k2-0905-A,dx-self-3-deictic,0shot,free,0.579,19,0.281,0.113,,other_participant=0.421
or-moonshotai/kimi-k2-0905-A,dx-self-3-third,0shot,all,1.000,31,0.285,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-self-2-deictic,0shot,all,0.484,31,0.282,0.090,,other=0.484;other_object=0.032
or-moonshotai/kimi-k2-0905-A,dx-self-2-deictic,0shot,free,0.500,22,0.277,0.107,,other=0.455;other_object=0.045
or-moonshotai/kimi-k2-0905-A,dx-self-2-third,0shot,all,0.893,28,0.295,0.059,,
or-moonshotai/kimi-k2-0905-A,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-moonshotai/kimi-k2-0905-A,dx-addressee-3-deictic,0shot,all,0.667,33,0.283,0.082,,other_participant=0.121;speaker_confusion=0.212
or-moonshotai/kimi-k2-0905-A,dx-addressee-3-deictic,0shot,free,0.727,22,0.280,0.095,,other_participant=0.091;speaker_confusion=0.182
or-moonshotai/kimi-k2-0905-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-moonshotai/kimi-k2-0905-A,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-moonshotai/kimi-k2-0905-A,dx-addressee-2-deictic,0shot,all,0.423,26,0.292,0.097,,other_participant=0.038;speaker_confusion=0.538
or-moonshotai/kimi-k2-0905-A,dx-addressee-2-deictic,0shot,free,0.588,17,0.289,0.119,,speaker_confusion=0.412
or-moonshotai/kimi-k2-0905-A,dx-addressee-2-third,0shot,all,0.970,33,0.285,0.030,,
or-moonshotai/kimi-k2-0905-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-moonshotai/kimi-k2-0905-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-moonshotai/kimi-k2-0905-A,an-narr,0shot,all,0.875,24,0.250,0.068,,
or-moonshotai/kimi-k2-0905-A,an-other-any,0shot,all,0.958,24,0.250,0.041,,
or-moonshotai/kimi-k2-0905-A,an-me-any,0shot,all,0.250,24,0.250,0.088,1.0,
or-moonshotai/kimi-k2-0905-A,an-you-voc,0shot,all,0.958,24,0.250,0.041,1.0,
or-moonshotai/kimi-k2-0905-A,an-you-novoc,0shot,all,0.792,24,0.250,0.083,0.0,
or-moonshotai/kimi-k2-0905-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-moonshotai/kimi-k2-0905-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-moonshotai/kimi-k2-0905-A,cm-rebind,0shot,all,1.000,20,0.200,0.000,,
or-moonshotai/kimi-k2-0905-A,cm-copy,0shot,all,0.950,20,0.200,0.049,,
or-meta-llama/llama-3.1-8b-instruct-A,attr,0shot,all,0.733,30,0.161,0.081,,
or-meta-llama/llama-3.1-8b-instruct-A,codevar,0shot,all,0.367,30,0.274,0.088,,
or-meta-llama/llama-3.1-8b-instruct-A,dialogue,0shot,all,0.900,30,0.311,0.055,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-owner-3-deictic,0shot,all,0.607,28,0.333,0.092,,other=0.393
or-meta-llama/llama-3.1-8b-instruct-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-meta-llama/llama-3.1-8b-instruct-A,dx-owner-3-third,0shot,all,0.786,28,0.333,0.077,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-3-deictic,0shot,all,0.444,27,0.284,0.096,,other_participant=0.556
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-3-deictic,0shot,free,0.526,19,0.281,0.115,,other_participant=0.474
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-3-third,0shot,all,0.688,32,0.286,0.082,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-3-third,0shot,free,0.727,22,0.284,0.095,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-2-deictic,0shot,all,0.333,33,0.280,0.082,,other=0.545;other_object=0.061;other_participant=0.061
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-2-deictic,0shot,free,0.435,23,0.275,0.103,,other=0.391;other_object=0.087;other_participant=0.087
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-2-third,0shot,all,0.571,28,0.295,0.093,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-self-2-third,0shot,free,0.737,19,0.289,0.101,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-3-deictic,0shot,all,0.303,33,0.283,0.080,,other_participant=0.242;speaker_confusion=0.455
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-3-deictic,0shot,free,0.318,22,0.280,0.099,,other_participant=0.273;speaker_confusion=0.409
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-3-third,0shot,all,0.667,27,0.287,0.091,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-3-third,0shot,free,0.684,19,0.285,0.107,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,other_participant=0.148;speaker_confusion=0.519
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-2-deictic,0shot,free,0.333,18,0.292,0.111,,other_participant=0.167;speaker_confusion=0.5
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-2-third,0shot,all,0.606,33,0.285,0.085,,
or-meta-llama/llama-3.1-8b-instruct-A,dx-addressee-2-third,0shot,free,0.696,23,0.279,0.096,,
or-meta-llama/llama-3.1-8b-instruct-A,sb-k1,0shot,all,0.917,24,0.125,0.056,,
or-meta-llama/llama-3.1-8b-instruct-A,an-narr,0shot,all,0.458,24,0.250,0.102,,
or-meta-llama/llama-3.1-8b-instruct-A,an-other-any,0shot,all,0.667,24,0.250,0.096,,
or-meta-llama/llama-3.1-8b-instruct-A,an-me-any,0shot,all,0.000,24,0.250,0.000,,
or-meta-llama/llama-3.1-8b-instruct-A,an-you-voc,0shot,all,0.167,24,0.250,0.076,1.0,
or-meta-llama/llama-3.1-8b-instruct-A,an-you-novoc,0shot,all,0.125,24,0.250,0.068,0.0,
or-meta-llama/llama-3.1-8b-instruct-A,adj-adj,0shot,all,0.840,25,0.333,0.073,,
or-meta-llama/llama-3.1-8b-instruct-A,adj-voc,0shot,all,0.800,25,0.333,0.080,,
or-meta-llama/llama-3.1-8b-instruct-A,adj-third,0shot,all,0.920,25,0.333,0.054,,
or-meta-llama/llama-3.1-8b-instruct-A,cm-mutate,0shot,all,0.650,20,0.200,0.107,,
or-meta-llama/llama-3.1-8b-instruct-A,cm-rebind,0shot,all,0.550,20,0.200,0.111,,
or-meta-llama/llama-3.1-8b-instruct-A,cm-copy,0shot,all,0.150,20,0.200,0.080,,
or-meta-llama/llama-3.2-1b-instruct-A,attr,0shot,all,0.767,30,0.161,0.077,,
or-meta-llama/llama-3.2-1b-instruct-A,codevar,0shot,all,0.133,30,0.274,0.062,,
or-meta-llama/llama-3.2-1b-instruct-A,dialogue,0shot,all,0.500,30,0.311,0.091,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-owner-3-deictic,0shot,all,0.214,28,0.333,0.077,,format=0.143;other=0.643
or-meta-llama/llama-3.2-1b-instruct-A,dx-owner-3-deictic,0shot,free,0.125,8,0.333,0.117,,format=0.125;other=0.75
or-meta-llama/llama-3.2-1b-instruct-A,dx-owner-3-third,0shot,all,0.321,28,0.333,0.088,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-owner-3-third,0shot,free,0.375,8,0.333,0.171,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-3-deictic,0shot,all,0.222,27,0.284,0.080,,format=0.259;other_participant=0.519
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-3-deictic,0shot,free,0.210,19,0.281,0.093,,format=0.263;other_participant=0.526
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-3-third,0shot,all,0.406,32,0.286,0.087,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-3-third,0shot,free,0.545,22,0.284,0.106,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-2-deictic,0shot,all,0.151,33,0.280,0.062,,format=0.121;other=0.515;other_object=0.091;other_participant=0.121
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-2-deictic,0shot,free,0.174,23,0.275,0.079,,format=0.174;other=0.391;other_object=0.087;other_participant=0.174
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-2-third,0shot,all,0.429,28,0.295,0.093,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-self-2-third,0shot,free,0.579,19,0.289,0.113,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-3-deictic,0shot,all,0.182,33,0.283,0.067,,format=0.152;other_participant=0.242;speaker_confusion=0.424
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-3-deictic,0shot,free,0.136,22,0.280,0.073,,format=0.136;other_participant=0.273;speaker_confusion=0.455
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-3-third,0shot,all,0.259,27,0.287,0.084,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-3-third,0shot,free,0.368,19,0.285,0.111,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-2-deictic,0shot,all,0.185,27,0.293,0.075,,format=0.259;other_participant=0.222;speaker_confusion=0.333
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-2-deictic,0shot,free,0.167,18,0.292,0.088,,format=0.222;other_participant=0.222;speaker_confusion=0.389
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-2-third,0shot,all,0.333,33,0.285,0.082,,
or-meta-llama/llama-3.2-1b-instruct-A,dx-addressee-2-third,0shot,free,0.435,23,0.279,0.103,,
or-meta-llama/llama-3.2-1b-instruct-A,sb-k1,0shot,all,0.542,24,0.125,0.102,,
or-meta-llama/llama-3.2-1b-instruct-A,an-narr,0shot,all,0.375,24,0.250,0.099,,
or-meta-llama/llama-3.2-1b-instruct-A,an-other-any,0shot,all,0.292,24,0.250,0.093,,
or-meta-llama/llama-3.2-1b-instruct-A,an-me-any,0shot,all,0.083,24,0.250,0.056,0.0,
or-meta-llama/llama-3.2-1b-instruct-A,an-you-voc,0shot,all,0.250,24,0.250,0.088,0.0,
or-meta-llama/llama-3.2-1b-instruct-A,an-you-novoc,0shot,all,0.250,24,0.250,0.088,0.0,
or-meta-llama/llama-3.2-1b-instruct-A,adj-adj,0shot,all,0.360,25,0.333,0.096,,
or-meta-llama/llama-3.2-1b-instruct-A,adj-voc,0shot,all,0.520,25,0.333,0.100,,
or-meta-llama/llama-3.2-1b-instruct-A,adj-third,0shot,all,0.640,25,0.333,0.096,,
or-meta-llama/llama-3.2-1b-instruct-A,cm-mutate,0shot,all,0.550,20,0.200,0.111,,
or-meta-llama/llama-3.2-1b-instruct-A,cm-rebind,0shot,all,0.000,20,0.200,0.000,,
or-meta-llama/llama-3.2-1b-instruct-A,cm-copy,0shot,all,0.050,20,0.200,0.049,,
or-meta-llama/llama-3.2-3b-instruct-A,attr,0shot,all,0.500,30,0.161,0.091,,
or-meta-llama/llama-3.2-3b-instruct-A,codevar,0shot,all,0.467,30,0.274,0.091,,
or-meta-llama/llama-3.2-3b-instruct-A,dialogue,0shot,all,0.800,30,0.311,0.073,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-owner-3-deictic,0shot,all,0.714,28,0.333,0.085,,other=0.286
or-meta-llama/llama-3.2-3b-instruct-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-meta-llama/llama-3.2-3b-instruct-A,dx-owner-3-third,0shot,all,0.536,28,0.333,0.094,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-owner-3-third,0shot,free,0.375,8,0.333,0.171,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-3-deictic,0shot,all,0.407,27,0.284,0.095,,format=0.037;other_participant=0.556
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-3-deictic,0shot,free,0.526,19,0.281,0.115,,format=0.053;other_participant=0.421
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-3-third,0shot,all,0.625,32,0.286,0.086,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-3-third,0shot,free,0.636,22,0.284,0.103,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-2-deictic,0shot,all,0.273,33,0.280,0.077,,other=0.545;other_object=0.152;other_participant=0.03
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-2-deictic,0shot,free,0.348,23,0.275,0.099,,other=0.435;other_object=0.217
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-2-third,0shot,all,0.571,28,0.295,0.093,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-self-2-third,0shot,free,0.632,19,0.289,0.111,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-3-deictic,0shot,all,0.364,33,0.283,0.084,,other_participant=0.212;speaker_confusion=0.424
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-3-deictic,0shot,free,0.409,22,0.280,0.105,,other_participant=0.273;speaker_confusion=0.318
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-3-third,0shot,all,0.593,27,0.287,0.095,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-3-third,0shot,free,0.579,19,0.285,0.113,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-2-deictic,0shot,all,0.259,27,0.293,0.084,,other_participant=0.111;speaker_confusion=0.63
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.556
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-2-third,0shot,all,0.576,33,0.285,0.086,,
or-meta-llama/llama-3.2-3b-instruct-A,dx-addressee-2-third,0shot,free,0.522,23,0.279,0.104,,
or-meta-llama/llama-3.2-3b-instruct-A,sb-k1,0shot,all,0.583,24,0.125,0.101,,
or-meta-llama/llama-3.2-3b-instruct-A,an-narr,0shot,all,0.458,24,0.250,0.102,,
or-meta-llama/llama-3.2-3b-instruct-A,an-other-any,0shot,all,0.458,24,0.250,0.102,,
or-meta-llama/llama-3.2-3b-instruct-A,an-me-any,0shot,all,0.500,24,0.250,0.102,1.0,
or-meta-llama/llama-3.2-3b-instruct-A,an-you-voc,0shot,all,0.000,24,0.250,0.000,,
or-meta-llama/llama-3.2-3b-instruct-A,an-you-novoc,0shot,all,0.125,24,0.250,0.068,0.0,
or-meta-llama/llama-3.2-3b-instruct-A,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-meta-llama/llama-3.2-3b-instruct-A,adj-voc,0shot,all,0.960,25,0.333,0.039,,
or-meta-llama/llama-3.2-3b-instruct-A,adj-third,0shot,all,0.880,25,0.333,0.065,,
or-meta-llama/llama-3.2-3b-instruct-A,cm-mutate,0shot,all,0.850,20,0.200,0.080,,
or-meta-llama/llama-3.2-3b-instruct-A,cm-rebind,0shot,all,0.200,20,0.200,0.089,,
or-meta-llama/llama-3.2-3b-instruct-A,cm-copy,0shot,all,0.100,20,0.200,0.067,,
or-meta-llama/llama-3.3-70b-instruct-A,attr,0shot,all,0.933,30,0.161,0.045,,
or-meta-llama/llama-3.3-70b-instruct-A,codevar,0shot,all,0.733,30,0.274,0.081,,
or-meta-llama/llama-3.3-70b-instruct-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-owner-3-deictic,0shot,all,0.786,28,0.333,0.077,,other=0.214
or-meta-llama/llama-3.3-70b-instruct-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-meta-llama/llama-3.3-70b-instruct-A,dx-owner-3-third,0shot,all,0.929,28,0.333,0.049,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-owner-3-third,0shot,free,0.750,8,0.333,0.153,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-3-deictic,0shot,all,0.704,27,0.284,0.088,,format=0.037;other_participant=0.259
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-3-deictic,0shot,free,0.789,19,0.281,0.093,,format=0.053;other_participant=0.158
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-3-third,0shot,all,0.969,32,0.286,0.031,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-2-deictic,0shot,all,0.485,33,0.280,0.087,,other=0.485;other_object=0.03
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,other=0.435;other_object=0.043
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-2-third,0shot,all,0.893,28,0.295,0.059,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-3-deictic,0shot,all,0.727,33,0.283,0.077,,other_participant=0.091;speaker_confusion=0.182
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-3-deictic,0shot,free,0.636,22,0.280,0.103,,other_participant=0.091;speaker_confusion=0.273
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-3-third,0shot,all,0.963,27,0.287,0.036,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-2-deictic,0shot,all,0.444,27,0.293,0.096,,other_participant=0.111;speaker_confusion=0.444
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-2-deictic,0shot,free,0.444,18,0.292,0.117,,other_participant=0.111;speaker_confusion=0.444
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-2-third,0shot,all,0.970,33,0.285,0.030,,
or-meta-llama/llama-3.3-70b-instruct-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-meta-llama/llama-3.3-70b-instruct-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,an-narr,0shot,all,0.875,24,0.250,0.068,,
or-meta-llama/llama-3.3-70b-instruct-A,an-other-any,0shot,all,0.917,24,0.250,0.056,,
or-meta-llama/llama-3.3-70b-instruct-A,an-me-any,0shot,all,0.250,24,0.250,0.088,1.0,
or-meta-llama/llama-3.3-70b-instruct-A,an-you-voc,0shot,all,0.542,24,0.250,0.102,1.0,
or-meta-llama/llama-3.3-70b-instruct-A,an-you-novoc,0shot,all,0.417,24,0.250,0.101,0.0,
or-meta-llama/llama-3.3-70b-instruct-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-meta-llama/llama-3.3-70b-instruct-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-meta-llama/llama-3.3-70b-instruct-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-meta-llama/llama-4-maverick-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-meta-llama/llama-4-maverick-A,codevar,0shot,all,1.000,30,0.274,0.000,,
or-meta-llama/llama-4-maverick-A,dialogue,0shot,all,1.000,30,0.311,0.000,,
or-meta-llama/llama-4-maverick-A,dx-owner-3-deictic,0shot,all,0.893,28,0.333,0.059,,other=0.107
or-meta-llama/llama-4-maverick-A,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-meta-llama/llama-4-maverick-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-meta-llama/llama-4-maverick-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-meta-llama/llama-4-maverick-A,dx-self-3-deictic,0shot,all,0.778,27,0.284,0.080,,format=0.037;other_participant=0.185
or-meta-llama/llama-4-maverick-A,dx-self-3-deictic,0shot,free,0.895,19,0.281,0.070,,format=0.053;other_participant=0.053
or-meta-llama/llama-4-maverick-A,dx-self-3-third,0shot,all,0.938,32,0.286,0.043,,
or-meta-llama/llama-4-maverick-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-meta-llama/llama-4-maverick-A,dx-self-2-deictic,0shot,all,0.636,33,0.280,0.084,,other=0.333;other_object=0.03
or-meta-llama/llama-4-maverick-A,dx-self-2-deictic,0shot,free,0.652,23,0.275,0.099,,other=0.304;other_object=0.043
or-meta-llama/llama-4-maverick-A,dx-self-2-third,0shot,all,0.964,28,0.295,0.035,,
or-meta-llama/llama-4-maverick-A,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-meta-llama/llama-4-maverick-A,dx-addressee-3-deictic,0shot,all,0.697,33,0.283,0.080,,other_participant=0.091;speaker_confusion=0.212
or-meta-llama/llama-4-maverick-A,dx-addressee-3-deictic,0shot,free,0.727,22,0.280,0.095,,other_participant=0.045;speaker_confusion=0.227
or-meta-llama/llama-4-maverick-A,dx-addressee-3-third,0shot,all,1.000,27,0.287,0.000,,
or-meta-llama/llama-4-maverick-A,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-meta-llama/llama-4-maverick-A,dx-addressee-2-deictic,0shot,all,0.741,27,0.293,0.084,,other_participant=0.074;speaker_confusion=0.185
or-meta-llama/llama-4-maverick-A,dx-addressee-2-deictic,0shot,free,0.833,18,0.292,0.088,,other_participant=0.056;speaker_confusion=0.111
or-meta-llama/llama-4-maverick-A,dx-addressee-2-third,0shot,all,0.939,33,0.285,0.042,,
or-meta-llama/llama-4-maverick-A,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-meta-llama/llama-4-maverick-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-meta-llama/llama-4-maverick-A,an-narr,0shot,all,0.833,24,0.250,0.076,,
or-meta-llama/llama-4-maverick-A,an-other-any,0shot,all,1.000,24,0.250,0.000,,
or-meta-llama/llama-4-maverick-A,an-me-any,0shot,all,0.542,24,0.250,0.102,1.0,
or-meta-llama/llama-4-maverick-A,an-you-voc,0shot,all,0.792,24,0.250,0.083,1.0,
or-meta-llama/llama-4-maverick-A,an-you-novoc,0shot,all,0.833,24,0.250,0.076,0.0,
or-meta-llama/llama-4-maverick-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-4-maverick-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-4-maverick-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-meta-llama/llama-4-maverick-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-meta-llama/llama-4-maverick-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-meta-llama/llama-4-maverick-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-mistralai/ministral-3b-2512-A,attr,0shot,all,0.800,30,0.161,0.073,,
or-mistralai/ministral-3b-2512-A,codevar,0shot,all,0.667,30,0.274,0.086,,
or-mistralai/ministral-3b-2512-A,dialogue,0shot,all,0.700,30,0.311,0.084,,
or-mistralai/ministral-3b-2512-A,dx-owner-3-deictic,0shot,all,0.536,28,0.333,0.094,,other=0.464
or-mistralai/ministral-3b-2512-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-mistralai/ministral-3b-2512-A,dx-owner-3-third,0shot,all,0.786,28,0.333,0.077,,
or-mistralai/ministral-3b-2512-A,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-mistralai/ministral-3b-2512-A,dx-self-3-deictic,0shot,all,0.222,27,0.284,0.080,,format=0.111;other_participant=0.667
or-mistralai/ministral-3b-2512-A,dx-self-3-deictic,0shot,free,0.316,19,0.281,0.107,,format=0.158;other_participant=0.526
or-mistralai/ministral-3b-2512-A,dx-self-3-third,0shot,all,0.688,32,0.286,0.082,,
or-mistralai/ministral-3b-2512-A,dx-self-3-third,0shot,free,0.818,22,0.284,0.082,,
or-mistralai/ministral-3b-2512-A,dx-self-2-deictic,0shot,all,0.394,33,0.280,0.085,,format=0.091;other=0.424;other_object=0.091
or-mistralai/ministral-3b-2512-A,dx-self-2-deictic,0shot,free,0.435,23,0.275,0.103,,format=0.13;other=0.304;other_object=0.13
or-mistralai/ministral-3b-2512-A,dx-self-2-third,0shot,all,0.679,28,0.295,0.088,,
or-mistralai/ministral-3b-2512-A,dx-self-2-third,0shot,free,0.842,19,0.289,0.084,,
or-mistralai/ministral-3b-2512-A,dx-addressee-3-deictic,0shot,all,0.303,33,0.283,0.080,,other_participant=0.212;speaker_confusion=0.485
or-mistralai/ministral-3b-2512-A,dx-addressee-3-deictic,0shot,free,0.409,22,0.280,0.105,,other_participant=0.182;speaker_confusion=0.409
or-mistralai/ministral-3b-2512-A,dx-addressee-3-third,0shot,all,0.593,27,0.287,0.095,,
or-mistralai/ministral-3b-2512-A,dx-addressee-3-third,0shot,free,0.579,19,0.285,0.113,,
or-mistralai/ministral-3b-2512-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,format=0.111;other_participant=0.074;speaker_confusion=0.481
or-mistralai/ministral-3b-2512-A,dx-addressee-2-deictic,0shot,free,0.500,18,0.292,0.118,,format=0.111;other_participant=0.111;speaker_confusion=0.278
or-mistralai/ministral-3b-2512-A,dx-addressee-2-third,0shot,all,0.545,33,0.285,0.087,,
or-mistralai/ministral-3b-2512-A,dx-addressee-2-third,0shot,free,0.739,23,0.279,0.092,,
or-mistralai/ministral-3b-2512-A,sb-k1,0shot,all,0.875,24,0.125,0.068,,
or-mistralai/ministral-3b-2512-A,an-narr,0shot,all,0.500,24,0.250,0.102,,
or-mistralai/ministral-3b-2512-A,an-other-any,0shot,all,0.417,24,0.250,0.101,,
or-mistralai/ministral-3b-2512-A,an-me-any,0shot,all,0.042,24,0.250,0.041,0.0,
or-mistralai/ministral-3b-2512-A,an-you-voc,0shot,all,0.042,24,0.250,0.041,1.0,
or-mistralai/ministral-3b-2512-A,an-you-novoc,0shot,all,0.125,24,0.250,0.068,0.0,
or-mistralai/ministral-3b-2512-A,adj-adj,0shot,all,0.560,25,0.333,0.099,,
or-mistralai/ministral-3b-2512-A,adj-voc,0shot,all,0.680,25,0.333,0.093,,
or-mistralai/ministral-3b-2512-A,adj-third,0shot,all,0.920,25,0.333,0.054,,
or-mistralai/ministral-3b-2512-A,cm-mutate,0shot,all,0.900,20,0.200,0.067,,
or-mistralai/ministral-3b-2512-A,cm-rebind,0shot,all,0.450,20,0.200,0.111,,
or-mistralai/ministral-3b-2512-A,cm-copy,0shot,all,0.350,20,0.200,0.107,,
or-mistralai/ministral-8b-2512-A,attr,0shot,all,0.600,30,0.161,0.089,,
or-mistralai/ministral-8b-2512-A,codevar,0shot,all,0.767,30,0.274,0.077,,
or-mistralai/ministral-8b-2512-A,dialogue,0shot,all,0.733,30,0.311,0.081,,
or-mistralai/ministral-8b-2512-A,dx-owner-3-deictic,0shot,all,0.786,28,0.333,0.077,,other=0.214
or-mistralai/ministral-8b-2512-A,dx-owner-3-deictic,0shot,free,0.875,8,0.333,0.117,,other=0.125
or-mistralai/ministral-8b-2512-A,dx-owner-3-third,0shot,all,0.607,28,0.333,0.092,,
or-mistralai/ministral-8b-2512-A,dx-owner-3-third,0shot,free,0.750,8,0.333,0.153,,
or-mistralai/ministral-8b-2512-A,dx-self-3-deictic,0shot,all,0.222,27,0.284,0.080,,other_participant=0.778
or-mistralai/ministral-8b-2512-A,dx-self-3-deictic,0shot,free,0.263,19,0.281,0.101,,other_participant=0.737
or-mistralai/ministral-8b-2512-A,dx-self-3-third,0shot,all,0.562,32,0.286,0.088,,
or-mistralai/ministral-8b-2512-A,dx-self-3-third,0shot,free,0.591,22,0.284,0.105,,
or-mistralai/ministral-8b-2512-A,dx-self-2-deictic,0shot,all,0.303,33,0.280,0.080,,format=0.091;other=0.576;other_object=0.03
or-mistralai/ministral-8b-2512-A,dx-self-2-deictic,0shot,free,0.391,23,0.275,0.102,,format=0.087;other=0.522
or-mistralai/ministral-8b-2512-A,dx-self-2-third,0shot,all,0.714,28,0.295,0.085,,
or-mistralai/ministral-8b-2512-A,dx-self-2-third,0shot,free,0.737,19,0.289,0.101,,
or-mistralai/ministral-8b-2512-A,dx-addressee-3-deictic,0shot,all,0.424,33,0.283,0.086,,other_participant=0.152;speaker_confusion=0.424
or-mistralai/ministral-8b-2512-A,dx-addressee-3-deictic,0shot,free,0.455,22,0.280,0.106,,other_participant=0.182;speaker_confusion=0.364
or-mistralai/ministral-8b-2512-A,dx-addressee-3-third,0shot,all,0.556,27,0.287,0.096,,
or-mistralai/ministral-8b-2512-A,dx-addressee-3-third,0shot,free,0.526,19,0.285,0.115,,
or-mistralai/ministral-8b-2512-A,dx-addressee-2-deictic,0shot,all,0.259,27,0.293,0.084,,other_participant=0.037;speaker_confusion=0.704
or-mistralai/ministral-8b-2512-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.556
or-mistralai/ministral-8b-2512-A,dx-addressee-2-third,0shot,all,0.455,33,0.285,0.087,,
or-mistralai/ministral-8b-2512-A,dx-addressee-2-third,0shot,free,0.522,23,0.279,0.104,,
or-mistralai/ministral-8b-2512-A,sb-k1,0shot,all,0.875,24,0.125,0.068,,
or-mistralai/ministral-8b-2512-A,an-narr,0shot,all,0.500,24,0.250,0.102,,
or-mistralai/ministral-8b-2512-A,an-other-any,0shot,all,0.625,24,0.250,0.099,,
or-mistralai/ministral-8b-2512-A,an-me-any,0shot,all,0.042,24,0.250,0.041,1.0,
or-mistralai/ministral-8b-2512-A,an-you-voc,0shot,all,0.458,24,0.250,0.102,1.0,
or-mistralai/ministral-8b-2512-A,an-you-novoc,0shot,all,0.000,24,0.250,0.000,,
or-mistralai/ministral-8b-2512-A,adj-adj,0shot,all,0.920,25,0.333,0.054,,
or-mistralai/ministral-8b-2512-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-mistralai/ministral-8b-2512-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-mistralai/ministral-8b-2512-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-mistralai/ministral-8b-2512-A,cm-rebind,0shot,all,0.800,20,0.200,0.089,,
or-mistralai/ministral-8b-2512-A,cm-copy,0shot,all,0.200,20,0.200,0.089,,
or-mistralai/mistral-large-2512-A,attr,0shot,all,0.812,16,0.161,0.098,,
or-mistralai/mistral-large-2512-A,codevar,0shot,all,0.917,12,0.274,0.080,,
or-mistralai/mistral-large-2512-A,dialogue,0shot,all,1.000,15,0.311,0.000,,
or-mistralai/mistral-large-2512-A,dx-owner-3-deictic,0shot,all,0.917,12,0.333,0.080,,other=0.083
or-mistralai/mistral-large-2512-A,dx-self-3-deictic,0shot,all,0.615,13,0.276,0.135,,format=0.077;other_participant=0.308
or-mistralai/mistral-large-2512-A,dx-self-3-deictic,0shot,free,0.500,10,0.283,0.158,,format=0.1;other_participant=0.4
or-mistralai/mistral-large-2512-A,dx-self-3-third,0shot,all,1.000,19,0.281,0.000,,
or-mistralai/mistral-large-2512-A,dx-self-3-third,0shot,free,1.000,11,0.273,0.000,,
or-mistralai/mistral-large-2512-A,dx-self-2-deictic,0shot,all,0.350,20,0.279,0.107,,format=0.05;other=0.5;other_object=0.05;other_participant=0.05
or-mistralai/mistral-large-2512-A,dx-self-2-deictic,0shot,free,0.357,14,0.274,0.128,,format=0.071;other=0.429;other_object=0.071;other_participant=0.071
or-mistralai/mistral-large-2512-A,dx-self-2-third,0shot,all,0.917,12,0.306,0.080,,
or-mistralai/mistral-large-2512-A,dx-addressee-3-deictic,0shot,all,0.562,16,0.276,0.124,,other_participant=0.25;speaker_confusion=0.188
or-mistralai/mistral-large-2512-A,dx-addressee-3-deictic,0shot,free,0.636,11,0.273,0.145,,other_participant=0.091;speaker_confusion=0.273
or-mistralai/mistral-large-2512-A,dx-addressee-3-third,0shot,all,0.941,17,0.294,0.057,,
or-mistralai/mistral-large-2512-A,dx-addressee-3-third,0shot,free,1.000,13,0.295,0.000,,
or-mistralai/mistral-large-2512-A,dx-addressee-2-deictic,0shot,all,0.625,16,0.286,0.121,,speaker_confusion=0.375
or-mistralai/mistral-large-2512-A,dx-addressee-2-deictic,0shot,free,0.727,11,0.288,0.134,,speaker_confusion=0.273
or-mistralai/mistral-large-2512-A,dx-addressee-2-third,0shot,all,0.938,16,0.292,0.060,,
or-mistralai/mistral-large-2512-A,dx-addressee-2-third,0shot,free,0.909,11,0.288,0.087,,
or-mistralai/mistral-large-2512-A,sb-k1,0shot,all,0.818,11,0.125,0.116,,
or-mistralai/mistral-small-3.2-24b-instruct-A,attr,0shot,all,0.867,30,0.161,0.062,,
or-mistralai/mistral-small-3.2-24b-instruct-A,codevar,0shot,all,0.933,30,0.274,0.045,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dialogue,0shot,all,0.933,30,0.311,0.045,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-owner-3-deictic,0shot,all,0.643,28,0.333,0.091,,other=0.357
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-owner-3-deictic,0shot,free,0.750,8,0.333,0.153,,other=0.25
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-owner-3-third,0shot,all,0.786,28,0.333,0.077,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-owner-3-third,0shot,free,0.625,8,0.333,0.171,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-3-deictic,0shot,all,0.407,27,0.284,0.095,,other_participant=0.593
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-3-deictic,0shot,free,0.421,19,0.281,0.113,,other_participant=0.579
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-3-third,0shot,all,0.781,32,0.286,0.073,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-2-deictic,0shot,all,0.364,33,0.280,0.084,,other=0.606;other_participant=0.03
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-2-deictic,0shot,free,0.391,23,0.275,0.102,,other=0.565;other_participant=0.043
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-2-third,0shot,all,0.714,28,0.295,0.085,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-self-2-third,0shot,free,0.789,19,0.289,0.093,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-3-deictic,0shot,all,0.455,33,0.283,0.087,,other_participant=0.182;speaker_confusion=0.364
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-3-deictic,0shot,free,0.545,22,0.280,0.106,,other_participant=0.091;speaker_confusion=0.364
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-3-third,0shot,all,0.741,27,0.287,0.084,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-3-third,0shot,free,0.737,19,0.285,0.101,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-2-deictic,0shot,all,0.259,27,0.293,0.084,,other_participant=0.074;speaker_confusion=0.667
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.556
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-2-third,0shot,all,0.758,33,0.285,0.075,,
or-mistralai/mistral-small-3.2-24b-instruct-A,dx-addressee-2-third,0shot,free,0.913,23,0.279,0.059,,
or-mistralai/mistral-small-3.2-24b-instruct-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-mistralai/mistral-small-3.2-24b-instruct-A,an-narr,0shot,all,0.542,24,0.250,0.102,,
or-mistralai/mistral-small-3.2-24b-instruct-A,an-other-any,0shot,all,0.792,24,0.250,0.083,,
or-mistralai/mistral-small-3.2-24b-instruct-A,an-me-any,0shot,all,0.375,24,0.250,0.099,1.0,
or-mistralai/mistral-small-3.2-24b-instruct-A,an-you-voc,0shot,all,0.750,24,0.250,0.088,1.0,
or-mistralai/mistral-small-3.2-24b-instruct-A,an-you-novoc,0shot,all,0.083,24,0.250,0.056,0.0,
or-mistralai/mistral-small-3.2-24b-instruct-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-mistralai/mistral-small-3.2-24b-instruct-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-mistralai/mistral-small-3.2-24b-instruct-A,adj-third,0shot,all,0.960,25,0.333,0.039,,
or-mistralai/mistral-small-3.2-24b-instruct-A,cm-mutate,0shot,all,0.950,20,0.200,0.049,,
or-mistralai/mistral-small-3.2-24b-instruct-A,cm-rebind,0shot,all,0.600,20,0.200,0.110,,
or-mistralai/mistral-small-3.2-24b-instruct-A,cm-copy,0shot,all,0.700,20,0.200,0.102,,
or-microsoft/phi-4-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-microsoft/phi-4-A,codevar,0shot,all,0.733,30,0.274,0.081,,
or-microsoft/phi-4-A,dialogue,0shot,all,0.900,30,0.311,0.055,,
or-microsoft/phi-4-A,dx-owner-3-deictic,0shot,all,0.857,28,0.333,0.066,,other=0.143
or-microsoft/phi-4-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-microsoft/phi-4-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-microsoft/phi-4-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-microsoft/phi-4-A,dx-self-3-deictic,0shot,all,0.518,27,0.284,0.096,,other_participant=0.481
or-microsoft/phi-4-A,dx-self-3-deictic,0shot,free,0.632,19,0.281,0.111,,other_participant=0.368
or-microsoft/phi-4-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-microsoft/phi-4-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-microsoft/phi-4-A,dx-self-2-deictic,0shot,all,0.455,33,0.280,0.087,,other=0.394;other_object=0.091;other_participant=0.061
or-microsoft/phi-4-A,dx-self-2-deictic,0shot,free,0.522,23,0.275,0.104,,other=0.348;other_object=0.13
or-microsoft/phi-4-A,dx-self-2-third,0shot,all,0.786,28,0.295,0.077,,
or-microsoft/phi-4-A,dx-self-2-third,0shot,free,0.737,19,0.289,0.101,,
or-microsoft/phi-4-A,dx-addressee-3-deictic,0shot,all,0.606,33,0.283,0.085,,other_participant=0.091;speaker_confusion=0.303
or-microsoft/phi-4-A,dx-addressee-3-deictic,0shot,free,0.591,22,0.280,0.105,,other_participant=0.136;speaker_confusion=0.273
or-microsoft/phi-4-A,dx-addressee-3-third,0shot,all,0.889,27,0.287,0.060,,
or-microsoft/phi-4-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-microsoft/phi-4-A,dx-addressee-2-deictic,0shot,all,0.370,27,0.293,0.093,,other_participant=0.074;speaker_confusion=0.556
or-microsoft/phi-4-A,dx-addressee-2-deictic,0shot,free,0.389,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.556
or-microsoft/phi-4-A,dx-addressee-2-third,0shot,all,0.758,33,0.285,0.075,,
or-microsoft/phi-4-A,dx-addressee-2-third,0shot,free,0.739,23,0.279,0.092,,
or-microsoft/phi-4-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-microsoft/phi-4-A,an-narr,0shot,all,0.708,24,0.250,0.093,,
or-microsoft/phi-4-A,an-other-any,0shot,all,0.708,24,0.250,0.093,,
or-microsoft/phi-4-A,an-me-any,0shot,all,0.292,24,0.250,0.093,1.0,
or-microsoft/phi-4-A,an-you-voc,0shot,all,0.708,24,0.250,0.093,1.0,
or-microsoft/phi-4-A,an-you-novoc,0shot,all,0.375,24,0.250,0.099,0.0,
or-microsoft/phi-4-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-microsoft/phi-4-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-microsoft/phi-4-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-microsoft/phi-4-A,cm-mutate,0shot,all,0.600,20,0.200,0.110,,
or-microsoft/phi-4-A,cm-rebind,0shot,all,0.950,20,0.200,0.049,,
or-microsoft/phi-4-A,cm-copy,0shot,all,0.850,20,0.200,0.080,,
or-qwen/qwen-2.5-72b-instruct-A,attr,0shot,all,1.000,30,0.161,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,codevar,0shot,all,0.833,30,0.274,0.068,,
or-qwen/qwen-2.5-72b-instruct-A,dialogue,0shot,all,0.967,30,0.311,0.033,,
or-qwen/qwen-2.5-72b-instruct-A,dx-owner-3-deictic,0shot,all,0.786,28,0.333,0.077,,other=0.214
or-qwen/qwen-2.5-72b-instruct-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,other=0.375
or-qwen/qwen-2.5-72b-instruct-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,dx-self-3-deictic,0shot,all,0.593,27,0.284,0.095,,other_participant=0.407
or-qwen/qwen-2.5-72b-instruct-A,dx-self-3-deictic,0shot,free,0.684,19,0.281,0.107,,other_participant=0.316
or-qwen/qwen-2.5-72b-instruct-A,dx-self-3-third,0shot,all,0.969,32,0.286,0.031,,
or-qwen/qwen-2.5-72b-instruct-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,dx-self-2-deictic,0shot,all,0.455,33,0.280,0.087,,other=0.485;other_object=0.03;other_participant=0.03
or-qwen/qwen-2.5-72b-instruct-A,dx-self-2-deictic,0shot,free,0.565,23,0.275,0.103,,other=0.391;other_object=0.043
or-qwen/qwen-2.5-72b-instruct-A,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-3-deictic,0shot,all,0.727,33,0.283,0.077,,other_participant=0.091;speaker_confusion=0.182
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-3-deictic,0shot,free,0.773,22,0.280,0.089,,other_participant=0.045;speaker_confusion=0.182
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-3-third,0shot,all,0.926,27,0.287,0.050,,
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-3-third,0shot,free,0.895,19,0.285,0.070,,
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-2-deictic,0shot,all,0.556,27,0.293,0.096,,other_participant=0.148;speaker_confusion=0.296
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,other_participant=0.111;speaker_confusion=0.278
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-2-third,0shot,all,0.939,33,0.285,0.042,,
or-qwen/qwen-2.5-72b-instruct-A,dx-addressee-2-third,0shot,free,0.957,23,0.279,0.043,,
or-qwen/qwen-2.5-72b-instruct-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,an-narr,0shot,all,0.708,24,0.250,0.093,,
or-qwen/qwen-2.5-72b-instruct-A,an-other-any,0shot,all,0.917,24,0.250,0.056,,
or-qwen/qwen-2.5-72b-instruct-A,an-me-any,0shot,all,0.250,24,0.250,0.088,0.833,
or-qwen/qwen-2.5-72b-instruct-A,an-you-voc,0shot,all,0.750,24,0.250,0.088,0.889,
or-qwen/qwen-2.5-72b-instruct-A,an-you-novoc,0shot,all,0.417,24,0.250,0.101,0.0,
or-qwen/qwen-2.5-72b-instruct-A,adj-adj,0shot,all,0.960,25,0.333,0.039,,
or-qwen/qwen-2.5-72b-instruct-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-qwen/qwen-2.5-72b-instruct-A,cm-rebind,0shot,all,0.750,20,0.200,0.097,,
or-qwen/qwen-2.5-72b-instruct-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
or-qwen/qwen-2.5-7b-instruct-A,attr,0shot,all,0.933,30,0.161,0.045,,
or-qwen/qwen-2.5-7b-instruct-A,codevar,0shot,all,0.667,30,0.274,0.086,,
or-qwen/qwen-2.5-7b-instruct-A,dialogue,0shot,all,0.900,30,0.311,0.055,,
or-qwen/qwen-2.5-7b-instruct-A,dx-owner-3-deictic,0shot,all,0.536,28,0.333,0.094,,format=0.036;other=0.429
or-qwen/qwen-2.5-7b-instruct-A,dx-owner-3-deictic,0shot,free,0.500,8,0.333,0.177,,other=0.5
or-qwen/qwen-2.5-7b-instruct-A,dx-owner-3-third,0shot,all,0.964,28,0.333,0.035,,
or-qwen/qwen-2.5-7b-instruct-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-qwen/qwen-2.5-7b-instruct-A,dx-self-3-deictic,0shot,all,0.370,27,0.284,0.093,,other_participant=0.63
or-qwen/qwen-2.5-7b-instruct-A,dx-self-3-deictic,0shot,free,0.474,19,0.281,0.115,,other_participant=0.526
or-qwen/qwen-2.5-7b-instruct-A,dx-self-3-third,0shot,all,0.719,32,0.286,0.080,,
or-qwen/qwen-2.5-7b-instruct-A,dx-self-3-third,0shot,free,0.818,22,0.284,0.082,,
or-qwen/qwen-2.5-7b-instruct-A,dx-self-2-deictic,0shot,all,0.455,33,0.280,0.087,,other=0.545
or-qwen/qwen-2.5-7b-instruct-A,dx-self-2-deictic,0shot,free,0.609,23,0.275,0.102,,other=0.391
or-qwen/qwen-2.5-7b-instruct-A,dx-self-2-third,0shot,all,0.714,28,0.295,0.085,,
or-qwen/qwen-2.5-7b-instruct-A,dx-self-2-third,0shot,free,0.842,19,0.289,0.084,,
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-3-deictic,0shot,all,0.455,33,0.283,0.087,,other_participant=0.182;speaker_confusion=0.364
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-3-deictic,0shot,free,0.636,22,0.280,0.103,,other_participant=0.091;speaker_confusion=0.273
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-3-third,0shot,all,0.741,27,0.287,0.084,,
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-3-third,0shot,free,0.842,19,0.285,0.084,,
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-2-deictic,0shot,all,0.444,27,0.293,0.096,,speaker_confusion=0.556
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,speaker_confusion=0.389
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-2-third,0shot,all,0.667,33,0.285,0.082,,
or-qwen/qwen-2.5-7b-instruct-A,dx-addressee-2-third,0shot,free,0.739,23,0.279,0.092,,
or-qwen/qwen-2.5-7b-instruct-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-qwen/qwen-2.5-7b-instruct-A,an-narr,0shot,all,0.500,24,0.250,0.102,,
or-qwen/qwen-2.5-7b-instruct-A,an-other-any,0shot,all,0.583,24,0.250,0.101,,
or-qwen/qwen-2.5-7b-instruct-A,an-me-any,0shot,all,0.125,24,0.250,0.068,0.667,
or-qwen/qwen-2.5-7b-instruct-A,an-you-voc,0shot,all,0.917,24,0.250,0.056,1.0,
or-qwen/qwen-2.5-7b-instruct-A,an-you-novoc,0shot,all,0.333,24,0.250,0.096,0.5,
or-qwen/qwen-2.5-7b-instruct-A,adj-adj,0shot,all,0.720,25,0.333,0.090,,
or-qwen/qwen-2.5-7b-instruct-A,adj-voc,0shot,all,0.840,25,0.333,0.073,,
or-qwen/qwen-2.5-7b-instruct-A,adj-third,0shot,all,0.960,25,0.333,0.039,,
or-qwen/qwen-2.5-7b-instruct-A,cm-mutate,0shot,all,0.700,20,0.200,0.102,,
or-qwen/qwen-2.5-7b-instruct-A,cm-rebind,0shot,all,0.500,20,0.200,0.112,,
or-qwen/qwen-2.5-7b-instruct-A,cm-copy,0shot,all,0.550,20,0.200,0.111,,
or-qwen/qwen3-235b-a22b-2507-A,attr,0shot,all,0.800,30,0.161,0.073,,
or-qwen/qwen3-235b-a22b-2507-A,codevar,0shot,all,0.733,30,0.274,0.081,,
or-qwen/qwen3-235b-a22b-2507-A,dialogue,0shot,all,0.867,30,0.311,0.062,,
or-qwen/qwen3-235b-a22b-2507-A,dx-owner-3-deictic,0shot,all,0.571,28,0.333,0.093,,format=0.107;other=0.321
or-qwen/qwen3-235b-a22b-2507-A,dx-owner-3-deictic,0shot,free,0.625,8,0.333,0.171,,format=0.125;other=0.25
or-qwen/qwen3-235b-a22b-2507-A,dx-owner-3-third,0shot,all,0.786,28,0.333,0.077,,
or-qwen/qwen3-235b-a22b-2507-A,dx-owner-3-third,0shot,free,0.750,8,0.333,0.153,,
or-qwen/qwen3-235b-a22b-2507-A,dx-self-3-deictic,0shot,all,0.296,27,0.284,0.088,,format=0.444;other_participant=0.259
or-qwen/qwen3-235b-a22b-2507-A,dx-self-3-deictic,0shot,free,0.263,19,0.281,0.101,,format=0.474;other_participant=0.263
or-qwen/qwen3-235b-a22b-2507-A,dx-self-3-third,0shot,all,0.938,32,0.286,0.043,,
or-qwen/qwen3-235b-a22b-2507-A,dx-self-3-third,0shot,free,0.909,22,0.284,0.061,,
or-qwen/qwen3-235b-a22b-2507-A,dx-self-2-deictic,0shot,all,0.303,33,0.280,0.080,,format=0.364;other=0.152;other_object=0.121;other_participant=0.061
or-qwen/qwen3-235b-a22b-2507-A,dx-self-2-deictic,0shot,free,0.217,23,0.275,0.086,,format=0.348;other=0.174;other_object=0.174;other_participant=0.087
or-qwen/qwen3-235b-a22b-2507-A,dx-self-2-third,0shot,all,0.714,28,0.295,0.085,,
or-qwen/qwen3-235b-a22b-2507-A,dx-self-2-third,0shot,free,0.789,19,0.289,0.093,,
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-3-deictic,0shot,all,0.303,33,0.283,0.080,,format=0.273;other_participant=0.212;speaker_confusion=0.212
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-3-deictic,0shot,free,0.318,22,0.280,0.099,,format=0.227;other_participant=0.273;speaker_confusion=0.182
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-3-third,0shot,all,0.926,27,0.287,0.050,,
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-3-third,0shot,free,0.895,19,0.285,0.070,,
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-2-deictic,0shot,all,0.333,27,0.293,0.091,,format=0.333;other_participant=0.222;speaker_confusion=0.111
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-2-deictic,0shot,free,0.278,18,0.292,0.106,,format=0.333;other_participant=0.222;speaker_confusion=0.167
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-2-third,0shot,all,0.909,33,0.285,0.050,,
or-qwen/qwen3-235b-a22b-2507-A,dx-addressee-2-third,0shot,free,0.913,23,0.279,0.059,,
or-qwen/qwen3-235b-a22b-2507-A,sb-k1,0shot,all,0.750,24,0.125,0.088,,
or-qwen/qwen3-235b-a22b-2507-A,an-narr,0shot,all,0.458,24,0.250,0.102,,
or-qwen/qwen3-235b-a22b-2507-A,an-other-any,0shot,all,0.708,24,0.250,0.093,,
or-qwen/qwen3-235b-a22b-2507-A,an-me-any,0shot,all,0.125,24,0.250,0.068,0.667,
or-qwen/qwen3-235b-a22b-2507-A,an-you-voc,0shot,all,0.833,24,0.250,0.076,0.9,
or-qwen/qwen3-235b-a22b-2507-A,an-you-novoc,0shot,all,0.333,24,0.250,0.096,0.25,
or-qwen/qwen3-235b-a22b-2507-A,adj-adj,0shot,all,0.840,25,0.333,0.073,,
or-qwen/qwen3-235b-a22b-2507-A,adj-voc,0shot,all,0.920,25,0.333,0.054,,
or-qwen/qwen3-235b-a22b-2507-A,adj-third,0shot,all,0.880,25,0.333,0.065,,
or-qwen/qwen3-235b-a22b-2507-A,cm-mutate,0shot,all,0.900,20,0.200,0.067,,
or-qwen/qwen3-235b-a22b-2507-A,cm-rebind,0shot,all,0.750,20,0.200,0.097,,
or-qwen/qwen3-235b-a22b-2507-A,cm-copy,0shot,all,0.800,20,0.200,0.089,,
or-qwen/qwen3-32b-A,attr,0shot,all,0.947,19,0.163,0.051,,
or-qwen/qwen3-32b-A,codevar,0shot,all,0.727,22,0.286,0.095,,
or-qwen/qwen3-32b-A,dialogue,0shot,all,1.000,19,0.303,0.000,,
or-qwen/qwen3-32b-A,dx-owner-3-deictic,0shot,all,0.684,19,0.333,0.107,,other=0.316
or-qwen/qwen3-32b-A,dx-owner-3-third,0shot,all,1.000,17,0.333,0.000,,
or-qwen/qwen3-32b-A,dx-self-3-deictic,0shot,all,0.368,19,0.281,0.111,,other_participant=0.632
or-qwen/qwen3-32b-A,dx-self-3-deictic,0shot,free,0.357,14,0.280,0.128,,other_participant=0.643
or-qwen/qwen3-32b-A,dx-self-3-third,0shot,all,0.818,22,0.280,0.082,,
or-qwen/qwen3-32b-A,dx-self-3-third,0shot,free,0.824,17,0.279,0.092,,
or-qwen/qwen3-32b-A,dx-self-2-deictic,0shot,all,0.471,17,0.265,0.121,,other=0.235;other_object=0.235;other_participant=0.059
or-qwen/qwen3-32b-A,dx-self-2-deictic,0shot,free,0.455,11,0.250,0.150,,other=0.091;other_object=0.364;other_participant=0.091
or-qwen/qwen3-32b-A,dx-self-2-third,0shot,all,0.800,20,0.292,0.089,,
or-qwen/qwen3-32b-A,dx-self-2-third,0shot,free,0.833,12,0.285,0.108,,
or-qwen/qwen3-32b-A,dx-addressee-3-deictic,0shot,all,0.280,25,0.287,0.090,,other_participant=0.28;speaker_confusion=0.44
or-qwen/qwen3-32b-A,dx-addressee-3-deictic,0shot,free,0.400,15,0.283,0.127,,other_participant=0.2;speaker_confusion=0.4
or-qwen/qwen3-32b-A,dx-addressee-3-third,0shot,all,0.905,21,0.282,0.064,,
or-qwen/qwen3-32b-A,dx-addressee-3-third,0shot,free,0.857,14,0.280,0.093,,
or-qwen/qwen3-32b-A,dx-addressee-2-deictic,0shot,all,0.210,19,0.303,0.093,,other_participant=0.158;speaker_confusion=0.632
or-qwen/qwen3-32b-A,dx-addressee-2-deictic,0shot,free,0.214,14,0.304,0.110,,other_participant=0.214;speaker_confusion=0.571
or-qwen/qwen3-32b-A,dx-addressee-2-third,0shot,all,0.714,21,0.286,0.099,,
or-qwen/qwen3-32b-A,dx-addressee-2-third,0shot,free,0.857,14,0.280,0.093,,
or-qwen/qwen3-32b-A,sb-k1,0shot,all,1.000,16,0.125,0.000,,
or-qwen/qwen3-32b-A,an-narr,0shot,all,0.867,15,0.250,0.088,,
or-qwen/qwen3-32b-A,an-other-any,0shot,all,0.786,14,0.250,0.110,,
or-qwen/qwen3-32b-A,an-me-any,0shot,all,0.118,17,0.250,0.078,0.5,
or-qwen/qwen3-32b-A,an-you-voc,0shot,all,0.909,11,0.250,0.087,1.0,
or-qwen/qwen3-32b-A,an-you-novoc,0shot,all,0.692,13,0.250,0.128,0.0,
or-qwen/qwen3-32b-A,adj-adj,0shot,all,1.000,14,0.333,0.000,,
or-qwen/qwen3-32b-A,adj-voc,0shot,all,1.000,16,0.333,0.000,,
or-qwen/qwen3-32b-A,adj-third,0shot,all,1.000,16,0.333,0.000,,
or-qwen/qwen3-32b-A,cm-mutate,0shot,all,1.000,10,0.200,0.000,,
or-qwen/qwen3-32b-A,cm-rebind,0shot,all,0.889,9,0.200,0.105,,
or-qwen/qwen3-32b-A,cm-copy,0shot,all,0.000,11,0.200,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-owner-3-third,0shot,all,0.964,28,0.333,0.035,,
or-qwen/qwen3.5-397b-a17b-B,dx-owner-3-third,0shot,free,0.875,8,0.333,0.117,,
or-qwen/qwen3.5-397b-a17b-B,dx-self-3-deictic,0shot,all,0.926,27,0.284,0.050,,format=0.074
or-qwen/qwen3.5-397b-a17b-B,dx-self-3-deictic,0shot,free,0.895,19,0.281,0.070,,format=0.105
or-qwen/qwen3.5-397b-a17b-B,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-self-2-deictic,0shot,all,0.849,33,0.280,0.062,,format=0.121;other_object=0.03
or-qwen/qwen3.5-397b-a17b-B,dx-self-2-deictic,0shot,free,0.870,23,0.275,0.070,,format=0.087;other_object=0.043
or-qwen/qwen3.5-397b-a17b-B,dx-self-2-third,0shot,all,1.000,28,0.295,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-self-2-third,0shot,free,1.000,19,0.289,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-3-deictic,0shot,all,0.909,33,0.283,0.050,,format=0.091
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-3-deictic,0shot,free,0.909,22,0.280,0.061,,format=0.091
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-3-third,0shot,all,0.926,27,0.287,0.050,,
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-3-third,0shot,free,0.947,19,0.285,0.051,,
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-2-deictic,0shot,all,0.889,27,0.293,0.060,,format=0.037;other_participant=0.037;speaker_confusion=0.037
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-2-deictic,0shot,free,0.833,18,0.292,0.088,,format=0.056;other_participant=0.056;speaker_confusion=0.056
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-2-third,0shot,all,1.000,33,0.285,0.000,,
or-qwen/qwen3.5-397b-a17b-B,dx-addressee-2-third,0shot,free,1.000,23,0.279,0.000,,
or-qwen/qwen3.5-397b-a17b-B,an-narr,0shot,all,1.000,24,0.250,0.000,,
or-qwen/qwen3.5-397b-a17b-B,an-other-any,0shot,all,0.917,24,0.250,0.056,,
or-qwen/qwen3.5-397b-a17b-B,an-me-any,0shot,all,0.750,24,0.250,0.088,0.944,
or-qwen/qwen3.5-397b-a17b-B,an-you-voc,0shot,all,0.875,24,0.250,0.068,0.81,
or-qwen/qwen3.5-397b-a17b-B,an-you-novoc,0shot,all,0.875,24,0.250,0.068,0.0,
or-qwen/qwen3.5-397b-a17b-B,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-B,adj-voc,0shot,all,0.960,25,0.333,0.039,,
or-qwen/qwen3.5-397b-a17b-B,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,attr,0shot,all,1.000,28,0.163,0.000,,
or-qwen/qwen3.5-397b-a17b-A,codevar,0shot,all,1.000,30,0.274,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dialogue,0shot,all,1.000,28,0.309,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-owner-3-deictic,0shot,all,1.000,28,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-owner-3-deictic,0shot,free,1.000,8,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-owner-3-third,0shot,all,1.000,28,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-owner-3-third,0shot,free,1.000,8,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-self-3-deictic,0shot,all,0.885,26,0.285,0.063,,other_participant=0.115
or-qwen/qwen3.5-397b-a17b-A,dx-self-3-deictic,0shot,free,0.895,19,0.281,0.070,,other_participant=0.105
or-qwen/qwen3.5-397b-a17b-A,dx-self-3-third,0shot,all,1.000,32,0.286,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-self-3-third,0shot,free,1.000,22,0.284,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-self-2-deictic,0shot,all,0.667,33,0.280,0.082,,other=0.303;other_participant=0.03
or-qwen/qwen3.5-397b-a17b-A,dx-self-2-deictic,0shot,free,0.696,23,0.275,0.096,,other=0.304
or-qwen/qwen3.5-397b-a17b-A,dx-self-2-third,0shot,all,0.964,28,0.295,0.035,,
or-qwen/qwen3.5-397b-a17b-A,dx-self-2-third,0shot,free,0.947,19,0.289,0.051,,
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-3-deictic,0shot,all,0.688,32,0.284,0.082,,other_participant=0.094;speaker_confusion=0.219
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-3-deictic,0shot,free,0.667,21,0.282,0.103,,other_participant=0.095;speaker_confusion=0.238
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-3-third,0shot,all,1.000,26,0.285,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-3-third,0shot,free,1.000,19,0.285,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-2-deictic,0shot,all,0.630,27,0.293,0.093,,other_participant=0.111;speaker_confusion=0.259
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-2-deictic,0shot,free,0.611,18,0.292,0.115,,other_participant=0.056;speaker_confusion=0.333
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-2-third,0shot,all,1.000,32,0.286,0.000,,
or-qwen/qwen3.5-397b-a17b-A,dx-addressee-2-third,0shot,free,1.000,22,0.280,0.000,,
or-qwen/qwen3.5-397b-a17b-A,sb-k1,0shot,all,1.000,24,0.125,0.000,,
or-qwen/qwen3.5-397b-a17b-A,an-narr,0shot,all,1.000,23,0.250,0.000,,
or-qwen/qwen3.5-397b-a17b-A,an-other-any,0shot,all,0.957,23,0.250,0.043,,
or-qwen/qwen3.5-397b-a17b-A,an-me-any,0shot,all,0.500,24,0.250,0.102,1.0,
or-qwen/qwen3.5-397b-a17b-A,an-you-voc,0shot,all,1.000,24,0.250,0.000,0.958,
or-qwen/qwen3.5-397b-a17b-A,an-you-novoc,0shot,all,0.958,24,0.250,0.041,0.0,
or-qwen/qwen3.5-397b-a17b-A,adj-adj,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,adj-voc,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,adj-third,0shot,all,1.000,25,0.333,0.000,,
or-qwen/qwen3.5-397b-a17b-A,cm-mutate,0shot,all,1.000,20,0.200,0.000,,
or-qwen/qwen3.5-397b-a17b-A,cm-rebind,0shot,all,0.900,20,0.200,0.067,,
or-qwen/qwen3.5-397b-a17b-A,cm-copy,0shot,all,0.900,20,0.200,0.067,,
smollm2-135m,attr,0shot,all,0.386,1210,0.168,0.014,,
smollm2-135m,attr,3shot_collide,all,0.474,1210,0.168,0.014,,
smollm2-135m,attr,3shot,all,0.461,1210,0.168,0.014,,
smollm2-135m,codevar,0shot,all,0.280,1200,0.267,0.013,,
smollm2-135m,codevar,3shot_collide,all,0.353,1200,0.267,0.014,,
smollm2-135m,codevar,3shot,all,0.349,1200,0.267,0.014,,
smollm2-135m,dialogue,0shot,all,0.432,1200,0.312,0.014,,
smollm2-135m,dialogue,3shot_collide,all,0.391,1200,0.312,0.014,,
smollm2-135m,dialogue,3shot,all,0.432,1200,0.312,0.014,,
smollm2-135m,dx-owner-3-deictic,0shot,all,0.500,120,0.333,0.046,,other_participant=0.383;speaker_confusion=0.108;tie=0.008
smollm2-135m,dx-owner-3-deictic,0shot,free,0.425,40,0.333,0.078,,other_participant=0.525;speaker_confusion=0.05
smollm2-135m,dx-owner-3-deictic,3shot_collide,all,0.267,120,0.333,0.040,,other_participant=0.458;speaker_confusion=0.275
smollm2-135m,dx-owner-3-deictic,3shot_collide,free,0.125,40,0.333,0.052,,other_participant=0.7;speaker_confusion=0.175
smollm2-135m,dx-owner-3-deictic,3shot,all,0.408,120,0.333,0.045,,other_participant=0.392;speaker_confusion=0.2
smollm2-135m,dx-owner-3-deictic,3shot,free,0.300,40,0.333,0.072,,other_participant=0.525;speaker_confusion=0.175
smollm2-135m,dx-owner-3-third,0shot,all,0.483,120,0.333,0.046,,
smollm2-135m,dx-owner-3-third,0shot,free,0.425,40,0.333,0.078,,
smollm2-135m,dx-owner-3-third,3shot_collide,all,0.350,120,0.333,0.043,,
smollm2-135m,dx-owner-3-third,3shot_collide,free,0.125,40,0.333,0.052,,
smollm2-135m,dx-owner-3-third,3shot,all,0.417,120,0.333,0.045,,
smollm2-135m,dx-owner-3-third,3shot,free,0.275,40,0.333,0.071,,
smollm2-135m,dx-self-3-deictic,0shot,all,0.294,180,0.287,0.034,,other_object=0.267;previous_owner=0.061;speaker_confusion=0.361;tie=0.017
smollm2-135m,dx-self-3-deictic,0shot,free,0.254,126,0.286,0.039,,other_object=0.333;previous_owner=0.04;speaker_confusion=0.365;tie=0.008
smollm2-135m,dx-self-3-deictic,3shot_collide,all,0.228,180,0.287,0.031,,other_object=0.256;previous_owner=0.072;speaker_confusion=0.433;tie=0.011
smollm2-135m,dx-self-3-deictic,3shot_collide,free,0.175,126,0.286,0.034,,other_object=0.286;previous_owner=0.048;speaker_confusion=0.484;tie=0.008
smollm2-135m,dx-self-3-deictic,3shot,all,0.261,180,0.287,0.033,,other_object=0.239;previous_owner=0.072;speaker_confusion=0.417;tie=0.011
smollm2-135m,dx-self-3-deictic,3shot,free,0.191,126,0.286,0.035,,other_object=0.294;previous_owner=0.048;speaker_confusion=0.468
smollm2-135m,dx-self-3-third,0shot,all,0.261,180,0.287,0.033,,
smollm2-135m,dx-self-3-third,0shot,free,0.214,126,0.286,0.037,,
smollm2-135m,dx-self-3-third,3shot_collide,all,0.272,180,0.287,0.033,,
smollm2-135m,dx-self-3-third,3shot_collide,free,0.198,126,0.286,0.035,,
smollm2-135m,dx-self-3-third,3shot,all,0.244,180,0.287,0.032,,
smollm2-135m,dx-self-3-third,3shot,free,0.135,126,0.286,0.030,,
smollm2-135m,dx-self-2-deictic,0shot,all,0.250,180,0.287,0.032,,speaker_confusion=0.733;tie=0.017
smollm2-135m,dx-self-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.77;tie=0.016
smollm2-135m,dx-self-2-deictic,3shot_collide,all,0.272,180,0.287,0.033,,speaker_confusion=0.722;tie=0.006
smollm2-135m,dx-self-2-deictic,3shot_collide,free,0.214,126,0.286,0.037,,speaker_confusion=0.778;tie=0.008
smollm2-135m,dx-self-2-deictic,3shot,all,0.272,180,0.287,0.033,,speaker_confusion=0.722;tie=0.006
smollm2-135m,dx-self-2-deictic,3shot,free,0.222,126,0.286,0.037,,speaker_confusion=0.77;tie=0.008
smollm2-135m,dx-self-2-third,0shot,all,0.311,180,0.287,0.035,,
smollm2-135m,dx-self-2-third,0shot,free,0.246,126,0.286,0.038,,
smollm2-135m,dx-self-2-third,3shot_collide,all,0.333,180,0.287,0.035,,
smollm2-135m,dx-self-2-third,3shot_collide,free,0.254,126,0.286,0.039,,
smollm2-135m,dx-self-2-third,3shot,all,0.328,180,0.287,0.035,,
smollm2-135m,dx-self-2-third,3shot,free,0.214,126,0.286,0.037,,
smollm2-135m,dx-addressee-3-deictic,0shot,all,0.328,180,0.287,0.035,,other_object=0.122;previous_owner=0.178;speaker_confusion=0.356;tie=0.017
smollm2-135m,dx-addressee-3-deictic,0shot,free,0.270,126,0.286,0.040,,other_object=0.103;previous_owner=0.222;speaker_confusion=0.389;tie=0.016
smollm2-135m,dx-addressee-3-deictic,3shot_collide,all,0.311,180,0.287,0.035,,other_object=0.156;previous_owner=0.183;speaker_confusion=0.344;tie=0.006
smollm2-135m,dx-addressee-3-deictic,3shot_collide,free,0.238,126,0.286,0.038,,other_object=0.167;previous_owner=0.23;speaker_confusion=0.365
smollm2-135m,dx-addressee-3-deictic,3shot,all,0.306,180,0.287,0.034,,other_object=0.144;previous_owner=0.217;speaker_confusion=0.322;tie=0.011
smollm2-135m,dx-addressee-3-deictic,3shot,free,0.230,126,0.286,0.037,,other_object=0.143;previous_owner=0.27;speaker_confusion=0.341;tie=0.016
smollm2-135m,dx-addressee-3-third,0shot,all,0.278,180,0.287,0.033,,
smollm2-135m,dx-addressee-3-third,0shot,free,0.294,126,0.286,0.041,,
smollm2-135m,dx-addressee-3-third,3shot_collide,all,0.272,180,0.287,0.033,,
smollm2-135m,dx-addressee-3-third,3shot_collide,free,0.214,126,0.286,0.037,,
smollm2-135m,dx-addressee-3-third,3shot,all,0.267,180,0.287,0.033,,
smollm2-135m,dx-addressee-3-third,3shot,free,0.182,126,0.286,0.034,,
smollm2-135m,dx-addressee-2-deictic,0shot,all,0.194,180,0.287,0.029,,speaker_confusion=0.8;tie=0.006
smollm2-135m,dx-addressee-2-deictic,0shot,free,0.167,126,0.286,0.033,,speaker_confusion=0.833
smollm2-135m,dx-addressee-2-deictic,3shot_collide,all,0.194,180,0.287,0.029,,speaker_confusion=0.8;tie=0.006
smollm2-135m,dx-addressee-2-deictic,3shot_collide,free,0.111,126,0.286,0.028,,speaker_confusion=0.881;tie=0.008
smollm2-135m,dx-addressee-2-deictic,3shot,all,0.183,180,0.287,0.029,,speaker_confusion=0.8;tie=0.017
smollm2-135m,dx-addressee-2-deictic,3shot,free,0.111,126,0.286,0.028,,speaker_confusion=0.881;tie=0.008
smollm2-135m,dx-addressee-2-third,0shot,all,0.206,180,0.287,0.030,,
smollm2-135m,dx-addressee-2-third,0shot,free,0.143,126,0.286,0.031,,
smollm2-135m,dx-addressee-2-third,3shot_collide,all,0.267,180,0.287,0.033,,
smollm2-135m,dx-addressee-2-third,3shot_collide,free,0.175,126,0.286,0.034,,
smollm2-135m,dx-addressee-2-third,3shot,all,0.283,180,0.287,0.034,,
smollm2-135m,dx-addressee-2-third,3shot,free,0.151,126,0.286,0.032,,
smollm2-135m,sb-k1,0shot,all,0.328,64,0.125,0.059,,
smollm2-135m,sb-k1,3shot_collide,all,0.219,64,0.125,0.052,,
smollm2-135m,sb-k1,3shot,all,0.234,64,0.125,0.053,,
smollm2-135m,sb-k23,0shot,all,0.078,128,0.027,0.024,,
smollm2-135m,sb-k23,3shot_collide,all,0.102,128,0.027,0.027,,
smollm2-135m,sb-k23,3shot,all,0.094,128,0.027,0.026,,
smollm2-135m,sb-split,0shot,all,0.070,128,0.027,0.023,,
smollm2-135m,sb-split,3shot_collide,all,0.016,128,0.027,0.011,,
smollm2-135m,sb-split,3shot,all,0.016,128,0.027,0.011,,
smollm2-135m,sb-dyn,0shot,all,0.055,256,0.027,0.014,,
smollm2-135m,sb-dyn,3shot_collide,all,0.016,256,0.027,0.008,,
smollm2-135m,sb-dyn,3shot,all,0.019,256,0.027,0.009,,
smollm2-135m,an-narr,0shot,all,0.416,640,0.250,0.019,,
smollm2-135m,an-narr,3shot_collide,all,0.247,640,0.250,0.017,,
smollm2-135m,an-narr,3shot,all,0.314,640,0.250,0.018,,
smollm2-135m,an-other-any,0shot,all,0.372,320,0.250,0.027,,
smollm2-135m,an-other-any,3shot_collide,all,0.303,320,0.250,0.026,,
smollm2-135m,an-other-any,3shot,all,0.319,320,0.250,0.026,,
smollm2-135m,an-me-any,0shot,all,0.419,160,0.250,0.039,,
smollm2-135m,an-me-any,3shot_collide,all,0.362,160,0.250,0.038,,
smollm2-135m,an-me-any,3shot,all,0.419,160,0.250,0.039,,
smollm2-135m,an-you-voc,0shot,all,0.263,80,0.250,0.049,,
smollm2-135m,an-you-voc,3shot_collide,all,0.150,80,0.250,0.040,,
smollm2-135m,an-you-voc,3shot,all,0.025,80,0.250,0.018,,
smollm2-135m,an-you-novoc,0shot,all,0.275,80,0.250,0.050,,
smollm2-135m,an-you-novoc,3shot_collide,all,0.087,80,0.250,0.032,,
smollm2-135m,an-you-novoc,3shot,all,0.025,80,0.250,0.018,,
smollm2-135m,adj-adj,0shot,all,0.381,260,0.333,0.030,,
smollm2-135m,adj-adj,3shot_collide,all,0.119,260,0.333,0.020,,
smollm2-135m,adj-adj,3shot,all,0.354,260,0.333,0.030,,
smollm2-135m,adj-voc,0shot,all,0.546,260,0.333,0.031,,
smollm2-135m,adj-voc,3shot_collide,all,0.265,260,0.333,0.027,,
smollm2-135m,adj-voc,3shot,all,0.492,260,0.333,0.031,,
smollm2-135m,adj-third,0shot,all,0.519,260,0.333,0.031,,
smollm2-135m,adj-third,3shot_collide,all,0.408,260,0.333,0.030,,
smollm2-135m,adj-third,3shot,all,0.511,260,0.333,0.031,,
smollm2-135m,cm-mutate,0shot,all,0.425,200,0.200,0.035,,
smollm2-135m,cm-mutate,3shot_collide,all,0.270,200,0.200,0.031,,
smollm2-135m,cm-mutate,3shot,all,0.270,200,0.200,0.031,,
smollm2-135m,cm-rebind,0shot,all,0.025,200,0.200,0.011,,
smollm2-135m,cm-rebind,3shot_collide,all,0.070,200,0.200,0.018,,
smollm2-135m,cm-rebind,3shot,all,0.020,200,0.200,0.010,,
smollm2-135m,cm-copy,0shot,all,0.040,200,0.200,0.014,,
smollm2-135m,cm-copy,3shot_collide,all,0.055,200,0.200,0.016,,
smollm2-135m,cm-copy,3shot,all,0.025,200,0.200,0.011,,
pythia-160m,attr,0shot,all,0.052,1210,0.168,0.006,,
pythia-160m,attr,3shot_collide,all,0.096,1210,0.168,0.009,,
pythia-160m,attr,3shot,all,0.125,1210,0.168,0.009,,
pythia-160m,codevar,0shot,all,0.097,1200,0.267,0.009,,
pythia-160m,codevar,3shot_collide,all,0.211,1200,0.267,0.012,,
pythia-160m,codevar,3shot,all,0.199,1200,0.267,0.011,,
pythia-160m,dialogue,0shot,all,0.102,1200,0.312,0.009,,
pythia-160m,dialogue,3shot_collide,all,0.113,1200,0.312,0.009,,
pythia-160m,dialogue,3shot,all,0.158,1200,0.312,0.011,,
pythia-160m,dx-owner-3-deictic,0shot,all,0.058,120,0.333,0.021,,other_participant=0.483;speaker_confusion=0.217;tie=0.242
pythia-160m,dx-owner-3-deictic,0shot,free,0.025,40,0.333,0.025,,other_participant=0.5;speaker_confusion=0.175;tie=0.3
pythia-160m,dx-owner-3-deictic,3shot_collide,all,0.025,120,0.333,0.014,,other_participant=0.575;speaker_confusion=0.275;tie=0.125
pythia-160m,dx-owner-3-deictic,3shot_collide,free,0.025,40,0.333,0.025,,other_participant=0.675;speaker_confusion=0.2;tie=0.1
pythia-160m,dx-owner-3-deictic,3shot,all,0.067,120,0.333,0.023,,other_participant=0.392;speaker_confusion=0.358;tie=0.183
pythia-160m,dx-owner-3-deictic,3shot,free,0.025,40,0.333,0.025,,other_participant=0.5;speaker_confusion=0.325;tie=0.15
pythia-160m,dx-owner-3-third,0shot,all,0.200,120,0.333,0.036,,
pythia-160m,dx-owner-3-third,0shot,free,0.175,40,0.333,0.060,,
pythia-160m,dx-owner-3-third,3shot_collide,all,0.067,120,0.333,0.023,,
pythia-160m,dx-owner-3-third,3shot_collide,free,0.050,40,0.333,0.035,,
pythia-160m,dx-owner-3-third,3shot,all,0.192,120,0.333,0.036,,
pythia-160m,dx-owner-3-third,3shot,free,0.125,40,0.333,0.052,,
pythia-160m,dx-self-3-deictic,0shot,all,0.050,180,0.287,0.016,,other_object=0.256;previous_owner=0.106;speaker_confusion=0.4;tie=0.189
pythia-160m,dx-self-3-deictic,0shot,free,0.032,126,0.286,0.016,,other_object=0.286;previous_owner=0.079;speaker_confusion=0.452;tie=0.151
pythia-160m,dx-self-3-deictic,3shot_collide,all,0.178,180,0.287,0.029,,other_object=0.2;previous_owner=0.106;speaker_confusion=0.394;tie=0.122
pythia-160m,dx-self-3-deictic,3shot_collide,free,0.111,126,0.286,0.028,,other_object=0.246;previous_owner=0.095;speaker_confusion=0.476;tie=0.071
pythia-160m,dx-self-3-deictic,3shot,all,0.161,180,0.287,0.027,,other_object=0.183;previous_owner=0.094;speaker_confusion=0.433;tie=0.128
pythia-160m,dx-self-3-deictic,3shot,free,0.095,126,0.286,0.026,,other_object=0.198;previous_owner=0.087;speaker_confusion=0.54;tie=0.079
pythia-160m,dx-self-3-third,0shot,all,0.078,180,0.287,0.020,,
pythia-160m,dx-self-3-third,0shot,free,0.087,126,0.286,0.025,,
pythia-160m,dx-self-3-third,3shot_collide,all,0.117,180,0.287,0.024,,
pythia-160m,dx-self-3-third,3shot_collide,free,0.103,126,0.286,0.027,,
pythia-160m,dx-self-3-third,3shot,all,0.122,180,0.287,0.024,,
pythia-160m,dx-self-3-third,3shot,free,0.087,126,0.286,0.025,,
pythia-160m,dx-self-2-deictic,0shot,all,0.050,180,0.287,0.016,,speaker_confusion=0.717;tie=0.233
pythia-160m,dx-self-2-deictic,0shot,free,0.040,126,0.286,0.017,,speaker_confusion=0.738;tie=0.222
pythia-160m,dx-self-2-deictic,3shot_collide,all,0.178,180,0.287,0.029,,speaker_confusion=0.711;tie=0.111
pythia-160m,dx-self-2-deictic,3shot_collide,free,0.087,126,0.286,0.025,,speaker_confusion=0.794;tie=0.119
pythia-160m,dx-self-2-deictic,3shot,all,0.150,180,0.287,0.027,,speaker_confusion=0.7;tie=0.15
pythia-160m,dx-self-2-deictic,3shot,free,0.071,126,0.286,0.023,,speaker_confusion=0.794;tie=0.135
pythia-160m,dx-self-2-third,0shot,all,0.067,180,0.287,0.019,,
pythia-160m,dx-self-2-third,0shot,free,0.064,126,0.286,0.022,,
pythia-160m,dx-self-2-third,3shot_collide,all,0.139,180,0.287,0.026,,
pythia-160m,dx-self-2-third,3shot_collide,free,0.119,126,0.286,0.029,,
pythia-160m,dx-self-2-third,3shot,all,0.089,180,0.287,0.021,,
pythia-160m,dx-self-2-third,3shot,free,0.079,126,0.286,0.024,,
pythia-160m,dx-addressee-3-deictic,0shot,all,0.117,180,0.287,0.024,,other_object=0.139;previous_owner=0.206;speaker_confusion=0.333;tie=0.206
pythia-160m,dx-addressee-3-deictic,0shot,free,0.071,126,0.286,0.023,,other_object=0.135;previous_owner=0.254;speaker_confusion=0.365;tie=0.175
pythia-160m,dx-addressee-3-deictic,3shot_collide,all,0.172,180,0.287,0.028,,other_object=0.178;previous_owner=0.206;speaker_confusion=0.344;tie=0.1
pythia-160m,dx-addressee-3-deictic,3shot_collide,free,0.087,126,0.286,0.025,,other_object=0.167;previous_owner=0.262;speaker_confusion=0.405;tie=0.079
pythia-160m,dx-addressee-3-deictic,3shot,all,0.150,180,0.287,0.027,,other_object=0.128;previous_owner=0.222;speaker_confusion=0.344;tie=0.156
pythia-160m,dx-addressee-3-deictic,3shot,free,0.087,126,0.286,0.025,,other_object=0.111;previous_owner=0.278;speaker_confusion=0.397;tie=0.127
pythia-160m,dx-addressee-3-third,0shot,all,0.083,180,0.287,0.021,,
pythia-160m,dx-addressee-3-third,0shot,free,0.064,126,0.286,0.022,,
pythia-160m,dx-addressee-3-third,3shot_collide,all,0.128,180,0.287,0.025,,
pythia-160m,dx-addressee-3-third,3shot_collide,free,0.111,126,0.286,0.028,,
pythia-160m,dx-addressee-3-third,3shot,all,0.139,180,0.287,0.026,,
pythia-160m,dx-addressee-3-third,3shot,free,0.103,126,0.286,0.027,,
pythia-160m,dx-addressee-2-deictic,0shot,all,0.089,180,0.287,0.021,,speaker_confusion=0.75;tie=0.161
pythia-160m,dx-addressee-2-deictic,0shot,free,0.056,126,0.286,0.020,,speaker_confusion=0.817;tie=0.127
pythia-160m,dx-addressee-2-deictic,3shot_collide,all,0.128,180,0.287,0.025,,speaker_confusion=0.733;tie=0.139
pythia-160m,dx-addressee-2-deictic,3shot_collide,free,0.087,126,0.286,0.025,,speaker_confusion=0.841;tie=0.071
pythia-160m,dx-addressee-2-deictic,3shot,all,0.117,180,0.287,0.024,,speaker_confusion=0.761;tie=0.122
pythia-160m,dx-addressee-2-deictic,3shot,free,0.040,126,0.286,0.017,,speaker_confusion=0.857;tie=0.103
pythia-160m,dx-addressee-2-third,0shot,all,0.067,180,0.287,0.019,,
pythia-160m,dx-addressee-2-third,0shot,free,0.087,126,0.286,0.025,,
pythia-160m,dx-addressee-2-third,3shot_collide,all,0.139,180,0.287,0.026,,
pythia-160m,dx-addressee-2-third,3shot_collide,free,0.111,126,0.286,0.028,,
pythia-160m,dx-addressee-2-third,3shot,all,0.100,180,0.287,0.022,,
pythia-160m,dx-addressee-2-third,3shot,free,0.064,126,0.286,0.022,,
pythia-160m,sb-k1,0shot,all,0.219,64,0.125,0.052,,
pythia-160m,sb-k1,3shot_collide,all,0.219,64,0.125,0.052,,
pythia-160m,sb-k1,3shot,all,0.281,64,0.125,0.056,,
pythia-160m,sb-k23,0shot,all,0.094,128,0.027,0.026,,
pythia-160m,sb-k23,3shot_collide,all,0.047,128,0.027,0.019,,
pythia-160m,sb-k23,3shot,all,0.039,128,0.027,0.017,,
pythia-160m,sb-split,0shot,all,0.047,128,0.027,0.019,,
pythia-160m,sb-split,3shot_collide,all,0.055,128,0.027,0.020,,
pythia-160m,sb-split,3shot,all,0.031,128,0.027,0.015,,
pythia-160m,sb-dyn,0shot,all,0.070,256,0.027,0.016,,
pythia-160m,sb-dyn,3shot_collide,all,0.043,256,0.027,0.013,,
pythia-160m,sb-dyn,3shot,all,0.047,256,0.027,0.013,,
pythia-160m,an-narr,0shot,all,0.061,640,0.250,0.009,,
pythia-160m,an-narr,3shot_collide,all,0.016,640,0.250,0.005,,
pythia-160m,an-narr,3shot,all,0.080,640,0.250,0.011,,
pythia-160m,an-other-any,0shot,all,0.209,320,0.250,0.023,,
pythia-160m,an-other-any,3shot_collide,all,0.234,320,0.250,0.024,,
pythia-160m,an-other-any,3shot,all,0.244,320,0.250,0.024,,
pythia-160m,an-me-any,0shot,all,0.575,160,0.250,0.039,,
pythia-160m,an-me-any,3shot_collide,all,0.456,160,0.250,0.039,,
pythia-160m,an-me-any,3shot,all,0.450,160,0.250,0.039,,
pythia-160m,an-you-voc,0shot,all,0.013,80,0.250,0.012,,
pythia-160m,an-you-voc,3shot_collide,all,0.300,80,0.250,0.051,,
pythia-160m,an-you-voc,3shot,all,0.375,80,0.250,0.054,,
pythia-160m,an-you-novoc,0shot,all,0.175,80,0.250,0.043,,
pythia-160m,an-you-novoc,3shot_collide,all,0.237,80,0.250,0.048,,
pythia-160m,an-you-novoc,3shot,all,0.412,80,0.250,0.055,,
pythia-160m,adj-adj,0shot,all,0.108,260,0.333,0.019,,
pythia-160m,adj-adj,3shot_collide,all,0.015,260,0.333,0.008,,
pythia-160m,adj-adj,3shot,all,0.096,260,0.333,0.018,,
pythia-160m,adj-voc,0shot,all,0.139,260,0.333,0.021,,
pythia-160m,adj-voc,3shot_collide,all,0.015,260,0.333,0.008,,
pythia-160m,adj-voc,3shot,all,0.119,260,0.333,0.020,,
pythia-160m,adj-third,0shot,all,0.211,260,0.333,0.025,,
pythia-160m,adj-third,3shot_collide,all,0.046,260,0.333,0.013,,
pythia-160m,adj-third,3shot,all,0.131,260,0.333,0.021,,
pythia-160m,cm-mutate,0shot,all,0.240,200,0.200,0.030,,
pythia-160m,cm-mutate,3shot_collide,all,0.160,200,0.200,0.026,,
pythia-160m,cm-mutate,3shot,all,0.170,200,0.200,0.027,,
pythia-160m,cm-rebind,0shot,all,0.100,200,0.200,0.021,,
pythia-160m,cm-rebind,3shot_collide,all,0.090,200,0.200,0.020,,
pythia-160m,cm-rebind,3shot,all,0.040,200,0.200,0.014,,
pythia-160m,cm-copy,0shot,all,0.040,200,0.200,0.014,,
pythia-160m,cm-copy,3shot_collide,all,0.095,200,0.200,0.021,,
pythia-160m,cm-copy,3shot,all,0.060,200,0.200,0.017,,
smollm2-360m,attr,0shot,all,0.452,1210,0.168,0.014,,
smollm2-360m,attr,3shot_collide,all,0.598,1210,0.168,0.014,,
smollm2-360m,attr,3shot,all,0.607,1210,0.168,0.014,,
smollm2-360m,codevar,0shot,all,0.310,1200,0.267,0.013,,
smollm2-360m,codevar,3shot_collide,all,0.398,1200,0.267,0.014,,
smollm2-360m,codevar,3shot,all,0.383,1200,0.267,0.014,,
smollm2-360m,dialogue,0shot,all,0.576,1200,0.312,0.014,,
smollm2-360m,dialogue,3shot_collide,all,0.462,1200,0.312,0.014,,
smollm2-360m,dialogue,3shot,all,0.496,1200,0.312,0.014,,
smollm2-360m,dx-owner-3-deictic,0shot,all,0.317,120,0.333,0.043,,other_participant=0.35;speaker_confusion=0.308;tie=0.025
smollm2-360m,dx-owner-3-deictic,0shot,free,0.250,40,0.333,0.069,,other_participant=0.525;speaker_confusion=0.225
smollm2-360m,dx-owner-3-deictic,3shot_collide,all,0.125,120,0.333,0.030,,other_participant=0.483;speaker_confusion=0.392
smollm2-360m,dx-owner-3-deictic,3shot_collide,free,0.025,40,0.333,0.025,,other_participant=0.65;speaker_confusion=0.325
smollm2-360m,dx-owner-3-deictic,3shot,all,0.225,120,0.333,0.038,,other_participant=0.35;speaker_confusion=0.417;tie=0.008
smollm2-360m,dx-owner-3-deictic,3shot,free,0.175,40,0.333,0.060,,other_participant=0.5;speaker_confusion=0.325
smollm2-360m,dx-owner-3-third,0shot,all,0.533,120,0.333,0.045,,
smollm2-360m,dx-owner-3-third,0shot,free,0.600,40,0.333,0.077,,
smollm2-360m,dx-owner-3-third,3shot_collide,all,0.358,120,0.333,0.044,,
smollm2-360m,dx-owner-3-third,3shot_collide,free,0.175,40,0.333,0.060,,
smollm2-360m,dx-owner-3-third,3shot,all,0.408,120,0.333,0.045,,
smollm2-360m,dx-owner-3-third,3shot,free,0.425,40,0.333,0.078,,
smollm2-360m,dx-self-3-deictic,0shot,all,0.350,180,0.287,0.036,,other_object=0.233;previous_owner=0.039;speaker_confusion=0.372;tie=0.006
smollm2-360m,dx-self-3-deictic,0shot,free,0.302,126,0.286,0.041,,other_object=0.238;previous_owner=0.04;speaker_confusion=0.421
smollm2-360m,dx-self-3-deictic,3shot_collide,all,0.361,180,0.287,0.036,,other_object=0.272;previous_owner=0.061;speaker_confusion=0.283;tie=0.022
smollm2-360m,dx-self-3-deictic,3shot_collide,free,0.444,126,0.286,0.044,,other_object=0.254;previous_owner=0.032;speaker_confusion=0.246;tie=0.024
smollm2-360m,dx-self-3-deictic,3shot,all,0.339,180,0.287,0.035,,other_object=0.283;previous_owner=0.061;speaker_confusion=0.306;tie=0.011
smollm2-360m,dx-self-3-deictic,3shot,free,0.397,126,0.286,0.044,,other_object=0.286;previous_owner=0.032;speaker_confusion=0.27;tie=0.016
smollm2-360m,dx-self-3-third,0shot,all,0.300,180,0.287,0.034,,
smollm2-360m,dx-self-3-third,0shot,free,0.206,126,0.286,0.036,,
smollm2-360m,dx-self-3-third,3shot_collide,all,0.372,180,0.287,0.036,,
smollm2-360m,dx-self-3-third,3shot_collide,free,0.349,126,0.286,0.043,,
smollm2-360m,dx-self-3-third,3shot,all,0.339,180,0.287,0.035,,
smollm2-360m,dx-self-3-third,3shot,free,0.286,126,0.286,0.040,,
smollm2-360m,dx-self-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.683;tie=0.011
smollm2-360m,dx-self-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.77;tie=0.016
smollm2-360m,dx-self-2-deictic,3shot_collide,all,0.322,180,0.287,0.035,,speaker_confusion=0.667;tie=0.011
smollm2-360m,dx-self-2-deictic,3shot_collide,free,0.381,126,0.286,0.043,,speaker_confusion=0.603;tie=0.016
smollm2-360m,dx-self-2-deictic,3shot,all,0.261,180,0.287,0.033,,speaker_confusion=0.706;tie=0.033
smollm2-360m,dx-self-2-deictic,3shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.643;tie=0.048
smollm2-360m,dx-self-2-third,0shot,all,0.322,180,0.287,0.035,,
smollm2-360m,dx-self-2-third,0shot,free,0.254,126,0.286,0.039,,
smollm2-360m,dx-self-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
smollm2-360m,dx-self-2-third,3shot_collide,free,0.349,126,0.286,0.043,,
smollm2-360m,dx-self-2-third,3shot,all,0.294,180,0.287,0.034,,
smollm2-360m,dx-self-2-third,3shot,free,0.294,126,0.286,0.041,,
smollm2-360m,dx-addressee-3-deictic,0shot,all,0.211,180,0.287,0.030,,other_object=0.183;previous_owner=0.139;speaker_confusion=0.45;tie=0.017
smollm2-360m,dx-addressee-3-deictic,0shot,free,0.198,126,0.286,0.035,,other_object=0.143;previous_owner=0.183;speaker_confusion=0.452;tie=0.024
smollm2-360m,dx-addressee-3-deictic,3shot_collide,all,0.283,180,0.287,0.034,,other_object=0.233;previous_owner=0.117;speaker_confusion=0.339;tie=0.028
smollm2-360m,dx-addressee-3-deictic,3shot_collide,free,0.309,126,0.286,0.041,,other_object=0.198;previous_owner=0.151;speaker_confusion=0.31;tie=0.032
smollm2-360m,dx-addressee-3-deictic,3shot,all,0.283,180,0.287,0.034,,other_object=0.233;previous_owner=0.089;speaker_confusion=0.383;tie=0.011
smollm2-360m,dx-addressee-3-deictic,3shot,free,0.294,126,0.286,0.041,,other_object=0.183;previous_owner=0.119;speaker_confusion=0.389;tie=0.016
smollm2-360m,dx-addressee-3-third,0shot,all,0.272,180,0.287,0.033,,
smollm2-360m,dx-addressee-3-third,0shot,free,0.222,126,0.286,0.037,,
smollm2-360m,dx-addressee-3-third,3shot_collide,all,0.272,180,0.287,0.033,,
smollm2-360m,dx-addressee-3-third,3shot_collide,free,0.254,126,0.286,0.039,,
smollm2-360m,dx-addressee-3-third,3shot,all,0.300,180,0.287,0.034,,
smollm2-360m,dx-addressee-3-third,3shot,free,0.262,126,0.286,0.039,,
smollm2-360m,dx-addressee-2-deictic,0shot,all,0.250,180,0.287,0.032,,speaker_confusion=0.744;tie=0.006
smollm2-360m,dx-addressee-2-deictic,0shot,free,0.151,126,0.286,0.032,,speaker_confusion=0.841;tie=0.008
smollm2-360m,dx-addressee-2-deictic,3shot_collide,all,0.339,180,0.287,0.035,,speaker_confusion=0.644;tie=0.017
smollm2-360m,dx-addressee-2-deictic,3shot_collide,free,0.389,126,0.286,0.043,,speaker_confusion=0.603;tie=0.008
smollm2-360m,dx-addressee-2-deictic,3shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.672;tie=0.011
smollm2-360m,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
smollm2-360m,dx-addressee-2-third,0shot,all,0.333,180,0.287,0.035,,
smollm2-360m,dx-addressee-2-third,0shot,free,0.246,126,0.286,0.038,,
smollm2-360m,dx-addressee-2-third,3shot_collide,all,0.328,180,0.287,0.035,,
smollm2-360m,dx-addressee-2-third,3shot_collide,free,0.357,126,0.286,0.043,,
smollm2-360m,dx-addressee-2-third,3shot,all,0.344,180,0.287,0.035,,
smollm2-360m,dx-addressee-2-third,3shot,free,0.349,126,0.286,0.043,,
smollm2-360m,sb-k1,0shot,all,0.391,64,0.125,0.061,,
smollm2-360m,sb-k1,3shot_collide,all,0.562,64,0.125,0.062,,
smollm2-360m,sb-k1,3shot,all,0.500,64,0.125,0.062,,
smollm2-360m,sb-k23,0shot,all,0.156,128,0.027,0.032,,
smollm2-360m,sb-k23,3shot_collide,all,0.227,128,0.027,0.037,,
smollm2-360m,sb-k23,3shot,all,0.258,128,0.027,0.039,,
smollm2-360m,sb-split,0shot,all,0.102,128,0.027,0.027,,
smollm2-360m,sb-split,3shot_collide,all,0.031,128,0.027,0.015,,
smollm2-360m,sb-split,3shot,all,0.047,128,0.027,0.019,,
smollm2-360m,sb-dyn,0shot,all,0.086,256,0.027,0.018,,
smollm2-360m,sb-dyn,3shot_collide,all,0.035,256,0.027,0.011,,
smollm2-360m,sb-dyn,3shot,all,0.047,256,0.027,0.013,,
smollm2-360m,an-narr,0shot,all,0.317,640,0.250,0.018,,
smollm2-360m,an-narr,3shot_collide,all,0.159,640,0.250,0.015,,
smollm2-360m,an-narr,3shot,all,0.278,640,0.250,0.018,,
smollm2-360m,an-other-any,0shot,all,0.181,320,0.250,0.021,,
smollm2-360m,an-other-any,3shot_collide,all,0.338,320,0.250,0.026,,
smollm2-360m,an-other-any,3shot,all,0.266,320,0.250,0.025,,
smollm2-360m,an-me-any,0shot,all,0.438,160,0.250,0.039,,
smollm2-360m,an-me-any,3shot_collide,all,0.006,160,0.250,0.006,,
smollm2-360m,an-me-any,3shot,all,0.019,160,0.250,0.011,,
smollm2-360m,an-you-voc,0shot,all,0.600,80,0.250,0.055,,
smollm2-360m,an-you-voc,3shot_collide,all,0.263,80,0.250,0.049,,
smollm2-360m,an-you-voc,3shot,all,0.263,80,0.250,0.049,,
smollm2-360m,an-you-novoc,0shot,all,0.400,80,0.250,0.055,,
smollm2-360m,an-you-novoc,3shot_collide,all,0.463,80,0.250,0.056,,
smollm2-360m,an-you-novoc,3shot,all,0.588,80,0.250,0.055,,
smollm2-360m,adj-adj,0shot,all,0.400,260,0.333,0.030,,
smollm2-360m,adj-adj,3shot_collide,all,0.158,260,0.333,0.023,,
smollm2-360m,adj-adj,3shot,all,0.258,260,0.333,0.027,,
smollm2-360m,adj-voc,0shot,all,0.423,260,0.333,0.031,,
smollm2-360m,adj-voc,3shot_collide,all,0.235,260,0.333,0.026,,
smollm2-360m,adj-voc,3shot,all,0.277,260,0.333,0.028,,
smollm2-360m,adj-third,0shot,all,0.581,260,0.333,0.031,,
smollm2-360m,adj-third,3shot_collide,all,0.288,260,0.333,0.028,,
smollm2-360m,adj-third,3shot,all,0.354,260,0.333,0.030,,
smollm2-360m,cm-mutate,0shot,all,0.565,200,0.200,0.035,,
smollm2-360m,cm-mutate,3shot_collide,all,0.465,200,0.200,0.035,,
smollm2-360m,cm-mutate,3shot,all,0.370,200,0.200,0.034,,
smollm2-360m,cm-rebind,0shot,all,0.035,200,0.200,0.013,,
smollm2-360m,cm-rebind,3shot_collide,all,0.260,200,0.200,0.031,,
smollm2-360m,cm-rebind,3shot,all,0.110,200,0.200,0.022,,
smollm2-360m,cm-copy,0shot,all,0.010,200,0.200,0.007,,
smollm2-360m,cm-copy,3shot_collide,all,0.315,200,0.200,0.033,,
smollm2-360m,cm-copy,3shot,all,0.150,200,0.200,0.025,,
pythia-410m,attr,0shot,all,0.526,1210,0.168,0.014,,
pythia-410m,attr,3shot_collide,all,0.567,1210,0.168,0.014,,
pythia-410m,attr,3shot,all,0.619,1210,0.168,0.014,,
pythia-410m,codevar,0shot,all,0.250,1200,0.267,0.013,,
pythia-410m,codevar,3shot_collide,all,0.383,1200,0.267,0.014,,
pythia-410m,codevar,3shot,all,0.348,1200,0.267,0.014,,
pythia-410m,dialogue,0shot,all,0.391,1200,0.312,0.014,,
pythia-410m,dialogue,3shot_collide,all,0.409,1200,0.312,0.014,,
pythia-410m,dialogue,3shot,all,0.451,1200,0.312,0.014,,
pythia-410m,dx-owner-3-deictic,0shot,all,0.408,120,0.333,0.045,,other_participant=0.433;speaker_confusion=0.15;tie=0.008
pythia-410m,dx-owner-3-deictic,0shot,free,0.400,40,0.333,0.077,,other_participant=0.45;speaker_confusion=0.125;tie=0.025
pythia-410m,dx-owner-3-deictic,3shot_collide,all,0.117,120,0.333,0.029,,other_participant=0.483;speaker_confusion=0.4
pythia-410m,dx-owner-3-deictic,3shot_collide,free,0.100,40,0.333,0.047,,other_participant=0.6;speaker_confusion=0.3
pythia-410m,dx-owner-3-deictic,3shot,all,0.400,120,0.333,0.045,,other_participant=0.367;speaker_confusion=0.217;tie=0.017
pythia-410m,dx-owner-3-deictic,3shot,free,0.525,40,0.333,0.079,,other_participant=0.325;speaker_confusion=0.15
pythia-410m,dx-owner-3-third,0shot,all,0.450,120,0.333,0.045,,
pythia-410m,dx-owner-3-third,0shot,free,0.450,40,0.333,0.079,,
pythia-410m,dx-owner-3-third,3shot_collide,all,0.233,120,0.333,0.039,,
pythia-410m,dx-owner-3-third,3shot_collide,free,0.175,40,0.333,0.060,,
pythia-410m,dx-owner-3-third,3shot,all,0.467,120,0.333,0.045,,
pythia-410m,dx-owner-3-third,3shot,free,0.450,40,0.333,0.079,,
pythia-410m,dx-self-3-deictic,0shot,all,0.283,180,0.287,0.034,,other_object=0.244;previous_owner=0.128;speaker_confusion=0.339;tie=0.006
pythia-410m,dx-self-3-deictic,0shot,free,0.318,126,0.286,0.042,,other_object=0.262;previous_owner=0.087;speaker_confusion=0.333
pythia-410m,dx-self-3-deictic,3shot_collide,all,0.294,180,0.287,0.034,,other_object=0.278;previous_owner=0.094;speaker_confusion=0.311;tie=0.022
pythia-410m,dx-self-3-deictic,3shot_collide,free,0.333,126,0.286,0.042,,other_object=0.302;previous_owner=0.056;speaker_confusion=0.286;tie=0.024
pythia-410m,dx-self-3-deictic,3shot,all,0.322,180,0.287,0.035,,other_object=0.289;previous_owner=0.072;speaker_confusion=0.317
pythia-410m,dx-self-3-deictic,3shot,free,0.373,126,0.286,0.043,,other_object=0.31;previous_owner=0.048;speaker_confusion=0.27
pythia-410m,dx-self-3-third,0shot,all,0.311,180,0.287,0.035,,
pythia-410m,dx-self-3-third,0shot,free,0.365,126,0.286,0.043,,
pythia-410m,dx-self-3-third,3shot_collide,all,0.333,180,0.287,0.035,,
pythia-410m,dx-self-3-third,3shot_collide,free,0.389,126,0.286,0.043,,
pythia-410m,dx-self-3-third,3shot,all,0.294,180,0.287,0.034,,
pythia-410m,dx-self-3-third,3shot,free,0.365,126,0.286,0.043,,
pythia-410m,dx-self-2-deictic,0shot,all,0.239,180,0.287,0.032,,speaker_confusion=0.75;tie=0.011
pythia-410m,dx-self-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.802;tie=0.008
pythia-410m,dx-self-2-deictic,3shot_collide,all,0.306,180,0.287,0.034,,speaker_confusion=0.689;tie=0.006
pythia-410m,dx-self-2-deictic,3shot_collide,free,0.333,126,0.286,0.042,,speaker_confusion=0.659;tie=0.008
pythia-410m,dx-self-2-deictic,3shot,all,0.294,180,0.287,0.034,,speaker_confusion=0.706
pythia-410m,dx-self-2-deictic,3shot,free,0.302,126,0.286,0.041,,speaker_confusion=0.698
pythia-410m,dx-self-2-third,0shot,all,0.256,180,0.287,0.033,,
pythia-410m,dx-self-2-third,0shot,free,0.278,126,0.286,0.040,,
pythia-410m,dx-self-2-third,3shot_collide,all,0.283,180,0.287,0.034,,
pythia-410m,dx-self-2-third,3shot_collide,free,0.325,126,0.286,0.042,,
pythia-410m,dx-self-2-third,3shot,all,0.328,180,0.287,0.035,,
pythia-410m,dx-self-2-third,3shot,free,0.373,126,0.286,0.043,,
pythia-410m,dx-addressee-3-deictic,0shot,all,0.361,180,0.287,0.036,,other_object=0.133;previous_owner=0.172;speaker_confusion=0.333
pythia-410m,dx-addressee-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.103;previous_owner=0.214;speaker_confusion=0.373
pythia-410m,dx-addressee-3-deictic,3shot_collide,all,0.283,180,0.287,0.034,,other_object=0.178;previous_owner=0.15;speaker_confusion=0.383;tie=0.006
pythia-410m,dx-addressee-3-deictic,3shot_collide,free,0.278,126,0.286,0.040,,other_object=0.151;previous_owner=0.167;speaker_confusion=0.397;tie=0.008
pythia-410m,dx-addressee-3-deictic,3shot,all,0.322,180,0.287,0.035,,other_object=0.2;previous_owner=0.122;speaker_confusion=0.35;tie=0.006
pythia-410m,dx-addressee-3-deictic,3shot,free,0.333,126,0.286,0.042,,other_object=0.151;previous_owner=0.159;speaker_confusion=0.349;tie=0.008
pythia-410m,dx-addressee-3-third,0shot,all,0.283,180,0.287,0.034,,
pythia-410m,dx-addressee-3-third,0shot,free,0.318,126,0.286,0.042,,
pythia-410m,dx-addressee-3-third,3shot_collide,all,0.350,180,0.287,0.036,,
pythia-410m,dx-addressee-3-third,3shot_collide,free,0.381,126,0.286,0.043,,
pythia-410m,dx-addressee-3-third,3shot,all,0.322,180,0.287,0.035,,
pythia-410m,dx-addressee-3-third,3shot,free,0.397,126,0.286,0.044,,
pythia-410m,dx-addressee-2-deictic,0shot,all,0.233,180,0.287,0.032,,speaker_confusion=0.761;tie=0.006
pythia-410m,dx-addressee-2-deictic,0shot,free,0.151,126,0.286,0.032,,speaker_confusion=0.849
pythia-410m,dx-addressee-2-deictic,3shot_collide,all,0.278,180,0.287,0.033,,speaker_confusion=0.717;tie=0.006
pythia-410m,dx-addressee-2-deictic,3shot_collide,free,0.222,126,0.286,0.037,,speaker_confusion=0.77;tie=0.008
pythia-410m,dx-addressee-2-deictic,3shot,all,0.239,180,0.287,0.032,,speaker_confusion=0.756;tie=0.006
pythia-410m,dx-addressee-2-deictic,3shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.81
pythia-410m,dx-addressee-2-third,0shot,all,0.261,180,0.287,0.033,,
pythia-410m,dx-addressee-2-third,0shot,free,0.302,126,0.286,0.041,,
pythia-410m,dx-addressee-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
pythia-410m,dx-addressee-2-third,3shot_collide,free,0.365,126,0.286,0.043,,
pythia-410m,dx-addressee-2-third,3shot,all,0.350,180,0.287,0.036,,
pythia-410m,dx-addressee-2-third,3shot,free,0.381,126,0.286,0.043,,
pythia-410m,sb-k1,0shot,all,0.578,64,0.125,0.062,,
pythia-410m,sb-k1,3shot_collide,all,0.375,64,0.125,0.060,,
pythia-410m,sb-k1,3shot,all,0.391,64,0.125,0.061,,
pythia-410m,sb-k23,0shot,all,0.102,128,0.027,0.027,,
pythia-410m,sb-k23,3shot_collide,all,0.227,128,0.027,0.037,,
pythia-410m,sb-k23,3shot,all,0.219,128,0.027,0.036,,
pythia-410m,sb-split,0shot,all,0.125,128,0.027,0.029,,
pythia-410m,sb-split,3shot_collide,all,0.039,128,0.027,0.017,,
pythia-410m,sb-split,3shot,all,0.078,128,0.027,0.024,,
pythia-410m,sb-dyn,0shot,all,0.105,256,0.027,0.019,,
pythia-410m,sb-dyn,3shot_collide,all,0.027,256,0.027,0.010,,
pythia-410m,sb-dyn,3shot,all,0.055,256,0.027,0.014,,
pythia-410m,an-narr,0shot,all,0.372,640,0.250,0.019,,
pythia-410m,an-narr,3shot_collide,all,0.119,640,0.250,0.013,,
pythia-410m,an-narr,3shot,all,0.356,640,0.250,0.019,,
pythia-410m,an-other-any,0shot,all,0.241,320,0.250,0.024,,
pythia-410m,an-other-any,3shot_collide,all,0.356,320,0.250,0.027,,
pythia-410m,an-other-any,3shot,all,0.256,320,0.250,0.024,,
pythia-410m,an-me-any,0shot,all,0.219,160,0.250,0.033,,
pythia-410m,an-me-any,3shot_collide,all,0.231,160,0.250,0.033,,
pythia-410m,an-me-any,3shot,all,0.206,160,0.250,0.032,,
pythia-410m,an-you-voc,0shot,all,0.300,80,0.250,0.051,,
pythia-410m,an-you-voc,3shot_collide,all,0.113,80,0.250,0.035,,
pythia-410m,an-you-voc,3shot,all,0.075,80,0.250,0.029,,
pythia-410m,an-you-novoc,0shot,all,0.787,80,0.250,0.046,,
pythia-410m,an-you-novoc,3shot_collide,all,0.087,80,0.250,0.032,,
pythia-410m,an-you-novoc,3shot,all,0.163,80,0.250,0.041,,
pythia-410m,adj-adj,0shot,all,0.354,260,0.333,0.030,,
pythia-410m,adj-adj,3shot_collide,all,0.088,260,0.333,0.018,,
pythia-410m,adj-adj,3shot,all,0.396,260,0.333,0.030,,
pythia-410m,adj-voc,0shot,all,0.415,260,0.333,0.031,,
pythia-410m,adj-voc,3shot_collide,all,0.112,260,0.333,0.019,,
pythia-410m,adj-voc,3shot,all,0.396,260,0.333,0.030,,
pythia-410m,adj-third,0shot,all,0.488,260,0.333,0.031,,
pythia-410m,adj-third,3shot_collide,all,0.342,260,0.333,0.029,,
pythia-410m,adj-third,3shot,all,0.469,260,0.333,0.031,,
pythia-410m,cm-mutate,0shot,all,0.320,200,0.200,0.033,,
pythia-410m,cm-mutate,3shot_collide,all,0.415,200,0.200,0.035,,
pythia-410m,cm-mutate,3shot,all,0.340,200,0.200,0.034,,
pythia-410m,cm-rebind,0shot,all,0.130,200,0.200,0.024,,
pythia-410m,cm-rebind,3shot_collide,all,0.325,200,0.200,0.033,,
pythia-410m,cm-rebind,3shot,all,0.155,200,0.200,0.026,,
pythia-410m,cm-copy,0shot,all,0.125,200,0.200,0.023,,
pythia-410m,cm-copy,3shot_collide,all,0.315,200,0.200,0.033,,
pythia-410m,cm-copy,3shot,all,0.270,200,0.200,0.031,,
qwen2.5-0.5b,attr,0shot,all,0.388,1210,0.168,0.014,,
qwen2.5-0.5b,attr,3shot_collide,all,0.786,1210,0.168,0.012,,
qwen2.5-0.5b,attr,3shot,all,0.946,1210,0.168,0.006,,
qwen2.5-0.5b,codevar,0shot,all,0.326,1200,0.267,0.013,,
qwen2.5-0.5b,codevar,3shot_collide,all,0.422,1200,0.267,0.014,,
qwen2.5-0.5b,codevar,3shot,all,0.417,1200,0.267,0.014,,
qwen2.5-0.5b,dialogue,0shot,all,0.450,1200,0.312,0.014,,
qwen2.5-0.5b,dialogue,3shot_collide,all,0.473,1200,0.312,0.014,,
qwen2.5-0.5b,dialogue,3shot,all,0.463,1200,0.312,0.014,,
qwen2.5-0.5b,dx-owner-3-deictic,0shot,all,0.450,120,0.333,0.045,,other_participant=0.325;speaker_confusion=0.217;tie=0.008
qwen2.5-0.5b,dx-owner-3-deictic,0shot,free,0.475,40,0.333,0.079,,other_participant=0.4;speaker_confusion=0.1;tie=0.025
qwen2.5-0.5b,dx-owner-3-deictic,3shot_collide,all,0.133,120,0.333,0.031,,other_participant=0.525;speaker_confusion=0.333;tie=0.008
qwen2.5-0.5b,dx-owner-3-deictic,3shot_collide,free,0.025,40,0.333,0.025,,other_participant=0.725;speaker_confusion=0.25
qwen2.5-0.5b,dx-owner-3-deictic,3shot,all,0.225,120,0.333,0.038,,other_participant=0.417;speaker_confusion=0.342;tie=0.017
qwen2.5-0.5b,dx-owner-3-deictic,3shot,free,0.150,40,0.333,0.057,,other_participant=0.575;speaker_confusion=0.275
qwen2.5-0.5b,dx-owner-3-third,0shot,all,0.492,120,0.333,0.046,,
qwen2.5-0.5b,dx-owner-3-third,0shot,free,0.500,40,0.333,0.079,,
qwen2.5-0.5b,dx-owner-3-third,3shot_collide,all,0.317,120,0.333,0.043,,
qwen2.5-0.5b,dx-owner-3-third,3shot_collide,free,0.150,40,0.333,0.057,,
qwen2.5-0.5b,dx-owner-3-third,3shot,all,0.467,120,0.333,0.045,,
qwen2.5-0.5b,dx-owner-3-third,3shot,free,0.450,40,0.333,0.079,,
qwen2.5-0.5b,dx-self-3-deictic,0shot,all,0.300,180,0.287,0.034,,other_object=0.283;previous_owner=0.061;speaker_confusion=0.35;tie=0.006
qwen2.5-0.5b,dx-self-3-deictic,0shot,free,0.373,126,0.286,0.043,,other_object=0.278;previous_owner=0.024;speaker_confusion=0.317;tie=0.008
qwen2.5-0.5b,dx-self-3-deictic,3shot_collide,all,0.306,180,0.287,0.034,,other_object=0.306;previous_owner=0.05;speaker_confusion=0.339
qwen2.5-0.5b,dx-self-3-deictic,3shot_collide,free,0.341,126,0.286,0.042,,other_object=0.317;previous_owner=0.032;speaker_confusion=0.31
qwen2.5-0.5b,dx-self-3-deictic,3shot,all,0.311,180,0.287,0.035,,other_object=0.322;previous_owner=0.033;speaker_confusion=0.333
qwen2.5-0.5b,dx-self-3-deictic,3shot,free,0.325,126,0.286,0.042,,other_object=0.349;previous_owner=0.024;speaker_confusion=0.302
qwen2.5-0.5b,dx-self-3-third,0shot,all,0.239,180,0.287,0.032,,
qwen2.5-0.5b,dx-self-3-third,0shot,free,0.095,126,0.286,0.026,,
qwen2.5-0.5b,dx-self-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
qwen2.5-0.5b,dx-self-3-third,3shot_collide,free,0.381,126,0.286,0.043,,
qwen2.5-0.5b,dx-self-3-third,3shot,all,0.272,180,0.287,0.033,,
qwen2.5-0.5b,dx-self-3-third,3shot,free,0.302,126,0.286,0.041,,
qwen2.5-0.5b,dx-self-2-deictic,0shot,all,0.272,180,0.287,0.033,,speaker_confusion=0.711;tie=0.017
qwen2.5-0.5b,dx-self-2-deictic,0shot,free,0.318,126,0.286,0.042,,speaker_confusion=0.667;tie=0.016
qwen2.5-0.5b,dx-self-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.661;tie=0.011
qwen2.5-0.5b,dx-self-2-deictic,3shot_collide,free,0.389,126,0.286,0.043,,speaker_confusion=0.603;tie=0.008
qwen2.5-0.5b,dx-self-2-deictic,3shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.694
qwen2.5-0.5b,dx-self-2-deictic,3shot,free,0.373,126,0.286,0.043,,speaker_confusion=0.627
qwen2.5-0.5b,dx-self-2-third,0shot,all,0.333,180,0.287,0.035,,
qwen2.5-0.5b,dx-self-2-third,0shot,free,0.214,126,0.286,0.037,,
qwen2.5-0.5b,dx-self-2-third,3shot_collide,all,0.317,180,0.287,0.035,,
qwen2.5-0.5b,dx-self-2-third,3shot_collide,free,0.349,126,0.286,0.043,,
qwen2.5-0.5b,dx-self-2-third,3shot,all,0.322,180,0.287,0.035,,
qwen2.5-0.5b,dx-self-2-third,3shot,free,0.341,126,0.286,0.042,,
qwen2.5-0.5b,dx-addressee-3-deictic,0shot,all,0.306,180,0.287,0.034,,other_object=0.194;previous_owner=0.15;speaker_confusion=0.333;tie=0.017
qwen2.5-0.5b,dx-addressee-3-deictic,0shot,free,0.349,126,0.286,0.043,,other_object=0.127;previous_owner=0.167;speaker_confusion=0.341;tie=0.016
qwen2.5-0.5b,dx-addressee-3-deictic,3shot_collide,all,0.278,180,0.287,0.033,,other_object=0.222;previous_owner=0.139;speaker_confusion=0.344;tie=0.017
qwen2.5-0.5b,dx-addressee-3-deictic,3shot_collide,free,0.302,126,0.286,0.041,,other_object=0.198;previous_owner=0.159;speaker_confusion=0.325;tie=0.016
qwen2.5-0.5b,dx-addressee-3-deictic,3shot,all,0.306,180,0.287,0.034,,other_object=0.233;previous_owner=0.111;speaker_confusion=0.339;tie=0.011
qwen2.5-0.5b,dx-addressee-3-deictic,3shot,free,0.318,126,0.286,0.042,,other_object=0.19;previous_owner=0.111;speaker_confusion=0.365;tie=0.016
qwen2.5-0.5b,dx-addressee-3-third,0shot,all,0.317,180,0.287,0.035,,
qwen2.5-0.5b,dx-addressee-3-third,0shot,free,0.159,126,0.286,0.033,,
qwen2.5-0.5b,dx-addressee-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
qwen2.5-0.5b,dx-addressee-3-third,3shot_collide,free,0.302,126,0.286,0.041,,
qwen2.5-0.5b,dx-addressee-3-third,3shot,all,0.311,180,0.287,0.035,,
qwen2.5-0.5b,dx-addressee-3-third,3shot,free,0.302,126,0.286,0.041,,
qwen2.5-0.5b,dx-addressee-2-deictic,0shot,all,0.233,180,0.287,0.032,,speaker_confusion=0.761;tie=0.006
qwen2.5-0.5b,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.786
qwen2.5-0.5b,dx-addressee-2-deictic,3shot_collide,all,0.394,180,0.287,0.036,,speaker_confusion=0.583;tie=0.022
qwen2.5-0.5b,dx-addressee-2-deictic,3shot_collide,free,0.429,126,0.286,0.044,,speaker_confusion=0.556;tie=0.016
qwen2.5-0.5b,dx-addressee-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.583;tie=0.039
qwen2.5-0.5b,dx-addressee-2-deictic,3shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.556;tie=0.04
qwen2.5-0.5b,dx-addressee-2-third,0shot,all,0.272,180,0.287,0.033,,
qwen2.5-0.5b,dx-addressee-2-third,0shot,free,0.159,126,0.286,0.033,,
qwen2.5-0.5b,dx-addressee-2-third,3shot_collide,all,0.344,180,0.287,0.035,,
qwen2.5-0.5b,dx-addressee-2-third,3shot_collide,free,0.373,126,0.286,0.043,,
qwen2.5-0.5b,dx-addressee-2-third,3shot,all,0.350,180,0.287,0.036,,
qwen2.5-0.5b,dx-addressee-2-third,3shot,free,0.373,126,0.286,0.043,,
qwen2.5-0.5b,sb-k1,0shot,all,0.469,64,0.125,0.062,,
qwen2.5-0.5b,sb-k1,3shot_collide,all,0.484,64,0.125,0.062,,
qwen2.5-0.5b,sb-k1,3shot,all,0.828,64,0.125,0.047,,
qwen2.5-0.5b,sb-k23,0shot,all,0.125,128,0.027,0.029,,
qwen2.5-0.5b,sb-k23,3shot_collide,all,0.266,128,0.027,0.039,,
qwen2.5-0.5b,sb-k23,3shot,all,0.352,128,0.027,0.042,,
qwen2.5-0.5b,sb-split,0shot,all,0.164,128,0.027,0.033,,
qwen2.5-0.5b,sb-split,3shot_collide,all,0.117,128,0.027,0.028,,
qwen2.5-0.5b,sb-split,3shot,all,0.148,128,0.027,0.031,,
qwen2.5-0.5b,sb-dyn,0shot,all,0.137,256,0.027,0.021,,
qwen2.5-0.5b,sb-dyn,3shot_collide,all,0.047,256,0.027,0.013,,
qwen2.5-0.5b,sb-dyn,3shot,all,0.059,256,0.027,0.015,,
qwen2.5-0.5b,an-narr,0shot,all,0.342,640,0.250,0.019,,
qwen2.5-0.5b,an-narr,3shot_collide,all,0.189,640,0.250,0.015,,
qwen2.5-0.5b,an-narr,3shot,all,0.212,640,0.250,0.016,,
qwen2.5-0.5b,an-other-any,0shot,all,0.403,320,0.250,0.027,,
qwen2.5-0.5b,an-other-any,3shot_collide,all,0.250,320,0.250,0.024,,
qwen2.5-0.5b,an-other-any,3shot,all,0.269,320,0.250,0.025,,
qwen2.5-0.5b,an-me-any,0shot,all,0.469,160,0.250,0.040,,
qwen2.5-0.5b,an-me-any,3shot_collide,all,0.019,160,0.250,0.011,,
qwen2.5-0.5b,an-me-any,3shot,all,0.013,160,0.250,0.009,,
qwen2.5-0.5b,an-you-voc,0shot,all,0.113,80,0.250,0.035,,
qwen2.5-0.5b,an-you-voc,3shot_collide,all,0.163,80,0.250,0.041,,
qwen2.5-0.5b,an-you-voc,3shot,all,0.188,80,0.250,0.044,,
qwen2.5-0.5b,an-you-novoc,0shot,all,0.150,80,0.250,0.040,,
qwen2.5-0.5b,an-you-novoc,3shot_collide,all,0.275,80,0.250,0.050,,
qwen2.5-0.5b,an-you-novoc,3shot,all,0.263,80,0.250,0.049,,
qwen2.5-0.5b,adj-adj,0shot,all,0.373,260,0.333,0.030,,
qwen2.5-0.5b,adj-adj,3shot_collide,all,0.123,260,0.333,0.020,,
qwen2.5-0.5b,adj-adj,3shot,all,0.235,260,0.333,0.026,,
qwen2.5-0.5b,adj-voc,0shot,all,0.419,260,0.333,0.031,,
qwen2.5-0.5b,adj-voc,3shot_collide,all,0.146,260,0.333,0.022,,
qwen2.5-0.5b,adj-voc,3shot,all,0.238,260,0.333,0.026,,
qwen2.5-0.5b,adj-third,0shot,all,0.523,260,0.333,0.031,,
qwen2.5-0.5b,adj-third,3shot_collide,all,0.504,260,0.333,0.031,,
qwen2.5-0.5b,adj-third,3shot,all,0.550,260,0.333,0.031,,
qwen2.5-0.5b,cm-mutate,0shot,all,0.580,200,0.200,0.035,,
qwen2.5-0.5b,cm-mutate,3shot_collide,all,0.660,200,0.200,0.034,,
qwen2.5-0.5b,cm-mutate,3shot,all,0.760,200,0.200,0.030,,
qwen2.5-0.5b,cm-rebind,0shot,all,0.050,200,0.200,0.015,,
qwen2.5-0.5b,cm-rebind,3shot_collide,all,0.455,200,0.200,0.035,,
qwen2.5-0.5b,cm-rebind,3shot,all,0.385,200,0.200,0.034,,
qwen2.5-0.5b,cm-copy,0shot,all,0.035,200,0.200,0.013,,
qwen2.5-0.5b,cm-copy,3shot_collide,all,0.400,200,0.200,0.035,,
qwen2.5-0.5b,cm-copy,3shot,all,0.435,200,0.200,0.035,,
qwen3-0.6b-base,attr,0shot,all,0.626,1210,0.168,0.014,,
qwen3-0.6b-base,attr,3shot_collide,all,0.842,1210,0.168,0.011,,
qwen3-0.6b-base,attr,3shot,all,0.946,1210,0.168,0.006,,
qwen3-0.6b-base,codevar,0shot,all,0.397,1200,0.267,0.014,,
qwen3-0.6b-base,codevar,3shot_collide,all,0.512,1200,0.267,0.014,,
qwen3-0.6b-base,codevar,3shot,all,0.517,1200,0.267,0.014,,
qwen3-0.6b-base,dialogue,0shot,all,0.550,1200,0.312,0.014,,
qwen3-0.6b-base,dialogue,3shot_collide,all,0.569,1200,0.312,0.014,,
qwen3-0.6b-base,dialogue,3shot,all,0.579,1200,0.312,0.014,,
qwen3-0.6b-base,dx-owner-3-deictic,0shot,all,0.467,120,0.333,0.045,,other_participant=0.433;speaker_confusion=0.1
qwen3-0.6b-base,dx-owner-3-deictic,0shot,free,0.400,40,0.333,0.077,,other_participant=0.575;speaker_confusion=0.025
qwen3-0.6b-base,dx-owner-3-deictic,3shot_collide,all,0.308,120,0.333,0.042,,other_participant=0.408;speaker_confusion=0.275;tie=0.008
qwen3-0.6b-base,dx-owner-3-deictic,3shot_collide,free,0.200,40,0.333,0.063,,other_participant=0.625;speaker_confusion=0.175
qwen3-0.6b-base,dx-owner-3-deictic,3shot,all,0.442,120,0.333,0.045,,other_participant=0.325;speaker_confusion=0.225;tie=0.008
qwen3-0.6b-base,dx-owner-3-deictic,3shot,free,0.425,40,0.333,0.078,,other_participant=0.4;speaker_confusion=0.15;tie=0.025
qwen3-0.6b-base,dx-owner-3-third,0shot,all,0.542,120,0.333,0.045,,
qwen3-0.6b-base,dx-owner-3-third,0shot,free,0.600,40,0.333,0.077,,
qwen3-0.6b-base,dx-owner-3-third,3shot_collide,all,0.417,120,0.333,0.045,,
qwen3-0.6b-base,dx-owner-3-third,3shot_collide,free,0.375,40,0.333,0.076,,
qwen3-0.6b-base,dx-owner-3-third,3shot,all,0.517,120,0.333,0.046,,
qwen3-0.6b-base,dx-owner-3-third,3shot,free,0.575,40,0.333,0.078,,
qwen3-0.6b-base,dx-self-3-deictic,0shot,all,0.389,180,0.287,0.036,,other_object=0.189;previous_owner=0.067;speaker_confusion=0.322;tie=0.033
qwen3-0.6b-base,dx-self-3-deictic,0shot,free,0.397,126,0.286,0.044,,other_object=0.167;previous_owner=0.071;speaker_confusion=0.349;tie=0.016
qwen3-0.6b-base,dx-self-3-deictic,3shot_collide,all,0.367,180,0.287,0.036,,other_object=0.261;previous_owner=0.022;speaker_confusion=0.344;tie=0.006
qwen3-0.6b-base,dx-self-3-deictic,3shot_collide,free,0.436,126,0.286,0.044,,other_object=0.27;previous_owner=0.032;speaker_confusion=0.254;tie=0.008
qwen3-0.6b-base,dx-self-3-deictic,3shot,all,0.356,180,0.287,0.036,,other_object=0.278;previous_owner=0.039;speaker_confusion=0.317;tie=0.011
qwen3-0.6b-base,dx-self-3-deictic,3shot,free,0.436,126,0.286,0.044,,other_object=0.286;previous_owner=0.04;speaker_confusion=0.222;tie=0.016
qwen3-0.6b-base,dx-self-3-third,0shot,all,0.294,180,0.287,0.034,,
qwen3-0.6b-base,dx-self-3-third,0shot,free,0.309,126,0.286,0.041,,
qwen3-0.6b-base,dx-self-3-third,3shot_collide,all,0.289,180,0.287,0.034,,
qwen3-0.6b-base,dx-self-3-third,3shot_collide,free,0.381,126,0.286,0.043,,
qwen3-0.6b-base,dx-self-3-third,3shot,all,0.306,180,0.287,0.034,,
qwen3-0.6b-base,dx-self-3-third,3shot,free,0.397,126,0.286,0.044,,
qwen3-0.6b-base,dx-self-2-deictic,0shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.622;tie=0.006
qwen3-0.6b-base,dx-self-2-deictic,0shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.548;tie=0.008
qwen3-0.6b-base,dx-self-2-deictic,3shot_collide,all,0.417,180,0.287,0.037,,speaker_confusion=0.578;tie=0.006
qwen3-0.6b-base,dx-self-2-deictic,3shot_collide,free,0.532,126,0.286,0.044,,speaker_confusion=0.46;tie=0.008
qwen3-0.6b-base,dx-self-2-deictic,3shot,all,0.439,180,0.287,0.037,,speaker_confusion=0.544;tie=0.017
qwen3-0.6b-base,dx-self-2-deictic,3shot,free,0.556,126,0.286,0.044,,speaker_confusion=0.429;tie=0.016
qwen3-0.6b-base,dx-self-2-third,0shot,all,0.367,180,0.287,0.036,,
qwen3-0.6b-base,dx-self-2-third,0shot,free,0.436,126,0.286,0.044,,
qwen3-0.6b-base,dx-self-2-third,3shot_collide,all,0.428,180,0.287,0.037,,
qwen3-0.6b-base,dx-self-2-third,3shot_collide,free,0.540,126,0.286,0.044,,
qwen3-0.6b-base,dx-self-2-third,3shot,all,0.450,180,0.287,0.037,,
qwen3-0.6b-base,dx-self-2-third,3shot,free,0.548,126,0.286,0.044,,
qwen3-0.6b-base,dx-addressee-3-deictic,0shot,all,0.278,180,0.287,0.033,,other_object=0.189;previous_owner=0.106;speaker_confusion=0.417;tie=0.011
qwen3-0.6b-base,dx-addressee-3-deictic,0shot,free,0.278,126,0.286,0.040,,other_object=0.19;previous_owner=0.135;speaker_confusion=0.389;tie=0.008
qwen3-0.6b-base,dx-addressee-3-deictic,3shot_collide,all,0.311,180,0.287,0.035,,other_object=0.244;previous_owner=0.072;speaker_confusion=0.356;tie=0.017
qwen3-0.6b-base,dx-addressee-3-deictic,3shot_collide,free,0.373,126,0.286,0.043,,other_object=0.183;previous_owner=0.103;speaker_confusion=0.317;tie=0.024
qwen3-0.6b-base,dx-addressee-3-deictic,3shot,all,0.311,180,0.287,0.035,,other_object=0.244;previous_owner=0.05;speaker_confusion=0.378;tie=0.017
qwen3-0.6b-base,dx-addressee-3-deictic,3shot,free,0.357,126,0.286,0.043,,other_object=0.198;previous_owner=0.071;speaker_confusion=0.357;tie=0.016
qwen3-0.6b-base,dx-addressee-3-third,0shot,all,0.333,180,0.287,0.035,,
qwen3-0.6b-base,dx-addressee-3-third,0shot,free,0.373,126,0.286,0.043,,
qwen3-0.6b-base,dx-addressee-3-third,3shot_collide,all,0.344,180,0.287,0.035,,
qwen3-0.6b-base,dx-addressee-3-third,3shot_collide,free,0.436,126,0.286,0.044,,
qwen3-0.6b-base,dx-addressee-3-third,3shot,all,0.339,180,0.287,0.035,,
qwen3-0.6b-base,dx-addressee-3-third,3shot,free,0.436,126,0.286,0.044,,
qwen3-0.6b-base,dx-addressee-2-deictic,0shot,all,0.261,180,0.287,0.033,,speaker_confusion=0.717;tie=0.022
qwen3-0.6b-base,dx-addressee-2-deictic,0shot,free,0.302,126,0.286,0.041,,speaker_confusion=0.675;tie=0.024
qwen3-0.6b-base,dx-addressee-2-deictic,3shot_collide,all,0.372,180,0.287,0.036,,speaker_confusion=0.611;tie=0.017
qwen3-0.6b-base,dx-addressee-2-deictic,3shot_collide,free,0.429,126,0.286,0.044,,speaker_confusion=0.548;tie=0.024
qwen3-0.6b-base,dx-addressee-2-deictic,3shot,all,0.389,180,0.287,0.036,,speaker_confusion=0.611
qwen3-0.6b-base,dx-addressee-2-deictic,3shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
qwen3-0.6b-base,dx-addressee-2-third,0shot,all,0.433,180,0.287,0.037,,
qwen3-0.6b-base,dx-addressee-2-third,0shot,free,0.532,126,0.286,0.044,,
qwen3-0.6b-base,dx-addressee-2-third,3shot_collide,all,0.472,180,0.287,0.037,,
qwen3-0.6b-base,dx-addressee-2-third,3shot_collide,free,0.635,126,0.286,0.043,,
qwen3-0.6b-base,dx-addressee-2-third,3shot,all,0.494,180,0.287,0.037,,
qwen3-0.6b-base,dx-addressee-2-third,3shot,free,0.651,126,0.286,0.043,,
qwen3-0.6b-base,sb-k1,0shot,all,0.578,64,0.125,0.062,,
qwen3-0.6b-base,sb-k1,3shot_collide,all,0.891,64,0.125,0.039,,
qwen3-0.6b-base,sb-k1,3shot,all,0.953,64,0.125,0.026,,
qwen3-0.6b-base,sb-k23,0shot,all,0.414,128,0.027,0.043,,
qwen3-0.6b-base,sb-k23,3shot_collide,all,0.586,128,0.027,0.043,,
qwen3-0.6b-base,sb-k23,3shot,all,0.688,128,0.027,0.041,,
qwen3-0.6b-base,sb-split,0shot,all,0.273,128,0.027,0.039,,
qwen3-0.6b-base,sb-split,3shot_collide,all,0.305,128,0.027,0.041,,
qwen3-0.6b-base,sb-split,3shot,all,0.359,128,0.027,0.042,,
qwen3-0.6b-base,sb-dyn,0shot,all,0.117,256,0.027,0.020,,
qwen3-0.6b-base,sb-dyn,3shot_collide,all,0.113,256,0.027,0.020,,
qwen3-0.6b-base,sb-dyn,3shot,all,0.113,256,0.027,0.020,,
qwen3-0.6b-base,an-narr,0shot,all,0.388,640,0.250,0.019,,
qwen3-0.6b-base,an-narr,3shot_collide,all,0.316,640,0.250,0.018,,
qwen3-0.6b-base,an-narr,3shot,all,0.391,640,0.250,0.019,,
qwen3-0.6b-base,an-other-any,0shot,all,0.366,320,0.250,0.027,,
qwen3-0.6b-base,an-other-any,3shot_collide,all,0.312,320,0.250,0.026,,
qwen3-0.6b-base,an-other-any,3shot,all,0.244,320,0.250,0.024,,
qwen3-0.6b-base,an-me-any,0shot,all,0.194,160,0.250,0.031,,
qwen3-0.6b-base,an-me-any,3shot_collide,all,0.100,160,0.250,0.024,,
qwen3-0.6b-base,an-me-any,3shot,all,0.056,160,0.250,0.018,,
qwen3-0.6b-base,an-you-voc,0shot,all,0.350,80,0.250,0.053,,
qwen3-0.6b-base,an-you-voc,3shot_collide,all,0.350,80,0.250,0.053,,
qwen3-0.6b-base,an-you-voc,3shot,all,0.312,80,0.250,0.052,,
qwen3-0.6b-base,an-you-novoc,0shot,all,0.500,80,0.250,0.056,,
qwen3-0.6b-base,an-you-novoc,3shot_collide,all,0.150,80,0.250,0.040,,
qwen3-0.6b-base,an-you-novoc,3shot,all,0.087,80,0.250,0.032,,
qwen3-0.6b-base,adj-adj,0shot,all,0.439,260,0.333,0.031,,
qwen3-0.6b-base,adj-adj,3shot_collide,all,0.254,260,0.333,0.027,,
qwen3-0.6b-base,adj-adj,3shot,all,0.311,260,0.333,0.029,,
qwen3-0.6b-base,adj-voc,0shot,all,0.565,260,0.333,0.031,,
qwen3-0.6b-base,adj-voc,3shot_collide,all,0.235,260,0.333,0.026,,
qwen3-0.6b-base,adj-voc,3shot,all,0.288,260,0.333,0.028,,
qwen3-0.6b-base,adj-third,0shot,all,0.761,260,0.333,0.026,,
qwen3-0.6b-base,adj-third,3shot_collide,all,0.431,260,0.333,0.031,,
qwen3-0.6b-base,adj-third,3shot,all,0.442,260,0.333,0.031,,
qwen3-0.6b-base,cm-mutate,0shot,all,0.800,200,0.200,0.028,,
qwen3-0.6b-base,cm-mutate,3shot_collide,all,0.770,200,0.200,0.030,,
qwen3-0.6b-base,cm-mutate,3shot,all,0.780,200,0.200,0.029,,
qwen3-0.6b-base,cm-rebind,0shot,all,0.070,200,0.200,0.018,,
qwen3-0.6b-base,cm-rebind,3shot_collide,all,0.310,200,0.200,0.033,,
qwen3-0.6b-base,cm-rebind,3shot,all,0.355,200,0.200,0.034,,
qwen3-0.6b-base,cm-copy,0shot,all,0.000,200,0.200,0.000,,
qwen3-0.6b-base,cm-copy,3shot_collide,all,0.360,200,0.200,0.034,,
qwen3-0.6b-base,cm-copy,3shot,all,0.290,200,0.200,0.032,,
llama3.2-1b,attr,0shot,all,0.566,1210,0.168,0.014,,
llama3.2-1b,attr,3shot_collide,all,0.861,1210,0.168,0.010,,
llama3.2-1b,attr,3shot,all,0.954,1210,0.168,0.006,,
llama3.2-1b,codevar,0shot,all,0.309,1200,0.267,0.013,,
llama3.2-1b,codevar,3shot_collide,all,0.427,1200,0.267,0.014,,
llama3.2-1b,codevar,3shot,all,0.421,1200,0.267,0.014,,
llama3.2-1b,dialogue,0shot,all,0.529,1200,0.312,0.014,,
llama3.2-1b,dialogue,3shot_collide,all,0.531,1200,0.312,0.014,,
llama3.2-1b,dialogue,3shot,all,0.557,1200,0.312,0.014,,
llama3.2-1b,dx-owner-3-deictic,0shot,all,0.408,120,0.333,0.045,,other_participant=0.425;speaker_confusion=0.158;tie=0.008
llama3.2-1b,dx-owner-3-deictic,0shot,free,0.300,40,0.333,0.072,,other_participant=0.55;speaker_confusion=0.15
llama3.2-1b,dx-owner-3-deictic,3shot_collide,all,0.125,120,0.333,0.030,,other_participant=0.475;speaker_confusion=0.383;tie=0.017
llama3.2-1b,dx-owner-3-deictic,3shot_collide,free,0.125,40,0.333,0.052,,other_participant=0.55;speaker_confusion=0.325
llama3.2-1b,dx-owner-3-deictic,3shot,all,0.383,120,0.333,0.044,,other_participant=0.3;speaker_confusion=0.283;tie=0.033
llama3.2-1b,dx-owner-3-deictic,3shot,free,0.500,40,0.333,0.079,,other_participant=0.275;speaker_confusion=0.175;tie=0.05
llama3.2-1b,dx-owner-3-third,0shot,all,0.375,120,0.333,0.044,,
llama3.2-1b,dx-owner-3-third,0shot,free,0.225,40,0.333,0.066,,
llama3.2-1b,dx-owner-3-third,3shot_collide,all,0.308,120,0.333,0.042,,
llama3.2-1b,dx-owner-3-third,3shot_collide,free,0.225,40,0.333,0.066,,
llama3.2-1b,dx-owner-3-third,3shot,all,0.442,120,0.333,0.045,,
llama3.2-1b,dx-owner-3-third,3shot,free,0.450,40,0.333,0.079,,
llama3.2-1b,dx-self-3-deictic,0shot,all,0.311,180,0.287,0.035,,other_object=0.228;previous_owner=0.089;speaker_confusion=0.372
llama3.2-1b,dx-self-3-deictic,0shot,free,0.365,126,0.286,0.043,,other_object=0.222;previous_owner=0.04;speaker_confusion=0.373
llama3.2-1b,dx-self-3-deictic,3shot_collide,all,0.300,180,0.287,0.034,,other_object=0.328;previous_owner=0.017;speaker_confusion=0.339;tie=0.017
llama3.2-1b,dx-self-3-deictic,3shot_collide,free,0.405,126,0.286,0.044,,other_object=0.317;previous_owner=0.008;speaker_confusion=0.246;tie=0.024
llama3.2-1b,dx-self-3-deictic,3shot,all,0.294,180,0.287,0.034,,other_object=0.344;previous_owner=0.017;speaker_confusion=0.339;tie=0.006
llama3.2-1b,dx-self-3-deictic,3shot,free,0.397,126,0.286,0.044,,other_object=0.333;previous_owner=0.016;speaker_confusion=0.246;tie=0.008
llama3.2-1b,dx-self-3-third,0shot,all,0.306,180,0.287,0.034,,
llama3.2-1b,dx-self-3-third,0shot,free,0.254,126,0.286,0.039,,
llama3.2-1b,dx-self-3-third,3shot_collide,all,0.311,180,0.287,0.035,,
llama3.2-1b,dx-self-3-third,3shot_collide,free,0.365,126,0.286,0.043,,
llama3.2-1b,dx-self-3-third,3shot,all,0.239,180,0.287,0.032,,
llama3.2-1b,dx-self-3-third,3shot,free,0.270,126,0.286,0.040,,
llama3.2-1b,dx-self-2-deictic,0shot,all,0.278,180,0.287,0.033,,speaker_confusion=0.7;tie=0.022
llama3.2-1b,dx-self-2-deictic,0shot,free,0.270,126,0.286,0.040,,speaker_confusion=0.706;tie=0.024
llama3.2-1b,dx-self-2-deictic,3shot_collide,all,0.361,180,0.287,0.036,,speaker_confusion=0.622;tie=0.017
llama3.2-1b,dx-self-2-deictic,3shot_collide,free,0.476,126,0.286,0.044,,speaker_confusion=0.516;tie=0.008
llama3.2-1b,dx-self-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
llama3.2-1b,dx-self-2-deictic,3shot,free,0.468,126,0.286,0.044,,speaker_confusion=0.524;tie=0.008
llama3.2-1b,dx-self-2-third,0shot,all,0.322,180,0.287,0.035,,
llama3.2-1b,dx-self-2-third,0shot,free,0.341,126,0.286,0.042,,
llama3.2-1b,dx-self-2-third,3shot_collide,all,0.344,180,0.287,0.035,,
llama3.2-1b,dx-self-2-third,3shot_collide,free,0.429,126,0.286,0.044,,
llama3.2-1b,dx-self-2-third,3shot,all,0.339,180,0.287,0.035,,
llama3.2-1b,dx-self-2-third,3shot,free,0.413,126,0.286,0.044,,
llama3.2-1b,dx-addressee-3-deictic,0shot,all,0.272,180,0.287,0.033,,other_object=0.133;previous_owner=0.139;speaker_confusion=0.45;tie=0.006
llama3.2-1b,dx-addressee-3-deictic,0shot,free,0.302,126,0.286,0.041,,other_object=0.079;previous_owner=0.167;speaker_confusion=0.444;tie=0.008
llama3.2-1b,dx-addressee-3-deictic,3shot_collide,all,0.306,180,0.287,0.034,,other_object=0.289;previous_owner=0.044;speaker_confusion=0.361
llama3.2-1b,dx-addressee-3-deictic,3shot_collide,free,0.397,126,0.286,0.044,,other_object=0.246;previous_owner=0.04;speaker_confusion=0.317
llama3.2-1b,dx-addressee-3-deictic,3shot,all,0.294,180,0.287,0.034,,other_object=0.283;previous_owner=0.039;speaker_confusion=0.378;tie=0.006
llama3.2-1b,dx-addressee-3-deictic,3shot,free,0.405,126,0.286,0.044,,other_object=0.222;previous_owner=0.04;speaker_confusion=0.325;tie=0.008
llama3.2-1b,dx-addressee-3-third,0shot,all,0.278,180,0.287,0.033,,
llama3.2-1b,dx-addressee-3-third,0shot,free,0.262,126,0.286,0.039,,
llama3.2-1b,dx-addressee-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
llama3.2-1b,dx-addressee-3-third,3shot_collide,free,0.381,126,0.286,0.043,,
llama3.2-1b,dx-addressee-3-third,3shot,all,0.294,180,0.287,0.034,,
llama3.2-1b,dx-addressee-3-third,3shot,free,0.373,126,0.286,0.043,,
llama3.2-1b,dx-addressee-2-deictic,0shot,all,0.244,180,0.287,0.032,,speaker_confusion=0.733;tie=0.022
llama3.2-1b,dx-addressee-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.794;tie=0.016
llama3.2-1b,dx-addressee-2-deictic,3shot_collide,all,0.306,180,0.287,0.034,,speaker_confusion=0.689;tie=0.006
llama3.2-1b,dx-addressee-2-deictic,3shot_collide,free,0.413,126,0.286,0.044,,speaker_confusion=0.579;tie=0.008
llama3.2-1b,dx-addressee-2-deictic,3shot,all,0.311,180,0.287,0.035,,speaker_confusion=0.683;tie=0.006
llama3.2-1b,dx-addressee-2-deictic,3shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.595
llama3.2-1b,dx-addressee-2-third,0shot,all,0.350,180,0.287,0.036,,
llama3.2-1b,dx-addressee-2-third,0shot,free,0.357,126,0.286,0.043,,
llama3.2-1b,dx-addressee-2-third,3shot_collide,all,0.339,180,0.287,0.035,,
llama3.2-1b,dx-addressee-2-third,3shot_collide,free,0.413,126,0.286,0.044,,
llama3.2-1b,dx-addressee-2-third,3shot,all,0.339,180,0.287,0.035,,
llama3.2-1b,dx-addressee-2-third,3shot,free,0.421,126,0.286,0.044,,
llama3.2-1b,sb-k1,0shot,all,0.562,64,0.125,0.062,,
llama3.2-1b,sb-k1,3shot_collide,all,0.734,64,0.125,0.055,,
llama3.2-1b,sb-k1,3shot,all,0.938,64,0.125,0.030,,
llama3.2-1b,sb-k23,0shot,all,0.164,128,0.027,0.033,,
llama3.2-1b,sb-k23,3shot_collide,all,0.492,128,0.027,0.044,,
llama3.2-1b,sb-k23,3shot,all,0.602,128,0.027,0.043,,
llama3.2-1b,sb-split,0shot,all,0.281,128,0.027,0.040,,
llama3.2-1b,sb-split,3shot_collide,all,0.164,128,0.027,0.033,,
llama3.2-1b,sb-split,3shot,all,0.133,128,0.027,0.030,,
llama3.2-1b,sb-dyn,0shot,all,0.117,256,0.027,0.020,,
llama3.2-1b,sb-dyn,3shot_collide,all,0.109,256,0.027,0.019,,
llama3.2-1b,sb-dyn,3shot,all,0.129,256,0.027,0.021,,
llama3.2-1b,an-narr,0shot,all,0.405,640,0.250,0.019,,
llama3.2-1b,an-narr,3shot_collide,all,0.222,640,0.250,0.016,,
llama3.2-1b,an-narr,3shot,all,0.359,640,0.250,0.019,,
llama3.2-1b,an-other-any,0shot,all,0.434,320,0.250,0.028,,
llama3.2-1b,an-other-any,3shot_collide,all,0.103,320,0.250,0.017,,
llama3.2-1b,an-other-any,3shot,all,0.119,320,0.250,0.018,,
llama3.2-1b,an-me-any,0shot,all,0.094,160,0.250,0.023,,
llama3.2-1b,an-me-any,3shot_collide,all,0.006,160,0.250,0.006,,
llama3.2-1b,an-me-any,3shot,all,0.050,160,0.250,0.017,,
llama3.2-1b,an-you-voc,0shot,all,0.338,80,0.250,0.053,,
llama3.2-1b,an-you-voc,3shot_collide,all,0.362,80,0.250,0.054,,
llama3.2-1b,an-you-voc,3shot,all,0.150,80,0.250,0.040,,
llama3.2-1b,an-you-novoc,0shot,all,0.138,80,0.250,0.038,,
llama3.2-1b,an-you-novoc,3shot_collide,all,0.263,80,0.250,0.049,,
llama3.2-1b,an-you-novoc,3shot,all,0.087,80,0.250,0.032,,
llama3.2-1b,adj-adj,0shot,all,0.554,260,0.333,0.031,,
llama3.2-1b,adj-adj,3shot_collide,all,0.108,260,0.333,0.019,,
llama3.2-1b,adj-adj,3shot,all,0.311,260,0.333,0.029,,
llama3.2-1b,adj-voc,0shot,all,0.746,260,0.333,0.027,,
llama3.2-1b,adj-voc,3shot_collide,all,0.158,260,0.333,0.023,,
llama3.2-1b,adj-voc,3shot,all,0.381,260,0.333,0.030,,
llama3.2-1b,adj-third,0shot,all,0.546,260,0.333,0.031,,
llama3.2-1b,adj-third,3shot_collide,all,0.361,260,0.333,0.030,,
llama3.2-1b,adj-third,3shot,all,0.508,260,0.333,0.031,,
llama3.2-1b,cm-mutate,0shot,all,0.870,200,0.200,0.024,,
llama3.2-1b,cm-mutate,3shot_collide,all,0.780,200,0.200,0.029,,
llama3.2-1b,cm-mutate,3shot,all,0.830,200,0.200,0.027,,
llama3.2-1b,cm-rebind,0shot,all,0.110,200,0.200,0.022,,
llama3.2-1b,cm-rebind,3shot_collide,all,0.290,200,0.200,0.032,,
llama3.2-1b,cm-rebind,3shot,all,0.185,200,0.200,0.028,,
llama3.2-1b,cm-copy,0shot,all,0.020,200,0.200,0.010,,
llama3.2-1b,cm-copy,3shot_collide,all,0.305,200,0.200,0.033,,
llama3.2-1b,cm-copy,3shot,all,0.195,200,0.200,0.028,,
olmo1-1b-0724,attr,0shot,all,0.432,1210,0.168,0.014,,
olmo1-1b-0724,attr,3shot_collide,all,0.778,1210,0.168,0.012,,
olmo1-1b-0724,attr,3shot,all,0.826,1210,0.168,0.011,,
olmo1-1b-0724,codevar,0shot,all,0.318,1200,0.267,0.013,,
olmo1-1b-0724,codevar,3shot_collide,all,0.400,1200,0.267,0.014,,
olmo1-1b-0724,codevar,3shot,all,0.381,1200,0.267,0.014,,
olmo1-1b-0724,dialogue,0shot,all,0.491,1200,0.312,0.014,,
olmo1-1b-0724,dialogue,3shot_collide,all,0.503,1200,0.312,0.014,,
olmo1-1b-0724,dialogue,3shot,all,0.514,1200,0.312,0.014,,
olmo1-1b-0724,dx-owner-3-deictic,0shot,all,0.325,120,0.333,0.043,,other_participant=0.358;speaker_confusion=0.283;tie=0.033
olmo1-1b-0724,dx-owner-3-deictic,0shot,free,0.225,40,0.333,0.066,,other_participant=0.45;speaker_confusion=0.25;tie=0.075
olmo1-1b-0724,dx-owner-3-deictic,3shot_collide,all,0.192,120,0.333,0.036,,other_participant=0.492;speaker_confusion=0.3;tie=0.017
olmo1-1b-0724,dx-owner-3-deictic,3shot_collide,free,0.150,40,0.333,0.057,,other_participant=0.675;speaker_confusion=0.15;tie=0.025
olmo1-1b-0724,dx-owner-3-deictic,3shot,all,0.308,120,0.333,0.042,,other_participant=0.35;speaker_confusion=0.333;tie=0.008
olmo1-1b-0724,dx-owner-3-deictic,3shot,free,0.325,40,0.333,0.074,,other_participant=0.425;speaker_confusion=0.225;tie=0.025
olmo1-1b-0724,dx-owner-3-third,0shot,all,0.483,120,0.333,0.046,,
olmo1-1b-0724,dx-owner-3-third,0shot,free,0.500,40,0.333,0.079,,
olmo1-1b-0724,dx-owner-3-third,3shot_collide,all,0.350,120,0.333,0.043,,
olmo1-1b-0724,dx-owner-3-third,3shot_collide,free,0.350,40,0.333,0.075,,
olmo1-1b-0724,dx-owner-3-third,3shot,all,0.433,120,0.333,0.045,,
olmo1-1b-0724,dx-owner-3-third,3shot,free,0.600,40,0.333,0.077,,
olmo1-1b-0724,dx-self-3-deictic,0shot,all,0.322,180,0.287,0.035,,other_object=0.256;previous_owner=0.061;speaker_confusion=0.35;tie=0.011
olmo1-1b-0724,dx-self-3-deictic,0shot,free,0.349,126,0.286,0.043,,other_object=0.23;previous_owner=0.032;speaker_confusion=0.373;tie=0.016
olmo1-1b-0724,dx-self-3-deictic,3shot_collide,all,0.328,180,0.287,0.035,,other_object=0.328;previous_owner=0.017;speaker_confusion=0.317;tie=0.011
olmo1-1b-0724,dx-self-3-deictic,3shot_collide,free,0.397,126,0.286,0.044,,other_object=0.341;previous_owner=0.008;speaker_confusion=0.246;tie=0.008
olmo1-1b-0724,dx-self-3-deictic,3shot,all,0.350,180,0.287,0.036,,other_object=0.339;previous_owner=0.006;speaker_confusion=0.294;tie=0.011
olmo1-1b-0724,dx-self-3-deictic,3shot,free,0.444,126,0.286,0.044,,other_object=0.325;speaker_confusion=0.214;tie=0.016
olmo1-1b-0724,dx-self-3-third,0shot,all,0.272,180,0.287,0.033,,
olmo1-1b-0724,dx-self-3-third,0shot,free,0.341,126,0.286,0.042,,
olmo1-1b-0724,dx-self-3-third,3shot_collide,all,0.289,180,0.287,0.034,,
olmo1-1b-0724,dx-self-3-third,3shot_collide,free,0.357,126,0.286,0.043,,
olmo1-1b-0724,dx-self-3-third,3shot,all,0.272,180,0.287,0.033,,
olmo1-1b-0724,dx-self-3-third,3shot,free,0.309,126,0.286,0.041,,
olmo1-1b-0724,dx-self-2-deictic,0shot,all,0.394,180,0.287,0.036,,speaker_confusion=0.6;tie=0.006
olmo1-1b-0724,dx-self-2-deictic,0shot,free,0.397,126,0.286,0.044,,speaker_confusion=0.595;tie=0.008
olmo1-1b-0724,dx-self-2-deictic,3shot_collide,all,0.367,180,0.287,0.036,,speaker_confusion=0.617;tie=0.017
olmo1-1b-0724,dx-self-2-deictic,3shot_collide,free,0.397,126,0.286,0.044,,speaker_confusion=0.579;tie=0.024
olmo1-1b-0724,dx-self-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.594;tie=0.028
olmo1-1b-0724,dx-self-2-deictic,3shot,free,0.429,126,0.286,0.044,,speaker_confusion=0.556;tie=0.016
olmo1-1b-0724,dx-self-2-third,0shot,all,0.289,180,0.287,0.034,,
olmo1-1b-0724,dx-self-2-third,0shot,free,0.357,126,0.286,0.043,,
olmo1-1b-0724,dx-self-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
olmo1-1b-0724,dx-self-2-third,3shot_collide,free,0.429,126,0.286,0.044,,
olmo1-1b-0724,dx-self-2-third,3shot,all,0.361,180,0.287,0.036,,
olmo1-1b-0724,dx-self-2-third,3shot,free,0.444,126,0.286,0.044,,
olmo1-1b-0724,dx-addressee-3-deictic,0shot,all,0.250,180,0.287,0.032,,other_object=0.183;previous_owner=0.128;speaker_confusion=0.422;tie=0.017
olmo1-1b-0724,dx-addressee-3-deictic,0shot,free,0.270,126,0.286,0.040,,other_object=0.143;previous_owner=0.159;speaker_confusion=0.413;tie=0.016
olmo1-1b-0724,dx-addressee-3-deictic,3shot_collide,all,0.283,180,0.287,0.034,,other_object=0.261;previous_owner=0.106;speaker_confusion=0.35
olmo1-1b-0724,dx-addressee-3-deictic,3shot_collide,free,0.309,126,0.286,0.041,,other_object=0.222;previous_owner=0.135;speaker_confusion=0.333
olmo1-1b-0724,dx-addressee-3-deictic,3shot,all,0.267,180,0.287,0.033,,other_object=0.256;previous_owner=0.1;speaker_confusion=0.367;tie=0.011
olmo1-1b-0724,dx-addressee-3-deictic,3shot,free,0.309,126,0.286,0.041,,other_object=0.214;previous_owner=0.135;speaker_confusion=0.333;tie=0.008
olmo1-1b-0724,dx-addressee-3-third,0shot,all,0.250,180,0.287,0.032,,
olmo1-1b-0724,dx-addressee-3-third,0shot,free,0.302,126,0.286,0.041,,
olmo1-1b-0724,dx-addressee-3-third,3shot_collide,all,0.344,180,0.287,0.035,,
olmo1-1b-0724,dx-addressee-3-third,3shot_collide,free,0.397,126,0.286,0.044,,
olmo1-1b-0724,dx-addressee-3-third,3shot,all,0.322,180,0.287,0.035,,
olmo1-1b-0724,dx-addressee-3-third,3shot,free,0.365,126,0.286,0.043,,
olmo1-1b-0724,dx-addressee-2-deictic,0shot,all,0.200,180,0.287,0.030,,speaker_confusion=0.789;tie=0.011
olmo1-1b-0724,dx-addressee-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.81
olmo1-1b-0724,dx-addressee-2-deictic,3shot_collide,all,0.333,180,0.287,0.035,,speaker_confusion=0.65;tie=0.017
olmo1-1b-0724,dx-addressee-2-deictic,3shot_collide,free,0.373,126,0.286,0.043,,speaker_confusion=0.611;tie=0.016
olmo1-1b-0724,dx-addressee-2-deictic,3shot,all,0.294,180,0.287,0.034,,speaker_confusion=0.694;tie=0.011
olmo1-1b-0724,dx-addressee-2-deictic,3shot,free,0.318,126,0.286,0.042,,speaker_confusion=0.667;tie=0.016
olmo1-1b-0724,dx-addressee-2-third,0shot,all,0.300,180,0.287,0.034,,
olmo1-1b-0724,dx-addressee-2-third,0shot,free,0.357,126,0.286,0.043,,
olmo1-1b-0724,dx-addressee-2-third,3shot_collide,all,0.372,180,0.287,0.036,,
olmo1-1b-0724,dx-addressee-2-third,3shot_collide,free,0.452,126,0.286,0.044,,
olmo1-1b-0724,dx-addressee-2-third,3shot,all,0.394,180,0.287,0.036,,
olmo1-1b-0724,dx-addressee-2-third,3shot,free,0.476,126,0.286,0.044,,
olmo1-1b-0724,sb-k1,0shot,all,0.391,64,0.125,0.061,,
olmo1-1b-0724,sb-k1,3shot_collide,all,0.781,64,0.125,0.052,,
olmo1-1b-0724,sb-k1,3shot,all,0.750,64,0.125,0.054,,
olmo1-1b-0724,sb-k23,0shot,all,0.164,128,0.027,0.033,,
olmo1-1b-0724,sb-k23,3shot_collide,all,0.352,128,0.027,0.042,,
olmo1-1b-0724,sb-k23,3shot,all,0.320,128,0.027,0.041,,
olmo1-1b-0724,sb-split,0shot,all,0.125,128,0.027,0.029,,
olmo1-1b-0724,sb-split,3shot_collide,all,0.117,128,0.027,0.028,,
olmo1-1b-0724,sb-split,3shot,all,0.055,128,0.027,0.020,,
olmo1-1b-0724,sb-dyn,0shot,all,0.062,256,0.027,0.015,,
olmo1-1b-0724,sb-dyn,3shot_collide,all,0.066,256,0.027,0.016,,
olmo1-1b-0724,sb-dyn,3shot,all,0.055,256,0.027,0.014,,
olmo1-1b-0724,an-narr,0shot,all,0.306,640,0.250,0.018,,
olmo1-1b-0724,an-narr,3shot_collide,all,0.177,640,0.250,0.015,,
olmo1-1b-0724,an-narr,3shot,all,0.348,640,0.250,0.019,,
olmo1-1b-0724,an-other-any,0shot,all,0.444,320,0.250,0.028,,
olmo1-1b-0724,an-other-any,3shot_collide,all,0.256,320,0.250,0.024,,
olmo1-1b-0724,an-other-any,3shot,all,0.266,320,0.250,0.025,,
olmo1-1b-0724,an-me-any,0shot,all,0.194,160,0.250,0.031,,
olmo1-1b-0724,an-me-any,3shot_collide,all,0.031,160,0.250,0.014,,
olmo1-1b-0724,an-me-any,3shot,all,0.013,160,0.250,0.009,,
olmo1-1b-0724,an-you-voc,0shot,all,0.200,80,0.250,0.045,,
olmo1-1b-0724,an-you-voc,3shot_collide,all,0.212,80,0.250,0.046,,
olmo1-1b-0724,an-you-voc,3shot,all,0.138,80,0.250,0.038,,
olmo1-1b-0724,an-you-novoc,0shot,all,0.113,80,0.250,0.035,,
olmo1-1b-0724,an-you-novoc,3shot_collide,all,0.087,80,0.250,0.032,,
olmo1-1b-0724,an-you-novoc,3shot,all,0.013,80,0.250,0.012,,
olmo1-1b-0724,adj-adj,0shot,all,0.427,260,0.333,0.031,,
olmo1-1b-0724,adj-adj,3shot_collide,all,0.142,260,0.333,0.022,,
olmo1-1b-0724,adj-adj,3shot,all,0.308,260,0.333,0.029,,
olmo1-1b-0724,adj-voc,0shot,all,0.550,260,0.333,0.031,,
olmo1-1b-0724,adj-voc,3shot_collide,all,0.246,260,0.333,0.027,,
olmo1-1b-0724,adj-voc,3shot,all,0.446,260,0.333,0.031,,
olmo1-1b-0724,adj-third,0shot,all,0.546,260,0.333,0.031,,
olmo1-1b-0724,adj-third,3shot_collide,all,0.300,260,0.333,0.028,,
olmo1-1b-0724,adj-third,3shot,all,0.381,260,0.333,0.030,,
olmo1-1b-0724,cm-mutate,0shot,all,0.635,200,0.200,0.034,,
olmo1-1b-0724,cm-mutate,3shot_collide,all,0.590,200,0.200,0.035,,
olmo1-1b-0724,cm-mutate,3shot,all,0.570,200,0.200,0.035,,
olmo1-1b-0724,cm-rebind,0shot,all,0.030,200,0.200,0.012,,
olmo1-1b-0724,cm-rebind,3shot_collide,all,0.400,200,0.200,0.035,,
olmo1-1b-0724,cm-rebind,3shot,all,0.185,200,0.200,0.028,,
olmo1-1b-0724,cm-copy,0shot,all,0.010,200,0.200,0.007,,
olmo1-1b-0724,cm-copy,3shot_collide,all,0.415,200,0.200,0.035,,
olmo1-1b-0724,cm-copy,3shot,all,0.190,200,0.200,0.028,,
pythia-1b,attr,0shot,all,0.463,1210,0.168,0.014,,
pythia-1b,attr,3shot_collide,all,0.659,1210,0.168,0.014,,
pythia-1b,attr,3shot,all,0.707,1210,0.168,0.013,,
pythia-1b,codevar,0shot,all,0.268,1200,0.267,0.013,,
pythia-1b,codevar,3shot_collide,all,0.409,1200,0.267,0.014,,
pythia-1b,codevar,3shot,all,0.364,1200,0.267,0.014,,
pythia-1b,dialogue,0shot,all,0.517,1200,0.312,0.014,,
pythia-1b,dialogue,3shot_collide,all,0.474,1200,0.312,0.014,,
pythia-1b,dialogue,3shot,all,0.532,1200,0.312,0.014,,
pythia-1b,dx-owner-3-deictic,0shot,all,0.467,120,0.333,0.045,,other_participant=0.35;speaker_confusion=0.167;tie=0.017
pythia-1b,dx-owner-3-deictic,0shot,free,0.500,40,0.333,0.079,,other_participant=0.375;speaker_confusion=0.1;tie=0.025
pythia-1b,dx-owner-3-deictic,3shot_collide,all,0.208,120,0.333,0.037,,other_participant=0.5;speaker_confusion=0.283;tie=0.008
pythia-1b,dx-owner-3-deictic,3shot_collide,free,0.175,40,0.333,0.060,,other_participant=0.575;speaker_confusion=0.25
pythia-1b,dx-owner-3-deictic,3shot,all,0.375,120,0.333,0.044,,other_participant=0.375;speaker_confusion=0.225;tie=0.025
pythia-1b,dx-owner-3-deictic,3shot,free,0.375,40,0.333,0.076,,other_participant=0.45;speaker_confusion=0.15;tie=0.025
pythia-1b,dx-owner-3-third,0shot,all,0.442,120,0.333,0.045,,
pythia-1b,dx-owner-3-third,0shot,free,0.375,40,0.333,0.076,,
pythia-1b,dx-owner-3-third,3shot_collide,all,0.333,120,0.333,0.043,,
pythia-1b,dx-owner-3-third,3shot_collide,free,0.100,40,0.333,0.047,,
pythia-1b,dx-owner-3-third,3shot,all,0.458,120,0.333,0.045,,
pythia-1b,dx-owner-3-third,3shot,free,0.400,40,0.333,0.077,,
pythia-1b,dx-self-3-deictic,0shot,all,0.356,180,0.287,0.036,,other_object=0.217;previous_owner=0.094;speaker_confusion=0.322;tie=0.011
pythia-1b,dx-self-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.198;previous_owner=0.056;speaker_confusion=0.341;tie=0.016
pythia-1b,dx-self-3-deictic,3shot_collide,all,0.267,180,0.287,0.033,,other_object=0.289;previous_owner=0.061;speaker_confusion=0.367;tie=0.017
pythia-1b,dx-self-3-deictic,3shot_collide,free,0.309,126,0.286,0.041,,other_object=0.31;previous_owner=0.04;speaker_confusion=0.333;tie=0.008
pythia-1b,dx-self-3-deictic,3shot,all,0.256,180,0.287,0.033,,other_object=0.311;previous_owner=0.094;speaker_confusion=0.311;tie=0.028
pythia-1b,dx-self-3-deictic,3shot,free,0.294,126,0.286,0.041,,other_object=0.317;previous_owner=0.056;speaker_confusion=0.302;tie=0.032
pythia-1b,dx-self-3-third,0shot,all,0.322,180,0.287,0.035,,
pythia-1b,dx-self-3-third,0shot,free,0.373,126,0.286,0.043,,
pythia-1b,dx-self-3-third,3shot_collide,all,0.300,180,0.287,0.034,,
pythia-1b,dx-self-3-third,3shot_collide,free,0.365,126,0.286,0.043,,
pythia-1b,dx-self-3-third,3shot,all,0.256,180,0.287,0.033,,
pythia-1b,dx-self-3-third,3shot,free,0.325,126,0.286,0.042,,
pythia-1b,dx-self-2-deictic,0shot,all,0.300,180,0.287,0.034,,speaker_confusion=0.689;tie=0.011
pythia-1b,dx-self-2-deictic,0shot,free,0.325,126,0.286,0.042,,speaker_confusion=0.659;tie=0.016
pythia-1b,dx-self-2-deictic,3shot_collide,all,0.339,180,0.287,0.035,,speaker_confusion=0.656;tie=0.006
pythia-1b,dx-self-2-deictic,3shot_collide,free,0.405,126,0.286,0.044,,speaker_confusion=0.595
pythia-1b,dx-self-2-deictic,3shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.667;tie=0.011
pythia-1b,dx-self-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.619;tie=0.016
pythia-1b,dx-self-2-third,0shot,all,0.322,180,0.287,0.035,,
pythia-1b,dx-self-2-third,0shot,free,0.349,126,0.286,0.043,,
pythia-1b,dx-self-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
pythia-1b,dx-self-2-third,3shot_collide,free,0.436,126,0.286,0.044,,
pythia-1b,dx-self-2-third,3shot,all,0.372,180,0.287,0.036,,
pythia-1b,dx-self-2-third,3shot,free,0.476,126,0.286,0.044,,
pythia-1b,dx-addressee-3-deictic,0shot,all,0.322,180,0.287,0.035,,other_object=0.128;previous_owner=0.144;speaker_confusion=0.406
pythia-1b,dx-addressee-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.103;previous_owner=0.175;speaker_confusion=0.413
pythia-1b,dx-addressee-3-deictic,3shot_collide,all,0.233,180,0.287,0.032,,other_object=0.217;previous_owner=0.111;speaker_confusion=0.428;tie=0.011
pythia-1b,dx-addressee-3-deictic,3shot_collide,free,0.238,126,0.286,0.038,,other_object=0.206;previous_owner=0.127;speaker_confusion=0.421;tie=0.008
pythia-1b,dx-addressee-3-deictic,3shot,all,0.233,180,0.287,0.032,,other_object=0.217;previous_owner=0.111;speaker_confusion=0.439
pythia-1b,dx-addressee-3-deictic,3shot,free,0.254,126,0.286,0.039,,other_object=0.214;previous_owner=0.119;speaker_confusion=0.413
pythia-1b,dx-addressee-3-third,0shot,all,0.328,180,0.287,0.035,,
pythia-1b,dx-addressee-3-third,0shot,free,0.325,126,0.286,0.042,,
pythia-1b,dx-addressee-3-third,3shot_collide,all,0.317,180,0.287,0.035,,
pythia-1b,dx-addressee-3-third,3shot_collide,free,0.373,126,0.286,0.043,,
pythia-1b,dx-addressee-3-third,3shot,all,0.311,180,0.287,0.035,,
pythia-1b,dx-addressee-3-third,3shot,free,0.389,126,0.286,0.043,,
pythia-1b,dx-addressee-2-deictic,0shot,all,0.189,180,0.287,0.029,,speaker_confusion=0.8;tie=0.011
pythia-1b,dx-addressee-2-deictic,0shot,free,0.151,126,0.286,0.032,,speaker_confusion=0.841;tie=0.008
pythia-1b,dx-addressee-2-deictic,3shot_collide,all,0.289,180,0.287,0.034,,speaker_confusion=0.694;tie=0.017
pythia-1b,dx-addressee-2-deictic,3shot_collide,free,0.318,126,0.286,0.042,,speaker_confusion=0.667;tie=0.016
pythia-1b,dx-addressee-2-deictic,3shot,all,0.261,180,0.287,0.033,,speaker_confusion=0.733;tie=0.006
pythia-1b,dx-addressee-2-deictic,3shot,free,0.294,126,0.286,0.041,,speaker_confusion=0.706
pythia-1b,dx-addressee-2-third,0shot,all,0.339,180,0.287,0.035,,
pythia-1b,dx-addressee-2-third,0shot,free,0.333,126,0.286,0.042,,
pythia-1b,dx-addressee-2-third,3shot_collide,all,0.322,180,0.287,0.035,,
pythia-1b,dx-addressee-2-third,3shot_collide,free,0.413,126,0.286,0.044,,
pythia-1b,dx-addressee-2-third,3shot,all,0.361,180,0.287,0.036,,
pythia-1b,dx-addressee-2-third,3shot,free,0.444,126,0.286,0.044,,
pythia-1b,sb-k1,0shot,all,0.281,64,0.125,0.056,,
pythia-1b,sb-k1,3shot_collide,all,0.484,64,0.125,0.062,,
pythia-1b,sb-k1,3shot,all,0.469,64,0.125,0.062,,
pythia-1b,sb-k23,0shot,all,0.203,128,0.027,0.036,,
pythia-1b,sb-k23,3shot_collide,all,0.250,128,0.027,0.038,,
pythia-1b,sb-k23,3shot,all,0.266,128,0.027,0.039,,
pythia-1b,sb-split,0shot,all,0.156,128,0.027,0.032,,
pythia-1b,sb-split,3shot_collide,all,0.086,128,0.027,0.025,,
pythia-1b,sb-split,3shot,all,0.109,128,0.027,0.028,,
pythia-1b,sb-dyn,0shot,all,0.125,256,0.027,0.021,,
pythia-1b,sb-dyn,3shot_collide,all,0.051,256,0.027,0.014,,
pythia-1b,sb-dyn,3shot,all,0.082,256,0.027,0.017,,
pythia-1b,an-narr,0shot,all,0.358,640,0.250,0.019,,
pythia-1b,an-narr,3shot_collide,all,0.202,640,0.250,0.016,,
pythia-1b,an-narr,3shot,all,0.294,640,0.250,0.018,,
pythia-1b,an-other-any,0shot,all,0.275,320,0.250,0.025,,
pythia-1b,an-other-any,3shot_collide,all,0.253,320,0.250,0.024,,
pythia-1b,an-other-any,3shot,all,0.247,320,0.250,0.024,,
pythia-1b,an-me-any,0shot,all,0.275,160,0.250,0.035,,
pythia-1b,an-me-any,3shot_collide,all,0.037,160,0.250,0.015,,
pythia-1b,an-me-any,3shot,all,0.025,160,0.250,0.012,,
pythia-1b,an-you-voc,0shot,all,0.512,80,0.250,0.056,,
pythia-1b,an-you-voc,3shot_collide,all,0.087,80,0.250,0.032,,
pythia-1b,an-you-voc,3shot,all,0.050,80,0.250,0.024,,
pythia-1b,an-you-novoc,0shot,all,0.438,80,0.250,0.056,,
pythia-1b,an-you-novoc,3shot_collide,all,0.350,80,0.250,0.053,,
pythia-1b,an-you-novoc,3shot,all,0.225,80,0.250,0.047,,
pythia-1b,adj-adj,0shot,all,0.335,260,0.333,0.029,,
pythia-1b,adj-adj,3shot_collide,all,0.123,260,0.333,0.020,,
pythia-1b,adj-adj,3shot,all,0.377,260,0.333,0.030,,
pythia-1b,adj-voc,0shot,all,0.408,260,0.333,0.030,,
pythia-1b,adj-voc,3shot_collide,all,0.262,260,0.333,0.027,,
pythia-1b,adj-voc,3shot,all,0.431,260,0.333,0.031,,
pythia-1b,adj-third,0shot,all,0.523,260,0.333,0.031,,
pythia-1b,adj-third,3shot_collide,all,0.300,260,0.333,0.028,,
pythia-1b,adj-third,3shot,all,0.419,260,0.333,0.031,,
pythia-1b,cm-mutate,0shot,all,0.525,200,0.200,0.035,,
pythia-1b,cm-mutate,3shot_collide,all,0.535,200,0.200,0.035,,
pythia-1b,cm-mutate,3shot,all,0.560,200,0.200,0.035,,
pythia-1b,cm-rebind,0shot,all,0.100,200,0.200,0.021,,
pythia-1b,cm-rebind,3shot_collide,all,0.280,200,0.200,0.032,,
pythia-1b,cm-rebind,3shot,all,0.290,200,0.200,0.032,,
pythia-1b,cm-copy,0shot,all,0.045,200,0.200,0.015,,
pythia-1b,cm-copy,3shot_collide,all,0.225,200,0.200,0.029,,
pythia-1b,cm-copy,3shot,all,0.250,200,0.200,0.031,,
tsumugi-release,attr,0shot,all,0.551,1210,0.168,0.014,,
tsumugi-release,attr,3shot_collide,all,0.896,1210,0.168,0.009,,
tsumugi-release,attr,3shot,all,0.986,1210,0.168,0.003,,
tsumugi-release,codevar,0shot,all,0.320,1200,0.267,0.013,,
tsumugi-release,codevar,3shot_collide,all,0.453,1200,0.267,0.014,,
tsumugi-release,codevar,3shot,all,0.427,1200,0.267,0.014,,
tsumugi-release,dialogue,0shot,all,0.597,1200,0.312,0.014,,
tsumugi-release,dialogue,3shot_collide,all,0.624,1200,0.312,0.014,,
tsumugi-release,dialogue,3shot,all,0.607,1200,0.312,0.014,,
tsumugi-release,dx-owner-3-deictic,0shot,all,0.492,120,0.333,0.046,,other_participant=0.342;speaker_confusion=0.142;tie=0.025
tsumugi-release,dx-owner-3-deictic,0shot,free,0.625,40,0.333,0.076,,other_participant=0.3;speaker_confusion=0.075
tsumugi-release,dx-owner-3-deictic,3shot_collide,all,0.358,120,0.333,0.044,,other_participant=0.292;speaker_confusion=0.35
tsumugi-release,dx-owner-3-deictic,3shot_collide,free,0.350,40,0.333,0.075,,other_participant=0.35;speaker_confusion=0.3
tsumugi-release,dx-owner-3-deictic,3shot,all,0.425,120,0.333,0.045,,other_participant=0.183;speaker_confusion=0.392
tsumugi-release,dx-owner-3-deictic,3shot,free,0.525,40,0.333,0.079,,other_participant=0.175;speaker_confusion=0.3
tsumugi-release,dx-owner-3-third,0shot,all,0.525,120,0.333,0.046,,
tsumugi-release,dx-owner-3-third,0shot,free,0.625,40,0.333,0.076,,
tsumugi-release,dx-owner-3-third,3shot_collide,all,0.675,120,0.333,0.043,,
tsumugi-release,dx-owner-3-third,3shot_collide,free,0.700,40,0.333,0.072,,
tsumugi-release,dx-owner-3-third,3shot,all,0.583,120,0.333,0.045,,
tsumugi-release,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
tsumugi-release,dx-self-3-deictic,0shot,all,0.328,180,0.287,0.035,,other_object=0.2;previous_owner=0.089;speaker_confusion=0.367;tie=0.017
tsumugi-release,dx-self-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.206;previous_owner=0.056;speaker_confusion=0.421;tie=0.008
tsumugi-release,dx-self-3-deictic,3shot_collide,all,0.383,180,0.287,0.036,,other_object=0.267;previous_owner=0.067;speaker_confusion=0.272;tie=0.011
tsumugi-release,dx-self-3-deictic,3shot_collide,free,0.468,126,0.286,0.044,,other_object=0.222;previous_owner=0.048;speaker_confusion=0.254;tie=0.008
tsumugi-release,dx-self-3-deictic,3shot,all,0.356,180,0.287,0.036,,other_object=0.261;previous_owner=0.067;speaker_confusion=0.317
tsumugi-release,dx-self-3-deictic,3shot,free,0.429,126,0.286,0.044,,other_object=0.222;previous_owner=0.056;speaker_confusion=0.294
tsumugi-release,dx-self-3-third,0shot,all,0.400,180,0.287,0.036,,
tsumugi-release,dx-self-3-third,0shot,free,0.373,126,0.286,0.043,,
tsumugi-release,dx-self-3-third,3shot_collide,all,0.428,180,0.287,0.037,,
tsumugi-release,dx-self-3-third,3shot_collide,free,0.500,126,0.286,0.044,,
tsumugi-release,dx-self-3-third,3shot,all,0.406,180,0.287,0.037,,
tsumugi-release,dx-self-3-third,3shot,free,0.460,126,0.286,0.044,,
tsumugi-release,dx-self-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.689;tie=0.006
tsumugi-release,dx-self-2-deictic,0shot,free,0.278,126,0.286,0.040,,speaker_confusion=0.714;tie=0.008
tsumugi-release,dx-self-2-deictic,3shot_collide,all,0.344,180,0.287,0.035,,speaker_confusion=0.65;tie=0.006
tsumugi-release,dx-self-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.571;tie=0.008
tsumugi-release,dx-self-2-deictic,3shot,all,0.361,180,0.287,0.036,,speaker_confusion=0.633;tie=0.006
tsumugi-release,dx-self-2-deictic,3shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
tsumugi-release,dx-self-2-third,0shot,all,0.378,180,0.287,0.036,,
tsumugi-release,dx-self-2-third,0shot,free,0.397,126,0.286,0.044,,
tsumugi-release,dx-self-2-third,3shot_collide,all,0.433,180,0.287,0.037,,
tsumugi-release,dx-self-2-third,3shot_collide,free,0.532,126,0.286,0.044,,
tsumugi-release,dx-self-2-third,3shot,all,0.439,180,0.287,0.037,,
tsumugi-release,dx-self-2-third,3shot,free,0.524,126,0.286,0.044,,
tsumugi-release,dx-addressee-3-deictic,0shot,all,0.328,180,0.287,0.035,,other_object=0.156;previous_owner=0.139;speaker_confusion=0.372;tie=0.006
tsumugi-release,dx-addressee-3-deictic,0shot,free,0.270,126,0.286,0.040,,other_object=0.103;previous_owner=0.183;speaker_confusion=0.437;tie=0.008
tsumugi-release,dx-addressee-3-deictic,3shot_collide,all,0.322,180,0.287,0.035,,other_object=0.233;previous_owner=0.094;speaker_confusion=0.35
tsumugi-release,dx-addressee-3-deictic,3shot_collide,free,0.365,126,0.286,0.043,,other_object=0.19;previous_owner=0.127;speaker_confusion=0.317
tsumugi-release,dx-addressee-3-deictic,3shot,all,0.344,180,0.287,0.035,,other_object=0.222;previous_owner=0.106;speaker_confusion=0.328
tsumugi-release,dx-addressee-3-deictic,3shot,free,0.373,126,0.286,0.043,,other_object=0.183;previous_owner=0.135;speaker_confusion=0.31
tsumugi-release,dx-addressee-3-third,0shot,all,0.428,180,0.287,0.037,,
tsumugi-release,dx-addressee-3-third,0shot,free,0.397,126,0.286,0.044,,
tsumugi-release,dx-addressee-3-third,3shot_collide,all,0.361,180,0.287,0.036,,
tsumugi-release,dx-addressee-3-third,3shot_collide,free,0.405,126,0.286,0.044,,
tsumugi-release,dx-addressee-3-third,3shot,all,0.406,180,0.287,0.037,,
tsumugi-release,dx-addressee-3-third,3shot,free,0.421,126,0.286,0.044,,
tsumugi-release,dx-addressee-2-deictic,0shot,all,0.250,180,0.287,0.032,,speaker_confusion=0.75
tsumugi-release,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.786
tsumugi-release,dx-addressee-2-deictic,3shot_collide,all,0.367,180,0.287,0.036,,speaker_confusion=0.622;tie=0.011
tsumugi-release,dx-addressee-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.571;tie=0.008
tsumugi-release,dx-addressee-2-deictic,3shot,all,0.389,180,0.287,0.036,,speaker_confusion=0.606;tie=0.006
tsumugi-release,dx-addressee-2-deictic,3shot,free,0.436,126,0.286,0.044,,speaker_confusion=0.556;tie=0.008
tsumugi-release,dx-addressee-2-third,0shot,all,0.378,180,0.287,0.036,,
tsumugi-release,dx-addressee-2-third,0shot,free,0.421,126,0.286,0.044,,
tsumugi-release,dx-addressee-2-third,3shot_collide,all,0.439,180,0.287,0.037,,
tsumugi-release,dx-addressee-2-third,3shot_collide,free,0.540,126,0.286,0.044,,
tsumugi-release,dx-addressee-2-third,3shot,all,0.422,180,0.287,0.037,,
tsumugi-release,dx-addressee-2-third,3shot,free,0.524,126,0.286,0.044,,
tsumugi-release,sb-k1,0shot,all,0.500,64,0.125,0.062,,
tsumugi-release,sb-k1,3shot_collide,all,0.953,64,0.125,0.026,,
tsumugi-release,sb-k1,3shot,all,0.984,64,0.125,0.015,,
tsumugi-release,sb-k23,0shot,all,0.094,128,0.027,0.026,,
tsumugi-release,sb-k23,3shot_collide,all,0.758,128,0.027,0.038,,
tsumugi-release,sb-k23,3shot,all,0.664,128,0.027,0.042,,
tsumugi-release,sb-split,0shot,all,0.164,128,0.027,0.033,,
tsumugi-release,sb-split,3shot_collide,all,0.328,128,0.027,0.042,,
tsumugi-release,sb-split,3shot,all,0.312,128,0.027,0.041,,
tsumugi-release,sb-dyn,0shot,all,0.117,256,0.027,0.020,,
tsumugi-release,sb-dyn,3shot_collide,all,0.144,256,0.027,0.022,,
tsumugi-release,sb-dyn,3shot,all,0.156,256,0.027,0.023,,
tsumugi-release,an-narr,0shot,all,0.444,640,0.250,0.020,,
tsumugi-release,an-narr,3shot_collide,all,0.506,640,0.250,0.020,,
tsumugi-release,an-narr,3shot,all,0.516,640,0.250,0.020,,
tsumugi-release,an-other-any,0shot,all,0.197,320,0.250,0.022,,
tsumugi-release,an-other-any,3shot_collide,all,0.409,320,0.250,0.028,,
tsumugi-release,an-other-any,3shot,all,0.388,320,0.250,0.027,,
tsumugi-release,an-me-any,0shot,all,0.212,160,0.250,0.032,,
tsumugi-release,an-me-any,3shot_collide,all,0.037,160,0.250,0.015,,
tsumugi-release,an-me-any,3shot,all,0.044,160,0.250,0.016,,
tsumugi-release,an-you-voc,0shot,all,0.500,80,0.250,0.056,,
tsumugi-release,an-you-voc,3shot_collide,all,0.100,80,0.250,0.034,,
tsumugi-release,an-you-voc,3shot,all,0.263,80,0.250,0.049,,
tsumugi-release,an-you-novoc,0shot,all,0.825,80,0.250,0.043,,
tsumugi-release,an-you-novoc,3shot_collide,all,0.175,80,0.250,0.043,,
tsumugi-release,an-you-novoc,3shot,all,0.225,80,0.250,0.047,,
tsumugi-release,adj-adj,0shot,all,0.592,260,0.333,0.030,,
tsumugi-release,adj-adj,3shot_collide,all,0.350,260,0.333,0.030,,
tsumugi-release,adj-adj,3shot,all,0.381,260,0.333,0.030,,
tsumugi-release,adj-voc,0shot,all,0.835,260,0.333,0.023,,
tsumugi-release,adj-voc,3shot_collide,all,0.485,260,0.333,0.031,,
tsumugi-release,adj-voc,3shot,all,0.465,260,0.333,0.031,,
tsumugi-release,adj-third,0shot,all,0.719,260,0.333,0.028,,
tsumugi-release,adj-third,3shot_collide,all,0.573,260,0.333,0.031,,
tsumugi-release,adj-third,3shot,all,0.519,260,0.333,0.031,,
tsumugi-release,cm-mutate,0shot,all,0.585,200,0.200,0.035,,
tsumugi-release,cm-mutate,3shot_collide,all,0.595,200,0.200,0.035,,
tsumugi-release,cm-mutate,3shot,all,0.655,200,0.200,0.034,,
tsumugi-release,cm-rebind,0shot,all,0.090,200,0.200,0.020,,
tsumugi-release,cm-rebind,3shot_collide,all,0.455,200,0.200,0.035,,
tsumugi-release,cm-rebind,3shot,all,0.465,200,0.200,0.035,,
tsumugi-release,cm-copy,0shot,all,0.070,200,0.200,0.018,,
tsumugi-release,cm-copy,3shot_collide,all,0.575,200,0.200,0.035,,
tsumugi-release,cm-copy,3shot,all,0.605,200,0.200,0.035,,
pythia-1.4b,attr,0shot,all,0.517,1210,0.168,0.014,,
pythia-1.4b,attr,3shot_collide,all,0.659,1210,0.168,0.014,,
pythia-1.4b,attr,3shot,all,0.717,1210,0.168,0.013,,
pythia-1.4b,codevar,0shot,all,0.241,1200,0.267,0.012,,
pythia-1.4b,codevar,3shot_collide,all,0.386,1200,0.267,0.014,,
pythia-1.4b,codevar,3shot,all,0.383,1200,0.267,0.014,,
pythia-1.4b,dialogue,0shot,all,0.478,1200,0.312,0.014,,
pythia-1.4b,dialogue,3shot_collide,all,0.413,1200,0.312,0.014,,
pythia-1.4b,dialogue,3shot,all,0.433,1200,0.312,0.014,,
pythia-1.4b,dx-owner-3-deictic,0shot,all,0.358,120,0.333,0.044,,other_participant=0.383;speaker_confusion=0.233;tie=0.025
pythia-1.4b,dx-owner-3-deictic,0shot,free,0.425,40,0.333,0.078,,other_participant=0.425;speaker_confusion=0.125;tie=0.025
pythia-1.4b,dx-owner-3-deictic,3shot_collide,all,0.150,120,0.333,0.033,,other_participant=0.508;speaker_confusion=0.308;tie=0.033
pythia-1.4b,dx-owner-3-deictic,3shot_collide,free,0.125,40,0.333,0.052,,other_participant=0.625;speaker_confusion=0.2;tie=0.05
pythia-1.4b,dx-owner-3-deictic,3shot,all,0.367,120,0.333,0.044,,other_participant=0.342;speaker_confusion=0.275;tie=0.017
pythia-1.4b,dx-owner-3-deictic,3shot,free,0.400,40,0.333,0.077,,other_participant=0.375;speaker_confusion=0.225
pythia-1.4b,dx-owner-3-third,0shot,all,0.517,120,0.333,0.046,,
pythia-1.4b,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
pythia-1.4b,dx-owner-3-third,3shot_collide,all,0.342,120,0.333,0.043,,
pythia-1.4b,dx-owner-3-third,3shot_collide,free,0.300,40,0.333,0.072,,
pythia-1.4b,dx-owner-3-third,3shot,all,0.442,120,0.333,0.045,,
pythia-1.4b,dx-owner-3-third,3shot,free,0.575,40,0.333,0.078,,
pythia-1.4b,dx-self-3-deictic,0shot,all,0.356,180,0.287,0.036,,other_object=0.306;previous_owner=0.094;speaker_confusion=0.233;tie=0.011
pythia-1.4b,dx-self-3-deictic,0shot,free,0.460,126,0.286,0.044,,other_object=0.286;previous_owner=0.04;speaker_confusion=0.198;tie=0.016
pythia-1.4b,dx-self-3-deictic,3shot_collide,all,0.333,180,0.287,0.035,,other_object=0.256;previous_owner=0.061;speaker_confusion=0.344;tie=0.006
pythia-1.4b,dx-self-3-deictic,3shot_collide,free,0.421,126,0.286,0.044,,other_object=0.246;previous_owner=0.04;speaker_confusion=0.286;tie=0.008
pythia-1.4b,dx-self-3-deictic,3shot,all,0.356,180,0.287,0.036,,other_object=0.233;previous_owner=0.072;speaker_confusion=0.322;tie=0.017
pythia-1.4b,dx-self-3-deictic,3shot,free,0.444,126,0.286,0.044,,other_object=0.222;previous_owner=0.048;speaker_confusion=0.27;tie=0.016
pythia-1.4b,dx-self-3-third,0shot,all,0.406,180,0.287,0.037,,
pythia-1.4b,dx-self-3-third,0shot,free,0.500,126,0.286,0.044,,
pythia-1.4b,dx-self-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
pythia-1.4b,dx-self-3-third,3shot_collide,free,0.397,126,0.286,0.044,,
pythia-1.4b,dx-self-3-third,3shot,all,0.356,180,0.287,0.036,,
pythia-1.4b,dx-self-3-third,3shot,free,0.444,126,0.286,0.044,,
pythia-1.4b,dx-self-2-deictic,0shot,all,0.333,180,0.287,0.035,,speaker_confusion=0.656;tie=0.011
pythia-1.4b,dx-self-2-deictic,0shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.563;tie=0.016
pythia-1.4b,dx-self-2-deictic,3shot_collide,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
pythia-1.4b,dx-self-2-deictic,3shot_collide,free,0.444,126,0.286,0.044,,speaker_confusion=0.556
pythia-1.4b,dx-self-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
pythia-1.4b,dx-self-2-deictic,3shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.548;tie=0.008
pythia-1.4b,dx-self-2-third,0shot,all,0.361,180,0.287,0.036,,
pythia-1.4b,dx-self-2-third,0shot,free,0.492,126,0.286,0.044,,
pythia-1.4b,dx-self-2-third,3shot_collide,all,0.411,180,0.287,0.037,,
pythia-1.4b,dx-self-2-third,3shot_collide,free,0.516,126,0.286,0.044,,
pythia-1.4b,dx-self-2-third,3shot,all,0.400,180,0.287,0.036,,
pythia-1.4b,dx-self-2-third,3shot,free,0.484,126,0.286,0.044,,
pythia-1.4b,dx-addressee-3-deictic,0shot,all,0.339,180,0.287,0.035,,other_object=0.15;previous_owner=0.078;speaker_confusion=0.411;tie=0.022
pythia-1.4b,dx-addressee-3-deictic,0shot,free,0.413,126,0.286,0.044,,other_object=0.103;previous_owner=0.095;speaker_confusion=0.381;tie=0.008
pythia-1.4b,dx-addressee-3-deictic,3shot_collide,all,0.317,180,0.287,0.035,,other_object=0.217;previous_owner=0.083;speaker_confusion=0.378;tie=0.006
pythia-1.4b,dx-addressee-3-deictic,3shot_collide,free,0.397,126,0.286,0.044,,other_object=0.143;previous_owner=0.111;speaker_confusion=0.341;tie=0.008
pythia-1.4b,dx-addressee-3-deictic,3shot,all,0.311,180,0.287,0.035,,other_object=0.222;previous_owner=0.094;speaker_confusion=0.367;tie=0.006
pythia-1.4b,dx-addressee-3-deictic,3shot,free,0.381,126,0.286,0.043,,other_object=0.151;previous_owner=0.127;speaker_confusion=0.333;tie=0.008
pythia-1.4b,dx-addressee-3-third,0shot,all,0.389,180,0.287,0.036,,
pythia-1.4b,dx-addressee-3-third,0shot,free,0.508,126,0.286,0.044,,
pythia-1.4b,dx-addressee-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
pythia-1.4b,dx-addressee-3-third,3shot_collide,free,0.429,126,0.286,0.044,,
pythia-1.4b,dx-addressee-3-third,3shot,all,0.383,180,0.287,0.036,,
pythia-1.4b,dx-addressee-3-third,3shot,free,0.484,126,0.286,0.044,,
pythia-1.4b,dx-addressee-2-deictic,0shot,all,0.206,180,0.287,0.030,,speaker_confusion=0.778;tie=0.017
pythia-1.4b,dx-addressee-2-deictic,0shot,free,0.246,126,0.286,0.038,,speaker_confusion=0.73;tie=0.024
pythia-1.4b,dx-addressee-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.65;tie=0.022
pythia-1.4b,dx-addressee-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.548;tie=0.032
pythia-1.4b,dx-addressee-2-deictic,3shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.711;tie=0.006
pythia-1.4b,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.627;tie=0.008
pythia-1.4b,dx-addressee-2-third,0shot,all,0.356,180,0.287,0.036,,
pythia-1.4b,dx-addressee-2-third,0shot,free,0.484,126,0.286,0.044,,
pythia-1.4b,dx-addressee-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
pythia-1.4b,dx-addressee-2-third,3shot_collide,free,0.460,126,0.286,0.044,,
pythia-1.4b,dx-addressee-2-third,3shot,all,0.394,180,0.287,0.036,,
pythia-1.4b,dx-addressee-2-third,3shot,free,0.500,126,0.286,0.044,,
pythia-1.4b,sb-k1,0shot,all,0.422,64,0.125,0.062,,
pythia-1.4b,sb-k1,3shot_collide,all,0.547,64,0.125,0.062,,
pythia-1.4b,sb-k1,3shot,all,0.500,64,0.125,0.062,,
pythia-1.4b,sb-k23,0shot,all,0.117,128,0.027,0.028,,
pythia-1.4b,sb-k23,3shot_collide,all,0.352,128,0.027,0.042,,
pythia-1.4b,sb-k23,3shot,all,0.352,128,0.027,0.042,,
pythia-1.4b,sb-split,0shot,all,0.180,128,0.027,0.034,,
pythia-1.4b,sb-split,3shot_collide,all,0.125,128,0.027,0.029,,
pythia-1.4b,sb-split,3shot,all,0.203,128,0.027,0.036,,
pythia-1.4b,sb-dyn,0shot,all,0.129,256,0.027,0.021,,
pythia-1.4b,sb-dyn,3shot_collide,all,0.098,256,0.027,0.019,,
pythia-1.4b,sb-dyn,3shot,all,0.137,256,0.027,0.021,,
pythia-1.4b,an-narr,0shot,all,0.322,640,0.250,0.018,,
pythia-1.4b,an-narr,3shot_collide,all,0.163,640,0.250,0.015,,
pythia-1.4b,an-narr,3shot,all,0.255,640,0.250,0.017,,
pythia-1.4b,an-other-any,0shot,all,0.216,320,0.250,0.023,,
pythia-1.4b,an-other-any,3shot_collide,all,0.278,320,0.250,0.025,,
pythia-1.4b,an-other-any,3shot,all,0.328,320,0.250,0.026,,
pythia-1.4b,an-me-any,0shot,all,0.069,160,0.250,0.020,,
pythia-1.4b,an-me-any,3shot_collide,all,0.006,160,0.250,0.006,,
pythia-1.4b,an-me-any,3shot,all,0.000,160,0.250,0.000,,
pythia-1.4b,an-you-voc,0shot,all,0.887,80,0.250,0.035,,
pythia-1.4b,an-you-voc,3shot_collide,all,0.188,80,0.250,0.044,,
pythia-1.4b,an-you-voc,3shot,all,0.287,80,0.250,0.051,,
pythia-1.4b,an-you-novoc,0shot,all,0.512,80,0.250,0.056,,
pythia-1.4b,an-you-novoc,3shot_collide,all,0.275,80,0.250,0.050,,
pythia-1.4b,an-you-novoc,3shot,all,0.237,80,0.250,0.048,,
pythia-1.4b,adj-adj,0shot,all,0.404,260,0.333,0.030,,
pythia-1.4b,adj-adj,3shot_collide,all,0.146,260,0.333,0.022,,
pythia-1.4b,adj-adj,3shot,all,0.369,260,0.333,0.030,,
pythia-1.4b,adj-voc,0shot,all,0.400,260,0.333,0.030,,
pythia-1.4b,adj-voc,3shot_collide,all,0.196,260,0.333,0.025,,
pythia-1.4b,adj-voc,3shot,all,0.435,260,0.333,0.031,,
pythia-1.4b,adj-third,0shot,all,0.631,260,0.333,0.030,,
pythia-1.4b,adj-third,3shot_collide,all,0.323,260,0.333,0.029,,
pythia-1.4b,adj-third,3shot,all,0.369,260,0.333,0.030,,
pythia-1.4b,cm-mutate,0shot,all,0.265,200,0.200,0.031,,
pythia-1.4b,cm-mutate,3shot_collide,all,0.385,200,0.200,0.034,,
pythia-1.4b,cm-mutate,3shot,all,0.470,200,0.200,0.035,,
pythia-1.4b,cm-rebind,0shot,all,0.215,200,0.200,0.029,,
pythia-1.4b,cm-rebind,3shot_collide,all,0.240,200,0.200,0.030,,
pythia-1.4b,cm-rebind,3shot,all,0.230,200,0.200,0.030,,
pythia-1.4b,cm-copy,0shot,all,0.140,200,0.200,0.025,,
pythia-1.4b,cm-copy,3shot_collide,all,0.325,200,0.200,0.033,,
pythia-1.4b,cm-copy,3shot,all,0.260,200,0.200,0.031,,
olmo2-1b,attr,0shot,all,0.326,1210,0.168,0.013,,
olmo2-1b,attr,3shot_collide,all,0.582,1210,0.168,0.014,,
olmo2-1b,attr,3shot,all,0.646,1210,0.168,0.014,,
olmo2-1b,codevar,0shot,all,0.344,1200,0.267,0.014,,
olmo2-1b,codevar,3shot_collide,all,0.420,1200,0.267,0.014,,
olmo2-1b,codevar,3shot,all,0.414,1200,0.267,0.014,,
olmo2-1b,dialogue,0shot,all,0.600,1200,0.312,0.014,,
olmo2-1b,dialogue,3shot_collide,all,0.625,1200,0.312,0.014,,
olmo2-1b,dialogue,3shot,all,0.619,1200,0.312,0.014,,
olmo2-1b,dx-owner-3-deictic,0shot,all,0.492,120,0.333,0.046,,other_participant=0.3;speaker_confusion=0.192;tie=0.017
olmo2-1b,dx-owner-3-deictic,0shot,free,0.475,40,0.333,0.079,,other_participant=0.4;speaker_confusion=0.1;tie=0.025
olmo2-1b,dx-owner-3-deictic,3shot_collide,all,0.367,120,0.333,0.044,,other_participant=0.375;speaker_confusion=0.258
olmo2-1b,dx-owner-3-deictic,3shot_collide,free,0.275,40,0.333,0.071,,other_participant=0.5;speaker_confusion=0.225
olmo2-1b,dx-owner-3-deictic,3shot,all,0.458,120,0.333,0.045,,other_participant=0.292;speaker_confusion=0.25
olmo2-1b,dx-owner-3-deictic,3shot,free,0.475,40,0.333,0.079,,other_participant=0.325;speaker_confusion=0.2
olmo2-1b,dx-owner-3-third,0shot,all,0.575,120,0.333,0.045,,
olmo2-1b,dx-owner-3-third,0shot,free,0.575,40,0.333,0.078,,
olmo2-1b,dx-owner-3-third,3shot_collide,all,0.483,120,0.333,0.046,,
olmo2-1b,dx-owner-3-third,3shot_collide,free,0.350,40,0.333,0.075,,
olmo2-1b,dx-owner-3-third,3shot,all,0.533,120,0.333,0.045,,
olmo2-1b,dx-owner-3-third,3shot,free,0.525,40,0.333,0.079,,
olmo2-1b,dx-self-3-deictic,0shot,all,0.378,180,0.287,0.036,,other_object=0.217;previous_owner=0.067;speaker_confusion=0.333;tie=0.006
olmo2-1b,dx-self-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.214;previous_owner=0.048;speaker_confusion=0.341;tie=0.008
olmo2-1b,dx-self-3-deictic,3shot_collide,all,0.294,180,0.287,0.034,,other_object=0.35;previous_owner=0.05;speaker_confusion=0.306
olmo2-1b,dx-self-3-deictic,3shot_collide,free,0.357,126,0.286,0.043,,other_object=0.373;previous_owner=0.04;speaker_confusion=0.23
olmo2-1b,dx-self-3-deictic,3shot,all,0.317,180,0.287,0.035,,other_object=0.333;previous_owner=0.056;speaker_confusion=0.294
olmo2-1b,dx-self-3-deictic,3shot,free,0.389,126,0.286,0.043,,other_object=0.349;previous_owner=0.048;speaker_confusion=0.214
olmo2-1b,dx-self-3-third,0shot,all,0.278,180,0.287,0.033,,
olmo2-1b,dx-self-3-third,0shot,free,0.254,126,0.286,0.039,,
olmo2-1b,dx-self-3-third,3shot_collide,all,0.361,180,0.287,0.036,,
olmo2-1b,dx-self-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
olmo2-1b,dx-self-3-third,3shot,all,0.361,180,0.287,0.036,,
olmo2-1b,dx-self-3-third,3shot,free,0.389,126,0.286,0.043,,
olmo2-1b,dx-self-2-deictic,0shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.622
olmo2-1b,dx-self-2-deictic,0shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
olmo2-1b,dx-self-2-deictic,3shot_collide,all,0.361,180,0.287,0.036,,speaker_confusion=0.639
olmo2-1b,dx-self-2-deictic,3shot_collide,free,0.460,126,0.286,0.044,,speaker_confusion=0.54
olmo2-1b,dx-self-2-deictic,3shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.628
olmo2-1b,dx-self-2-deictic,3shot,free,0.468,126,0.286,0.044,,speaker_confusion=0.532
olmo2-1b,dx-self-2-third,0shot,all,0.422,180,0.287,0.037,,
olmo2-1b,dx-self-2-third,0shot,free,0.429,126,0.286,0.044,,
olmo2-1b,dx-self-2-third,3shot_collide,all,0.383,180,0.287,0.036,,
olmo2-1b,dx-self-2-third,3shot_collide,free,0.429,126,0.286,0.044,,
olmo2-1b,dx-self-2-third,3shot,all,0.389,180,0.287,0.036,,
olmo2-1b,dx-self-2-third,3shot,free,0.381,126,0.286,0.043,,
olmo2-1b,dx-addressee-3-deictic,0shot,all,0.317,180,0.287,0.035,,other_object=0.122;previous_owner=0.161;speaker_confusion=0.4
olmo2-1b,dx-addressee-3-deictic,0shot,free,0.254,126,0.286,0.039,,other_object=0.119;previous_owner=0.198;speaker_confusion=0.429
olmo2-1b,dx-addressee-3-deictic,3shot_collide,all,0.322,180,0.287,0.035,,other_object=0.244;previous_owner=0.083;speaker_confusion=0.35
olmo2-1b,dx-addressee-3-deictic,3shot_collide,free,0.357,126,0.286,0.043,,other_object=0.19;previous_owner=0.111;speaker_confusion=0.341
olmo2-1b,dx-addressee-3-deictic,3shot,all,0.317,180,0.287,0.035,,other_object=0.2;previous_owner=0.067;speaker_confusion=0.411;tie=0.006
olmo2-1b,dx-addressee-3-deictic,3shot,free,0.341,126,0.286,0.042,,other_object=0.167;previous_owner=0.087;speaker_confusion=0.397;tie=0.008
olmo2-1b,dx-addressee-3-third,0shot,all,0.383,180,0.287,0.036,,
olmo2-1b,dx-addressee-3-third,0shot,free,0.325,126,0.286,0.042,,
olmo2-1b,dx-addressee-3-third,3shot_collide,all,0.378,180,0.287,0.036,,
olmo2-1b,dx-addressee-3-third,3shot_collide,free,0.405,126,0.286,0.044,,
olmo2-1b,dx-addressee-3-third,3shot,all,0.344,180,0.287,0.035,,
olmo2-1b,dx-addressee-3-third,3shot,free,0.325,126,0.286,0.042,,
olmo2-1b,dx-addressee-2-deictic,0shot,all,0.189,180,0.287,0.029,,speaker_confusion=0.806;tie=0.006
olmo2-1b,dx-addressee-2-deictic,0shot,free,0.159,126,0.286,0.033,,speaker_confusion=0.833;tie=0.008
olmo2-1b,dx-addressee-2-deictic,3shot_collide,all,0.317,180,0.287,0.035,,speaker_confusion=0.678;tie=0.006
olmo2-1b,dx-addressee-2-deictic,3shot_collide,free,0.381,126,0.286,0.043,,speaker_confusion=0.611;tie=0.008
olmo2-1b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.644
olmo2-1b,dx-addressee-2-deictic,3shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.556
olmo2-1b,dx-addressee-2-third,0shot,all,0.433,180,0.287,0.037,,
olmo2-1b,dx-addressee-2-third,0shot,free,0.436,126,0.286,0.044,,
olmo2-1b,dx-addressee-2-third,3shot_collide,all,0.417,180,0.287,0.037,,
olmo2-1b,dx-addressee-2-third,3shot_collide,free,0.508,126,0.286,0.044,,
olmo2-1b,dx-addressee-2-third,3shot,all,0.417,180,0.287,0.037,,
olmo2-1b,dx-addressee-2-third,3shot,free,0.492,126,0.286,0.044,,
olmo2-1b,sb-k1,0shot,all,0.328,64,0.125,0.059,,
olmo2-1b,sb-k1,3shot_collide,all,0.359,64,0.125,0.060,,
olmo2-1b,sb-k1,3shot,all,0.422,64,0.125,0.062,,
olmo2-1b,sb-k23,0shot,all,0.242,128,0.027,0.038,,
olmo2-1b,sb-k23,3shot_collide,all,0.180,128,0.027,0.034,,
olmo2-1b,sb-k23,3shot,all,0.242,128,0.027,0.038,,
olmo2-1b,sb-split,0shot,all,0.055,128,0.027,0.020,,
olmo2-1b,sb-split,3shot_collide,all,0.070,128,0.027,0.023,,
olmo2-1b,sb-split,3shot,all,0.039,128,0.027,0.017,,
olmo2-1b,sb-dyn,0shot,all,0.113,256,0.027,0.020,,
olmo2-1b,sb-dyn,3shot_collide,all,0.117,256,0.027,0.020,,
olmo2-1b,sb-dyn,3shot,all,0.102,256,0.027,0.019,,
olmo2-1b,an-narr,0shot,all,0.477,640,0.250,0.020,,
olmo2-1b,an-narr,3shot_collide,all,0.373,640,0.250,0.019,,
olmo2-1b,an-narr,3shot,all,0.392,640,0.250,0.019,,
olmo2-1b,an-other-any,0shot,all,0.434,320,0.250,0.028,,
olmo2-1b,an-other-any,3shot_collide,all,0.266,320,0.250,0.025,,
olmo2-1b,an-other-any,3shot,all,0.272,320,0.250,0.025,,
olmo2-1b,an-me-any,0shot,all,0.094,160,0.250,0.023,,
olmo2-1b,an-me-any,3shot_collide,all,0.025,160,0.250,0.012,,
olmo2-1b,an-me-any,3shot,all,0.013,160,0.250,0.009,,
olmo2-1b,an-you-voc,0shot,all,0.275,80,0.250,0.050,,
olmo2-1b,an-you-voc,3shot_collide,all,0.338,80,0.250,0.053,,
olmo2-1b,an-you-voc,3shot,all,0.237,80,0.250,0.048,,
olmo2-1b,an-you-novoc,0shot,all,0.325,80,0.250,0.052,,
olmo2-1b,an-you-novoc,3shot_collide,all,0.400,80,0.250,0.055,,
olmo2-1b,an-you-novoc,3shot,all,0.362,80,0.250,0.054,,
olmo2-1b,adj-adj,0shot,all,0.642,260,0.333,0.030,,
olmo2-1b,adj-adj,3shot_collide,all,0.427,260,0.333,0.031,,
olmo2-1b,adj-adj,3shot,all,0.473,260,0.333,0.031,,
olmo2-1b,adj-voc,0shot,all,0.781,260,0.333,0.026,,
olmo2-1b,adj-voc,3shot_collide,all,0.492,260,0.333,0.031,,
olmo2-1b,adj-voc,3shot,all,0.439,260,0.333,0.031,,
olmo2-1b,adj-third,0shot,all,0.635,260,0.333,0.030,,
olmo2-1b,adj-third,3shot_collide,all,0.488,260,0.333,0.031,,
olmo2-1b,adj-third,3shot,all,0.469,260,0.333,0.031,,
olmo2-1b,cm-mutate,0shot,all,0.595,200,0.200,0.035,,
olmo2-1b,cm-mutate,3shot_collide,all,0.460,200,0.200,0.035,,
olmo2-1b,cm-mutate,3shot,all,0.425,200,0.200,0.035,,
olmo2-1b,cm-rebind,0shot,all,0.010,200,0.200,0.007,,
olmo2-1b,cm-rebind,3shot_collide,all,0.190,200,0.200,0.028,,
olmo2-1b,cm-rebind,3shot,all,0.130,200,0.200,0.024,,
olmo2-1b,cm-copy,0shot,all,0.035,200,0.200,0.013,,
olmo2-1b,cm-copy,3shot_collide,all,0.165,200,0.200,0.026,,
olmo2-1b,cm-copy,3shot,all,0.080,200,0.200,0.019,,
qwen2.5-1.5b,attr,0shot,all,0.712,1210,0.168,0.013,,
qwen2.5-1.5b,attr,3shot_collide,all,0.870,1210,0.168,0.010,,
qwen2.5-1.5b,attr,3shot,all,0.977,1210,0.168,0.004,,
qwen2.5-1.5b,codevar,0shot,all,0.389,1200,0.267,0.014,,
qwen2.5-1.5b,codevar,3shot_collide,all,0.470,1200,0.267,0.014,,
qwen2.5-1.5b,codevar,3shot,all,0.461,1200,0.267,0.014,,
qwen2.5-1.5b,dialogue,0shot,all,0.637,1200,0.312,0.014,,
qwen2.5-1.5b,dialogue,3shot_collide,all,0.588,1200,0.312,0.014,,
qwen2.5-1.5b,dialogue,3shot,all,0.600,1200,0.312,0.014,,
qwen2.5-1.5b,dx-owner-3-deictic,0shot,all,0.433,120,0.333,0.045,,other_participant=0.408;speaker_confusion=0.142;tie=0.017
qwen2.5-1.5b,dx-owner-3-deictic,0shot,free,0.325,40,0.333,0.074,,other_participant=0.55;speaker_confusion=0.125
qwen2.5-1.5b,dx-owner-3-deictic,3shot_collide,all,0.283,120,0.333,0.041,,other_participant=0.517;speaker_confusion=0.2
qwen2.5-1.5b,dx-owner-3-deictic,3shot_collide,free,0.125,40,0.333,0.052,,other_participant=0.7;speaker_confusion=0.175
qwen2.5-1.5b,dx-owner-3-deictic,3shot,all,0.392,120,0.333,0.045,,other_participant=0.367;speaker_confusion=0.233;tie=0.008
qwen2.5-1.5b,dx-owner-3-deictic,3shot,free,0.500,40,0.333,0.079,,other_participant=0.375;speaker_confusion=0.1;tie=0.025
qwen2.5-1.5b,dx-owner-3-third,0shot,all,0.525,120,0.333,0.046,,
qwen2.5-1.5b,dx-owner-3-third,0shot,free,0.550,40,0.333,0.079,,
qwen2.5-1.5b,dx-owner-3-third,3shot_collide,all,0.458,120,0.333,0.045,,
qwen2.5-1.5b,dx-owner-3-third,3shot_collide,free,0.400,40,0.333,0.077,,
qwen2.5-1.5b,dx-owner-3-third,3shot,all,0.483,120,0.333,0.046,,
qwen2.5-1.5b,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
qwen2.5-1.5b,dx-self-3-deictic,0shot,all,0.350,180,0.287,0.036,,other_object=0.317;previous_owner=0.056;speaker_confusion=0.244;tie=0.033
qwen2.5-1.5b,dx-self-3-deictic,0shot,free,0.452,126,0.286,0.044,,other_object=0.278;previous_owner=0.048;speaker_confusion=0.183;tie=0.04
qwen2.5-1.5b,dx-self-3-deictic,3shot_collide,all,0.339,180,0.287,0.035,,other_object=0.333;previous_owner=0.044;speaker_confusion=0.267;tie=0.017
qwen2.5-1.5b,dx-self-3-deictic,3shot_collide,free,0.381,126,0.286,0.043,,other_object=0.357;previous_owner=0.04;speaker_confusion=0.198;tie=0.024
qwen2.5-1.5b,dx-self-3-deictic,3shot,all,0.339,180,0.287,0.035,,other_object=0.317;previous_owner=0.05;speaker_confusion=0.278;tie=0.017
qwen2.5-1.5b,dx-self-3-deictic,3shot,free,0.389,126,0.286,0.043,,other_object=0.325;previous_owner=0.048;speaker_confusion=0.214;tie=0.024
qwen2.5-1.5b,dx-self-3-third,0shot,all,0.350,180,0.287,0.036,,
qwen2.5-1.5b,dx-self-3-third,0shot,free,0.302,126,0.286,0.041,,
qwen2.5-1.5b,dx-self-3-third,3shot_collide,all,0.278,180,0.287,0.033,,
qwen2.5-1.5b,dx-self-3-third,3shot_collide,free,0.349,126,0.286,0.043,,
qwen2.5-1.5b,dx-self-3-third,3shot,all,0.289,180,0.287,0.034,,
qwen2.5-1.5b,dx-self-3-third,3shot,free,0.365,126,0.286,0.043,,
qwen2.5-1.5b,dx-self-2-deictic,0shot,all,0.361,180,0.287,0.036,,speaker_confusion=0.6;tie=0.039
qwen2.5-1.5b,dx-self-2-deictic,0shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.468;tie=0.048
qwen2.5-1.5b,dx-self-2-deictic,3shot_collide,all,0.428,180,0.287,0.037,,speaker_confusion=0.567;tie=0.006
qwen2.5-1.5b,dx-self-2-deictic,3shot_collide,free,0.516,126,0.286,0.044,,speaker_confusion=0.476;tie=0.008
qwen2.5-1.5b,dx-self-2-deictic,3shot,all,0.406,180,0.287,0.037,,speaker_confusion=0.572;tie=0.022
qwen2.5-1.5b,dx-self-2-deictic,3shot,free,0.508,126,0.286,0.044,,speaker_confusion=0.476;tie=0.016
qwen2.5-1.5b,dx-self-2-third,0shot,all,0.439,180,0.287,0.037,,
qwen2.5-1.5b,dx-self-2-third,0shot,free,0.492,126,0.286,0.044,,
qwen2.5-1.5b,dx-self-2-third,3shot_collide,all,0.444,180,0.287,0.037,,
qwen2.5-1.5b,dx-self-2-third,3shot_collide,free,0.516,126,0.286,0.044,,
qwen2.5-1.5b,dx-self-2-third,3shot,all,0.450,180,0.287,0.037,,
qwen2.5-1.5b,dx-self-2-third,3shot,free,0.524,126,0.286,0.044,,
qwen2.5-1.5b,dx-addressee-3-deictic,0shot,all,0.317,180,0.287,0.035,,other_object=0.228;previous_owner=0.072;speaker_confusion=0.367;tie=0.017
qwen2.5-1.5b,dx-addressee-3-deictic,0shot,free,0.333,126,0.286,0.042,,other_object=0.183;previous_owner=0.079;speaker_confusion=0.397;tie=0.008
qwen2.5-1.5b,dx-addressee-3-deictic,3shot_collide,all,0.244,180,0.287,0.032,,other_object=0.233;previous_owner=0.1;speaker_confusion=0.417;tie=0.006
qwen2.5-1.5b,dx-addressee-3-deictic,3shot_collide,free,0.238,126,0.286,0.038,,other_object=0.183;previous_owner=0.127;speaker_confusion=0.444;tie=0.008
qwen2.5-1.5b,dx-addressee-3-deictic,3shot,all,0.261,180,0.287,0.033,,other_object=0.211;previous_owner=0.1;speaker_confusion=0.417;tie=0.011
qwen2.5-1.5b,dx-addressee-3-deictic,3shot,free,0.270,126,0.286,0.040,,other_object=0.19;previous_owner=0.135;speaker_confusion=0.389;tie=0.016
qwen2.5-1.5b,dx-addressee-3-third,0shot,all,0.378,180,0.287,0.036,,
qwen2.5-1.5b,dx-addressee-3-third,0shot,free,0.333,126,0.286,0.042,,
qwen2.5-1.5b,dx-addressee-3-third,3shot_collide,all,0.339,180,0.287,0.035,,
qwen2.5-1.5b,dx-addressee-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
qwen2.5-1.5b,dx-addressee-3-third,3shot,all,0.289,180,0.287,0.034,,
qwen2.5-1.5b,dx-addressee-3-third,3shot,free,0.365,126,0.286,0.043,,
qwen2.5-1.5b,dx-addressee-2-deictic,0shot,all,0.278,180,0.287,0.033,,speaker_confusion=0.717;tie=0.006
qwen2.5-1.5b,dx-addressee-2-deictic,0shot,free,0.318,126,0.286,0.042,,speaker_confusion=0.675;tie=0.008
qwen2.5-1.5b,dx-addressee-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.661;tie=0.011
qwen2.5-1.5b,dx-addressee-2-deictic,3shot_collide,free,0.341,126,0.286,0.042,,speaker_confusion=0.651;tie=0.008
qwen2.5-1.5b,dx-addressee-2-deictic,3shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.661;tie=0.022
qwen2.5-1.5b,dx-addressee-2-deictic,3shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.659;tie=0.032
qwen2.5-1.5b,dx-addressee-2-third,0shot,all,0.544,180,0.287,0.037,,
qwen2.5-1.5b,dx-addressee-2-third,0shot,free,0.556,126,0.286,0.044,,
qwen2.5-1.5b,dx-addressee-2-third,3shot_collide,all,0.411,180,0.287,0.037,,
qwen2.5-1.5b,dx-addressee-2-third,3shot_collide,free,0.452,126,0.286,0.044,,
qwen2.5-1.5b,dx-addressee-2-third,3shot,all,0.439,180,0.287,0.037,,
qwen2.5-1.5b,dx-addressee-2-third,3shot,free,0.468,126,0.286,0.044,,
qwen2.5-1.5b,sb-k1,0shot,all,0.781,64,0.125,0.052,,
qwen2.5-1.5b,sb-k1,3shot_collide,all,0.875,64,0.125,0.041,,
qwen2.5-1.5b,sb-k1,3shot,all,0.969,64,0.125,0.022,,
qwen2.5-1.5b,sb-k23,0shot,all,0.352,128,0.027,0.042,,
qwen2.5-1.5b,sb-k23,3shot_collide,all,0.398,128,0.027,0.043,,
qwen2.5-1.5b,sb-k23,3shot,all,0.406,128,0.027,0.043,,
qwen2.5-1.5b,sb-split,0shot,all,0.289,128,0.027,0.040,,
qwen2.5-1.5b,sb-split,3shot_collide,all,0.219,128,0.027,0.036,,
qwen2.5-1.5b,sb-split,3shot,all,0.234,128,0.027,0.037,,
qwen2.5-1.5b,sb-dyn,0shot,all,0.211,256,0.027,0.025,,
qwen2.5-1.5b,sb-dyn,3shot_collide,all,0.117,256,0.027,0.020,,
qwen2.5-1.5b,sb-dyn,3shot,all,0.168,256,0.027,0.023,,
qwen2.5-1.5b,an-narr,0shot,all,0.392,640,0.250,0.019,,
qwen2.5-1.5b,an-narr,3shot_collide,all,0.256,640,0.250,0.017,,
qwen2.5-1.5b,an-narr,3shot,all,0.395,640,0.250,0.019,,
qwen2.5-1.5b,an-other-any,0shot,all,0.300,320,0.250,0.026,,
qwen2.5-1.5b,an-other-any,3shot_collide,all,0.328,320,0.250,0.026,,
qwen2.5-1.5b,an-other-any,3shot,all,0.319,320,0.250,0.026,,
qwen2.5-1.5b,an-me-any,0shot,all,0.225,160,0.250,0.033,,
qwen2.5-1.5b,an-me-any,3shot_collide,all,0.013,160,0.250,0.009,,
qwen2.5-1.5b,an-me-any,3shot,all,0.050,160,0.250,0.017,,
qwen2.5-1.5b,an-you-voc,0shot,all,0.375,80,0.250,0.054,,
qwen2.5-1.5b,an-you-voc,3shot_collide,all,0.050,80,0.250,0.024,,
qwen2.5-1.5b,an-you-voc,3shot,all,0.100,80,0.250,0.034,,
qwen2.5-1.5b,an-you-novoc,0shot,all,0.412,80,0.250,0.055,,
qwen2.5-1.5b,an-you-novoc,3shot_collide,all,0.150,80,0.250,0.040,,
qwen2.5-1.5b,an-you-novoc,3shot,all,0.237,80,0.250,0.048,,
qwen2.5-1.5b,adj-adj,0shot,all,0.546,260,0.333,0.031,,
qwen2.5-1.5b,adj-adj,3shot_collide,all,0.296,260,0.333,0.028,,
qwen2.5-1.5b,adj-adj,3shot,all,0.465,260,0.333,0.031,,
qwen2.5-1.5b,adj-voc,0shot,all,0.654,260,0.333,0.029,,
qwen2.5-1.5b,adj-voc,3shot_collide,all,0.369,260,0.333,0.030,,
qwen2.5-1.5b,adj-voc,3shot,all,0.469,260,0.333,0.031,,
qwen2.5-1.5b,adj-third,0shot,all,0.665,260,0.333,0.029,,
qwen2.5-1.5b,adj-third,3shot_collide,all,0.558,260,0.333,0.031,,
qwen2.5-1.5b,adj-third,3shot,all,0.592,260,0.333,0.030,,
qwen2.5-1.5b,cm-mutate,0shot,all,0.790,200,0.200,0.029,,
qwen2.5-1.5b,cm-mutate,3shot_collide,all,0.800,200,0.200,0.028,,
qwen2.5-1.5b,cm-mutate,3shot,all,0.850,200,0.200,0.025,,
qwen2.5-1.5b,cm-rebind,0shot,all,0.020,200,0.200,0.010,,
qwen2.5-1.5b,cm-rebind,3shot_collide,all,0.370,200,0.200,0.034,,
qwen2.5-1.5b,cm-rebind,3shot,all,0.370,200,0.200,0.034,,
qwen2.5-1.5b,cm-copy,0shot,all,0.005,200,0.200,0.005,,
qwen2.5-1.5b,cm-copy,3shot_collide,all,0.565,200,0.200,0.035,,
qwen2.5-1.5b,cm-copy,3shot,all,0.550,200,0.200,0.035,,
qwen3-1.7b-base,attr,0shot,all,0.719,1210,0.168,0.013,,
qwen3-1.7b-base,attr,3shot_collide,all,0.957,1210,0.168,0.006,,
qwen3-1.7b-base,attr,3shot,all,0.997,1210,0.168,0.002,,
qwen3-1.7b-base,codevar,0shot,all,0.459,1200,0.267,0.014,,
qwen3-1.7b-base,codevar,3shot_collide,all,0.624,1200,0.267,0.014,,
qwen3-1.7b-base,codevar,3shot,all,0.619,1200,0.267,0.014,,
qwen3-1.7b-base,dialogue,0shot,all,0.711,1200,0.312,0.013,,
qwen3-1.7b-base,dialogue,3shot_collide,all,0.757,1200,0.312,0.012,,
qwen3-1.7b-base,dialogue,3shot,all,0.742,1200,0.312,0.013,,
qwen3-1.7b-base,dx-owner-3-deictic,0shot,all,0.492,120,0.333,0.046,,other_participant=0.267;speaker_confusion=0.217;tie=0.025
qwen3-1.7b-base,dx-owner-3-deictic,0shot,free,0.475,40,0.333,0.079,,other_participant=0.375;speaker_confusion=0.1;tie=0.05
qwen3-1.7b-base,dx-owner-3-deictic,3shot_collide,all,0.433,120,0.333,0.045,,other_participant=0.325;speaker_confusion=0.233;tie=0.008
qwen3-1.7b-base,dx-owner-3-deictic,3shot_collide,free,0.450,40,0.333,0.079,,other_participant=0.425;speaker_confusion=0.125
qwen3-1.7b-base,dx-owner-3-deictic,3shot,all,0.533,120,0.333,0.045,,other_participant=0.233;speaker_confusion=0.225;tie=0.008
qwen3-1.7b-base,dx-owner-3-deictic,3shot,free,0.525,40,0.333,0.079,,other_participant=0.35;speaker_confusion=0.125
qwen3-1.7b-base,dx-owner-3-third,0shot,all,0.558,120,0.333,0.045,,
qwen3-1.7b-base,dx-owner-3-third,0shot,free,0.600,40,0.333,0.077,,
qwen3-1.7b-base,dx-owner-3-third,3shot_collide,all,0.700,120,0.333,0.042,,
qwen3-1.7b-base,dx-owner-3-third,3shot_collide,free,0.650,40,0.333,0.075,,
qwen3-1.7b-base,dx-owner-3-third,3shot,all,0.767,120,0.333,0.039,,
qwen3-1.7b-base,dx-owner-3-third,3shot,free,0.750,40,0.333,0.069,,
qwen3-1.7b-base,dx-self-3-deictic,0shot,all,0.361,180,0.287,0.036,,other_object=0.233;previous_owner=0.072;speaker_confusion=0.3;tie=0.033
qwen3-1.7b-base,dx-self-3-deictic,0shot,free,0.500,126,0.286,0.044,,other_object=0.19;previous_owner=0.048;speaker_confusion=0.214;tie=0.048
qwen3-1.7b-base,dx-self-3-deictic,3shot_collide,all,0.378,180,0.287,0.036,,other_object=0.194;previous_owner=0.083;speaker_confusion=0.333;tie=0.011
qwen3-1.7b-base,dx-self-3-deictic,3shot_collide,free,0.476,126,0.286,0.044,,other_object=0.167;previous_owner=0.063;speaker_confusion=0.278;tie=0.016
qwen3-1.7b-base,dx-self-3-deictic,3shot,all,0.378,180,0.287,0.036,,other_object=0.228;previous_owner=0.067;speaker_confusion=0.317;tie=0.011
qwen3-1.7b-base,dx-self-3-deictic,3shot,free,0.492,126,0.286,0.044,,other_object=0.206;previous_owner=0.063;speaker_confusion=0.222;tie=0.016
qwen3-1.7b-base,dx-self-3-third,0shot,all,0.467,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-3-third,0shot,free,0.444,126,0.286,0.044,,
qwen3-1.7b-base,dx-self-3-third,3shot_collide,all,0.511,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-3-third,3shot_collide,free,0.595,126,0.286,0.044,,
qwen3-1.7b-base,dx-self-3-third,3shot,all,0.472,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-3-third,3shot,free,0.564,126,0.286,0.044,,
qwen3-1.7b-base,dx-self-2-deictic,0shot,all,0.428,180,0.287,0.037,,speaker_confusion=0.539;tie=0.033
qwen3-1.7b-base,dx-self-2-deictic,0shot,free,0.516,126,0.286,0.044,,speaker_confusion=0.437;tie=0.048
qwen3-1.7b-base,dx-self-2-deictic,3shot_collide,all,0.411,180,0.287,0.037,,speaker_confusion=0.578;tie=0.011
qwen3-1.7b-base,dx-self-2-deictic,3shot_collide,free,0.516,126,0.286,0.044,,speaker_confusion=0.468;tie=0.016
qwen3-1.7b-base,dx-self-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.506;tie=0.011
qwen3-1.7b-base,dx-self-2-deictic,3shot,free,0.619,126,0.286,0.043,,speaker_confusion=0.365;tie=0.016
qwen3-1.7b-base,dx-self-2-third,0shot,all,0.578,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-2-third,0shot,free,0.667,126,0.286,0.042,,
qwen3-1.7b-base,dx-self-2-third,3shot_collide,all,0.572,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-2-third,3shot_collide,free,0.691,126,0.286,0.041,,
qwen3-1.7b-base,dx-self-2-third,3shot,all,0.544,180,0.287,0.037,,
qwen3-1.7b-base,dx-self-2-third,3shot,free,0.635,126,0.286,0.043,,
qwen3-1.7b-base,dx-addressee-3-deictic,0shot,all,0.278,180,0.287,0.033,,other_object=0.172;previous_owner=0.111;speaker_confusion=0.422;tie=0.017
qwen3-1.7b-base,dx-addressee-3-deictic,0shot,free,0.318,126,0.286,0.042,,other_object=0.151;previous_owner=0.143;speaker_confusion=0.373;tie=0.016
qwen3-1.7b-base,dx-addressee-3-deictic,3shot_collide,all,0.311,180,0.287,0.035,,other_object=0.156;previous_owner=0.094;speaker_confusion=0.422;tie=0.017
qwen3-1.7b-base,dx-addressee-3-deictic,3shot_collide,free,0.318,126,0.286,0.042,,other_object=0.127;previous_owner=0.111;speaker_confusion=0.421;tie=0.024
qwen3-1.7b-base,dx-addressee-3-deictic,3shot,all,0.317,180,0.287,0.035,,other_object=0.161;previous_owner=0.078;speaker_confusion=0.411;tie=0.033
qwen3-1.7b-base,dx-addressee-3-deictic,3shot,free,0.349,126,0.286,0.043,,other_object=0.127;previous_owner=0.103;speaker_confusion=0.373;tie=0.048
qwen3-1.7b-base,dx-addressee-3-third,0shot,all,0.483,180,0.287,0.037,,
qwen3-1.7b-base,dx-addressee-3-third,0shot,free,0.484,126,0.286,0.044,,
qwen3-1.7b-base,dx-addressee-3-third,3shot_collide,all,0.517,180,0.287,0.037,,
qwen3-1.7b-base,dx-addressee-3-third,3shot_collide,free,0.579,126,0.286,0.044,,
qwen3-1.7b-base,dx-addressee-3-third,3shot,all,0.478,180,0.287,0.037,,
qwen3-1.7b-base,dx-addressee-3-third,3shot,free,0.548,126,0.286,0.044,,
qwen3-1.7b-base,dx-addressee-2-deictic,0shot,all,0.211,180,0.287,0.030,,speaker_confusion=0.75;tie=0.039
qwen3-1.7b-base,dx-addressee-2-deictic,0shot,free,0.254,126,0.286,0.039,,speaker_confusion=0.69;tie=0.056
qwen3-1.7b-base,dx-addressee-2-deictic,3shot_collide,all,0.383,180,0.287,0.036,,speaker_confusion=0.594;tie=0.022
qwen3-1.7b-base,dx-addressee-2-deictic,3shot_collide,free,0.405,126,0.286,0.044,,speaker_confusion=0.563;tie=0.032
qwen3-1.7b-base,dx-addressee-2-deictic,3shot,all,0.394,180,0.287,0.036,,speaker_confusion=0.583;tie=0.022
qwen3-1.7b-base,dx-addressee-2-deictic,3shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.548;tie=0.032
qwen3-1.7b-base,dx-addressee-2-third,0shot,all,0.589,180,0.287,0.037,,
qwen3-1.7b-base,dx-addressee-2-third,0shot,free,0.667,126,0.286,0.042,,
qwen3-1.7b-base,dx-addressee-2-third,3shot_collide,all,0.622,180,0.287,0.036,,
qwen3-1.7b-base,dx-addressee-2-third,3shot_collide,free,0.738,126,0.286,0.039,,
qwen3-1.7b-base,dx-addressee-2-third,3shot,all,0.611,180,0.287,0.036,,
qwen3-1.7b-base,dx-addressee-2-third,3shot,free,0.698,126,0.286,0.041,,
qwen3-1.7b-base,sb-k1,0shot,all,0.766,64,0.125,0.053,,
qwen3-1.7b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
qwen3-1.7b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen3-1.7b-base,sb-k23,0shot,all,0.383,128,0.027,0.043,,
qwen3-1.7b-base,sb-k23,3shot_collide,all,0.930,128,0.027,0.023,,
qwen3-1.7b-base,sb-k23,3shot,all,0.961,128,0.027,0.017,,
qwen3-1.7b-base,sb-split,0shot,all,0.406,128,0.027,0.043,,
qwen3-1.7b-base,sb-split,3shot_collide,all,0.570,128,0.027,0.044,,
qwen3-1.7b-base,sb-split,3shot,all,0.555,128,0.027,0.044,,
qwen3-1.7b-base,sb-dyn,0shot,all,0.340,256,0.027,0.030,,
qwen3-1.7b-base,sb-dyn,3shot_collide,all,0.191,256,0.027,0.025,,
qwen3-1.7b-base,sb-dyn,3shot,all,0.188,256,0.027,0.024,,
qwen3-1.7b-base,an-narr,0shot,all,0.453,640,0.250,0.020,,
qwen3-1.7b-base,an-narr,3shot_collide,all,0.487,640,0.250,0.020,,
qwen3-1.7b-base,an-narr,3shot,all,0.517,640,0.250,0.020,,
qwen3-1.7b-base,an-other-any,0shot,all,0.525,320,0.250,0.028,,
qwen3-1.7b-base,an-other-any,3shot_collide,all,0.372,320,0.250,0.027,,
qwen3-1.7b-base,an-other-any,3shot,all,0.397,320,0.250,0.027,,
qwen3-1.7b-base,an-me-any,0shot,all,0.062,160,0.250,0.019,,
qwen3-1.7b-base,an-me-any,3shot_collide,all,0.075,160,0.250,0.021,,
qwen3-1.7b-base,an-me-any,3shot,all,0.113,160,0.250,0.025,,
qwen3-1.7b-base,an-you-voc,0shot,all,0.275,80,0.250,0.050,,
qwen3-1.7b-base,an-you-voc,3shot_collide,all,0.250,80,0.250,0.048,,
qwen3-1.7b-base,an-you-voc,3shot,all,0.212,80,0.250,0.046,,
qwen3-1.7b-base,an-you-novoc,0shot,all,0.350,80,0.250,0.053,,
qwen3-1.7b-base,an-you-novoc,3shot_collide,all,0.225,80,0.250,0.047,,
qwen3-1.7b-base,an-you-novoc,3shot,all,0.300,80,0.250,0.051,,
qwen3-1.7b-base,adj-adj,0shot,all,0.761,260,0.333,0.026,,
qwen3-1.7b-base,adj-adj,3shot_collide,all,0.565,260,0.333,0.031,,
qwen3-1.7b-base,adj-adj,3shot,all,0.669,260,0.333,0.029,,
qwen3-1.7b-base,adj-voc,0shot,all,0.927,260,0.333,0.016,,
qwen3-1.7b-base,adj-voc,3shot_collide,all,0.785,260,0.333,0.025,,
qwen3-1.7b-base,adj-voc,3shot,all,0.788,260,0.333,0.025,,
qwen3-1.7b-base,adj-third,0shot,all,0.554,260,0.333,0.031,,
qwen3-1.7b-base,adj-third,3shot_collide,all,0.804,260,0.333,0.025,,
qwen3-1.7b-base,adj-third,3shot,all,0.900,260,0.333,0.019,,
qwen3-1.7b-base,cm-mutate,0shot,all,0.750,200,0.200,0.031,,
qwen3-1.7b-base,cm-mutate,3shot_collide,all,0.945,200,0.200,0.016,,
qwen3-1.7b-base,cm-mutate,3shot,all,0.910,200,0.200,0.020,,
qwen3-1.7b-base,cm-rebind,0shot,all,0.215,200,0.200,0.029,,
qwen3-1.7b-base,cm-rebind,3shot_collide,all,0.890,200,0.200,0.022,,
qwen3-1.7b-base,cm-rebind,3shot,all,0.835,200,0.200,0.026,,
qwen3-1.7b-base,cm-copy,0shot,all,0.110,200,0.200,0.022,,
qwen3-1.7b-base,cm-copy,3shot_collide,all,0.960,200,0.200,0.014,,
qwen3-1.7b-base,cm-copy,3shot,all,0.925,200,0.200,0.019,,
smollm2-1.7b,attr,0shot,all,0.406,1210,0.168,0.014,,
smollm2-1.7b,attr,3shot_collide,all,0.779,1210,0.168,0.012,,
smollm2-1.7b,attr,3shot,all,0.831,1210,0.168,0.011,,
smollm2-1.7b,codevar,0shot,all,0.355,1200,0.267,0.014,,
smollm2-1.7b,codevar,3shot_collide,all,0.449,1200,0.267,0.014,,
smollm2-1.7b,codevar,3shot,all,0.463,1200,0.267,0.014,,
smollm2-1.7b,dialogue,0shot,all,0.599,1200,0.312,0.014,,
smollm2-1.7b,dialogue,3shot_collide,all,0.614,1200,0.312,0.014,,
smollm2-1.7b,dialogue,3shot,all,0.645,1200,0.312,0.014,,
smollm2-1.7b,dx-owner-3-deictic,0shot,all,0.375,120,0.333,0.044,,other_participant=0.392;speaker_confusion=0.225;tie=0.008
smollm2-1.7b,dx-owner-3-deictic,0shot,free,0.325,40,0.333,0.074,,other_participant=0.5;speaker_confusion=0.15;tie=0.025
smollm2-1.7b,dx-owner-3-deictic,3shot_collide,all,0.200,120,0.333,0.036,,other_participant=0.492;speaker_confusion=0.308
smollm2-1.7b,dx-owner-3-deictic,3shot_collide,free,0.150,40,0.333,0.057,,other_participant=0.675;speaker_confusion=0.175
smollm2-1.7b,dx-owner-3-deictic,3shot,all,0.292,120,0.333,0.042,,other_participant=0.35;speaker_confusion=0.342;tie=0.017
smollm2-1.7b,dx-owner-3-deictic,3shot,free,0.200,40,0.333,0.063,,other_participant=0.6;speaker_confusion=0.2
smollm2-1.7b,dx-owner-3-third,0shot,all,0.492,120,0.333,0.046,,
smollm2-1.7b,dx-owner-3-third,0shot,free,0.450,40,0.333,0.079,,
smollm2-1.7b,dx-owner-3-third,3shot_collide,all,0.358,120,0.333,0.044,,
smollm2-1.7b,dx-owner-3-third,3shot_collide,free,0.300,40,0.333,0.072,,
smollm2-1.7b,dx-owner-3-third,3shot,all,0.508,120,0.333,0.046,,
smollm2-1.7b,dx-owner-3-third,3shot,free,0.475,40,0.333,0.079,,
smollm2-1.7b,dx-self-3-deictic,0shot,all,0.378,180,0.287,0.036,,other_object=0.222;previous_owner=0.028;speaker_confusion=0.367;tie=0.006
smollm2-1.7b,dx-self-3-deictic,0shot,free,0.421,126,0.286,0.044,,other_object=0.222;previous_owner=0.016;speaker_confusion=0.333;tie=0.008
smollm2-1.7b,dx-self-3-deictic,3shot_collide,all,0.350,180,0.287,0.036,,other_object=0.228;previous_owner=0.056;speaker_confusion=0.344;tie=0.022
smollm2-1.7b,dx-self-3-deictic,3shot_collide,free,0.429,126,0.286,0.044,,other_object=0.246;previous_owner=0.04;speaker_confusion=0.262;tie=0.024
smollm2-1.7b,dx-self-3-deictic,3shot,all,0.367,180,0.287,0.036,,other_object=0.261;previous_owner=0.039;speaker_confusion=0.328;tie=0.006
smollm2-1.7b,dx-self-3-deictic,3shot,free,0.460,126,0.286,0.044,,other_object=0.27;previous_owner=0.04;speaker_confusion=0.222;tie=0.008
smollm2-1.7b,dx-self-3-third,0shot,all,0.344,180,0.287,0.035,,
smollm2-1.7b,dx-self-3-third,0shot,free,0.349,126,0.286,0.043,,
smollm2-1.7b,dx-self-3-third,3shot_collide,all,0.294,180,0.287,0.034,,
smollm2-1.7b,dx-self-3-third,3shot_collide,free,0.333,126,0.286,0.042,,
smollm2-1.7b,dx-self-3-third,3shot,all,0.317,180,0.287,0.035,,
smollm2-1.7b,dx-self-3-third,3shot,free,0.325,126,0.286,0.042,,
smollm2-1.7b,dx-self-2-deictic,0shot,all,0.367,180,0.287,0.036,,speaker_confusion=0.622;tie=0.011
smollm2-1.7b,dx-self-2-deictic,0shot,free,0.436,126,0.286,0.044,,speaker_confusion=0.563
smollm2-1.7b,dx-self-2-deictic,3shot_collide,all,0.394,180,0.287,0.036,,speaker_confusion=0.606
smollm2-1.7b,dx-self-2-deictic,3shot_collide,free,0.492,126,0.286,0.044,,speaker_confusion=0.508
smollm2-1.7b,dx-self-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.611;tie=0.011
smollm2-1.7b,dx-self-2-deictic,3shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.508;tie=0.008
smollm2-1.7b,dx-self-2-third,0shot,all,0.417,180,0.287,0.037,,
smollm2-1.7b,dx-self-2-third,0shot,free,0.500,126,0.286,0.044,,
smollm2-1.7b,dx-self-2-third,3shot_collide,all,0.400,180,0.287,0.036,,
smollm2-1.7b,dx-self-2-third,3shot_collide,free,0.429,126,0.286,0.044,,
smollm2-1.7b,dx-self-2-third,3shot,all,0.422,180,0.287,0.037,,
smollm2-1.7b,dx-self-2-third,3shot,free,0.436,126,0.286,0.044,,
smollm2-1.7b,dx-addressee-3-deictic,0shot,all,0.283,180,0.287,0.034,,other_object=0.222;previous_owner=0.05;speaker_confusion=0.428;tie=0.017
smollm2-1.7b,dx-addressee-3-deictic,0shot,free,0.341,126,0.286,0.042,,other_object=0.198;previous_owner=0.056;speaker_confusion=0.381;tie=0.024
smollm2-1.7b,dx-addressee-3-deictic,3shot_collide,all,0.361,180,0.287,0.036,,other_object=0.256;previous_owner=0.067;speaker_confusion=0.317
smollm2-1.7b,dx-addressee-3-deictic,3shot_collide,free,0.381,126,0.286,0.043,,other_object=0.19;previous_owner=0.095;speaker_confusion=0.333
smollm2-1.7b,dx-addressee-3-deictic,3shot,all,0.328,180,0.287,0.035,,other_object=0.267;previous_owner=0.072;speaker_confusion=0.328;tie=0.006
smollm2-1.7b,dx-addressee-3-deictic,3shot,free,0.357,126,0.286,0.043,,other_object=0.206;previous_owner=0.103;speaker_confusion=0.325;tie=0.008
smollm2-1.7b,dx-addressee-3-third,0shot,all,0.306,180,0.287,0.034,,
smollm2-1.7b,dx-addressee-3-third,0shot,free,0.318,126,0.286,0.042,,
smollm2-1.7b,dx-addressee-3-third,3shot_collide,all,0.394,180,0.287,0.036,,
smollm2-1.7b,dx-addressee-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
smollm2-1.7b,dx-addressee-3-third,3shot,all,0.389,180,0.287,0.036,,
smollm2-1.7b,dx-addressee-3-third,3shot,free,0.413,126,0.286,0.044,,
smollm2-1.7b,dx-addressee-2-deictic,0shot,all,0.222,180,0.287,0.031,,speaker_confusion=0.778
smollm2-1.7b,dx-addressee-2-deictic,0shot,free,0.230,126,0.286,0.037,,speaker_confusion=0.77
smollm2-1.7b,dx-addressee-2-deictic,3shot_collide,all,0.378,180,0.287,0.036,,speaker_confusion=0.611;tie=0.011
smollm2-1.7b,dx-addressee-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.571;tie=0.008
smollm2-1.7b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
smollm2-1.7b,dx-addressee-2-deictic,3shot,free,0.381,126,0.286,0.043,,speaker_confusion=0.611;tie=0.008
smollm2-1.7b,dx-addressee-2-third,0shot,all,0.450,180,0.287,0.037,,
smollm2-1.7b,dx-addressee-2-third,0shot,free,0.548,126,0.286,0.044,,
smollm2-1.7b,dx-addressee-2-third,3shot_collide,all,0.406,180,0.287,0.037,,
smollm2-1.7b,dx-addressee-2-third,3shot_collide,free,0.460,126,0.286,0.044,,
smollm2-1.7b,dx-addressee-2-third,3shot,all,0.433,180,0.287,0.037,,
smollm2-1.7b,dx-addressee-2-third,3shot,free,0.484,126,0.286,0.044,,
smollm2-1.7b,sb-k1,0shot,all,0.422,64,0.125,0.062,,
smollm2-1.7b,sb-k1,3shot_collide,all,0.719,64,0.125,0.056,,
smollm2-1.7b,sb-k1,3shot,all,0.719,64,0.125,0.056,,
smollm2-1.7b,sb-k23,0shot,all,0.203,128,0.027,0.036,,
smollm2-1.7b,sb-k23,3shot_collide,all,0.367,128,0.027,0.043,,
smollm2-1.7b,sb-k23,3shot,all,0.398,128,0.027,0.043,,
smollm2-1.7b,sb-split,0shot,all,0.211,128,0.027,0.036,,
smollm2-1.7b,sb-split,3shot_collide,all,0.156,128,0.027,0.032,,
smollm2-1.7b,sb-split,3shot,all,0.141,128,0.027,0.031,,
smollm2-1.7b,sb-dyn,0shot,all,0.102,256,0.027,0.019,,
smollm2-1.7b,sb-dyn,3shot_collide,all,0.094,256,0.027,0.018,,
smollm2-1.7b,sb-dyn,3shot,all,0.070,256,0.027,0.016,,
smollm2-1.7b,an-narr,0shot,all,0.277,640,0.250,0.018,,
smollm2-1.7b,an-narr,3shot_collide,all,0.239,640,0.250,0.017,,
smollm2-1.7b,an-narr,3shot,all,0.294,640,0.250,0.018,,
smollm2-1.7b,an-other-any,0shot,all,0.391,320,0.250,0.027,,
smollm2-1.7b,an-other-any,3shot_collide,all,0.175,320,0.250,0.021,,
smollm2-1.7b,an-other-any,3shot,all,0.141,320,0.250,0.019,,
smollm2-1.7b,an-me-any,0shot,all,0.144,160,0.250,0.028,,
smollm2-1.7b,an-me-any,3shot_collide,all,0.069,160,0.250,0.020,,
smollm2-1.7b,an-me-any,3shot,all,0.013,160,0.250,0.009,,
smollm2-1.7b,an-you-voc,0shot,all,0.425,80,0.250,0.055,,
smollm2-1.7b,an-you-voc,3shot_collide,all,0.087,80,0.250,0.032,,
smollm2-1.7b,an-you-voc,3shot,all,0.312,80,0.250,0.052,,
smollm2-1.7b,an-you-novoc,0shot,all,0.113,80,0.250,0.035,,
smollm2-1.7b,an-you-novoc,3shot_collide,all,0.175,80,0.250,0.043,,
smollm2-1.7b,an-you-novoc,3shot,all,0.300,80,0.250,0.051,,
smollm2-1.7b,adj-adj,0shot,all,0.535,260,0.333,0.031,,
smollm2-1.7b,adj-adj,3shot_collide,all,0.254,260,0.333,0.027,,
smollm2-1.7b,adj-adj,3shot,all,0.373,260,0.333,0.030,,
smollm2-1.7b,adj-voc,0shot,all,0.542,260,0.333,0.031,,
smollm2-1.7b,adj-voc,3shot_collide,all,0.262,260,0.333,0.027,,
smollm2-1.7b,adj-voc,3shot,all,0.358,260,0.333,0.030,,
smollm2-1.7b,adj-third,0shot,all,0.411,260,0.333,0.030,,
smollm2-1.7b,adj-third,3shot_collide,all,0.323,260,0.333,0.029,,
smollm2-1.7b,adj-third,3shot,all,0.481,260,0.333,0.031,,
smollm2-1.7b,cm-mutate,0shot,all,0.765,200,0.200,0.030,,
smollm2-1.7b,cm-mutate,3shot_collide,all,0.835,200,0.200,0.026,,
smollm2-1.7b,cm-mutate,3shot,all,0.780,200,0.200,0.029,,
smollm2-1.7b,cm-rebind,0shot,all,0.065,200,0.200,0.017,,
smollm2-1.7b,cm-rebind,3shot_collide,all,0.490,200,0.200,0.035,,
smollm2-1.7b,cm-rebind,3shot,all,0.515,200,0.200,0.035,,
smollm2-1.7b,cm-copy,0shot,all,0.020,200,0.200,0.010,,
smollm2-1.7b,cm-copy,3shot_collide,all,0.505,200,0.200,0.035,,
smollm2-1.7b,cm-copy,3shot,all,0.430,200,0.200,0.035,,
pythia-2.8b,attr,0shot,all,0.612,1210,0.168,0.014,,
pythia-2.8b,attr,3shot_collide,all,0.785,1210,0.168,0.012,,
pythia-2.8b,attr,3shot,all,0.847,1210,0.168,0.010,,
pythia-2.8b,codevar,0shot,all,0.329,1200,0.267,0.014,,
pythia-2.8b,codevar,3shot_collide,all,0.412,1200,0.267,0.014,,
pythia-2.8b,codevar,3shot,all,0.390,1200,0.267,0.014,,
pythia-2.8b,dialogue,0shot,all,0.568,1200,0.312,0.014,,
pythia-2.8b,dialogue,3shot_collide,all,0.510,1200,0.312,0.014,,
pythia-2.8b,dialogue,3shot,all,0.544,1200,0.312,0.014,,
pythia-2.8b,dx-owner-3-deictic,0shot,all,0.483,120,0.333,0.046,,other_participant=0.317;speaker_confusion=0.183;tie=0.017
pythia-2.8b,dx-owner-3-deictic,0shot,free,0.475,40,0.333,0.079,,other_participant=0.375;speaker_confusion=0.15
pythia-2.8b,dx-owner-3-deictic,3shot_collide,all,0.192,120,0.333,0.036,,other_participant=0.467;speaker_confusion=0.325;tie=0.017
pythia-2.8b,dx-owner-3-deictic,3shot_collide,free,0.050,40,0.333,0.035,,other_participant=0.7;speaker_confusion=0.25
pythia-2.8b,dx-owner-3-deictic,3shot,all,0.358,120,0.333,0.044,,other_participant=0.358;speaker_confusion=0.283
pythia-2.8b,dx-owner-3-deictic,3shot,free,0.125,40,0.333,0.052,,other_participant=0.6;speaker_confusion=0.275
pythia-2.8b,dx-owner-3-third,0shot,all,0.558,120,0.333,0.045,,
pythia-2.8b,dx-owner-3-third,0shot,free,0.725,40,0.333,0.071,,
pythia-2.8b,dx-owner-3-third,3shot_collide,all,0.292,120,0.333,0.042,,
pythia-2.8b,dx-owner-3-third,3shot_collide,free,0.200,40,0.333,0.063,,
pythia-2.8b,dx-owner-3-third,3shot,all,0.408,120,0.333,0.045,,
pythia-2.8b,dx-owner-3-third,3shot,free,0.450,40,0.333,0.079,,
pythia-2.8b,dx-self-3-deictic,0shot,all,0.289,180,0.287,0.034,,other_object=0.233;previous_owner=0.089;speaker_confusion=0.383;tie=0.006
pythia-2.8b,dx-self-3-deictic,0shot,free,0.270,126,0.286,0.040,,other_object=0.23;previous_owner=0.056;speaker_confusion=0.437;tie=0.008
pythia-2.8b,dx-self-3-deictic,3shot_collide,all,0.289,180,0.287,0.034,,other_object=0.272;previous_owner=0.067;speaker_confusion=0.35;tie=0.022
pythia-2.8b,dx-self-3-deictic,3shot_collide,free,0.357,126,0.286,0.043,,other_object=0.278;previous_owner=0.056;speaker_confusion=0.278;tie=0.032
pythia-2.8b,dx-self-3-deictic,3shot,all,0.317,180,0.287,0.035,,other_object=0.256;previous_owner=0.061;speaker_confusion=0.361;tie=0.006
pythia-2.8b,dx-self-3-deictic,3shot,free,0.381,126,0.286,0.043,,other_object=0.27;previous_owner=0.048;speaker_confusion=0.294;tie=0.008
pythia-2.8b,dx-self-3-third,0shot,all,0.328,180,0.287,0.035,,
pythia-2.8b,dx-self-3-third,0shot,free,0.341,126,0.286,0.042,,
pythia-2.8b,dx-self-3-third,3shot_collide,all,0.328,180,0.287,0.035,,
pythia-2.8b,dx-self-3-third,3shot_collide,free,0.373,126,0.286,0.043,,
pythia-2.8b,dx-self-3-third,3shot,all,0.350,180,0.287,0.036,,
pythia-2.8b,dx-self-3-third,3shot,free,0.436,126,0.286,0.044,,
pythia-2.8b,dx-self-2-deictic,0shot,all,0.272,180,0.287,0.033,,speaker_confusion=0.728
pythia-2.8b,dx-self-2-deictic,0shot,free,0.246,126,0.286,0.038,,speaker_confusion=0.754
pythia-2.8b,dx-self-2-deictic,3shot_collide,all,0.372,180,0.287,0.036,,speaker_confusion=0.628
pythia-2.8b,dx-self-2-deictic,3shot_collide,free,0.436,126,0.286,0.044,,speaker_confusion=0.563
pythia-2.8b,dx-self-2-deictic,3shot,all,0.367,180,0.287,0.036,,speaker_confusion=0.628;tie=0.006
pythia-2.8b,dx-self-2-deictic,3shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.54;tie=0.008
pythia-2.8b,dx-self-2-third,0shot,all,0.344,180,0.287,0.035,,
pythia-2.8b,dx-self-2-third,0shot,free,0.381,126,0.286,0.043,,
pythia-2.8b,dx-self-2-third,3shot_collide,all,0.350,180,0.287,0.036,,
pythia-2.8b,dx-self-2-third,3shot_collide,free,0.405,126,0.286,0.044,,
pythia-2.8b,dx-self-2-third,3shot,all,0.394,180,0.287,0.036,,
pythia-2.8b,dx-self-2-third,3shot,free,0.492,126,0.286,0.044,,
pythia-2.8b,dx-addressee-3-deictic,0shot,all,0.333,180,0.287,0.035,,other_object=0.183;previous_owner=0.15;speaker_confusion=0.322;tie=0.011
pythia-2.8b,dx-addressee-3-deictic,0shot,free,0.325,126,0.286,0.042,,other_object=0.143;previous_owner=0.206;speaker_confusion=0.31;tie=0.016
pythia-2.8b,dx-addressee-3-deictic,3shot_collide,all,0.306,180,0.287,0.034,,other_object=0.206;previous_owner=0.122;speaker_confusion=0.356;tie=0.011
pythia-2.8b,dx-addressee-3-deictic,3shot_collide,free,0.333,126,0.286,0.042,,other_object=0.183;previous_owner=0.159;speaker_confusion=0.31;tie=0.016
pythia-2.8b,dx-addressee-3-deictic,3shot,all,0.328,180,0.287,0.035,,other_object=0.178;previous_owner=0.133;speaker_confusion=0.339;tie=0.022
pythia-2.8b,dx-addressee-3-deictic,3shot,free,0.365,126,0.286,0.043,,other_object=0.143;previous_owner=0.167;speaker_confusion=0.294;tie=0.032
pythia-2.8b,dx-addressee-3-third,0shot,all,0.422,180,0.287,0.037,,
pythia-2.8b,dx-addressee-3-third,0shot,free,0.436,126,0.286,0.044,,
pythia-2.8b,dx-addressee-3-third,3shot_collide,all,0.367,180,0.287,0.036,,
pythia-2.8b,dx-addressee-3-third,3shot_collide,free,0.429,126,0.286,0.044,,
pythia-2.8b,dx-addressee-3-third,3shot,all,0.400,180,0.287,0.036,,
pythia-2.8b,dx-addressee-3-third,3shot,free,0.484,126,0.286,0.044,,
pythia-2.8b,dx-addressee-2-deictic,0shot,all,0.211,180,0.287,0.030,,speaker_confusion=0.789
pythia-2.8b,dx-addressee-2-deictic,0shot,free,0.159,126,0.286,0.033,,speaker_confusion=0.841
pythia-2.8b,dx-addressee-2-deictic,3shot_collide,all,0.378,180,0.287,0.036,,speaker_confusion=0.622
pythia-2.8b,dx-addressee-2-deictic,3shot_collide,free,0.413,126,0.286,0.044,,speaker_confusion=0.587
pythia-2.8b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
pythia-2.8b,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.627;tie=0.008
pythia-2.8b,dx-addressee-2-third,0shot,all,0.400,180,0.287,0.036,,
pythia-2.8b,dx-addressee-2-third,0shot,free,0.365,126,0.286,0.043,,
pythia-2.8b,dx-addressee-2-third,3shot_collide,all,0.406,180,0.287,0.037,,
pythia-2.8b,dx-addressee-2-third,3shot_collide,free,0.508,126,0.286,0.044,,
pythia-2.8b,dx-addressee-2-third,3shot,all,0.400,180,0.287,0.036,,
pythia-2.8b,dx-addressee-2-third,3shot,free,0.468,126,0.286,0.044,,
pythia-2.8b,sb-k1,0shot,all,0.641,64,0.125,0.060,,
pythia-2.8b,sb-k1,3shot_collide,all,0.719,64,0.125,0.056,,
pythia-2.8b,sb-k1,3shot,all,0.734,64,0.125,0.055,,
pythia-2.8b,sb-k23,0shot,all,0.352,128,0.027,0.042,,
pythia-2.8b,sb-k23,3shot_collide,all,0.508,128,0.027,0.044,,
pythia-2.8b,sb-k23,3shot,all,0.469,128,0.027,0.044,,
pythia-2.8b,sb-split,0shot,all,0.266,128,0.027,0.039,,
pythia-2.8b,sb-split,3shot_collide,all,0.258,128,0.027,0.039,,
pythia-2.8b,sb-split,3shot,all,0.188,128,0.027,0.035,,
pythia-2.8b,sb-dyn,0shot,all,0.144,256,0.027,0.022,,
pythia-2.8b,sb-dyn,3shot_collide,all,0.117,256,0.027,0.020,,
pythia-2.8b,sb-dyn,3shot,all,0.117,256,0.027,0.020,,
pythia-2.8b,an-narr,0shot,all,0.403,640,0.250,0.019,,
pythia-2.8b,an-narr,3shot_collide,all,0.242,640,0.250,0.017,,
pythia-2.8b,an-narr,3shot,all,0.348,640,0.250,0.019,,
pythia-2.8b,an-other-any,0shot,all,0.138,320,0.250,0.019,,
pythia-2.8b,an-other-any,3shot_collide,all,0.487,320,0.250,0.028,,
pythia-2.8b,an-other-any,3shot,all,0.512,320,0.250,0.028,,
pythia-2.8b,an-me-any,0shot,all,0.675,160,0.250,0.037,,
pythia-2.8b,an-me-any,3shot_collide,all,0.069,160,0.250,0.020,,
pythia-2.8b,an-me-any,3shot,all,0.019,160,0.250,0.011,,
pythia-2.8b,an-you-voc,0shot,all,0.200,80,0.250,0.045,,
pythia-2.8b,an-you-voc,3shot_collide,all,0.025,80,0.250,0.018,,
pythia-2.8b,an-you-voc,3shot,all,0.013,80,0.250,0.012,,
pythia-2.8b,an-you-novoc,0shot,all,0.125,80,0.250,0.037,,
pythia-2.8b,an-you-novoc,3shot_collide,all,0.087,80,0.250,0.032,,
pythia-2.8b,an-you-novoc,3shot,all,0.138,80,0.250,0.038,,
pythia-2.8b,adj-adj,0shot,all,0.535,260,0.333,0.031,,
pythia-2.8b,adj-adj,3shot_collide,all,0.238,260,0.333,0.026,,
pythia-2.8b,adj-adj,3shot,all,0.339,260,0.333,0.029,,
pythia-2.8b,adj-voc,0shot,all,0.677,260,0.333,0.029,,
pythia-2.8b,adj-voc,3shot_collide,all,0.327,260,0.333,0.029,,
pythia-2.8b,adj-voc,3shot,all,0.431,260,0.333,0.031,,
pythia-2.8b,adj-third,0shot,all,0.823,260,0.333,0.024,,
pythia-2.8b,adj-third,3shot_collide,all,0.458,260,0.333,0.031,,
pythia-2.8b,adj-third,3shot,all,0.581,260,0.333,0.031,,
pythia-2.8b,cm-mutate,0shot,all,0.765,200,0.200,0.030,,
pythia-2.8b,cm-mutate,3shot_collide,all,0.535,200,0.200,0.035,,
pythia-2.8b,cm-mutate,3shot,all,0.655,200,0.200,0.034,,
pythia-2.8b,cm-rebind,0shot,all,0.100,200,0.200,0.021,,
pythia-2.8b,cm-rebind,3shot_collide,all,0.430,200,0.200,0.035,,
pythia-2.8b,cm-rebind,3shot,all,0.290,200,0.200,0.032,,
pythia-2.8b,cm-copy,0shot,all,0.045,200,0.200,0.015,,
pythia-2.8b,cm-copy,3shot_collide,all,0.425,200,0.200,0.035,,
pythia-2.8b,cm-copy,3shot,all,0.385,200,0.200,0.034,,
falcon3-3b,attr,0shot,all,0.550,1210,0.168,0.014,,
falcon3-3b,attr,3shot_collide,all,0.918,1210,0.168,0.008,,
falcon3-3b,attr,3shot,all,0.953,1210,0.168,0.006,,
falcon3-3b,codevar,0shot,all,0.427,1200,0.267,0.014,,
falcon3-3b,codevar,3shot_collide,all,0.617,1200,0.267,0.014,,
falcon3-3b,codevar,3shot,all,0.625,1200,0.267,0.014,,
falcon3-3b,dialogue,0shot,all,0.718,1200,0.312,0.013,,
falcon3-3b,dialogue,3shot_collide,all,0.717,1200,0.312,0.013,,
falcon3-3b,dialogue,3shot,all,0.743,1200,0.312,0.013,,
falcon3-3b,dx-owner-3-deictic,0shot,all,0.450,120,0.333,0.045,,other_participant=0.358;speaker_confusion=0.183;tie=0.008
falcon3-3b,dx-owner-3-deictic,0shot,free,0.350,40,0.333,0.075,,other_participant=0.45;speaker_confusion=0.2
falcon3-3b,dx-owner-3-deictic,3shot,all,0.492,120,0.333,0.046,,other_participant=0.3;speaker_confusion=0.192;tie=0.017
falcon3-3b,dx-owner-3-deictic,3shot,free,0.350,40,0.333,0.075,,other_participant=0.425;speaker_confusion=0.2;tie=0.025
falcon3-3b,dx-owner-3-third,0shot,all,0.592,120,0.333,0.045,,
falcon3-3b,dx-owner-3-third,0shot,free,0.525,40,0.333,0.079,,
falcon3-3b,dx-owner-3-third,3shot,all,0.633,120,0.333,0.044,,
falcon3-3b,dx-owner-3-third,3shot,free,0.575,40,0.333,0.078,,
falcon3-3b,dx-self-3-deictic,0shot,all,0.283,180,0.287,0.034,,other_object=0.233;previous_owner=0.078;speaker_confusion=0.394;tie=0.011
falcon3-3b,dx-self-3-deictic,0shot,free,0.318,126,0.286,0.042,,other_object=0.198;previous_owner=0.024;speaker_confusion=0.46
falcon3-3b,dx-self-3-deictic,3shot,all,0.411,180,0.287,0.037,,other_object=0.256;previous_owner=0.072;speaker_confusion=0.233;tie=0.028
falcon3-3b,dx-self-3-deictic,3shot,free,0.500,126,0.286,0.044,,other_object=0.246;previous_owner=0.04;speaker_confusion=0.198;tie=0.016
falcon3-3b,dx-self-3-third,0shot,all,0.444,180,0.287,0.037,,
falcon3-3b,dx-self-3-third,0shot,free,0.389,126,0.286,0.043,,
falcon3-3b,dx-self-3-third,3shot,all,0.461,180,0.287,0.037,,
falcon3-3b,dx-self-3-third,3shot,free,0.468,126,0.286,0.044,,
falcon3-3b,dx-self-2-deictic,0shot,all,0.256,180,0.287,0.033,,speaker_confusion=0.722;tie=0.022
falcon3-3b,dx-self-2-deictic,0shot,free,0.222,126,0.286,0.037,,speaker_confusion=0.754;tie=0.024
falcon3-3b,dx-self-2-deictic,3shot,all,0.450,180,0.287,0.037,,speaker_confusion=0.528;tie=0.022
falcon3-3b,dx-self-2-deictic,3shot,free,0.540,126,0.286,0.044,,speaker_confusion=0.444;tie=0.016
falcon3-3b,dx-self-2-third,0shot,all,0.556,180,0.287,0.037,,
falcon3-3b,dx-self-2-third,0shot,free,0.556,126,0.286,0.044,,
falcon3-3b,dx-self-2-third,3shot,all,0.500,180,0.287,0.037,,
falcon3-3b,dx-self-2-third,3shot,free,0.556,126,0.286,0.044,,
falcon3-3b,dx-addressee-3-deictic,0shot,all,0.300,180,0.287,0.034,,other_object=0.167;previous_owner=0.128;speaker_confusion=0.394;tie=0.011
falcon3-3b,dx-addressee-3-deictic,0shot,free,0.286,126,0.286,0.040,,other_object=0.135;previous_owner=0.159;speaker_confusion=0.405;tie=0.016
falcon3-3b,dx-addressee-3-deictic,3shot,all,0.339,180,0.287,0.035,,other_object=0.189;previous_owner=0.117;speaker_confusion=0.35;tie=0.006
falcon3-3b,dx-addressee-3-deictic,3shot,free,0.318,126,0.286,0.042,,other_object=0.167;previous_owner=0.143;speaker_confusion=0.373
falcon3-3b,dx-addressee-3-third,0shot,all,0.533,180,0.287,0.037,,
falcon3-3b,dx-addressee-3-third,0shot,free,0.460,126,0.286,0.044,,
falcon3-3b,dx-addressee-3-third,3shot,all,0.433,180,0.287,0.037,,
falcon3-3b,dx-addressee-3-third,3shot,free,0.381,126,0.286,0.043,,
falcon3-3b,dx-addressee-2-deictic,0shot,all,0.239,180,0.287,0.032,,speaker_confusion=0.739;tie=0.022
falcon3-3b,dx-addressee-2-deictic,0shot,free,0.206,126,0.286,0.036,,speaker_confusion=0.778;tie=0.016
falcon3-3b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.633;tie=0.011
falcon3-3b,dx-addressee-2-deictic,3shot,free,0.397,126,0.286,0.044,,speaker_confusion=0.595;tie=0.008
falcon3-3b,dx-addressee-2-third,0shot,all,0.511,180,0.287,0.037,,
falcon3-3b,dx-addressee-2-third,0shot,free,0.484,126,0.286,0.044,,
falcon3-3b,dx-addressee-2-third,3shot,all,0.489,180,0.287,0.037,,
falcon3-3b,dx-addressee-2-third,3shot,free,0.516,126,0.286,0.044,,
qwen2.5-3b,attr,0shot,all,0.703,1210,0.168,0.013,,
qwen2.5-3b,attr,3shot_collide,all,0.959,1210,0.168,0.006,,
qwen2.5-3b,attr,3shot,all,0.993,1210,0.168,0.002,,
qwen2.5-3b,codevar,0shot,all,0.433,1200,0.267,0.014,,
qwen2.5-3b,codevar,3shot_collide,all,0.522,1200,0.267,0.014,,
qwen2.5-3b,codevar,3shot,all,0.483,1200,0.267,0.014,,
qwen2.5-3b,dialogue,0shot,all,0.667,1200,0.312,0.014,,
qwen2.5-3b,dialogue,3shot_collide,all,0.716,1200,0.312,0.013,,
qwen2.5-3b,dialogue,3shot,all,0.713,1200,0.312,0.013,,
qwen2.5-3b,dx-owner-3-deictic,0shot,all,0.492,120,0.333,0.046,,other_participant=0.392;speaker_confusion=0.083;tie=0.033
qwen2.5-3b,dx-owner-3-deictic,0shot,free,0.425,40,0.333,0.078,,other_participant=0.425;speaker_confusion=0.1;tie=0.05
qwen2.5-3b,dx-owner-3-deictic,3shot_collide,all,0.467,120,0.333,0.045,,other_participant=0.475;speaker_confusion=0.05;tie=0.008
qwen2.5-3b,dx-owner-3-deictic,3shot_collide,free,0.375,40,0.333,0.076,,other_participant=0.575;speaker_confusion=0.025;tie=0.025
qwen2.5-3b,dx-owner-3-deictic,3shot,all,0.433,120,0.333,0.045,,other_participant=0.492;speaker_confusion=0.067;tie=0.008
qwen2.5-3b,dx-owner-3-deictic,3shot,free,0.325,40,0.333,0.074,,other_participant=0.65;speaker_confusion=0.025
qwen2.5-3b,dx-owner-3-third,0shot,all,0.567,120,0.333,0.045,,
qwen2.5-3b,dx-owner-3-third,0shot,free,0.525,40,0.333,0.079,,
qwen2.5-3b,dx-owner-3-third,3shot_collide,all,0.692,120,0.333,0.042,,
qwen2.5-3b,dx-owner-3-third,3shot_collide,free,0.600,40,0.333,0.077,,
qwen2.5-3b,dx-owner-3-third,3shot,all,0.642,120,0.333,0.044,,
qwen2.5-3b,dx-owner-3-third,3shot,free,0.550,40,0.333,0.079,,
qwen2.5-3b,dx-self-3-deictic,0shot,all,0.328,180,0.287,0.035,,other_object=0.311;previous_owner=0.033;speaker_confusion=0.294;tie=0.033
qwen2.5-3b,dx-self-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.31;previous_owner=0.016;speaker_confusion=0.254;tie=0.032
qwen2.5-3b,dx-self-3-deictic,3shot_collide,all,0.444,180,0.287,0.037,,other_object=0.222;previous_owner=0.017;speaker_confusion=0.306;tie=0.011
qwen2.5-3b,dx-self-3-deictic,3shot_collide,free,0.564,126,0.286,0.044,,other_object=0.246;previous_owner=0.008;speaker_confusion=0.167;tie=0.016
qwen2.5-3b,dx-self-3-deictic,3shot,all,0.428,180,0.287,0.037,,other_object=0.267;previous_owner=0.033;speaker_confusion=0.267;tie=0.006
qwen2.5-3b,dx-self-3-deictic,3shot,free,0.524,126,0.286,0.044,,other_object=0.27;previous_owner=0.016;speaker_confusion=0.183;tie=0.008
qwen2.5-3b,dx-self-3-third,0shot,all,0.572,180,0.287,0.037,,
qwen2.5-3b,dx-self-3-third,0shot,free,0.643,126,0.286,0.043,,
qwen2.5-3b,dx-self-3-third,3shot_collide,all,0.528,180,0.287,0.037,,
qwen2.5-3b,dx-self-3-third,3shot_collide,free,0.579,126,0.286,0.044,,
qwen2.5-3b,dx-self-3-third,3shot,all,0.478,180,0.287,0.037,,
qwen2.5-3b,dx-self-3-third,3shot,free,0.548,126,0.286,0.044,,
qwen2.5-3b,dx-self-2-deictic,0shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.661;tie=0.011
qwen2.5-3b,dx-self-2-deictic,0shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.579;tie=0.016
qwen2.5-3b,dx-self-2-deictic,3shot_collide,all,0.472,180,0.287,0.037,,speaker_confusion=0.522;tie=0.006
qwen2.5-3b,dx-self-2-deictic,3shot_collide,free,0.579,126,0.286,0.044,,speaker_confusion=0.421
qwen2.5-3b,dx-self-2-deictic,3shot,all,0.478,180,0.287,0.037,,speaker_confusion=0.517;tie=0.006
qwen2.5-3b,dx-self-2-deictic,3shot,free,0.587,126,0.286,0.044,,speaker_confusion=0.405;tie=0.008
qwen2.5-3b,dx-self-2-third,0shot,all,0.561,180,0.287,0.037,,
qwen2.5-3b,dx-self-2-third,0shot,free,0.667,126,0.286,0.042,,
qwen2.5-3b,dx-self-2-third,3shot_collide,all,0.567,180,0.287,0.037,,
qwen2.5-3b,dx-self-2-third,3shot_collide,free,0.659,126,0.286,0.042,,
qwen2.5-3b,dx-self-2-third,3shot,all,0.583,180,0.287,0.037,,
qwen2.5-3b,dx-self-2-third,3shot,free,0.675,126,0.286,0.042,,
qwen2.5-3b,dx-addressee-3-deictic,0shot,all,0.228,180,0.287,0.031,,other_object=0.272;previous_owner=0.05;speaker_confusion=0.439;tie=0.011
qwen2.5-3b,dx-addressee-3-deictic,0shot,free,0.286,126,0.286,0.040,,other_object=0.238;previous_owner=0.071;speaker_confusion=0.389;tie=0.016
qwen2.5-3b,dx-addressee-3-deictic,3shot_collide,all,0.317,180,0.287,0.035,,other_object=0.161;previous_owner=0.044;speaker_confusion=0.467;tie=0.011
qwen2.5-3b,dx-addressee-3-deictic,3shot_collide,free,0.365,126,0.286,0.043,,other_object=0.135;previous_owner=0.056;speaker_confusion=0.437;tie=0.008
qwen2.5-3b,dx-addressee-3-deictic,3shot,all,0.322,180,0.287,0.035,,other_object=0.156;previous_owner=0.039;speaker_confusion=0.478;tie=0.006
qwen2.5-3b,dx-addressee-3-deictic,3shot,free,0.365,126,0.286,0.043,,other_object=0.151;previous_owner=0.048;speaker_confusion=0.429;tie=0.008
qwen2.5-3b,dx-addressee-3-third,0shot,all,0.572,180,0.287,0.037,,
qwen2.5-3b,dx-addressee-3-third,0shot,free,0.651,126,0.286,0.043,,
qwen2.5-3b,dx-addressee-3-third,3shot_collide,all,0.500,180,0.287,0.037,,
qwen2.5-3b,dx-addressee-3-third,3shot_collide,free,0.556,126,0.286,0.044,,
qwen2.5-3b,dx-addressee-3-third,3shot,all,0.467,180,0.287,0.037,,
qwen2.5-3b,dx-addressee-3-third,3shot,free,0.532,126,0.286,0.044,,
qwen2.5-3b,dx-addressee-2-deictic,0shot,all,0.244,180,0.287,0.032,,speaker_confusion=0.75;tie=0.006
qwen2.5-3b,dx-addressee-2-deictic,0shot,free,0.262,126,0.286,0.039,,speaker_confusion=0.73;tie=0.008
qwen2.5-3b,dx-addressee-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.65;tie=0.022
qwen2.5-3b,dx-addressee-2-deictic,3shot_collide,free,0.365,126,0.286,0.043,,speaker_confusion=0.619;tie=0.016
qwen2.5-3b,dx-addressee-2-deictic,3shot,all,0.344,180,0.287,0.035,,speaker_confusion=0.65;tie=0.006
qwen2.5-3b,dx-addressee-2-deictic,3shot,free,0.389,126,0.286,0.043,,speaker_confusion=0.611
qwen2.5-3b,dx-addressee-2-third,0shot,all,0.617,180,0.287,0.036,,
qwen2.5-3b,dx-addressee-2-third,0shot,free,0.762,126,0.286,0.038,,
qwen2.5-3b,dx-addressee-2-third,3shot_collide,all,0.600,180,0.287,0.036,,
qwen2.5-3b,dx-addressee-2-third,3shot_collide,free,0.691,126,0.286,0.041,,
qwen2.5-3b,dx-addressee-2-third,3shot,all,0.583,180,0.287,0.037,,
qwen2.5-3b,dx-addressee-2-third,3shot,free,0.675,126,0.286,0.042,,
qwen2.5-3b,sb-k1,0shot,all,0.812,64,0.125,0.049,,
qwen2.5-3b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
qwen2.5-3b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen2.5-3b,sb-k23,0shot,all,0.609,128,0.027,0.043,,
qwen2.5-3b,sb-k23,3shot_collide,all,0.844,128,0.027,0.032,,
qwen2.5-3b,sb-k23,3shot,all,0.859,128,0.027,0.031,,
qwen2.5-3b,sb-split,0shot,all,0.273,128,0.027,0.039,,
qwen2.5-3b,sb-split,3shot_collide,all,0.617,128,0.027,0.043,,
qwen2.5-3b,sb-split,3shot,all,0.578,128,0.027,0.044,,
qwen2.5-3b,sb-dyn,0shot,all,0.383,256,0.027,0.030,,
qwen2.5-3b,sb-dyn,3shot_collide,all,0.172,256,0.027,0.024,,
qwen2.5-3b,sb-dyn,3shot,all,0.211,256,0.027,0.025,,
qwen2.5-3b,an-narr,0shot,all,0.372,640,0.250,0.019,,
qwen2.5-3b,an-narr,3shot_collide,all,0.502,640,0.250,0.020,,
qwen2.5-3b,an-narr,3shot,all,0.455,640,0.250,0.020,,
qwen2.5-3b,an-other-any,0shot,all,0.400,320,0.250,0.027,,
qwen2.5-3b,an-other-any,3shot_collide,all,0.284,320,0.250,0.025,,
qwen2.5-3b,an-other-any,3shot,all,0.234,320,0.250,0.024,,
qwen2.5-3b,an-me-any,0shot,all,0.000,160,0.250,0.000,,
qwen2.5-3b,an-me-any,3shot_collide,all,0.113,160,0.250,0.025,,
qwen2.5-3b,an-me-any,3shot,all,0.138,160,0.250,0.027,,
qwen2.5-3b,an-you-voc,0shot,all,0.812,80,0.250,0.044,,
qwen2.5-3b,an-you-voc,3shot_collide,all,0.225,80,0.250,0.047,,
qwen2.5-3b,an-you-voc,3shot,all,0.438,80,0.250,0.056,,
qwen2.5-3b,an-you-novoc,0shot,all,0.400,80,0.250,0.055,,
qwen2.5-3b,an-you-novoc,3shot_collide,all,0.225,80,0.250,0.047,,
qwen2.5-3b,an-you-novoc,3shot,all,0.388,80,0.250,0.054,,
qwen2.5-3b,adj-adj,0shot,all,0.589,260,0.333,0.030,,
qwen2.5-3b,adj-adj,3shot_collide,all,0.561,260,0.333,0.031,,
qwen2.5-3b,adj-adj,3shot,all,0.661,260,0.333,0.029,,
qwen2.5-3b,adj-voc,0shot,all,0.677,260,0.333,0.029,,
qwen2.5-3b,adj-voc,3shot_collide,all,0.596,260,0.333,0.030,,
qwen2.5-3b,adj-voc,3shot,all,0.612,260,0.333,0.030,,
qwen2.5-3b,adj-third,0shot,all,0.592,260,0.333,0.030,,
qwen2.5-3b,adj-third,3shot_collide,all,0.669,260,0.333,0.029,,
qwen2.5-3b,adj-third,3shot,all,0.665,260,0.333,0.029,,
qwen2.5-3b,cm-mutate,0shot,all,0.890,200,0.200,0.022,,
qwen2.5-3b,cm-mutate,3shot_collide,all,0.930,200,0.200,0.018,,
qwen2.5-3b,cm-mutate,3shot,all,0.890,200,0.200,0.022,,
qwen2.5-3b,cm-rebind,0shot,all,0.110,200,0.200,0.022,,
qwen2.5-3b,cm-rebind,3shot_collide,all,0.650,200,0.200,0.034,,
qwen2.5-3b,cm-rebind,3shot,all,0.550,200,0.200,0.035,,
qwen2.5-3b,cm-copy,0shot,all,0.090,200,0.200,0.020,,
qwen2.5-3b,cm-copy,3shot_collide,all,0.820,200,0.200,0.027,,
qwen2.5-3b,cm-copy,3shot,all,0.800,200,0.200,0.028,,
qwen3-4b-base,attr,0shot,all,0.973,1210,0.168,0.005,,
qwen3-4b-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
qwen3-4b-base,attr,3shot,all,1.000,1210,0.168,0.000,,
qwen3-4b-base,codevar,0shot,all,0.604,1200,0.267,0.014,,
qwen3-4b-base,codevar,3shot_collide,all,0.693,1200,0.267,0.013,,
qwen3-4b-base,codevar,3shot,all,0.683,1200,0.267,0.013,,
qwen3-4b-base,dialogue,0shot,all,0.873,1200,0.312,0.010,,
qwen3-4b-base,dialogue,3shot_collide,all,0.842,1200,0.312,0.011,,
qwen3-4b-base,dialogue,3shot,all,0.824,1200,0.312,0.011,,
qwen3-4b-base,dx-owner-3-deictic,0shot,all,0.708,120,0.333,0.042,,other_participant=0.225;speaker_confusion=0.067
qwen3-4b-base,dx-owner-3-deictic,0shot,free,0.775,40,0.333,0.066,,other_participant=0.225
qwen3-4b-base,dx-owner-3-deictic,3shot_collide,all,0.900,120,0.333,0.027,,other_participant=0.075;speaker_confusion=0.017;tie=0.008
qwen3-4b-base,dx-owner-3-deictic,3shot_collide,free,0.900,40,0.333,0.047,,other_participant=0.075;tie=0.025
qwen3-4b-base,dx-owner-3-deictic,3shot,all,0.900,120,0.333,0.027,,other_participant=0.075;speaker_confusion=0.025
qwen3-4b-base,dx-owner-3-deictic,3shot,free,0.925,40,0.333,0.042,,other_participant=0.075
qwen3-4b-base,dx-owner-3-third,0shot,all,0.858,120,0.333,0.032,,
qwen3-4b-base,dx-owner-3-third,0shot,free,0.950,40,0.333,0.035,,
qwen3-4b-base,dx-owner-3-third,3shot_collide,all,0.967,120,0.333,0.016,,
qwen3-4b-base,dx-owner-3-third,3shot_collide,free,0.950,40,0.333,0.035,,
qwen3-4b-base,dx-owner-3-third,3shot,all,0.958,120,0.333,0.018,,
qwen3-4b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
qwen3-4b-base,dx-self-3-deictic,0shot,all,0.372,180,0.287,0.036,,other_object=0.156;previous_owner=0.122;speaker_confusion=0.333;tie=0.017
qwen3-4b-base,dx-self-3-deictic,0shot,free,0.468,126,0.286,0.044,,other_object=0.127;previous_owner=0.063;speaker_confusion=0.333;tie=0.008
qwen3-4b-base,dx-self-3-deictic,3shot_collide,all,0.511,180,0.287,0.037,,other_object=0.2;previous_owner=0.028;speaker_confusion=0.25;tie=0.011
qwen3-4b-base,dx-self-3-deictic,3shot_collide,free,0.611,126,0.286,0.043,,other_object=0.206;previous_owner=0.024;speaker_confusion=0.159
qwen3-4b-base,dx-self-3-deictic,3shot,all,0.522,180,0.287,0.037,,other_object=0.178;previous_owner=0.028;speaker_confusion=0.261;tie=0.011
qwen3-4b-base,dx-self-3-deictic,3shot,free,0.627,126,0.286,0.043,,other_object=0.175;previous_owner=0.016;speaker_confusion=0.175;tie=0.008
qwen3-4b-base,dx-self-3-third,0shot,all,0.717,180,0.287,0.034,,
qwen3-4b-base,dx-self-3-third,0shot,free,0.794,126,0.286,0.036,,
qwen3-4b-base,dx-self-3-third,3shot_collide,all,0.722,180,0.287,0.033,,
qwen3-4b-base,dx-self-3-third,3shot_collide,free,0.754,126,0.286,0.038,,
qwen3-4b-base,dx-self-3-third,3shot,all,0.711,180,0.287,0.034,,
qwen3-4b-base,dx-self-3-third,3shot,free,0.754,126,0.286,0.038,,
qwen3-4b-base,dx-self-2-deictic,0shot,all,0.444,180,0.287,0.037,,speaker_confusion=0.544;tie=0.011
qwen3-4b-base,dx-self-2-deictic,0shot,free,0.571,126,0.286,0.044,,speaker_confusion=0.421;tie=0.008
qwen3-4b-base,dx-self-2-deictic,3shot_collide,all,0.561,180,0.287,0.037,,speaker_confusion=0.411;tie=0.028
qwen3-4b-base,dx-self-2-deictic,3shot_collide,free,0.714,126,0.286,0.040,,speaker_confusion=0.278;tie=0.008
qwen3-4b-base,dx-self-2-deictic,3shot,all,0.533,180,0.287,0.037,,speaker_confusion=0.456;tie=0.011
qwen3-4b-base,dx-self-2-deictic,3shot,free,0.667,126,0.286,0.042,,speaker_confusion=0.317;tie=0.016
qwen3-4b-base,dx-self-2-third,0shot,all,0.678,180,0.287,0.035,,
qwen3-4b-base,dx-self-2-third,0shot,free,0.786,126,0.286,0.037,,
qwen3-4b-base,dx-self-2-third,3shot_collide,all,0.689,180,0.287,0.035,,
qwen3-4b-base,dx-self-2-third,3shot_collide,free,0.778,126,0.286,0.037,,
qwen3-4b-base,dx-self-2-third,3shot,all,0.689,180,0.287,0.035,,
qwen3-4b-base,dx-self-2-third,3shot,free,0.762,126,0.286,0.038,,
qwen3-4b-base,dx-addressee-3-deictic,0shot,all,0.383,180,0.287,0.036,,other_object=0.111;previous_owner=0.117;speaker_confusion=0.361;tie=0.028
qwen3-4b-base,dx-addressee-3-deictic,0shot,free,0.484,126,0.286,0.044,,other_object=0.063;previous_owner=0.119;speaker_confusion=0.302;tie=0.032
qwen3-4b-base,dx-addressee-3-deictic,3shot_collide,all,0.406,180,0.287,0.037,,other_object=0.178;previous_owner=0.044;speaker_confusion=0.372
qwen3-4b-base,dx-addressee-3-deictic,3shot_collide,free,0.468,126,0.286,0.044,,other_object=0.143;previous_owner=0.056;speaker_confusion=0.333
qwen3-4b-base,dx-addressee-3-deictic,3shot,all,0.428,180,0.287,0.037,,other_object=0.144;previous_owner=0.05;speaker_confusion=0.372;tie=0.006
qwen3-4b-base,dx-addressee-3-deictic,3shot,free,0.508,126,0.286,0.044,,other_object=0.111;previous_owner=0.056;speaker_confusion=0.325
qwen3-4b-base,dx-addressee-3-third,0shot,all,0.744,180,0.287,0.033,,
qwen3-4b-base,dx-addressee-3-third,0shot,free,0.786,126,0.286,0.037,,
qwen3-4b-base,dx-addressee-3-third,3shot_collide,all,0.733,180,0.287,0.033,,
qwen3-4b-base,dx-addressee-3-third,3shot_collide,free,0.770,126,0.286,0.037,,
qwen3-4b-base,dx-addressee-3-third,3shot,all,0.706,180,0.287,0.034,,
qwen3-4b-base,dx-addressee-3-third,3shot,free,0.738,126,0.286,0.039,,
qwen3-4b-base,dx-addressee-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.656;tie=0.039
qwen3-4b-base,dx-addressee-2-deictic,0shot,free,0.413,126,0.286,0.044,,speaker_confusion=0.54;tie=0.048
qwen3-4b-base,dx-addressee-2-deictic,3shot_collide,all,0.478,180,0.287,0.037,,speaker_confusion=0.494;tie=0.028
qwen3-4b-base,dx-addressee-2-deictic,3shot_collide,free,0.532,126,0.286,0.044,,speaker_confusion=0.437;tie=0.032
qwen3-4b-base,dx-addressee-2-deictic,3shot,all,0.456,180,0.287,0.037,,speaker_confusion=0.528;tie=0.017
qwen3-4b-base,dx-addressee-2-deictic,3shot,free,0.516,126,0.286,0.044,,speaker_confusion=0.468;tie=0.016
qwen3-4b-base,dx-addressee-2-third,0shot,all,0.661,180,0.287,0.035,,
qwen3-4b-base,dx-addressee-2-third,0shot,free,0.794,126,0.286,0.036,,
qwen3-4b-base,dx-addressee-2-third,3shot_collide,all,0.733,180,0.287,0.033,,
qwen3-4b-base,dx-addressee-2-third,3shot_collide,free,0.809,126,0.286,0.035,,
qwen3-4b-base,dx-addressee-2-third,3shot,all,0.722,180,0.287,0.033,,
qwen3-4b-base,dx-addressee-2-third,3shot,free,0.802,126,0.286,0.035,,
qwen3-4b-base,sb-k1,0shot,all,1.000,64,0.125,0.000,,
qwen3-4b-base,sb-k1,3shot_collide,all,0.984,64,0.125,0.015,,
qwen3-4b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen3-4b-base,sb-k23,0shot,all,0.742,128,0.027,0.039,,
qwen3-4b-base,sb-k23,3shot_collide,all,0.945,128,0.027,0.020,,
qwen3-4b-base,sb-k23,3shot,all,0.969,128,0.027,0.015,,
qwen3-4b-base,sb-split,0shot,all,0.688,128,0.027,0.041,,
qwen3-4b-base,sb-split,3shot_collide,all,0.797,128,0.027,0.036,,
qwen3-4b-base,sb-split,3shot,all,0.828,128,0.027,0.033,,
qwen3-4b-base,sb-dyn,0shot,all,0.547,256,0.027,0.031,,
qwen3-4b-base,sb-dyn,3shot_collide,all,0.371,256,0.027,0.030,,
qwen3-4b-base,sb-dyn,3shot,all,0.352,256,0.027,0.030,,
qwen3-4b-base,an-narr,0shot,all,0.655,640,0.250,0.019,,
qwen3-4b-base,an-narr,3shot_collide,all,0.903,640,0.250,0.012,,
qwen3-4b-base,an-narr,3shot,all,0.894,640,0.250,0.012,,
qwen3-4b-base,an-other-any,0shot,all,0.506,320,0.250,0.028,,
qwen3-4b-base,an-other-any,3shot_collide,all,0.653,320,0.250,0.027,,
qwen3-4b-base,an-other-any,3shot,all,0.637,320,0.250,0.027,,
qwen3-4b-base,an-me-any,0shot,all,0.131,160,0.250,0.027,,
qwen3-4b-base,an-me-any,3shot_collide,all,0.406,160,0.250,0.039,,
qwen3-4b-base,an-me-any,3shot,all,0.312,160,0.250,0.037,,
qwen3-4b-base,an-you-voc,0shot,all,0.412,80,0.250,0.055,,
qwen3-4b-base,an-you-voc,3shot_collide,all,0.800,80,0.250,0.045,,
qwen3-4b-base,an-you-voc,3shot,all,0.863,80,0.250,0.038,,
qwen3-4b-base,an-you-novoc,0shot,all,0.775,80,0.250,0.047,,
qwen3-4b-base,an-you-novoc,3shot_collide,all,0.450,80,0.250,0.056,,
qwen3-4b-base,an-you-novoc,3shot,all,0.537,80,0.250,0.056,,
qwen3-4b-base,adj-adj,0shot,all,0.888,260,0.333,0.019,,
qwen3-4b-base,adj-adj,3shot_collide,all,0.923,260,0.333,0.017,,
qwen3-4b-base,adj-adj,3shot,all,0.904,260,0.333,0.018,,
qwen3-4b-base,adj-voc,0shot,all,0.954,260,0.333,0.013,,
qwen3-4b-base,adj-voc,3shot_collide,all,0.977,260,0.333,0.009,,
qwen3-4b-base,adj-voc,3shot,all,0.985,260,0.333,0.008,,
qwen3-4b-base,adj-third,0shot,all,0.946,260,0.333,0.014,,
qwen3-4b-base,adj-third,3shot_collide,all,0.989,260,0.333,0.007,,
qwen3-4b-base,adj-third,3shot,all,0.985,260,0.333,0.008,,
qwen3-4b-base,cm-mutate,0shot,all,0.945,200,0.200,0.016,,
qwen3-4b-base,cm-mutate,3shot_collide,all,0.945,200,0.200,0.016,,
qwen3-4b-base,cm-mutate,3shot,all,0.965,200,0.200,0.013,,
qwen3-4b-base,cm-rebind,0shot,all,0.310,200,0.200,0.033,,
qwen3-4b-base,cm-rebind,3shot_collide,all,0.985,200,0.200,0.009,,
qwen3-4b-base,cm-rebind,3shot,all,0.965,200,0.200,0.013,,
qwen3-4b-base,cm-copy,0shot,all,0.295,200,0.200,0.032,,
qwen3-4b-base,cm-copy,3shot_collide,all,0.975,200,0.200,0.011,,
qwen3-4b-base,cm-copy,3shot,all,0.980,200,0.200,0.010,,
pythia-6.9b,attr,0shot,all,0.529,1210,0.168,0.014,,
pythia-6.9b,attr,3shot_collide,all,0.821,1210,0.168,0.011,,
pythia-6.9b,attr,3shot,all,0.872,1210,0.168,0.010,,
pythia-6.9b,codevar,0shot,all,0.370,1200,0.267,0.014,,
pythia-6.9b,codevar,3shot_collide,all,0.385,1200,0.267,0.014,,
pythia-6.9b,codevar,3shot,all,0.397,1200,0.267,0.014,,
pythia-6.9b,dialogue,0shot,all,0.566,1200,0.312,0.014,,
pythia-6.9b,dialogue,3shot_collide,all,0.473,1200,0.312,0.014,,
pythia-6.9b,dialogue,3shot,all,0.499,1200,0.312,0.014,,
pythia-6.9b,dx-owner-3-deictic,0shot,all,0.450,120,0.333,0.045,,other_participant=0.333;speaker_confusion=0.217
pythia-6.9b,dx-owner-3-deictic,0shot,free,0.525,40,0.333,0.079,,other_participant=0.35;speaker_confusion=0.125
pythia-6.9b,dx-owner-3-deictic,3shot_collide,all,0.283,120,0.333,0.041,,other_participant=0.367;speaker_confusion=0.333;tie=0.017
pythia-6.9b,dx-owner-3-deictic,3shot_collide,free,0.200,40,0.333,0.063,,other_participant=0.55;speaker_confusion=0.25
pythia-6.9b,dx-owner-3-deictic,3shot,all,0.442,120,0.333,0.045,,other_participant=0.283;speaker_confusion=0.267;tie=0.008
pythia-6.9b,dx-owner-3-deictic,3shot,free,0.500,40,0.333,0.079,,other_participant=0.325;speaker_confusion=0.15;tie=0.025
pythia-6.9b,dx-owner-3-third,0shot,all,0.558,120,0.333,0.045,,
pythia-6.9b,dx-owner-3-third,0shot,free,0.775,40,0.333,0.066,,
pythia-6.9b,dx-owner-3-third,3shot_collide,all,0.367,120,0.333,0.044,,
pythia-6.9b,dx-owner-3-third,3shot_collide,free,0.325,40,0.333,0.074,,
pythia-6.9b,dx-owner-3-third,3shot,all,0.533,120,0.333,0.045,,
pythia-6.9b,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
pythia-6.9b,dx-self-3-deictic,0shot,all,0.300,180,0.287,0.034,,other_object=0.222;previous_owner=0.089;speaker_confusion=0.383;tie=0.006
pythia-6.9b,dx-self-3-deictic,0shot,free,0.333,126,0.286,0.042,,other_object=0.238;previous_owner=0.032;speaker_confusion=0.397
pythia-6.9b,dx-self-3-deictic,3shot_collide,all,0.344,180,0.287,0.035,,other_object=0.256;previous_owner=0.083;speaker_confusion=0.306;tie=0.011
pythia-6.9b,dx-self-3-deictic,3shot_collide,free,0.452,126,0.286,0.044,,other_object=0.27;previous_owner=0.063;speaker_confusion=0.198;tie=0.016
pythia-6.9b,dx-self-3-deictic,3shot,all,0.367,180,0.287,0.036,,other_object=0.267;previous_owner=0.089;speaker_confusion=0.272;tie=0.006
pythia-6.9b,dx-self-3-deictic,3shot,free,0.476,126,0.286,0.044,,other_object=0.246;previous_owner=0.063;speaker_confusion=0.206;tie=0.008
pythia-6.9b,dx-self-3-third,0shot,all,0.267,180,0.287,0.033,,
pythia-6.9b,dx-self-3-third,0shot,free,0.309,126,0.286,0.041,,
pythia-6.9b,dx-self-3-third,3shot_collide,all,0.372,180,0.287,0.036,,
pythia-6.9b,dx-self-3-third,3shot_collide,free,0.476,126,0.286,0.044,,
pythia-6.9b,dx-self-3-third,3shot,all,0.411,180,0.287,0.037,,
pythia-6.9b,dx-self-3-third,3shot,free,0.524,126,0.286,0.044,,
pythia-6.9b,dx-self-2-deictic,0shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.678;tie=0.006
pythia-6.9b,dx-self-2-deictic,0shot,free,0.349,126,0.286,0.043,,speaker_confusion=0.643;tie=0.008
pythia-6.9b,dx-self-2-deictic,3shot_collide,all,0.339,180,0.287,0.035,,speaker_confusion=0.661
pythia-6.9b,dx-self-2-deictic,3shot_collide,free,0.436,126,0.286,0.044,,speaker_confusion=0.563
pythia-6.9b,dx-self-2-deictic,3shot,all,0.383,180,0.287,0.036,,speaker_confusion=0.6;tie=0.017
pythia-6.9b,dx-self-2-deictic,3shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.492;tie=0.024
pythia-6.9b,dx-self-2-third,0shot,all,0.339,180,0.287,0.035,,
pythia-6.9b,dx-self-2-third,0shot,free,0.397,126,0.286,0.044,,
pythia-6.9b,dx-self-2-third,3shot_collide,all,0.400,180,0.287,0.036,,
pythia-6.9b,dx-self-2-third,3shot_collide,free,0.516,126,0.286,0.044,,
pythia-6.9b,dx-self-2-third,3shot,all,0.383,180,0.287,0.036,,
pythia-6.9b,dx-self-2-third,3shot,free,0.484,126,0.286,0.044,,
pythia-6.9b,dx-addressee-3-deictic,0shot,all,0.356,180,0.287,0.036,,other_object=0.128;previous_owner=0.167;speaker_confusion=0.35
pythia-6.9b,dx-addressee-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.119;previous_owner=0.214;speaker_confusion=0.357
pythia-6.9b,dx-addressee-3-deictic,3shot_collide,all,0.333,180,0.287,0.035,,other_object=0.211;previous_owner=0.128;speaker_confusion=0.328
pythia-6.9b,dx-addressee-3-deictic,3shot_collide,free,0.381,126,0.286,0.043,,other_object=0.159;previous_owner=0.135;speaker_confusion=0.325
pythia-6.9b,dx-addressee-3-deictic,3shot,all,0.350,180,0.287,0.036,,other_object=0.211;previous_owner=0.133;speaker_confusion=0.289;tie=0.017
pythia-6.9b,dx-addressee-3-deictic,3shot,free,0.405,126,0.286,0.044,,other_object=0.159;previous_owner=0.151;speaker_confusion=0.262;tie=0.024
pythia-6.9b,dx-addressee-3-third,0shot,all,0.350,180,0.287,0.036,,
pythia-6.9b,dx-addressee-3-third,0shot,free,0.397,126,0.286,0.044,,
pythia-6.9b,dx-addressee-3-third,3shot_collide,all,0.383,180,0.287,0.036,,
pythia-6.9b,dx-addressee-3-third,3shot_collide,free,0.468,126,0.286,0.044,,
pythia-6.9b,dx-addressee-3-third,3shot,all,0.333,180,0.287,0.035,,
pythia-6.9b,dx-addressee-3-third,3shot,free,0.389,126,0.286,0.043,,
pythia-6.9b,dx-addressee-2-deictic,0shot,all,0.206,180,0.287,0.030,,speaker_confusion=0.783;tie=0.011
pythia-6.9b,dx-addressee-2-deictic,0shot,free,0.182,126,0.286,0.034,,speaker_confusion=0.81;tie=0.008
pythia-6.9b,dx-addressee-2-deictic,3shot_collide,all,0.278,180,0.287,0.033,,speaker_confusion=0.689;tie=0.033
pythia-6.9b,dx-addressee-2-deictic,3shot_collide,free,0.341,126,0.286,0.042,,speaker_confusion=0.627;tie=0.032
pythia-6.9b,dx-addressee-2-deictic,3shot,all,0.278,180,0.287,0.033,,speaker_confusion=0.711;tie=0.011
pythia-6.9b,dx-addressee-2-deictic,3shot,free,0.318,126,0.286,0.042,,speaker_confusion=0.667;tie=0.016
pythia-6.9b,dx-addressee-2-third,0shot,all,0.389,180,0.287,0.036,,
pythia-6.9b,dx-addressee-2-third,0shot,free,0.436,126,0.286,0.044,,
pythia-6.9b,dx-addressee-2-third,3shot_collide,all,0.428,180,0.287,0.037,,
pythia-6.9b,dx-addressee-2-third,3shot_collide,free,0.540,126,0.286,0.044,,
pythia-6.9b,dx-addressee-2-third,3shot,all,0.400,180,0.287,0.036,,
pythia-6.9b,dx-addressee-2-third,3shot,free,0.492,126,0.286,0.044,,
pythia-6.9b,sb-k1,0shot,all,0.578,64,0.125,0.062,,
pythia-6.9b,sb-k1,3shot_collide,all,0.703,64,0.125,0.057,,
pythia-6.9b,sb-k1,3shot,all,0.766,64,0.125,0.053,,
pythia-6.9b,sb-k23,0shot,all,0.242,128,0.027,0.038,,
pythia-6.9b,sb-k23,3shot_collide,all,0.500,128,0.027,0.044,,
pythia-6.9b,sb-k23,3shot,all,0.531,128,0.027,0.044,,
pythia-6.9b,sb-split,0shot,all,0.102,128,0.027,0.027,,
pythia-6.9b,sb-split,3shot_collide,all,0.203,128,0.027,0.036,,
pythia-6.9b,sb-split,3shot,all,0.180,128,0.027,0.034,,
pythia-6.9b,sb-dyn,0shot,all,0.113,256,0.027,0.020,,
pythia-6.9b,sb-dyn,3shot_collide,all,0.117,256,0.027,0.020,,
pythia-6.9b,sb-dyn,3shot,all,0.078,256,0.027,0.017,,
pythia-6.9b,an-narr,0shot,all,0.402,640,0.250,0.019,,
pythia-6.9b,an-narr,3shot_collide,all,0.270,640,0.250,0.018,,
pythia-6.9b,an-narr,3shot,all,0.436,640,0.250,0.020,,
pythia-6.9b,an-other-any,0shot,all,0.650,320,0.250,0.027,,
pythia-6.9b,an-other-any,3shot_collide,all,0.341,320,0.250,0.026,,
pythia-6.9b,an-other-any,3shot,all,0.447,320,0.250,0.028,,
pythia-6.9b,an-me-any,0shot,all,0.150,160,0.250,0.028,,
pythia-6.9b,an-me-any,3shot_collide,all,0.069,160,0.250,0.020,,
pythia-6.9b,an-me-any,3shot,all,0.062,160,0.250,0.019,,
pythia-6.9b,an-you-voc,0shot,all,0.175,80,0.250,0.043,,
pythia-6.9b,an-you-voc,3shot_collide,all,0.075,80,0.250,0.029,,
pythia-6.9b,an-you-voc,3shot,all,0.062,80,0.250,0.027,,
pythia-6.9b,an-you-novoc,0shot,all,0.050,80,0.250,0.024,,
pythia-6.9b,an-you-novoc,3shot_collide,all,0.200,80,0.250,0.045,,
pythia-6.9b,an-you-novoc,3shot,all,0.138,80,0.250,0.038,,
pythia-6.9b,adj-adj,0shot,all,0.415,260,0.333,0.031,,
pythia-6.9b,adj-adj,3shot_collide,all,0.158,260,0.333,0.023,,
pythia-6.9b,adj-adj,3shot,all,0.262,260,0.333,0.027,,
pythia-6.9b,adj-voc,0shot,all,0.561,260,0.333,0.031,,
pythia-6.9b,adj-voc,3shot_collide,all,0.238,260,0.333,0.026,,
pythia-6.9b,adj-voc,3shot,all,0.508,260,0.333,0.031,,
pythia-6.9b,adj-third,0shot,all,0.673,260,0.333,0.029,,
pythia-6.9b,adj-third,3shot_collide,all,0.385,260,0.333,0.030,,
pythia-6.9b,adj-third,3shot,all,0.450,260,0.333,0.031,,
pythia-6.9b,cm-mutate,0shot,all,0.540,200,0.200,0.035,,
pythia-6.9b,cm-mutate,3shot_collide,all,0.425,200,0.200,0.035,,
pythia-6.9b,cm-mutate,3shot,all,0.625,200,0.200,0.034,,
pythia-6.9b,cm-rebind,0shot,all,0.180,200,0.200,0.027,,
pythia-6.9b,cm-rebind,3shot_collide,all,0.225,200,0.200,0.029,,
pythia-6.9b,cm-rebind,3shot,all,0.180,200,0.200,0.027,,
pythia-6.9b,cm-copy,0shot,all,0.105,200,0.200,0.022,,
pythia-6.9b,cm-copy,3shot_collide,all,0.330,200,0.200,0.033,,
pythia-6.9b,cm-copy,3shot,all,0.205,200,0.200,0.029,,
falcon3-7b,attr,0shot,all,0.752,1210,0.168,0.012,,
falcon3-7b,attr,3shot_collide,all,0.995,1210,0.168,0.002,,
falcon3-7b,attr,3shot,all,0.998,1210,0.168,0.001,,
falcon3-7b,codevar,0shot,all,0.624,1200,0.267,0.014,,
falcon3-7b,codevar,3shot_collide,all,0.829,1200,0.267,0.011,,
falcon3-7b,codevar,3shot,all,0.827,1200,0.267,0.011,,
falcon3-7b,dialogue,0shot,all,0.859,1200,0.312,0.010,,
falcon3-7b,dialogue,3shot_collide,all,0.877,1200,0.312,0.009,,
falcon3-7b,dialogue,3shot,all,0.843,1200,0.312,0.011,,
falcon3-7b,dx-owner-3-deictic,0shot,all,0.567,120,0.333,0.045,,other_participant=0.367;speaker_confusion=0.067
falcon3-7b,dx-owner-3-deictic,0shot,free,0.350,40,0.333,0.075,,other_participant=0.55;speaker_confusion=0.1
falcon3-7b,dx-owner-3-deictic,3shot,all,0.833,120,0.333,0.034,,other_participant=0.083;speaker_confusion=0.083
falcon3-7b,dx-owner-3-deictic,3shot,free,0.800,40,0.333,0.063,,other_participant=0.1;speaker_confusion=0.1
falcon3-7b,dx-owner-3-third,0shot,all,0.775,120,0.333,0.038,,
falcon3-7b,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
falcon3-7b,dx-owner-3-third,3shot,all,0.908,120,0.333,0.026,,
falcon3-7b,dx-owner-3-third,3shot,free,0.900,40,0.333,0.047,,
falcon3-7b,dx-self-3-deictic,0shot,all,0.456,180,0.287,0.037,,other_object=0.139;previous_owner=0.089;speaker_confusion=0.317
falcon3-7b,dx-self-3-deictic,0shot,free,0.476,126,0.286,0.044,,other_object=0.151;previous_owner=0.04;speaker_confusion=0.333
falcon3-7b,dx-self-3-deictic,3shot,all,0.517,180,0.287,0.037,,other_object=0.133;previous_owner=0.056;speaker_confusion=0.294
falcon3-7b,dx-self-3-deictic,3shot,free,0.627,126,0.286,0.043,,other_object=0.103;previous_owner=0.032;speaker_confusion=0.238
falcon3-7b,dx-self-3-third,0shot,all,0.728,180,0.287,0.033,,
falcon3-7b,dx-self-3-third,0shot,free,0.714,126,0.286,0.040,,
falcon3-7b,dx-self-3-third,3shot,all,0.683,180,0.287,0.035,,
falcon3-7b,dx-self-3-third,3shot,free,0.714,126,0.286,0.040,,
falcon3-7b,dx-self-2-deictic,0shot,all,0.344,180,0.287,0.035,,speaker_confusion=0.65;tie=0.006
falcon3-7b,dx-self-2-deictic,0shot,free,0.325,126,0.286,0.042,,speaker_confusion=0.667;tie=0.008
falcon3-7b,dx-self-2-deictic,3shot,all,0.494,180,0.287,0.037,,speaker_confusion=0.506
falcon3-7b,dx-self-2-deictic,3shot,free,0.595,126,0.286,0.044,,speaker_confusion=0.405
falcon3-7b,dx-self-2-third,0shot,all,0.694,180,0.287,0.034,,
falcon3-7b,dx-self-2-third,0shot,free,0.730,126,0.286,0.040,,
falcon3-7b,dx-self-2-third,3shot,all,0.683,180,0.287,0.035,,
falcon3-7b,dx-self-2-third,3shot,free,0.738,126,0.286,0.039,,
falcon3-7b,dx-addressee-3-deictic,0shot,all,0.361,180,0.287,0.036,,other_object=0.217;previous_owner=0.1;speaker_confusion=0.322
falcon3-7b,dx-addressee-3-deictic,0shot,free,0.318,126,0.286,0.042,,other_object=0.206;previous_owner=0.135;speaker_confusion=0.341
falcon3-7b,dx-addressee-3-deictic,3shot,all,0.378,180,0.287,0.036,,other_object=0.133;previous_owner=0.111;speaker_confusion=0.378
falcon3-7b,dx-addressee-3-deictic,3shot,free,0.405,126,0.286,0.044,,other_object=0.103;previous_owner=0.143;speaker_confusion=0.349
falcon3-7b,dx-addressee-3-third,0shot,all,0.717,180,0.287,0.034,,
falcon3-7b,dx-addressee-3-third,0shot,free,0.691,126,0.286,0.041,,
falcon3-7b,dx-addressee-3-third,3shot,all,0.633,180,0.287,0.036,,
falcon3-7b,dx-addressee-3-third,3shot,free,0.603,126,0.286,0.044,,
falcon3-7b,dx-addressee-2-deictic,0shot,all,0.278,180,0.287,0.033,,speaker_confusion=0.722
falcon3-7b,dx-addressee-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.81
falcon3-7b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.644
falcon3-7b,dx-addressee-2-deictic,3shot,free,0.357,126,0.286,0.043,,speaker_confusion=0.643
falcon3-7b,dx-addressee-2-third,0shot,all,0.628,180,0.287,0.036,,
falcon3-7b,dx-addressee-2-third,0shot,free,0.682,126,0.286,0.042,,
falcon3-7b,dx-addressee-2-third,3shot,all,0.706,180,0.287,0.034,,
falcon3-7b,dx-addressee-2-third,3shot,free,0.778,126,0.286,0.037,,
falcon3-7b,sb-k1,0shot,all,0.703,64,0.125,0.057,,
falcon3-7b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
falcon3-7b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
falcon3-7b,sb-k23,0shot,all,0.414,128,0.027,0.043,,
falcon3-7b,sb-k23,3shot_collide,all,0.930,128,0.027,0.023,,
falcon3-7b,sb-k23,3shot,all,0.930,128,0.027,0.023,,
falcon3-7b,sb-split,0shot,all,0.492,128,0.027,0.044,,
falcon3-7b,sb-split,3shot_collide,all,0.836,128,0.027,0.033,,
falcon3-7b,sb-split,3shot,all,0.789,128,0.027,0.036,,
falcon3-7b,sb-dyn,0shot,all,0.410,256,0.027,0.031,,
falcon3-7b,sb-dyn,3shot_collide,all,0.223,256,0.027,0.026,,
falcon3-7b,sb-dyn,3shot,all,0.203,256,0.027,0.025,,
falcon3-7b,an-narr,0shot,all,0.505,640,0.250,0.020,,
falcon3-7b,an-narr,3shot_collide,all,0.831,640,0.250,0.015,,
falcon3-7b,an-narr,3shot,all,0.873,640,0.250,0.013,,
falcon3-7b,an-other-any,0shot,all,0.428,320,0.250,0.028,,
falcon3-7b,an-other-any,3shot_collide,all,0.463,320,0.250,0.028,,
falcon3-7b,an-other-any,3shot,all,0.494,320,0.250,0.028,,
falcon3-7b,an-me-any,0shot,all,0.094,160,0.250,0.023,,
falcon3-7b,an-me-any,3shot_collide,all,0.125,160,0.250,0.026,,
falcon3-7b,an-me-any,3shot,all,0.075,160,0.250,0.021,,
falcon3-7b,an-you-voc,0shot,all,0.400,80,0.250,0.055,,
falcon3-7b,an-you-voc,3shot_collide,all,0.463,80,0.250,0.056,,
falcon3-7b,an-you-voc,3shot,all,0.662,80,0.250,0.053,,
falcon3-7b,an-you-novoc,0shot,all,0.438,80,0.250,0.056,,
falcon3-7b,an-you-novoc,3shot_collide,all,0.138,80,0.250,0.038,,
falcon3-7b,an-you-novoc,3shot,all,0.250,80,0.250,0.048,,
falcon3-7b,adj-adj,0shot,all,0.858,260,0.333,0.022,,
falcon3-7b,adj-adj,3shot_collide,all,0.819,260,0.333,0.024,,
falcon3-7b,adj-adj,3shot,all,0.885,260,0.333,0.020,,
falcon3-7b,adj-voc,0shot,all,0.896,260,0.333,0.019,,
falcon3-7b,adj-voc,3shot_collide,all,0.977,260,0.333,0.009,,
falcon3-7b,adj-voc,3shot,all,0.981,260,0.333,0.009,,
falcon3-7b,adj-third,0shot,all,0.989,260,0.333,0.007,,
falcon3-7b,adj-third,3shot_collide,all,0.965,260,0.333,0.011,,
falcon3-7b,adj-third,3shot,all,0.981,260,0.333,0.009,,
falcon3-7b,cm-mutate,0shot,all,0.840,200,0.200,0.026,,
falcon3-7b,cm-mutate,3shot_collide,all,0.950,200,0.200,0.015,,
falcon3-7b,cm-mutate,3shot,all,0.970,200,0.200,0.012,,
falcon3-7b,cm-rebind,0shot,all,0.225,200,0.200,0.029,,
falcon3-7b,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
falcon3-7b,cm-rebind,3shot,all,0.990,200,0.200,0.007,,
falcon3-7b,cm-copy,0shot,all,0.120,200,0.200,0.023,,
falcon3-7b,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
falcon3-7b,cm-copy,3shot,all,0.990,200,0.200,0.007,,
olmo2-7b-base,attr,0shot,all,0.456,1210,0.168,0.014,,
olmo2-7b-base,attr,3shot_collide,all,0.881,1210,0.168,0.009,,
olmo2-7b-base,attr,3shot,all,0.919,1210,0.168,0.008,,
olmo2-7b-base,codevar,0shot,all,0.372,1200,0.267,0.014,,
olmo2-7b-base,codevar,3shot_collide,all,0.472,1200,0.267,0.014,,
olmo2-7b-base,codevar,3shot,all,0.485,1200,0.267,0.014,,
olmo2-7b-base,dialogue,0shot,all,0.767,1200,0.312,0.012,,
olmo2-7b-base,dialogue,3shot_collide,all,0.797,1200,0.312,0.012,,
olmo2-7b-base,dialogue,3shot,all,0.803,1200,0.312,0.011,,
olmo2-7b-base,dx-owner-3-deictic,0shot,all,0.592,120,0.333,0.045,,other_participant=0.317;speaker_confusion=0.092
olmo2-7b-base,dx-owner-3-deictic,0shot,free,0.400,40,0.333,0.077,,other_participant=0.525;speaker_confusion=0.075
olmo2-7b-base,dx-owner-3-deictic,3shot_collide,all,0.692,120,0.333,0.042,,other_participant=0.283;speaker_confusion=0.025
olmo2-7b-base,dx-owner-3-deictic,3shot_collide,free,0.600,40,0.333,0.077,,other_participant=0.375;speaker_confusion=0.025
olmo2-7b-base,dx-owner-3-deictic,3shot,all,0.692,120,0.333,0.042,,other_participant=0.275;speaker_confusion=0.033
olmo2-7b-base,dx-owner-3-deictic,3shot,free,0.625,40,0.333,0.076,,other_participant=0.375
olmo2-7b-base,dx-owner-3-third,0shot,all,0.667,120,0.333,0.043,,
olmo2-7b-base,dx-owner-3-third,0shot,free,0.475,40,0.333,0.079,,
olmo2-7b-base,dx-owner-3-third,3shot_collide,all,0.617,120,0.333,0.044,,
olmo2-7b-base,dx-owner-3-third,3shot_collide,free,0.525,40,0.333,0.079,,
olmo2-7b-base,dx-owner-3-third,3shot,all,0.742,120,0.333,0.040,,
olmo2-7b-base,dx-owner-3-third,3shot,free,0.650,40,0.333,0.075,,
olmo2-7b-base,dx-self-3-deictic,0shot,all,0.383,180,0.287,0.036,,other_object=0.311;previous_owner=0.056;speaker_confusion=0.25
olmo2-7b-base,dx-self-3-deictic,0shot,free,0.365,126,0.286,0.043,,other_object=0.341;previous_owner=0.016;speaker_confusion=0.278
olmo2-7b-base,dx-self-3-deictic,3shot_collide,all,0.422,180,0.287,0.037,,other_object=0.267;previous_owner=0.061;speaker_confusion=0.25
olmo2-7b-base,dx-self-3-deictic,3shot_collide,free,0.452,126,0.286,0.044,,other_object=0.294;previous_owner=0.048;speaker_confusion=0.206
olmo2-7b-base,dx-self-3-deictic,3shot,all,0.411,180,0.287,0.037,,other_object=0.294;previous_owner=0.039;speaker_confusion=0.256
olmo2-7b-base,dx-self-3-deictic,3shot,free,0.413,126,0.286,0.044,,other_object=0.317;previous_owner=0.024;speaker_confusion=0.246
olmo2-7b-base,dx-self-3-third,0shot,all,0.461,180,0.287,0.037,,
olmo2-7b-base,dx-self-3-third,0shot,free,0.357,126,0.286,0.043,,
olmo2-7b-base,dx-self-3-third,3shot_collide,all,0.489,180,0.287,0.037,,
olmo2-7b-base,dx-self-3-third,3shot_collide,free,0.460,126,0.286,0.044,,
olmo2-7b-base,dx-self-3-third,3shot,all,0.550,180,0.287,0.037,,
olmo2-7b-base,dx-self-3-third,3shot,free,0.524,126,0.286,0.044,,
olmo2-7b-base,dx-self-2-deictic,0shot,all,0.339,180,0.287,0.035,,speaker_confusion=0.656;tie=0.006
olmo2-7b-base,dx-self-2-deictic,0shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.683;tie=0.008
olmo2-7b-base,dx-self-2-deictic,3shot_collide,all,0.428,180,0.287,0.037,,speaker_confusion=0.572
olmo2-7b-base,dx-self-2-deictic,3shot_collide,free,0.508,126,0.286,0.044,,speaker_confusion=0.492
olmo2-7b-base,dx-self-2-deictic,3shot,all,0.361,180,0.287,0.036,,speaker_confusion=0.639
olmo2-7b-base,dx-self-2-deictic,3shot,free,0.413,126,0.286,0.044,,speaker_confusion=0.587
olmo2-7b-base,dx-self-2-third,0shot,all,0.456,180,0.287,0.037,,
olmo2-7b-base,dx-self-2-third,0shot,free,0.476,126,0.286,0.044,,
olmo2-7b-base,dx-self-2-third,3shot_collide,all,0.550,180,0.287,0.037,,
olmo2-7b-base,dx-self-2-third,3shot_collide,free,0.556,126,0.286,0.044,,
olmo2-7b-base,dx-self-2-third,3shot,all,0.550,180,0.287,0.037,,
olmo2-7b-base,dx-self-2-third,3shot,free,0.571,126,0.286,0.044,,
olmo2-7b-base,dx-addressee-3-deictic,0shot,all,0.333,180,0.287,0.035,,other_object=0.206;previous_owner=0.089;speaker_confusion=0.372
olmo2-7b-base,dx-addressee-3-deictic,0shot,free,0.341,126,0.286,0.042,,other_object=0.183;previous_owner=0.103;speaker_confusion=0.373
olmo2-7b-base,dx-addressee-3-deictic,3shot_collide,all,0.317,180,0.287,0.035,,other_object=0.211;previous_owner=0.094;speaker_confusion=0.378
olmo2-7b-base,dx-addressee-3-deictic,3shot_collide,free,0.333,126,0.286,0.042,,other_object=0.151;previous_owner=0.119;speaker_confusion=0.397
olmo2-7b-base,dx-addressee-3-deictic,3shot,all,0.300,180,0.287,0.034,,other_object=0.206;previous_owner=0.117;speaker_confusion=0.378
olmo2-7b-base,dx-addressee-3-deictic,3shot,free,0.278,126,0.286,0.040,,other_object=0.159;previous_owner=0.135;speaker_confusion=0.429
olmo2-7b-base,dx-addressee-3-third,0shot,all,0.494,180,0.287,0.037,,
olmo2-7b-base,dx-addressee-3-third,0shot,free,0.389,126,0.286,0.043,,
olmo2-7b-base,dx-addressee-3-third,3shot_collide,all,0.533,180,0.287,0.037,,
olmo2-7b-base,dx-addressee-3-third,3shot_collide,free,0.476,126,0.286,0.044,,
olmo2-7b-base,dx-addressee-3-third,3shot,all,0.572,180,0.287,0.037,,
olmo2-7b-base,dx-addressee-3-third,3shot,free,0.532,126,0.286,0.044,,
olmo2-7b-base,dx-addressee-2-deictic,0shot,all,0.272,180,0.287,0.033,,speaker_confusion=0.728
olmo2-7b-base,dx-addressee-2-deictic,0shot,free,0.222,126,0.286,0.037,,speaker_confusion=0.778
olmo2-7b-base,dx-addressee-2-deictic,3shot_collide,all,0.356,180,0.287,0.036,,speaker_confusion=0.644
olmo2-7b-base,dx-addressee-2-deictic,3shot_collide,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
olmo2-7b-base,dx-addressee-2-deictic,3shot,all,0.339,180,0.287,0.035,,speaker_confusion=0.661
olmo2-7b-base,dx-addressee-2-deictic,3shot,free,0.333,126,0.286,0.042,,speaker_confusion=0.667
olmo2-7b-base,dx-addressee-2-third,0shot,all,0.450,180,0.287,0.037,,
olmo2-7b-base,dx-addressee-2-third,0shot,free,0.444,126,0.286,0.044,,
olmo2-7b-base,dx-addressee-2-third,3shot_collide,all,0.594,180,0.287,0.037,,
olmo2-7b-base,dx-addressee-2-third,3shot_collide,free,0.619,126,0.286,0.043,,
olmo2-7b-base,dx-addressee-2-third,3shot,all,0.628,180,0.287,0.036,,
olmo2-7b-base,dx-addressee-2-third,3shot,free,0.619,126,0.286,0.043,,
olmo2-7b-base,sb-k1,0shot,all,0.516,64,0.125,0.062,,
olmo2-7b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
olmo2-7b-base,sb-k1,3shot,all,0.969,64,0.125,0.022,,
olmo2-7b-base,sb-k23,0shot,all,0.266,128,0.027,0.039,,
olmo2-7b-base,sb-k23,3shot_collide,all,0.820,128,0.027,0.034,,
olmo2-7b-base,sb-k23,3shot,all,0.820,128,0.027,0.034,,
olmo2-7b-base,sb-split,0shot,all,0.102,128,0.027,0.027,,
olmo2-7b-base,sb-split,3shot_collide,all,0.461,128,0.027,0.044,,
olmo2-7b-base,sb-split,3shot,all,0.484,128,0.027,0.044,,
olmo2-7b-base,sb-dyn,0shot,all,0.098,256,0.027,0.019,,
olmo2-7b-base,sb-dyn,3shot_collide,all,0.168,256,0.027,0.023,,
olmo2-7b-base,sb-dyn,3shot,all,0.168,256,0.027,0.023,,
olmo2-7b-base,an-narr,0shot,all,0.489,640,0.250,0.020,,
olmo2-7b-base,an-narr,3shot_collide,all,0.698,640,0.250,0.018,,
olmo2-7b-base,an-narr,3shot,all,0.662,640,0.250,0.019,,
olmo2-7b-base,an-other-any,0shot,all,0.072,320,0.250,0.014,,
olmo2-7b-base,an-other-any,3shot_collide,all,0.506,320,0.250,0.028,,
olmo2-7b-base,an-other-any,3shot,all,0.494,320,0.250,0.028,,
olmo2-7b-base,an-me-any,0shot,all,0.656,160,0.250,0.037,,
olmo2-7b-base,an-me-any,3shot_collide,all,0.294,160,0.250,0.036,,
olmo2-7b-base,an-me-any,3shot,all,0.131,160,0.250,0.027,,
olmo2-7b-base,an-you-voc,0shot,all,0.725,80,0.250,0.050,,
olmo2-7b-base,an-you-voc,3shot_collide,all,0.362,80,0.250,0.054,,
olmo2-7b-base,an-you-voc,3shot,all,0.400,80,0.250,0.055,,
olmo2-7b-base,an-you-novoc,0shot,all,0.300,80,0.250,0.051,,
olmo2-7b-base,an-you-novoc,3shot_collide,all,0.050,80,0.250,0.024,,
olmo2-7b-base,an-you-novoc,3shot,all,0.013,80,0.250,0.012,,
olmo2-7b-base,adj-adj,0shot,all,0.869,260,0.333,0.021,,
olmo2-7b-base,adj-adj,3shot_collide,all,0.915,260,0.333,0.017,,
olmo2-7b-base,adj-adj,3shot,all,0.896,260,0.333,0.019,,
olmo2-7b-base,adj-voc,0shot,all,0.900,260,0.333,0.019,,
olmo2-7b-base,adj-voc,3shot_collide,all,0.935,260,0.333,0.015,,
olmo2-7b-base,adj-voc,3shot,all,0.923,260,0.333,0.017,,
olmo2-7b-base,adj-third,0shot,all,0.731,260,0.333,0.028,,
olmo2-7b-base,adj-third,3shot_collide,all,0.765,260,0.333,0.026,,
olmo2-7b-base,adj-third,3shot,all,0.800,260,0.333,0.025,,
olmo2-7b-base,cm-mutate,0shot,all,0.830,200,0.200,0.027,,
olmo2-7b-base,cm-mutate,3shot_collide,all,0.755,200,0.200,0.030,,
olmo2-7b-base,cm-mutate,3shot,all,0.770,200,0.200,0.030,,
olmo2-7b-base,cm-rebind,0shot,all,0.095,200,0.200,0.021,,
olmo2-7b-base,cm-rebind,3shot_collide,all,0.620,200,0.200,0.034,,
olmo2-7b-base,cm-rebind,3shot,all,0.530,200,0.200,0.035,,
olmo2-7b-base,cm-copy,0shot,all,0.030,200,0.200,0.012,,
olmo2-7b-base,cm-copy,3shot_collide,all,0.735,200,0.200,0.031,,
olmo2-7b-base,cm-copy,3shot,all,0.670,200,0.200,0.033,,
qwen2.5-7b,attr,0shot,all,0.791,1210,0.168,0.012,,
qwen2.5-7b,attr,3shot_collide,all,0.995,1210,0.168,0.002,,
qwen2.5-7b,attr,3shot,all,0.998,1210,0.168,0.001,,
qwen2.5-7b,codevar,0shot,all,0.614,1200,0.267,0.014,,
qwen2.5-7b,codevar,3shot_collide,all,0.746,1200,0.267,0.013,,
qwen2.5-7b,codevar,3shot,all,0.700,1200,0.267,0.013,,
qwen2.5-7b,dialogue,0shot,all,0.819,1200,0.312,0.011,,
qwen2.5-7b,dialogue,3shot_collide,all,0.852,1200,0.312,0.010,,
qwen2.5-7b,dialogue,3shot,all,0.836,1200,0.312,0.011,,
qwen2.5-7b,dx-owner-3-deictic,0shot,all,0.558,120,0.333,0.045,,other_participant=0.35;speaker_confusion=0.083;tie=0.008
qwen2.5-7b,dx-owner-3-deictic,0shot,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.05
qwen2.5-7b,dx-owner-3-deictic,3shot_collide,all,0.658,120,0.333,0.043,,other_participant=0.267;speaker_confusion=0.067;tie=0.008
qwen2.5-7b,dx-owner-3-deictic,3shot_collide,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.05
qwen2.5-7b,dx-owner-3-deictic,3shot,all,0.767,120,0.333,0.039,,other_participant=0.183;speaker_confusion=0.05
qwen2.5-7b,dx-owner-3-deictic,3shot,free,0.800,40,0.333,0.063,,other_participant=0.2
qwen2.5-7b,dx-owner-3-third,0shot,all,0.792,120,0.333,0.037,,
qwen2.5-7b,dx-owner-3-third,0shot,free,0.850,40,0.333,0.057,,
qwen2.5-7b,dx-owner-3-third,3shot_collide,all,0.783,120,0.333,0.038,,
qwen2.5-7b,dx-owner-3-third,3shot_collide,free,0.750,40,0.333,0.069,,
qwen2.5-7b,dx-owner-3-third,3shot,all,0.892,120,0.333,0.028,,
qwen2.5-7b,dx-owner-3-third,3shot,free,0.900,40,0.333,0.047,,
qwen2.5-7b,dx-self-3-deictic,0shot,all,0.417,180,0.287,0.037,,other_object=0.261;previous_owner=0.044;speaker_confusion=0.261;tie=0.017
qwen2.5-7b,dx-self-3-deictic,0shot,free,0.564,126,0.286,0.044,,other_object=0.246;previous_owner=0.016;speaker_confusion=0.151;tie=0.024
qwen2.5-7b,dx-self-3-deictic,3shot_collide,all,0.528,180,0.287,0.037,,other_object=0.167;previous_owner=0.061;speaker_confusion=0.228;tie=0.017
qwen2.5-7b,dx-self-3-deictic,3shot_collide,free,0.556,126,0.286,0.044,,other_object=0.167;previous_owner=0.063;speaker_confusion=0.198;tie=0.016
qwen2.5-7b,dx-self-3-deictic,3shot,all,0.483,180,0.287,0.037,,other_object=0.194;previous_owner=0.061;speaker_confusion=0.239;tie=0.022
qwen2.5-7b,dx-self-3-deictic,3shot,free,0.532,126,0.286,0.044,,other_object=0.198;previous_owner=0.048;speaker_confusion=0.198;tie=0.024
qwen2.5-7b,dx-self-3-third,0shot,all,0.506,180,0.287,0.037,,
qwen2.5-7b,dx-self-3-third,0shot,free,0.532,126,0.286,0.044,,
qwen2.5-7b,dx-self-3-third,3shot_collide,all,0.622,180,0.287,0.036,,
qwen2.5-7b,dx-self-3-third,3shot_collide,free,0.651,126,0.286,0.043,,
qwen2.5-7b,dx-self-3-third,3shot,all,0.556,180,0.287,0.037,,
qwen2.5-7b,dx-self-3-third,3shot,free,0.619,126,0.286,0.043,,
qwen2.5-7b,dx-self-2-deictic,0shot,all,0.433,180,0.287,0.037,,speaker_confusion=0.561;tie=0.006
qwen2.5-7b,dx-self-2-deictic,0shot,free,0.556,126,0.286,0.044,,speaker_confusion=0.437;tie=0.008
qwen2.5-7b,dx-self-2-deictic,3shot_collide,all,0.589,180,0.287,0.037,,speaker_confusion=0.4;tie=0.011
qwen2.5-7b,dx-self-2-deictic,3shot_collide,free,0.682,126,0.286,0.042,,speaker_confusion=0.317
qwen2.5-7b,dx-self-2-deictic,3shot,all,0.594,180,0.287,0.037,,speaker_confusion=0.394;tie=0.011
qwen2.5-7b,dx-self-2-deictic,3shot,free,0.706,126,0.286,0.041,,speaker_confusion=0.278;tie=0.016
qwen2.5-7b,dx-self-2-third,0shot,all,0.600,180,0.287,0.036,,
qwen2.5-7b,dx-self-2-third,0shot,free,0.682,126,0.286,0.042,,
qwen2.5-7b,dx-self-2-third,3shot_collide,all,0.650,180,0.287,0.036,,
qwen2.5-7b,dx-self-2-third,3shot_collide,free,0.730,126,0.286,0.040,,
qwen2.5-7b,dx-self-2-third,3shot,all,0.650,180,0.287,0.036,,
qwen2.5-7b,dx-self-2-third,3shot,free,0.722,126,0.286,0.040,,
qwen2.5-7b,dx-addressee-3-deictic,0shot,all,0.378,180,0.287,0.036,,other_object=0.306;previous_owner=0.017;speaker_confusion=0.294;tie=0.006
qwen2.5-7b,dx-addressee-3-deictic,0shot,free,0.516,126,0.286,0.044,,other_object=0.222;previous_owner=0.016;speaker_confusion=0.238;tie=0.008
qwen2.5-7b,dx-addressee-3-deictic,3shot_collide,all,0.356,180,0.287,0.036,,other_object=0.2;previous_owner=0.089;speaker_confusion=0.339;tie=0.017
qwen2.5-7b,dx-addressee-3-deictic,3shot_collide,free,0.389,126,0.286,0.043,,other_object=0.167;previous_owner=0.095;speaker_confusion=0.325;tie=0.024
qwen2.5-7b,dx-addressee-3-deictic,3shot,all,0.383,180,0.287,0.036,,other_object=0.2;previous_owner=0.1;speaker_confusion=0.311;tie=0.006
qwen2.5-7b,dx-addressee-3-deictic,3shot,free,0.436,126,0.286,0.044,,other_object=0.143;previous_owner=0.119;speaker_confusion=0.294;tie=0.008
qwen2.5-7b,dx-addressee-3-third,0shot,all,0.556,180,0.287,0.037,,
qwen2.5-7b,dx-addressee-3-third,0shot,free,0.564,126,0.286,0.044,,
qwen2.5-7b,dx-addressee-3-third,3shot_collide,all,0.600,180,0.287,0.036,,
qwen2.5-7b,dx-addressee-3-third,3shot_collide,free,0.635,126,0.286,0.043,,
qwen2.5-7b,dx-addressee-3-third,3shot,all,0.583,180,0.287,0.037,,
qwen2.5-7b,dx-addressee-3-third,3shot,free,0.651,126,0.286,0.043,,
qwen2.5-7b,dx-addressee-2-deictic,0shot,all,0.350,180,0.287,0.036,,speaker_confusion=0.639;tie=0.011
qwen2.5-7b,dx-addressee-2-deictic,0shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.54;tie=0.016
qwen2.5-7b,dx-addressee-2-deictic,3shot_collide,all,0.400,180,0.287,0.036,,speaker_confusion=0.589;tie=0.011
qwen2.5-7b,dx-addressee-2-deictic,3shot_collide,free,0.452,126,0.286,0.044,,speaker_confusion=0.532;tie=0.016
qwen2.5-7b,dx-addressee-2-deictic,3shot,all,0.428,180,0.287,0.037,,speaker_confusion=0.567;tie=0.006
qwen2.5-7b,dx-addressee-2-deictic,3shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.508;tie=0.008
qwen2.5-7b,dx-addressee-2-third,0shot,all,0.600,180,0.287,0.036,,
qwen2.5-7b,dx-addressee-2-third,0shot,free,0.714,126,0.286,0.040,,
qwen2.5-7b,dx-addressee-2-third,3shot_collide,all,0.700,180,0.287,0.034,,
qwen2.5-7b,dx-addressee-2-third,3shot_collide,free,0.786,126,0.286,0.037,,
qwen2.5-7b,dx-addressee-2-third,3shot,all,0.694,180,0.287,0.034,,
qwen2.5-7b,dx-addressee-2-third,3shot,free,0.754,126,0.286,0.038,,
qwen2.5-7b,sb-k1,0shot,all,1.000,64,0.125,0.000,,
qwen2.5-7b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
qwen2.5-7b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen2.5-7b,sb-k23,0shot,all,0.750,128,0.027,0.038,,
qwen2.5-7b,sb-k23,3shot_collide,all,0.977,128,0.027,0.013,,
qwen2.5-7b,sb-k23,3shot,all,0.992,128,0.027,0.008,,
qwen2.5-7b,sb-split,0shot,all,0.641,128,0.027,0.042,,
qwen2.5-7b,sb-split,3shot_collide,all,0.883,128,0.027,0.028,,
qwen2.5-7b,sb-split,3shot,all,0.875,128,0.027,0.029,,
qwen2.5-7b,sb-dyn,0shot,all,0.512,256,0.027,0.031,,
qwen2.5-7b,sb-dyn,3shot_collide,all,0.355,256,0.027,0.030,,
qwen2.5-7b,sb-dyn,3shot,all,0.379,256,0.027,0.030,,
qwen2.5-7b,an-narr,0shot,all,0.455,640,0.250,0.020,,
qwen2.5-7b,an-narr,3shot_collide,all,0.819,640,0.250,0.015,,
qwen2.5-7b,an-narr,3shot,all,0.820,640,0.250,0.015,,
qwen2.5-7b,an-other-any,0shot,all,0.144,320,0.250,0.020,,
qwen2.5-7b,an-other-any,3shot_collide,all,0.606,320,0.250,0.027,,
qwen2.5-7b,an-other-any,3shot,all,0.512,320,0.250,0.028,,
qwen2.5-7b,an-me-any,0shot,all,0.013,160,0.250,0.009,,
qwen2.5-7b,an-me-any,3shot_collide,all,0.256,160,0.250,0.035,,
qwen2.5-7b,an-me-any,3shot,all,0.156,160,0.250,0.029,,
qwen2.5-7b,an-you-voc,0shot,all,0.887,80,0.250,0.035,,
qwen2.5-7b,an-you-voc,3shot_collide,all,0.588,80,0.250,0.055,,
qwen2.5-7b,an-you-voc,3shot,all,0.613,80,0.250,0.054,,
qwen2.5-7b,an-you-novoc,0shot,all,0.988,80,0.250,0.012,,
qwen2.5-7b,an-you-novoc,3shot_collide,all,0.275,80,0.250,0.050,,
qwen2.5-7b,an-you-novoc,3shot,all,0.500,80,0.250,0.056,,
qwen2.5-7b,adj-adj,0shot,all,0.819,260,0.333,0.024,,
qwen2.5-7b,adj-adj,3shot_collide,all,0.927,260,0.333,0.016,,
qwen2.5-7b,adj-adj,3shot,all,0.915,260,0.333,0.017,,
qwen2.5-7b,adj-voc,0shot,all,0.896,260,0.333,0.019,,
qwen2.5-7b,adj-voc,3shot_collide,all,0.973,260,0.333,0.010,,
qwen2.5-7b,adj-voc,3shot,all,0.977,260,0.333,0.009,,
qwen2.5-7b,adj-third,0shot,all,0.985,260,0.333,0.008,,
qwen2.5-7b,adj-third,3shot_collide,all,0.873,260,0.333,0.021,,
qwen2.5-7b,adj-third,3shot,all,0.788,260,0.333,0.025,,
qwen2.5-7b,cm-mutate,0shot,all,0.895,200,0.200,0.022,,
qwen2.5-7b,cm-mutate,3shot_collide,all,0.990,200,0.200,0.007,,
qwen2.5-7b,cm-mutate,3shot,all,0.990,200,0.200,0.007,,
qwen2.5-7b,cm-rebind,0shot,all,0.505,200,0.200,0.035,,
qwen2.5-7b,cm-rebind,3shot_collide,all,0.985,200,0.200,0.009,,
qwen2.5-7b,cm-rebind,3shot,all,0.975,200,0.200,0.011,,
qwen2.5-7b,cm-copy,0shot,all,0.250,200,0.200,0.031,,
qwen2.5-7b,cm-copy,3shot_collide,all,0.975,200,0.200,0.011,,
qwen2.5-7b,cm-copy,3shot,all,0.985,200,0.200,0.009,,
llama31-8b-base,attr,0shot,all,0.662,1210,0.168,0.014,,
llama31-8b-base,attr,3shot_collide,all,0.993,1210,0.168,0.003,,
llama31-8b-base,attr,3shot,all,0.999,1210,0.168,0.001,,
llama31-8b-base,codevar,0shot,all,0.565,1200,0.267,0.014,,
llama31-8b-base,codevar,3shot_collide,all,0.686,1200,0.267,0.013,,
llama31-8b-base,codevar,3shot,all,0.706,1200,0.267,0.013,,
llama31-8b-base,dialogue,0shot,all,0.818,1200,0.312,0.011,,
llama31-8b-base,dialogue,3shot_collide,all,0.840,1200,0.312,0.011,,
llama31-8b-base,dialogue,3shot,all,0.825,1200,0.312,0.011,,
llama31-8b-base,dx-owner-3-deictic,0shot,all,0.608,120,0.333,0.045,,other_participant=0.342;speaker_confusion=0.017;tie=0.033
llama31-8b-base,dx-owner-3-deictic,0shot,free,0.575,40,0.333,0.078,,other_participant=0.425
llama31-8b-base,dx-owner-3-deictic,3shot_collide,all,0.808,120,0.333,0.036,,other_participant=0.158;speaker_confusion=0.025;tie=0.008
llama31-8b-base,dx-owner-3-deictic,3shot_collide,free,0.750,40,0.333,0.069,,other_participant=0.225;tie=0.025
llama31-8b-base,dx-owner-3-deictic,3shot,all,0.767,120,0.333,0.039,,other_participant=0.2;speaker_confusion=0.025;tie=0.008
llama31-8b-base,dx-owner-3-deictic,3shot,free,0.800,40,0.333,0.063,,other_participant=0.175;tie=0.025
llama31-8b-base,dx-owner-3-third,0shot,all,0.592,120,0.333,0.045,,
llama31-8b-base,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
llama31-8b-base,dx-owner-3-third,3shot_collide,all,0.708,120,0.333,0.042,,
llama31-8b-base,dx-owner-3-third,3shot_collide,free,0.800,40,0.333,0.063,,
llama31-8b-base,dx-owner-3-third,3shot,all,0.700,120,0.333,0.042,,
llama31-8b-base,dx-owner-3-third,3shot,free,0.825,40,0.333,0.060,,
llama31-8b-base,dx-self-3-deictic,0shot,all,0.422,180,0.287,0.037,,other_object=0.189;previous_owner=0.044;speaker_confusion=0.317;tie=0.028
llama31-8b-base,dx-self-3-deictic,0shot,free,0.468,126,0.286,0.044,,other_object=0.143;previous_owner=0.008;speaker_confusion=0.341;tie=0.04
llama31-8b-base,dx-self-3-deictic,3shot_collide,all,0.494,180,0.287,0.037,,other_object=0.183;previous_owner=0.033;speaker_confusion=0.278;tie=0.011
llama31-8b-base,dx-self-3-deictic,3shot_collide,free,0.564,126,0.286,0.044,,other_object=0.198;previous_owner=0.04;speaker_confusion=0.183;tie=0.016
llama31-8b-base,dx-self-3-deictic,3shot,all,0.517,180,0.287,0.037,,other_object=0.156;previous_owner=0.033;speaker_confusion=0.278;tie=0.017
llama31-8b-base,dx-self-3-deictic,3shot,free,0.595,126,0.286,0.044,,other_object=0.143;previous_owner=0.024;speaker_confusion=0.214;tie=0.024
llama31-8b-base,dx-self-3-third,0shot,all,0.644,180,0.287,0.036,,
llama31-8b-base,dx-self-3-third,0shot,free,0.706,126,0.286,0.041,,
llama31-8b-base,dx-self-3-third,3shot_collide,all,0.633,180,0.287,0.036,,
llama31-8b-base,dx-self-3-third,3shot_collide,free,0.746,126,0.286,0.039,,
llama31-8b-base,dx-self-3-third,3shot,all,0.644,180,0.287,0.036,,
llama31-8b-base,dx-self-3-third,3shot,free,0.746,126,0.286,0.039,,
llama31-8b-base,dx-self-2-deictic,0shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.661;tie=0.017
llama31-8b-base,dx-self-2-deictic,0shot,free,0.381,126,0.286,0.043,,speaker_confusion=0.595;tie=0.024
llama31-8b-base,dx-self-2-deictic,3shot_collide,all,0.478,180,0.287,0.037,,speaker_confusion=0.522
llama31-8b-base,dx-self-2-deictic,3shot_collide,free,0.564,126,0.286,0.044,,speaker_confusion=0.437
llama31-8b-base,dx-self-2-deictic,3shot,all,0.489,180,0.287,0.037,,speaker_confusion=0.489;tie=0.022
llama31-8b-base,dx-self-2-deictic,3shot,free,0.579,126,0.286,0.044,,speaker_confusion=0.389;tie=0.032
llama31-8b-base,dx-self-2-third,0shot,all,0.706,180,0.287,0.034,,
llama31-8b-base,dx-self-2-third,0shot,free,0.794,126,0.286,0.036,,
llama31-8b-base,dx-self-2-third,3shot_collide,all,0.700,180,0.287,0.034,,
llama31-8b-base,dx-self-2-third,3shot_collide,free,0.794,126,0.286,0.036,,
llama31-8b-base,dx-self-2-third,3shot,all,0.644,180,0.287,0.036,,
llama31-8b-base,dx-self-2-third,3shot,free,0.754,126,0.286,0.038,,
llama31-8b-base,dx-addressee-3-deictic,0shot,all,0.367,180,0.287,0.036,,other_object=0.122;previous_owner=0.106;speaker_confusion=0.367;tie=0.039
llama31-8b-base,dx-addressee-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.087;previous_owner=0.143;speaker_confusion=0.349;tie=0.032
llama31-8b-base,dx-addressee-3-deictic,3shot_collide,all,0.350,180,0.287,0.036,,other_object=0.272;previous_owner=0.072;speaker_confusion=0.289;tie=0.017
llama31-8b-base,dx-addressee-3-deictic,3shot_collide,free,0.413,126,0.286,0.044,,other_object=0.198;previous_owner=0.071;speaker_confusion=0.302;tie=0.016
llama31-8b-base,dx-addressee-3-deictic,3shot,all,0.411,180,0.287,0.037,,other_object=0.222;previous_owner=0.061;speaker_confusion=0.306
llama31-8b-base,dx-addressee-3-deictic,3shot,free,0.476,126,0.286,0.044,,other_object=0.167;previous_owner=0.071;speaker_confusion=0.286
llama31-8b-base,dx-addressee-3-third,0shot,all,0.650,180,0.287,0.036,,
llama31-8b-base,dx-addressee-3-third,0shot,free,0.691,126,0.286,0.041,,
llama31-8b-base,dx-addressee-3-third,3shot_collide,all,0.594,180,0.287,0.037,,
llama31-8b-base,dx-addressee-3-third,3shot_collide,free,0.675,126,0.286,0.042,,
llama31-8b-base,dx-addressee-3-third,3shot,all,0.633,180,0.287,0.036,,
llama31-8b-base,dx-addressee-3-third,3shot,free,0.706,126,0.286,0.041,,
llama31-8b-base,dx-addressee-2-deictic,0shot,all,0.200,180,0.287,0.030,,speaker_confusion=0.761;tie=0.039
llama31-8b-base,dx-addressee-2-deictic,0shot,free,0.159,126,0.286,0.033,,speaker_confusion=0.817;tie=0.024
llama31-8b-base,dx-addressee-2-deictic,3shot_collide,all,0.378,180,0.287,0.036,,speaker_confusion=0.606;tie=0.017
llama31-8b-base,dx-addressee-2-deictic,3shot_collide,free,0.444,126,0.286,0.044,,speaker_confusion=0.54;tie=0.016
llama31-8b-base,dx-addressee-2-deictic,3shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.6;tie=0.028
llama31-8b-base,dx-addressee-2-deictic,3shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.548;tie=0.032
llama31-8b-base,dx-addressee-2-third,0shot,all,0.683,180,0.287,0.035,,
llama31-8b-base,dx-addressee-2-third,0shot,free,0.754,126,0.286,0.038,,
llama31-8b-base,dx-addressee-2-third,3shot_collide,all,0.694,180,0.287,0.034,,
llama31-8b-base,dx-addressee-2-third,3shot_collide,free,0.802,126,0.286,0.035,,
llama31-8b-base,dx-addressee-2-third,3shot,all,0.694,180,0.287,0.034,,
llama31-8b-base,dx-addressee-2-third,3shot,free,0.818,126,0.286,0.034,,
llama31-8b-base,sb-k1,0shot,all,0.750,64,0.125,0.054,,
llama31-8b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
llama31-8b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
llama31-8b-base,sb-k23,0shot,all,0.547,128,0.027,0.044,,
llama31-8b-base,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
llama31-8b-base,sb-k23,3shot,all,0.977,128,0.027,0.013,,
llama31-8b-base,sb-split,0shot,all,0.438,128,0.027,0.044,,
llama31-8b-base,sb-split,3shot_collide,all,0.906,128,0.027,0.026,,
llama31-8b-base,sb-split,3shot,all,0.898,128,0.027,0.027,,
llama31-8b-base,sb-dyn,0shot,all,0.438,256,0.027,0.031,,
llama31-8b-base,sb-dyn,3shot_collide,all,0.434,256,0.027,0.031,,
llama31-8b-base,sb-dyn,3shot,all,0.398,256,0.027,0.031,,
llama31-8b-base,an-narr,0shot,all,0.570,640,0.250,0.020,,
llama31-8b-base,an-narr,3shot_collide,all,0.847,640,0.250,0.014,,
llama31-8b-base,an-narr,3shot,all,0.831,640,0.250,0.015,,
llama31-8b-base,an-other-any,0shot,all,0.575,320,0.250,0.028,,
llama31-8b-base,an-other-any,3shot_collide,all,0.575,320,0.250,0.028,,
llama31-8b-base,an-other-any,3shot,all,0.469,320,0.250,0.028,,
llama31-8b-base,an-me-any,0shot,all,0.006,160,0.250,0.006,,
llama31-8b-base,an-me-any,3shot_collide,all,0.356,160,0.250,0.038,,
llama31-8b-base,an-me-any,3shot,all,0.319,160,0.250,0.037,,
llama31-8b-base,an-you-voc,0shot,all,0.200,80,0.250,0.045,,
llama31-8b-base,an-you-voc,3shot_collide,all,0.463,80,0.250,0.056,,
llama31-8b-base,an-you-voc,3shot,all,0.700,80,0.250,0.051,,
llama31-8b-base,an-you-novoc,0shot,all,0.787,80,0.250,0.046,,
llama31-8b-base,an-you-novoc,3shot_collide,all,0.287,80,0.250,0.051,,
llama31-8b-base,an-you-novoc,3shot,all,0.487,80,0.250,0.056,,
llama31-8b-base,adj-adj,0shot,all,0.865,260,0.333,0.021,,
llama31-8b-base,adj-adj,3shot_collide,all,0.958,260,0.333,0.013,,
llama31-8b-base,adj-adj,3shot,all,0.935,260,0.333,0.015,,
llama31-8b-base,adj-voc,0shot,all,0.892,260,0.333,0.019,,
llama31-8b-base,adj-voc,3shot_collide,all,0.992,260,0.333,0.005,,
llama31-8b-base,adj-voc,3shot,all,0.981,260,0.333,0.009,,
llama31-8b-base,adj-third,0shot,all,0.846,260,0.333,0.022,,
llama31-8b-base,adj-third,3shot_collide,all,0.869,260,0.333,0.021,,
llama31-8b-base,adj-third,3shot,all,0.765,260,0.333,0.026,,
llama31-8b-base,cm-mutate,0shot,all,0.855,200,0.200,0.025,,
llama31-8b-base,cm-mutate,3shot_collide,all,0.965,200,0.200,0.013,,
llama31-8b-base,cm-mutate,3shot,all,0.935,200,0.200,0.017,,
llama31-8b-base,cm-rebind,0shot,all,0.275,200,0.200,0.032,,
llama31-8b-base,cm-rebind,3shot_collide,all,0.935,200,0.200,0.017,,
llama31-8b-base,cm-rebind,3shot,all,0.905,200,0.200,0.021,,
llama31-8b-base,cm-copy,0shot,all,0.120,200,0.200,0.023,,
llama31-8b-base,cm-copy,3shot_collide,all,0.920,200,0.200,0.019,,
llama31-8b-base,cm-copy,3shot,all,0.890,200,0.200,0.022,,
gemma2-9b,attr,0shot,all,0.762,1210,0.168,0.012,,
gemma2-9b,attr,3shot_collide,all,0.983,1210,0.168,0.004,,
gemma2-9b,attr,3shot,all,0.997,1210,0.168,0.002,,
gemma2-9b,codevar,0shot,all,0.614,1200,0.267,0.014,,
gemma2-9b,codevar,3shot_collide,all,0.739,1200,0.267,0.013,,
gemma2-9b,codevar,3shot,all,0.759,1200,0.267,0.012,,
gemma2-9b,dialogue,0shot,all,0.839,1200,0.312,0.011,,
gemma2-9b,dialogue,3shot_collide,all,0.903,1200,0.312,0.009,,
gemma2-9b,dialogue,3shot,all,0.895,1200,0.312,0.009,,
gemma2-9b,dx-owner-3-deictic,0shot,all,0.642,120,0.333,0.044,,other_participant=0.3;speaker_confusion=0.025;tie=0.033
gemma2-9b,dx-owner-3-deictic,0shot,free,0.600,40,0.333,0.077,,other_participant=0.35;tie=0.05
gemma2-9b,dx-owner-3-deictic,3shot_collide,all,0.942,120,0.333,0.021,,other_participant=0.042;speaker_confusion=0.008;tie=0.008
gemma2-9b,dx-owner-3-deictic,3shot_collide,free,0.975,40,0.333,0.025,,tie=0.025
gemma2-9b,dx-owner-3-deictic,3shot,all,0.867,120,0.333,0.031,,other_participant=0.117;tie=0.017
gemma2-9b,dx-owner-3-deictic,3shot,free,0.925,40,0.333,0.042,,other_participant=0.05;tie=0.025
gemma2-9b,dx-owner-3-third,0shot,all,0.733,120,0.333,0.040,,
gemma2-9b,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
gemma2-9b,dx-owner-3-third,3shot_collide,all,0.917,120,0.333,0.025,,
gemma2-9b,dx-owner-3-third,3shot_collide,free,0.975,40,0.333,0.025,,
gemma2-9b,dx-owner-3-third,3shot,all,0.900,120,0.333,0.027,,
gemma2-9b,dx-owner-3-third,3shot,free,0.975,40,0.333,0.025,,
gemma2-9b,dx-self-3-deictic,0shot,all,0.367,180,0.287,0.036,,other_object=0.233;previous_owner=0.072;speaker_confusion=0.289;tie=0.039
gemma2-9b,dx-self-3-deictic,0shot,free,0.413,126,0.286,0.044,,other_object=0.238;previous_owner=0.032;speaker_confusion=0.278;tie=0.04
gemma2-9b,dx-self-3-deictic,3shot_collide,all,0.483,180,0.287,0.037,,other_object=0.117;previous_owner=0.056;speaker_confusion=0.317;tie=0.028
gemma2-9b,dx-self-3-deictic,3shot_collide,free,0.548,126,0.286,0.044,,other_object=0.119;previous_owner=0.048;speaker_confusion=0.246;tie=0.04
gemma2-9b,dx-self-3-deictic,3shot,all,0.522,180,0.287,0.037,,other_object=0.089;previous_owner=0.061;speaker_confusion=0.306;tie=0.022
gemma2-9b,dx-self-3-deictic,3shot,free,0.603,126,0.286,0.044,,other_object=0.095;previous_owner=0.048;speaker_confusion=0.238;tie=0.016
gemma2-9b,dx-self-3-third,0shot,all,0.650,180,0.287,0.036,,
gemma2-9b,dx-self-3-third,0shot,free,0.706,126,0.286,0.041,,
gemma2-9b,dx-self-3-third,3shot_collide,all,0.694,180,0.287,0.034,,
gemma2-9b,dx-self-3-third,3shot_collide,free,0.682,126,0.286,0.042,,
gemma2-9b,dx-self-3-third,3shot,all,0.700,180,0.287,0.034,,
gemma2-9b,dx-self-3-third,3shot,free,0.706,126,0.286,0.041,,
gemma2-9b,dx-self-2-deictic,0shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.622;tie=0.022
gemma2-9b,dx-self-2-deictic,0shot,free,0.413,126,0.286,0.044,,speaker_confusion=0.571;tie=0.016
gemma2-9b,dx-self-2-deictic,3shot_collide,all,0.517,180,0.287,0.037,,speaker_confusion=0.422;tie=0.061
gemma2-9b,dx-self-2-deictic,3shot_collide,free,0.579,126,0.286,0.044,,speaker_confusion=0.349;tie=0.071
gemma2-9b,dx-self-2-deictic,3shot,all,0.511,180,0.287,0.037,,speaker_confusion=0.439;tie=0.05
gemma2-9b,dx-self-2-deictic,3shot,free,0.571,126,0.286,0.044,,speaker_confusion=0.365;tie=0.063
gemma2-9b,dx-self-2-third,0shot,all,0.656,180,0.287,0.035,,
gemma2-9b,dx-self-2-third,0shot,free,0.714,126,0.286,0.040,,
gemma2-9b,dx-self-2-third,3shot_collide,all,0.711,180,0.287,0.034,,
gemma2-9b,dx-self-2-third,3shot_collide,free,0.754,126,0.286,0.038,,
gemma2-9b,dx-self-2-third,3shot,all,0.700,180,0.287,0.034,,
gemma2-9b,dx-self-2-third,3shot,free,0.746,126,0.286,0.039,,
gemma2-9b,dx-addressee-3-deictic,0shot,all,0.272,180,0.287,0.033,,other_object=0.2;previous_owner=0.106;speaker_confusion=0.389;tie=0.033
gemma2-9b,dx-addressee-3-deictic,0shot,free,0.278,126,0.286,0.040,,other_object=0.175;previous_owner=0.127;speaker_confusion=0.381;tie=0.04
gemma2-9b,dx-addressee-3-deictic,3shot_collide,all,0.300,180,0.287,0.034,,other_object=0.15;previous_owner=0.133;speaker_confusion=0.372;tie=0.044
gemma2-9b,dx-addressee-3-deictic,3shot_collide,free,0.286,126,0.286,0.040,,other_object=0.087;previous_owner=0.167;speaker_confusion=0.413;tie=0.048
gemma2-9b,dx-addressee-3-deictic,3shot,all,0.344,180,0.287,0.035,,other_object=0.122;previous_owner=0.156;speaker_confusion=0.339;tie=0.039
gemma2-9b,dx-addressee-3-deictic,3shot,free,0.325,126,0.286,0.042,,other_object=0.079;previous_owner=0.175;speaker_confusion=0.365;tie=0.056
gemma2-9b,dx-addressee-3-third,0shot,all,0.644,180,0.287,0.036,,
gemma2-9b,dx-addressee-3-third,0shot,free,0.643,126,0.286,0.043,,
gemma2-9b,dx-addressee-3-third,3shot_collide,all,0.717,180,0.287,0.034,,
gemma2-9b,dx-addressee-3-third,3shot_collide,free,0.714,126,0.286,0.040,,
gemma2-9b,dx-addressee-3-third,3shot,all,0.694,180,0.287,0.034,,
gemma2-9b,dx-addressee-3-third,3shot,free,0.682,126,0.286,0.042,,
gemma2-9b,dx-addressee-2-deictic,0shot,all,0.200,180,0.287,0.030,,speaker_confusion=0.778;tie=0.022
gemma2-9b,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.778;tie=0.008
gemma2-9b,dx-addressee-2-deictic,3shot_collide,all,0.306,180,0.287,0.034,,speaker_confusion=0.644;tie=0.05
gemma2-9b,dx-addressee-2-deictic,3shot_collide,free,0.302,126,0.286,0.041,,speaker_confusion=0.651;tie=0.048
gemma2-9b,dx-addressee-2-deictic,3shot,all,0.333,180,0.287,0.035,,speaker_confusion=0.622;tie=0.044
gemma2-9b,dx-addressee-2-deictic,3shot,free,0.349,126,0.286,0.043,,speaker_confusion=0.603;tie=0.048
gemma2-9b,dx-addressee-2-third,0shot,all,0.667,180,0.287,0.035,,
gemma2-9b,dx-addressee-2-third,0shot,free,0.746,126,0.286,0.039,,
gemma2-9b,dx-addressee-2-third,3shot_collide,all,0.722,180,0.287,0.033,,
gemma2-9b,dx-addressee-2-third,3shot_collide,free,0.770,126,0.286,0.037,,
gemma2-9b,dx-addressee-2-third,3shot,all,0.733,180,0.287,0.033,,
gemma2-9b,dx-addressee-2-third,3shot,free,0.778,126,0.286,0.037,,
gemma2-9b,sb-k1,0shot,all,0.953,64,0.125,0.026,,
gemma2-9b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
gemma2-9b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
gemma2-9b,sb-k23,0shot,all,0.578,128,0.027,0.044,,
gemma2-9b,sb-k23,3shot_collide,all,0.992,128,0.027,0.008,,
gemma2-9b,sb-k23,3shot,all,0.977,128,0.027,0.013,,
gemma2-9b,sb-split,0shot,all,0.523,128,0.027,0.044,,
gemma2-9b,sb-split,3shot_collide,all,0.859,128,0.027,0.031,,
gemma2-9b,sb-split,3shot,all,0.805,128,0.027,0.035,,
gemma2-9b,sb-dyn,0shot,all,0.422,256,0.027,0.031,,
gemma2-9b,sb-dyn,3shot_collide,all,0.387,256,0.027,0.030,,
gemma2-9b,sb-dyn,3shot,all,0.367,256,0.027,0.030,,
gemma2-9b,an-narr,0shot,all,0.622,640,0.250,0.019,,
gemma2-9b,an-narr,3shot_collide,all,0.953,640,0.250,0.008,,
gemma2-9b,an-narr,3shot,all,0.928,640,0.250,0.010,,
gemma2-9b,an-other-any,0shot,all,0.219,320,0.250,0.023,,
gemma2-9b,an-other-any,3shot_collide,all,0.691,320,0.250,0.026,,
gemma2-9b,an-other-any,3shot,all,0.613,320,0.250,0.027,,
gemma2-9b,an-me-any,0shot,all,0.356,160,0.250,0.038,,
gemma2-9b,an-me-any,3shot_collide,all,0.331,160,0.250,0.037,,
gemma2-9b,an-me-any,3shot,all,0.300,160,0.250,0.036,,
gemma2-9b,an-you-voc,0shot,all,0.325,80,0.250,0.052,,
gemma2-9b,an-you-voc,3shot_collide,all,0.400,80,0.250,0.055,,
gemma2-9b,an-you-voc,3shot,all,0.450,80,0.250,0.056,,
gemma2-9b,an-you-novoc,0shot,all,0.400,80,0.250,0.055,,
gemma2-9b,an-you-novoc,3shot_collide,all,0.125,80,0.250,0.037,,
gemma2-9b,an-you-novoc,3shot,all,0.113,80,0.250,0.035,,
gemma2-9b,adj-adj,0shot,all,0.946,260,0.333,0.014,,
gemma2-9b,adj-adj,3shot_collide,all,0.985,260,0.333,0.008,,
gemma2-9b,adj-adj,3shot,all,0.996,260,0.333,0.004,,
gemma2-9b,adj-voc,0shot,all,0.973,260,0.333,0.010,,
gemma2-9b,adj-voc,3shot_collide,all,0.996,260,0.333,0.004,,
gemma2-9b,adj-voc,3shot,all,1.000,260,0.333,0.000,,
gemma2-9b,adj-third,0shot,all,0.985,260,0.333,0.008,,
gemma2-9b,adj-third,3shot_collide,all,0.969,260,0.333,0.011,,
gemma2-9b,adj-third,3shot,all,0.965,260,0.333,0.011,,
gemma2-9b,cm-mutate,0shot,all,0.860,200,0.200,0.025,,
gemma2-9b,cm-mutate,3shot_collide,all,0.960,200,0.200,0.014,,
gemma2-9b,cm-mutate,3shot,all,0.995,200,0.200,0.005,,
gemma2-9b,cm-rebind,0shot,all,0.670,200,0.200,0.033,,
gemma2-9b,cm-rebind,3shot_collide,all,0.990,200,0.200,0.007,,
gemma2-9b,cm-rebind,3shot,all,0.990,200,0.200,0.007,,
gemma2-9b,cm-copy,0shot,all,0.270,200,0.200,0.031,,
gemma2-9b,cm-copy,3shot_collide,all,0.980,200,0.200,0.010,,
gemma2-9b,cm-copy,3shot,all,0.985,200,0.200,0.009,,
olmo2-13b,dx-owner-3-deictic,0shot,all,0.683,120,0.333,0.043,,other_participant=0.308;speaker_confusion=0.008
olmo2-13b,dx-owner-3-deictic,0shot,free,0.650,40,0.333,0.075,,other_participant=0.35
olmo2-13b,dx-owner-3-deictic,3shot_collide,all,0.792,120,0.333,0.037,,other_participant=0.2;speaker_confusion=0.008
olmo2-13b,dx-owner-3-deictic,3shot_collide,free,0.600,40,0.333,0.077,,other_participant=0.4
olmo2-13b,dx-owner-3-deictic,3shot,all,0.733,120,0.333,0.040,,other_participant=0.242;speaker_confusion=0.025
olmo2-13b,dx-owner-3-deictic,3shot,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.05
olmo2-13b,dx-owner-3-third,0shot,all,0.667,120,0.333,0.043,,
olmo2-13b,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
olmo2-13b,dx-owner-3-third,3shot_collide,all,0.825,120,0.333,0.035,,
olmo2-13b,dx-owner-3-third,3shot_collide,free,0.825,40,0.333,0.060,,
olmo2-13b,dx-owner-3-third,3shot,all,0.900,120,0.333,0.027,,
olmo2-13b,dx-owner-3-third,3shot,free,0.925,40,0.333,0.042,,
olmo2-13b,dx-self-3-deictic,0shot,all,0.433,180,0.287,0.037,,other_object=0.3;previous_owner=0.056;speaker_confusion=0.211
olmo2-13b,dx-self-3-deictic,0shot,free,0.548,126,0.286,0.044,,other_object=0.27;previous_owner=0.016;speaker_confusion=0.167
olmo2-13b,dx-self-3-deictic,3shot_collide,all,0.639,180,0.287,0.036,,other_object=0.178;previous_owner=0.022;speaker_confusion=0.161
olmo2-13b,dx-self-3-deictic,3shot_collide,free,0.651,126,0.286,0.043,,other_object=0.206;previous_owner=0.016;speaker_confusion=0.127
olmo2-13b,dx-self-3-deictic,3shot,all,0.706,180,0.287,0.034,,other_object=0.128;previous_owner=0.022;speaker_confusion=0.144
olmo2-13b,dx-self-3-deictic,3shot,free,0.746,126,0.286,0.039,,other_object=0.119;previous_owner=0.016;speaker_confusion=0.119
olmo2-13b,dx-self-3-third,0shot,all,0.661,180,0.287,0.035,,
olmo2-13b,dx-self-3-third,0shot,free,0.675,126,0.286,0.042,,
olmo2-13b,dx-self-3-third,3shot_collide,all,0.767,180,0.287,0.032,,
olmo2-13b,dx-self-3-third,3shot_collide,free,0.786,126,0.286,0.037,,
olmo2-13b,dx-self-3-third,3shot,all,0.828,180,0.287,0.028,,
olmo2-13b,dx-self-3-third,3shot,free,0.841,126,0.286,0.033,,
olmo2-13b,dx-self-2-deictic,0shot,all,0.411,180,0.287,0.037,,speaker_confusion=0.589
olmo2-13b,dx-self-2-deictic,0shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.516
olmo2-13b,dx-self-2-deictic,3shot_collide,all,0.533,180,0.287,0.037,,speaker_confusion=0.467
olmo2-13b,dx-self-2-deictic,3shot_collide,free,0.587,126,0.286,0.044,,speaker_confusion=0.413
olmo2-13b,dx-self-2-deictic,3shot,all,0.594,180,0.287,0.037,,speaker_confusion=0.406
olmo2-13b,dx-self-2-deictic,3shot,free,0.651,126,0.286,0.043,,speaker_confusion=0.349
olmo2-13b,dx-self-2-third,0shot,all,0.567,180,0.287,0.037,,
olmo2-13b,dx-self-2-third,0shot,free,0.595,126,0.286,0.044,,
olmo2-13b,dx-self-2-third,3shot_collide,all,0.761,180,0.287,0.032,,
olmo2-13b,dx-self-2-third,3shot_collide,free,0.794,126,0.286,0.036,,
olmo2-13b,dx-self-2-third,3shot,all,0.800,180,0.287,0.030,,
olmo2-13b,dx-self-2-third,3shot,free,0.809,126,0.286,0.035,,
olmo2-13b,dx-addressee-3-deictic,0shot,all,0.433,180,0.287,0.037,,other_object=0.172;previous_owner=0.083;speaker_confusion=0.311
olmo2-13b,dx-addressee-3-deictic,0shot,free,0.484,126,0.286,0.044,,other_object=0.127;previous_owner=0.103;speaker_confusion=0.286
olmo2-13b,dx-addressee-3-deictic,3shot_collide,all,0.478,180,0.287,0.037,,other_object=0.139;previous_owner=0.061;speaker_confusion=0.322
olmo2-13b,dx-addressee-3-deictic,3shot_collide,free,0.484,126,0.286,0.044,,other_object=0.119;previous_owner=0.079;speaker_confusion=0.317
olmo2-13b,dx-addressee-3-deictic,3shot,all,0.533,180,0.287,0.037,,other_object=0.122;previous_owner=0.056;speaker_confusion=0.289
olmo2-13b,dx-addressee-3-deictic,3shot,free,0.508,126,0.286,0.044,,other_object=0.127;previous_owner=0.071;speaker_confusion=0.294
olmo2-13b,dx-addressee-3-third,0shot,all,0.689,180,0.287,0.035,,
olmo2-13b,dx-addressee-3-third,0shot,free,0.635,126,0.286,0.043,,
olmo2-13b,dx-addressee-3-third,3shot_collide,all,0.739,180,0.287,0.033,,
olmo2-13b,dx-addressee-3-third,3shot_collide,free,0.730,126,0.286,0.040,,
olmo2-13b,dx-addressee-3-third,3shot,all,0.789,180,0.287,0.030,,
olmo2-13b,dx-addressee-3-third,3shot,free,0.762,126,0.286,0.038,,
olmo2-13b,dx-addressee-2-deictic,0shot,all,0.294,180,0.287,0.034,,speaker_confusion=0.706
olmo2-13b,dx-addressee-2-deictic,0shot,free,0.381,126,0.286,0.043,,speaker_confusion=0.619
olmo2-13b,dx-addressee-2-deictic,3shot_collide,all,0.478,180,0.287,0.037,,speaker_confusion=0.522
olmo2-13b,dx-addressee-2-deictic,3shot_collide,free,0.556,126,0.286,0.044,,speaker_confusion=0.444
olmo2-13b,dx-addressee-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.517
olmo2-13b,dx-addressee-2-deictic,3shot,free,0.564,126,0.286,0.044,,speaker_confusion=0.437
olmo2-13b,dx-addressee-2-third,0shot,all,0.572,180,0.287,0.037,,
olmo2-13b,dx-addressee-2-third,0shot,free,0.579,126,0.286,0.044,,
olmo2-13b,dx-addressee-2-third,3shot_collide,all,0.772,180,0.287,0.031,,
olmo2-13b,dx-addressee-2-third,3shot_collide,free,0.825,126,0.286,0.034,,
olmo2-13b,dx-addressee-2-third,3shot,all,0.800,180,0.287,0.030,,
olmo2-13b,dx-addressee-2-third,3shot,free,0.825,126,0.286,0.034,,
qwen3-30b-a3b-base,attr,0shot,all,0.841,1210,0.168,0.011,,
qwen3-30b-a3b-base,attr,3shot_collide,all,0.998,1210,0.168,0.001,,
qwen3-30b-a3b-base,attr,3shot,all,1.000,1210,0.168,0.000,,
qwen3-30b-a3b-base,codevar,0shot,all,0.747,1200,0.267,0.013,,
qwen3-30b-a3b-base,codevar,3shot_collide,all,0.834,1200,0.267,0.011,,
qwen3-30b-a3b-base,codevar,3shot,all,0.829,1200,0.267,0.011,,
qwen3-30b-a3b-base,dialogue,0shot,all,0.812,1200,0.312,0.011,,
qwen3-30b-a3b-base,dialogue,3shot_collide,all,0.880,1200,0.312,0.009,,
qwen3-30b-a3b-base,dialogue,3shot,all,0.886,1200,0.312,0.009,,
qwen3-30b-a3b-base,dx-owner-3-deictic,0shot,all,0.617,120,0.333,0.044,,other_participant=0.317;speaker_confusion=0.025;tie=0.042
qwen3-30b-a3b-base,dx-owner-3-deictic,0shot,free,0.600,40,0.333,0.077,,other_participant=0.325;tie=0.075
qwen3-30b-a3b-base,dx-owner-3-deictic,3shot_collide,all,0.725,120,0.333,0.041,,other_participant=0.217;speaker_confusion=0.033;tie=0.025
qwen3-30b-a3b-base,dx-owner-3-deictic,3shot_collide,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.025;tie=0.025
qwen3-30b-a3b-base,dx-owner-3-deictic,3shot,all,0.800,120,0.333,0.036,,other_participant=0.15;speaker_confusion=0.042;tie=0.008
qwen3-30b-a3b-base,dx-owner-3-deictic,3shot,free,0.675,40,0.333,0.074,,other_participant=0.225;speaker_confusion=0.075;tie=0.025
qwen3-30b-a3b-base,dx-owner-3-third,0shot,all,0.700,120,0.333,0.042,,
qwen3-30b-a3b-base,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
qwen3-30b-a3b-base,dx-owner-3-third,3shot_collide,all,0.992,120,0.333,0.008,,
qwen3-30b-a3b-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
qwen3-30b-a3b-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
qwen3-30b-a3b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
qwen3-30b-a3b-base,dx-self-3-deictic,0shot,all,0.361,180,0.287,0.036,,other_object=0.25;previous_owner=0.061;speaker_confusion=0.294;tie=0.033
qwen3-30b-a3b-base,dx-self-3-deictic,0shot,free,0.421,126,0.286,0.044,,other_object=0.262;previous_owner=0.032;speaker_confusion=0.262;tie=0.024
qwen3-30b-a3b-base,dx-self-3-deictic,3shot_collide,all,0.506,180,0.287,0.037,,other_object=0.1;previous_owner=0.072;speaker_confusion=0.311;tie=0.011
qwen3-30b-a3b-base,dx-self-3-deictic,3shot_collide,free,0.619,126,0.286,0.043,,other_object=0.079;previous_owner=0.04;speaker_confusion=0.246;tie=0.016
qwen3-30b-a3b-base,dx-self-3-deictic,3shot,all,0.517,180,0.287,0.037,,other_object=0.133;previous_owner=0.05;speaker_confusion=0.294;tie=0.006
qwen3-30b-a3b-base,dx-self-3-deictic,3shot,free,0.651,126,0.286,0.043,,other_object=0.119;previous_owner=0.032;speaker_confusion=0.19;tie=0.008
qwen3-30b-a3b-base,dx-self-3-third,0shot,all,0.678,180,0.287,0.035,,
qwen3-30b-a3b-base,dx-self-3-third,0shot,free,0.706,126,0.286,0.041,,
qwen3-30b-a3b-base,dx-self-3-third,3shot_collide,all,0.806,180,0.287,0.029,,
qwen3-30b-a3b-base,dx-self-3-third,3shot_collide,free,0.818,126,0.286,0.034,,
qwen3-30b-a3b-base,dx-self-3-third,3shot,all,0.806,180,0.287,0.029,,
qwen3-30b-a3b-base,dx-self-3-third,3shot,free,0.825,126,0.286,0.034,,
qwen3-30b-a3b-base,dx-self-2-deictic,0shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.611;tie=0.033
qwen3-30b-a3b-base,dx-self-2-deictic,0shot,free,0.413,126,0.286,0.044,,speaker_confusion=0.548;tie=0.04
qwen3-30b-a3b-base,dx-self-2-deictic,3shot_collide,all,0.489,180,0.287,0.037,,speaker_confusion=0.506;tie=0.006
qwen3-30b-a3b-base,dx-self-2-deictic,3shot_collide,free,0.611,126,0.286,0.043,,speaker_confusion=0.381;tie=0.008
qwen3-30b-a3b-base,dx-self-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.511;tie=0.006
qwen3-30b-a3b-base,dx-self-2-deictic,3shot,free,0.611,126,0.286,0.043,,speaker_confusion=0.381;tie=0.008
qwen3-30b-a3b-base,dx-self-2-third,0shot,all,0.722,180,0.287,0.033,,
qwen3-30b-a3b-base,dx-self-2-third,0shot,free,0.778,126,0.286,0.037,,
qwen3-30b-a3b-base,dx-self-2-third,3shot_collide,all,0.761,180,0.287,0.032,,
qwen3-30b-a3b-base,dx-self-2-third,3shot_collide,free,0.809,126,0.286,0.035,,
qwen3-30b-a3b-base,dx-self-2-third,3shot,all,0.739,180,0.287,0.033,,
qwen3-30b-a3b-base,dx-self-2-third,3shot,free,0.786,126,0.286,0.037,,
qwen3-30b-a3b-base,dx-addressee-3-deictic,0shot,all,0.317,180,0.287,0.035,,other_object=0.178;previous_owner=0.089;speaker_confusion=0.406;tie=0.011
qwen3-30b-a3b-base,dx-addressee-3-deictic,0shot,free,0.325,126,0.286,0.042,,other_object=0.151;previous_owner=0.119;speaker_confusion=0.397;tie=0.008
qwen3-30b-a3b-base,dx-addressee-3-deictic,3shot_collide,all,0.422,180,0.287,0.037,,other_object=0.106;previous_owner=0.106;speaker_confusion=0.35;tie=0.017
qwen3-30b-a3b-base,dx-addressee-3-deictic,3shot_collide,free,0.460,126,0.286,0.044,,other_object=0.071;previous_owner=0.127;speaker_confusion=0.325;tie=0.016
qwen3-30b-a3b-base,dx-addressee-3-deictic,3shot,all,0.439,180,0.287,0.037,,other_object=0.094;previous_owner=0.094;speaker_confusion=0.372
qwen3-30b-a3b-base,dx-addressee-3-deictic,3shot,free,0.468,126,0.286,0.044,,other_object=0.071;previous_owner=0.119;speaker_confusion=0.341
qwen3-30b-a3b-base,dx-addressee-3-third,0shot,all,0.706,180,0.287,0.034,,
qwen3-30b-a3b-base,dx-addressee-3-third,0shot,free,0.698,126,0.286,0.041,,
qwen3-30b-a3b-base,dx-addressee-3-third,3shot_collide,all,0.811,180,0.287,0.029,,
qwen3-30b-a3b-base,dx-addressee-3-third,3shot_collide,free,0.881,126,0.286,0.029,,
qwen3-30b-a3b-base,dx-addressee-3-third,3shot,all,0.817,180,0.287,0.029,,
qwen3-30b-a3b-base,dx-addressee-3-third,3shot,free,0.897,126,0.286,0.027,,
qwen3-30b-a3b-base,dx-addressee-2-deictic,0shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.7;tie=0.017
qwen3-30b-a3b-base,dx-addressee-2-deictic,0shot,free,0.286,126,0.286,0.040,,speaker_confusion=0.69;tie=0.024
qwen3-30b-a3b-base,dx-addressee-2-deictic,3shot_collide,all,0.389,180,0.287,0.036,,speaker_confusion=0.594;tie=0.017
qwen3-30b-a3b-base,dx-addressee-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.556;tie=0.024
qwen3-30b-a3b-base,dx-addressee-2-deictic,3shot,all,0.428,180,0.287,0.037,,speaker_confusion=0.561;tie=0.011
qwen3-30b-a3b-base,dx-addressee-2-deictic,3shot,free,0.476,126,0.286,0.044,,speaker_confusion=0.508;tie=0.016
qwen3-30b-a3b-base,dx-addressee-2-third,0shot,all,0.728,180,0.287,0.033,,
qwen3-30b-a3b-base,dx-addressee-2-third,0shot,free,0.809,126,0.286,0.035,,
qwen3-30b-a3b-base,dx-addressee-2-third,3shot_collide,all,0.756,180,0.287,0.032,,
qwen3-30b-a3b-base,dx-addressee-2-third,3shot_collide,free,0.841,126,0.286,0.033,,
qwen3-30b-a3b-base,dx-addressee-2-third,3shot,all,0.756,180,0.287,0.032,,
qwen3-30b-a3b-base,dx-addressee-2-third,3shot,free,0.825,126,0.286,0.034,,
qwen3-30b-a3b-base,sb-k1,0shot,all,0.891,64,0.125,0.039,,
qwen3-30b-a3b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
qwen3-30b-a3b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen3-30b-a3b-base,sb-k23,0shot,all,0.594,128,0.027,0.043,,
qwen3-30b-a3b-base,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
qwen3-30b-a3b-base,sb-k23,3shot,all,0.977,128,0.027,0.013,,
qwen3-30b-a3b-base,sb-split,0shot,all,0.555,128,0.027,0.044,,
qwen3-30b-a3b-base,sb-split,3shot_collide,all,0.891,128,0.027,0.028,,
qwen3-30b-a3b-base,sb-split,3shot,all,0.820,128,0.027,0.034,,
qwen3-30b-a3b-base,sb-dyn,0shot,all,0.492,256,0.027,0.031,,
qwen3-30b-a3b-base,sb-dyn,3shot_collide,all,0.504,256,0.027,0.031,,
qwen3-30b-a3b-base,sb-dyn,3shot,all,0.535,256,0.027,0.031,,
qwen3-30b-a3b-base,an-narr,0shot,all,0.489,640,0.250,0.020,,
qwen3-30b-a3b-base,an-narr,3shot_collide,all,0.822,640,0.250,0.015,,
qwen3-30b-a3b-base,an-narr,3shot,all,0.841,640,0.250,0.015,,
qwen3-30b-a3b-base,an-other-any,0shot,all,0.416,320,0.250,0.028,,
qwen3-30b-a3b-base,an-other-any,3shot_collide,all,0.369,320,0.250,0.027,,
qwen3-30b-a3b-base,an-other-any,3shot,all,0.450,320,0.250,0.028,,
qwen3-30b-a3b-base,an-me-any,0shot,all,0.119,160,0.250,0.026,,
qwen3-30b-a3b-base,an-me-any,3shot_collide,all,0.119,160,0.250,0.026,,
qwen3-30b-a3b-base,an-me-any,3shot,all,0.075,160,0.250,0.021,,
qwen3-30b-a3b-base,an-you-voc,0shot,all,0.938,80,0.250,0.027,,
qwen3-30b-a3b-base,an-you-voc,3shot_collide,all,0.775,80,0.250,0.047,,
qwen3-30b-a3b-base,an-you-voc,3shot,all,0.738,80,0.250,0.049,,
qwen3-30b-a3b-base,an-you-novoc,0shot,all,0.637,80,0.250,0.054,,
qwen3-30b-a3b-base,an-you-novoc,3shot_collide,all,0.438,80,0.250,0.056,,
qwen3-30b-a3b-base,an-you-novoc,3shot,all,0.312,80,0.250,0.052,,
qwen3-30b-a3b-base,adj-adj,0shot,all,0.854,260,0.333,0.022,,
qwen3-30b-a3b-base,adj-adj,3shot_collide,all,0.858,260,0.333,0.022,,
qwen3-30b-a3b-base,adj-adj,3shot,all,0.888,260,0.333,0.019,,
qwen3-30b-a3b-base,adj-voc,0shot,all,0.950,260,0.333,0.013,,
qwen3-30b-a3b-base,adj-voc,3shot_collide,all,0.981,260,0.333,0.009,,
qwen3-30b-a3b-base,adj-voc,3shot,all,0.996,260,0.333,0.004,,
qwen3-30b-a3b-base,adj-third,0shot,all,0.969,260,0.333,0.011,,
qwen3-30b-a3b-base,adj-third,3shot_collide,all,0.992,260,0.333,0.005,,
qwen3-30b-a3b-base,adj-third,3shot,all,0.992,260,0.333,0.005,,
qwen3-30b-a3b-base,cm-mutate,0shot,all,0.915,200,0.200,0.020,,
qwen3-30b-a3b-base,cm-mutate,3shot_collide,all,0.990,200,0.200,0.007,,
qwen3-30b-a3b-base,cm-mutate,3shot,all,0.980,200,0.200,0.010,,
qwen3-30b-a3b-base,cm-rebind,0shot,all,0.725,200,0.200,0.032,,
qwen3-30b-a3b-base,cm-rebind,3shot_collide,all,0.905,200,0.200,0.021,,
qwen3-30b-a3b-base,cm-rebind,3shot,all,0.830,200,0.200,0.027,,
qwen3-30b-a3b-base,cm-copy,0shot,all,0.640,200,0.200,0.034,,
qwen3-30b-a3b-base,cm-copy,3shot_collide,all,0.940,200,0.200,0.017,,
qwen3-30b-a3b-base,cm-copy,3shot,all,0.990,200,0.200,0.007,,
gemma4-31b-base,attr,0shot,all,0.704,1210,0.168,0.013,,
gemma4-31b-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
gemma4-31b-base,attr,3shot,all,1.000,1210,0.168,0.000,,
gemma4-31b-base,codevar,0shot,all,0.867,1200,0.267,0.010,,
gemma4-31b-base,codevar,3shot_collide,all,0.943,1200,0.267,0.007,,
gemma4-31b-base,codevar,3shot,all,0.916,1200,0.267,0.008,,
gemma4-31b-base,dialogue,0shot,all,0.932,1200,0.312,0.007,,
gemma4-31b-base,dialogue,3shot_collide,all,0.976,1200,0.312,0.004,,
gemma4-31b-base,dialogue,3shot,all,0.963,1200,0.312,0.005,,
gemma4-31b-base,dx-owner-3-deictic,0shot,all,0.717,120,0.333,0.041,,other_participant=0.275;tie=0.008
gemma4-31b-base,dx-owner-3-deictic,0shot,free,0.625,40,0.333,0.076,,other_participant=0.35;tie=0.025
gemma4-31b-base,dx-owner-3-deictic,3shot_collide,all,0.992,120,0.333,0.008,,other_participant=0.008
gemma4-31b-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
gemma4-31b-base,dx-owner-3-deictic,3shot,all,0.992,120,0.333,0.008,,other_participant=0.008
gemma4-31b-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
gemma4-31b-base,dx-owner-3-third,0shot,all,0.883,120,0.333,0.029,,
gemma4-31b-base,dx-owner-3-third,0shot,free,0.800,40,0.333,0.063,,
gemma4-31b-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
gemma4-31b-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
gemma4-31b-base,dx-owner-3-third,3shot,all,0.992,120,0.333,0.008,,
gemma4-31b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
gemma4-31b-base,dx-self-3-deictic,0shot,all,0.689,180,0.287,0.035,,other_object=0.106;previous_owner=0.028;speaker_confusion=0.144;tie=0.033
gemma4-31b-base,dx-self-3-deictic,0shot,free,0.738,126,0.286,0.039,,other_object=0.135;previous_owner=0.024;speaker_confusion=0.095;tie=0.008
gemma4-31b-base,dx-self-3-deictic,3shot_collide,all,0.950,180,0.287,0.016,,other_object=0.017;previous_owner=0.011;speaker_confusion=0.017;tie=0.006
gemma4-31b-base,dx-self-3-deictic,3shot_collide,free,0.976,126,0.286,0.014,,other_object=0.016;speaker_confusion=0.008
gemma4-31b-base,dx-self-3-deictic,3shot,all,0.933,180,0.287,0.019,,other_object=0.028;previous_owner=0.028;speaker_confusion=0.011
gemma4-31b-base,dx-self-3-deictic,3shot,free,0.944,126,0.286,0.020,,other_object=0.04;previous_owner=0.016
gemma4-31b-base,dx-self-3-third,0shot,all,0.922,180,0.287,0.020,,
gemma4-31b-base,dx-self-3-third,0shot,free,0.936,126,0.286,0.022,,
gemma4-31b-base,dx-self-3-third,3shot_collide,all,0.983,180,0.287,0.009,,
gemma4-31b-base,dx-self-3-third,3shot_collide,free,0.984,126,0.286,0.011,,
gemma4-31b-base,dx-self-3-third,3shot,all,0.978,180,0.287,0.011,,
gemma4-31b-base,dx-self-3-third,3shot,free,0.984,126,0.286,0.011,,
gemma4-31b-base,dx-self-2-deictic,0shot,all,0.522,180,0.287,0.037,,speaker_confusion=0.45;tie=0.028
gemma4-31b-base,dx-self-2-deictic,0shot,free,0.603,126,0.286,0.044,,speaker_confusion=0.365;tie=0.032
gemma4-31b-base,dx-self-2-deictic,3shot_collide,all,0.772,180,0.287,0.031,,speaker_confusion=0.222;tie=0.006
gemma4-31b-base,dx-self-2-deictic,3shot_collide,free,0.825,126,0.286,0.034,,speaker_confusion=0.175
gemma4-31b-base,dx-self-2-deictic,3shot,all,0.756,180,0.287,0.032,,speaker_confusion=0.228;tie=0.017
gemma4-31b-base,dx-self-2-deictic,3shot,free,0.825,126,0.286,0.034,,speaker_confusion=0.151;tie=0.024
gemma4-31b-base,dx-self-2-third,0shot,all,0.889,180,0.287,0.023,,
gemma4-31b-base,dx-self-2-third,0shot,free,0.913,126,0.286,0.025,,
gemma4-31b-base,dx-self-2-third,3shot_collide,all,0.900,180,0.287,0.022,,
gemma4-31b-base,dx-self-2-third,3shot_collide,free,0.929,126,0.286,0.023,,
gemma4-31b-base,dx-self-2-third,3shot,all,0.917,180,0.287,0.021,,
gemma4-31b-base,dx-self-2-third,3shot,free,0.921,126,0.286,0.024,,
gemma4-31b-base,dx-addressee-3-deictic,0shot,all,0.378,180,0.287,0.036,,other_object=0.089;previous_owner=0.056;speaker_confusion=0.456;tie=0.022
gemma4-31b-base,dx-addressee-3-deictic,0shot,free,0.429,126,0.286,0.044,,other_object=0.079;previous_owner=0.063;speaker_confusion=0.413;tie=0.016
gemma4-31b-base,dx-addressee-3-deictic,3shot_collide,all,0.806,180,0.287,0.029,,other_object=0.044;previous_owner=0.044;speaker_confusion=0.1;tie=0.006
gemma4-31b-base,dx-addressee-3-deictic,3shot_collide,free,0.841,126,0.286,0.033,,other_object=0.024;previous_owner=0.056;speaker_confusion=0.071;tie=0.008
gemma4-31b-base,dx-addressee-3-deictic,3shot,all,0.706,180,0.287,0.034,,other_object=0.05;previous_owner=0.039;speaker_confusion=0.183;tie=0.022
gemma4-31b-base,dx-addressee-3-deictic,3shot,free,0.762,126,0.286,0.038,,other_object=0.024;previous_owner=0.04;speaker_confusion=0.143;tie=0.032
gemma4-31b-base,dx-addressee-3-third,0shot,all,0.939,180,0.287,0.018,,
gemma4-31b-base,dx-addressee-3-third,0shot,free,0.929,126,0.286,0.023,,
gemma4-31b-base,dx-addressee-3-third,3shot_collide,all,0.950,180,0.287,0.016,,
gemma4-31b-base,dx-addressee-3-third,3shot_collide,free,0.936,126,0.286,0.022,,
gemma4-31b-base,dx-addressee-3-third,3shot,all,0.956,180,0.287,0.015,,
gemma4-31b-base,dx-addressee-3-third,3shot,free,0.952,126,0.286,0.019,,
gemma4-31b-base,dx-addressee-2-deictic,0shot,all,0.311,180,0.287,0.035,,speaker_confusion=0.667;tie=0.022
gemma4-31b-base,dx-addressee-2-deictic,0shot,free,0.325,126,0.286,0.042,,speaker_confusion=0.643;tie=0.032
gemma4-31b-base,dx-addressee-2-deictic,3shot_collide,all,0.511,180,0.287,0.037,,speaker_confusion=0.467;tie=0.022
gemma4-31b-base,dx-addressee-2-deictic,3shot_collide,free,0.548,126,0.286,0.044,,speaker_confusion=0.437;tie=0.016
gemma4-31b-base,dx-addressee-2-deictic,3shot,all,0.500,180,0.287,0.037,,speaker_confusion=0.483;tie=0.017
gemma4-31b-base,dx-addressee-2-deictic,3shot,free,0.540,126,0.286,0.044,,speaker_confusion=0.444;tie=0.016
gemma4-31b-base,dx-addressee-2-third,0shot,all,0.894,180,0.287,0.023,,
gemma4-31b-base,dx-addressee-2-third,0shot,free,0.913,126,0.286,0.025,,
gemma4-31b-base,dx-addressee-2-third,3shot_collide,all,0.928,180,0.287,0.019,,
gemma4-31b-base,dx-addressee-2-third,3shot_collide,free,0.960,126,0.286,0.017,,
gemma4-31b-base,dx-addressee-2-third,3shot,all,0.906,180,0.287,0.022,,
gemma4-31b-base,dx-addressee-2-third,3shot,free,0.936,126,0.286,0.022,,
gemma4-31b-base,sb-k1,0shot,all,0.906,64,0.125,0.036,,
gemma4-31b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
gemma4-31b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
gemma4-31b-base,sb-k23,0shot,all,0.438,128,0.027,0.044,,
gemma4-31b-base,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
gemma4-31b-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
gemma4-31b-base,sb-split,0shot,all,0.719,128,0.027,0.040,,
gemma4-31b-base,sb-split,3shot_collide,all,0.977,128,0.027,0.013,,
gemma4-31b-base,sb-split,3shot,all,0.891,128,0.027,0.028,,
gemma4-31b-base,sb-dyn,0shot,all,0.664,256,0.027,0.029,,
gemma4-31b-base,sb-dyn,3shot_collide,all,0.691,256,0.027,0.029,,
gemma4-31b-base,sb-dyn,3shot,all,0.684,256,0.027,0.029,,
gemma4-31b-base,an-narr,0shot,all,0.647,640,0.250,0.019,,
gemma4-31b-base,an-narr,3shot_collide,all,0.992,640,0.250,0.004,,
gemma4-31b-base,an-narr,3shot,all,0.994,640,0.250,0.003,,
gemma4-31b-base,an-other-any,0shot,all,0.713,320,0.250,0.025,,
gemma4-31b-base,an-other-any,3shot_collide,all,0.959,320,0.250,0.011,,
gemma4-31b-base,an-other-any,3shot,all,0.956,320,0.250,0.011,,
gemma4-31b-base,an-me-any,0shot,all,0.675,160,0.250,0.037,,
gemma4-31b-base,an-me-any,3shot_collide,all,0.944,160,0.250,0.018,,
gemma4-31b-base,an-me-any,3shot,all,0.975,160,0.250,0.012,,
gemma4-31b-base,an-you-voc,0shot,all,0.575,80,0.250,0.055,,
gemma4-31b-base,an-you-voc,3shot_collide,all,0.963,80,0.250,0.021,,
gemma4-31b-base,an-you-voc,3shot,all,0.963,80,0.250,0.021,,
gemma4-31b-base,an-you-novoc,0shot,all,0.512,80,0.250,0.056,,
gemma4-31b-base,an-you-novoc,3shot_collide,all,0.875,80,0.250,0.037,,
gemma4-31b-base,an-you-novoc,3shot,all,0.875,80,0.250,0.037,,
gemma4-31b-base,adj-adj,0shot,all,0.996,260,0.333,0.004,,
gemma4-31b-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-voc,0shot,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-third,0shot,all,0.989,260,0.333,0.007,,
gemma4-31b-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
gemma4-31b-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
gemma4-31b-base,cm-mutate,0shot,all,0.990,200,0.200,0.007,,
gemma4-31b-base,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
gemma4-31b-base,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
gemma4-31b-base,cm-rebind,0shot,all,0.810,200,0.200,0.028,,
gemma4-31b-base,cm-rebind,3shot_collide,all,0.995,200,0.200,0.005,,
gemma4-31b-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
gemma4-31b-base,cm-copy,0shot,all,0.565,200,0.200,0.035,,
gemma4-31b-base,cm-copy,3shot_collide,all,0.985,200,0.200,0.009,,
gemma4-31b-base,cm-copy,3shot,all,0.980,200,0.200,0.010,,
glm4-32b-base,attr,0shot,all,0.774,1210,0.168,0.012,,
glm4-32b-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
glm4-32b-base,attr,3shot,all,0.999,1210,0.168,0.001,,
glm4-32b-base,codevar,0shot,all,0.788,1200,0.267,0.012,,
glm4-32b-base,codevar,3shot_collide,all,0.892,1200,0.267,0.009,,
glm4-32b-base,codevar,3shot,all,0.904,1200,0.267,0.009,,
glm4-32b-base,dialogue,0shot,all,0.906,1200,0.312,0.008,,
glm4-32b-base,dialogue,3shot_collide,all,0.955,1200,0.312,0.006,,
glm4-32b-base,dialogue,3shot,all,0.942,1200,0.312,0.007,,
glm4-32b-base,dx-owner-3-deictic,0shot,all,0.717,120,0.333,0.041,,other_participant=0.217;speaker_confusion=0.05;tie=0.017
glm4-32b-base,dx-owner-3-deictic,0shot,free,0.650,40,0.333,0.075,,other_participant=0.325;speaker_confusion=0.025
glm4-32b-base,dx-owner-3-deictic,3shot_collide,all,0.967,120,0.333,0.016,,other_participant=0.017;speaker_confusion=0.017
glm4-32b-base,dx-owner-3-deictic,3shot_collide,free,0.975,40,0.333,0.025,,other_participant=0.025
glm4-32b-base,dx-owner-3-deictic,3shot,all,0.975,120,0.333,0.014,,other_participant=0.008;speaker_confusion=0.017
glm4-32b-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
glm4-32b-base,dx-owner-3-third,0shot,all,0.900,120,0.333,0.027,,
glm4-32b-base,dx-owner-3-third,0shot,free,0.825,40,0.333,0.060,,
glm4-32b-base,dx-owner-3-third,3shot_collide,all,0.992,120,0.333,0.008,,
glm4-32b-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
glm4-32b-base,dx-owner-3-third,3shot,all,0.992,120,0.333,0.008,,
glm4-32b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
glm4-32b-base,dx-self-3-deictic,0shot,all,0.472,180,0.287,0.037,,other_object=0.211;previous_owner=0.067;speaker_confusion=0.222;tie=0.028
glm4-32b-base,dx-self-3-deictic,0shot,free,0.452,126,0.286,0.044,,other_object=0.262;previous_owner=0.032;speaker_confusion=0.222;tie=0.032
glm4-32b-base,dx-self-3-deictic,3shot_collide,all,0.806,180,0.287,0.029,,other_object=0.056;previous_owner=0.05;speaker_confusion=0.083;tie=0.006
glm4-32b-base,dx-self-3-deictic,3shot_collide,free,0.794,126,0.286,0.036,,other_object=0.079;previous_owner=0.024;speaker_confusion=0.103
glm4-32b-base,dx-self-3-deictic,3shot,all,0.772,180,0.287,0.031,,other_object=0.056;previous_owner=0.056;speaker_confusion=0.094;tie=0.022
glm4-32b-base,dx-self-3-deictic,3shot,free,0.754,126,0.286,0.038,,other_object=0.079;previous_owner=0.04;speaker_confusion=0.119;tie=0.008
glm4-32b-base,dx-self-3-third,0shot,all,0.844,180,0.287,0.027,,
glm4-32b-base,dx-self-3-third,0shot,free,0.857,126,0.286,0.031,,
glm4-32b-base,dx-self-3-third,3shot_collide,all,0.939,180,0.287,0.018,,
glm4-32b-base,dx-self-3-third,3shot_collide,free,0.944,126,0.286,0.020,,
glm4-32b-base,dx-self-3-third,3shot,all,0.939,180,0.287,0.018,,
glm4-32b-base,dx-self-3-third,3shot,free,0.944,126,0.286,0.020,,
glm4-32b-base,dx-self-2-deictic,0shot,all,0.478,180,0.287,0.037,,speaker_confusion=0.522
glm4-32b-base,dx-self-2-deictic,0shot,free,0.468,126,0.286,0.044,,speaker_confusion=0.532
glm4-32b-base,dx-self-2-deictic,3shot_collide,all,0.628,180,0.287,0.036,,speaker_confusion=0.367;tie=0.006
glm4-32b-base,dx-self-2-deictic,3shot_collide,free,0.706,126,0.286,0.041,,speaker_confusion=0.294
glm4-32b-base,dx-self-2-deictic,3shot,all,0.594,180,0.287,0.037,,speaker_confusion=0.394;tie=0.011
glm4-32b-base,dx-self-2-deictic,3shot,free,0.682,126,0.286,0.042,,speaker_confusion=0.31;tie=0.008
glm4-32b-base,dx-self-2-third,0shot,all,0.817,180,0.287,0.029,,
glm4-32b-base,dx-self-2-third,0shot,free,0.865,126,0.286,0.030,,
glm4-32b-base,dx-self-2-third,3shot_collide,all,0.933,180,0.287,0.019,,
glm4-32b-base,dx-self-2-third,3shot_collide,free,0.936,126,0.286,0.022,,
glm4-32b-base,dx-self-2-third,3shot,all,0.956,180,0.287,0.015,,
glm4-32b-base,dx-self-2-third,3shot,free,0.968,126,0.286,0.016,,
glm4-32b-base,dx-addressee-3-deictic,0shot,all,0.344,180,0.287,0.035,,other_object=0.106;previous_owner=0.117;speaker_confusion=0.422;tie=0.011
glm4-32b-base,dx-addressee-3-deictic,0shot,free,0.302,126,0.286,0.041,,other_object=0.103;previous_owner=0.151;speaker_confusion=0.444
glm4-32b-base,dx-addressee-3-deictic,3shot_collide,all,0.533,180,0.287,0.037,,other_object=0.072;previous_owner=0.089;speaker_confusion=0.289;tie=0.017
glm4-32b-base,dx-addressee-3-deictic,3shot_collide,free,0.500,126,0.286,0.044,,other_object=0.071;previous_owner=0.111;speaker_confusion=0.294;tie=0.024
glm4-32b-base,dx-addressee-3-deictic,3shot,all,0.567,180,0.287,0.037,,other_object=0.078;previous_owner=0.067;speaker_confusion=0.283;tie=0.006
glm4-32b-base,dx-addressee-3-deictic,3shot,free,0.524,126,0.286,0.044,,other_object=0.071;previous_owner=0.087;speaker_confusion=0.31;tie=0.008
glm4-32b-base,dx-addressee-3-third,0shot,all,0.833,180,0.287,0.028,,
glm4-32b-base,dx-addressee-3-third,0shot,free,0.841,126,0.286,0.033,,
glm4-32b-base,dx-addressee-3-third,3shot_collide,all,0.978,180,0.287,0.011,,
glm4-32b-base,dx-addressee-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
glm4-32b-base,dx-addressee-3-third,3shot,all,0.956,180,0.287,0.015,,
glm4-32b-base,dx-addressee-3-third,3shot,free,0.968,126,0.286,0.016,,
glm4-32b-base,dx-addressee-2-deictic,0shot,all,0.267,180,0.287,0.033,,speaker_confusion=0.711;tie=0.022
glm4-32b-base,dx-addressee-2-deictic,0shot,free,0.151,126,0.286,0.032,,speaker_confusion=0.825;tie=0.024
glm4-32b-base,dx-addressee-2-deictic,3shot_collide,all,0.306,180,0.287,0.034,,speaker_confusion=0.667;tie=0.028
glm4-32b-base,dx-addressee-2-deictic,3shot_collide,free,0.254,126,0.286,0.039,,speaker_confusion=0.73;tie=0.016
glm4-32b-base,dx-addressee-2-deictic,3shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.672;tie=0.011
glm4-32b-base,dx-addressee-2-deictic,3shot,free,0.270,126,0.286,0.040,,speaker_confusion=0.714;tie=0.016
glm4-32b-base,dx-addressee-2-third,0shot,all,0.839,180,0.287,0.027,,
glm4-32b-base,dx-addressee-2-third,0shot,free,0.849,126,0.286,0.032,,
glm4-32b-base,dx-addressee-2-third,3shot_collide,all,0.933,180,0.287,0.019,,
glm4-32b-base,dx-addressee-2-third,3shot_collide,free,0.960,126,0.286,0.017,,
glm4-32b-base,dx-addressee-2-third,3shot,all,0.933,180,0.287,0.019,,
glm4-32b-base,dx-addressee-2-third,3shot,free,0.952,126,0.286,0.019,,
glm4-32b-base,sb-k1,0shot,all,0.859,64,0.125,0.043,,
glm4-32b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
glm4-32b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
glm4-32b-base,sb-k23,0shot,all,0.711,128,0.027,0.040,,
glm4-32b-base,sb-k23,3shot_collide,all,0.992,128,0.027,0.008,,
glm4-32b-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
glm4-32b-base,sb-split,0shot,all,0.719,128,0.027,0.040,,
glm4-32b-base,sb-split,3shot_collide,all,0.945,128,0.027,0.020,,
glm4-32b-base,sb-split,3shot,all,0.828,128,0.027,0.033,,
glm4-32b-base,sb-dyn,0shot,all,0.711,256,0.027,0.028,,
glm4-32b-base,sb-dyn,3shot_collide,all,0.644,256,0.027,0.030,,
glm4-32b-base,sb-dyn,3shot,all,0.633,256,0.027,0.030,,
glm4-32b-base,an-narr,0shot,all,0.645,640,0.250,0.019,,
glm4-32b-base,an-narr,3shot_collide,all,0.991,640,0.250,0.004,,
glm4-32b-base,an-narr,3shot,all,0.991,640,0.250,0.004,,
glm4-32b-base,an-other-any,0shot,all,0.522,320,0.250,0.028,,
glm4-32b-base,an-other-any,3shot_collide,all,0.903,320,0.250,0.017,,
glm4-32b-base,an-other-any,3shot,all,0.838,320,0.250,0.021,,
glm4-32b-base,an-me-any,0shot,all,0.338,160,0.250,0.037,,
glm4-32b-base,an-me-any,3shot_collide,all,0.475,160,0.250,0.040,,
glm4-32b-base,an-me-any,3shot,all,0.412,160,0.250,0.039,,
glm4-32b-base,an-you-voc,0shot,all,0.537,80,0.250,0.056,,
glm4-32b-base,an-you-voc,3shot_collide,all,0.912,80,0.250,0.032,,
glm4-32b-base,an-you-voc,3shot,all,0.912,80,0.250,0.032,,
glm4-32b-base,an-you-novoc,0shot,all,0.450,80,0.250,0.056,,
glm4-32b-base,an-you-novoc,3shot_collide,all,0.775,80,0.250,0.047,,
glm4-32b-base,an-you-novoc,3shot,all,0.588,80,0.250,0.055,,
glm4-32b-base,adj-adj,0shot,all,0.992,260,0.333,0.005,,
glm4-32b-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
glm4-32b-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
glm4-32b-base,adj-voc,0shot,all,0.996,260,0.333,0.004,,
glm4-32b-base,adj-voc,3shot_collide,all,0.996,260,0.333,0.004,,
glm4-32b-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
glm4-32b-base,adj-third,0shot,all,0.992,260,0.333,0.005,,
glm4-32b-base,adj-third,3shot_collide,all,0.992,260,0.333,0.005,,
glm4-32b-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
glm4-32b-base,cm-mutate,0shot,all,0.955,200,0.200,0.015,,
glm4-32b-base,cm-mutate,3shot_collide,all,0.935,200,0.200,0.017,,
glm4-32b-base,cm-mutate,3shot,all,0.910,200,0.200,0.020,,
glm4-32b-base,cm-rebind,0shot,all,0.575,200,0.200,0.035,,
glm4-32b-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
glm4-32b-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
glm4-32b-base,cm-copy,0shot,all,0.090,200,0.200,0.020,,
glm4-32b-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
glm4-32b-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
olmo3-32b,attr,0shot,all,0.782,1210,0.168,0.012,,
olmo3-32b,attr,3shot_collide,all,0.997,1210,0.168,0.002,,
olmo3-32b,attr,3shot,all,1.000,1210,0.168,0.000,,
olmo3-32b,codevar,0shot,all,0.752,1200,0.267,0.013,,
olmo3-32b,codevar,3shot_collide,all,0.873,1200,0.267,0.010,,
olmo3-32b,codevar,3shot,all,0.873,1200,0.267,0.010,,
olmo3-32b,dialogue,0shot,all,0.889,1200,0.312,0.009,,
olmo3-32b,dialogue,3shot_collide,all,0.932,1200,0.312,0.007,,
olmo3-32b,dialogue,3shot,all,0.923,1200,0.312,0.008,,
olmo3-32b,dx-owner-3-deictic,0shot,all,0.717,120,0.333,0.041,,other_participant=0.258;speaker_confusion=0.025
olmo3-32b,dx-owner-3-deictic,0shot,free,0.650,40,0.333,0.075,,other_participant=0.35
olmo3-32b,dx-owner-3-deictic,3shot_collide,all,0.933,120,0.333,0.023,,other_participant=0.058;speaker_confusion=0.008
olmo3-32b,dx-owner-3-deictic,3shot_collide,free,0.900,40,0.333,0.047,,other_participant=0.075;speaker_confusion=0.025
olmo3-32b,dx-owner-3-deictic,3shot,all,0.958,120,0.333,0.018,,other_participant=0.017;speaker_confusion=0.025
olmo3-32b,dx-owner-3-deictic,3shot,free,0.975,40,0.333,0.025,,speaker_confusion=0.025
olmo3-32b,dx-owner-3-third,0shot,all,0.842,120,0.333,0.033,,
olmo3-32b,dx-owner-3-third,0shot,free,0.825,40,0.333,0.060,,
olmo3-32b,dx-owner-3-third,3shot_collide,all,0.933,120,0.333,0.023,,
olmo3-32b,dx-owner-3-third,3shot_collide,free,0.975,40,0.333,0.025,,
olmo3-32b,dx-owner-3-third,3shot,all,0.883,120,0.333,0.029,,
olmo3-32b,dx-owner-3-third,3shot,free,0.950,40,0.333,0.035,,
olmo3-32b,dx-self-3-deictic,0shot,all,0.500,180,0.287,0.037,,other_object=0.228;previous_owner=0.028;speaker_confusion=0.239;tie=0.006
olmo3-32b,dx-self-3-deictic,0shot,free,0.571,126,0.286,0.044,,other_object=0.206;previous_owner=0.016;speaker_confusion=0.198;tie=0.008
olmo3-32b,dx-self-3-deictic,3shot_collide,all,0.617,180,0.287,0.036,,other_object=0.111;previous_owner=0.033;speaker_confusion=0.233;tie=0.006
olmo3-32b,dx-self-3-deictic,3shot_collide,free,0.682,126,0.286,0.042,,other_object=0.135;previous_owner=0.032;speaker_confusion=0.151
olmo3-32b,dx-self-3-deictic,3shot,all,0.689,180,0.287,0.035,,other_object=0.072;previous_owner=0.044;speaker_confusion=0.194
olmo3-32b,dx-self-3-deictic,3shot,free,0.786,126,0.286,0.037,,other_object=0.079;previous_owner=0.016;speaker_confusion=0.119
olmo3-32b,dx-self-3-third,0shot,all,0.761,180,0.287,0.032,,
olmo3-32b,dx-self-3-third,0shot,free,0.841,126,0.286,0.033,,
olmo3-32b,dx-self-3-third,3shot_collide,all,0.806,180,0.287,0.029,,
olmo3-32b,dx-self-3-third,3shot_collide,free,0.881,126,0.286,0.029,,
olmo3-32b,dx-self-3-third,3shot,all,0.772,180,0.287,0.031,,
olmo3-32b,dx-self-3-third,3shot,free,0.849,126,0.286,0.032,,
olmo3-32b,dx-self-2-deictic,0shot,all,0.461,180,0.287,0.037,,speaker_confusion=0.522;tie=0.017
olmo3-32b,dx-self-2-deictic,0shot,free,0.571,126,0.286,0.044,,speaker_confusion=0.421;tie=0.008
olmo3-32b,dx-self-2-deictic,3shot_collide,all,0.511,180,0.287,0.037,,speaker_confusion=0.483;tie=0.006
olmo3-32b,dx-self-2-deictic,3shot_collide,free,0.611,126,0.286,0.043,,speaker_confusion=0.381;tie=0.008
olmo3-32b,dx-self-2-deictic,3shot,all,0.528,180,0.287,0.037,,speaker_confusion=0.472
olmo3-32b,dx-self-2-deictic,3shot,free,0.635,126,0.286,0.043,,speaker_confusion=0.365
olmo3-32b,dx-self-2-third,0shot,all,0.750,180,0.287,0.032,,
olmo3-32b,dx-self-2-third,0shot,free,0.841,126,0.286,0.033,,
olmo3-32b,dx-self-2-third,3shot_collide,all,0.733,180,0.287,0.033,,
olmo3-32b,dx-self-2-third,3shot_collide,free,0.802,126,0.286,0.035,,
olmo3-32b,dx-self-2-third,3shot,all,0.683,180,0.287,0.035,,
olmo3-32b,dx-self-2-third,3shot,free,0.754,126,0.286,0.038,,
olmo3-32b,dx-addressee-3-deictic,0shot,all,0.411,180,0.287,0.037,,other_object=0.178;previous_owner=0.022;speaker_confusion=0.372;tie=0.017
olmo3-32b,dx-addressee-3-deictic,0shot,free,0.508,126,0.286,0.044,,other_object=0.135;previous_owner=0.032;speaker_confusion=0.31;tie=0.016
olmo3-32b,dx-addressee-3-deictic,3shot_collide,all,0.439,180,0.287,0.037,,other_object=0.139;previous_owner=0.072;speaker_confusion=0.339;tie=0.011
olmo3-32b,dx-addressee-3-deictic,3shot_collide,free,0.500,126,0.286,0.044,,other_object=0.095;previous_owner=0.079;speaker_confusion=0.31;tie=0.016
olmo3-32b,dx-addressee-3-deictic,3shot,all,0.506,180,0.287,0.037,,other_object=0.094;previous_owner=0.078;speaker_confusion=0.317;tie=0.006
olmo3-32b,dx-addressee-3-deictic,3shot,free,0.587,126,0.286,0.044,,other_object=0.056;previous_owner=0.087;speaker_confusion=0.262;tie=0.008
olmo3-32b,dx-addressee-3-third,0shot,all,0.806,180,0.287,0.029,,
olmo3-32b,dx-addressee-3-third,0shot,free,0.818,126,0.286,0.034,,
olmo3-32b,dx-addressee-3-third,3shot_collide,all,0.783,180,0.287,0.031,,
olmo3-32b,dx-addressee-3-third,3shot_collide,free,0.857,126,0.286,0.031,,
olmo3-32b,dx-addressee-3-third,3shot,all,0.794,180,0.287,0.030,,
olmo3-32b,dx-addressee-3-third,3shot,free,0.857,126,0.286,0.031,,
olmo3-32b,dx-addressee-2-deictic,0shot,all,0.250,180,0.287,0.032,,speaker_confusion=0.739;tie=0.011
olmo3-32b,dx-addressee-2-deictic,0shot,free,0.294,126,0.286,0.041,,speaker_confusion=0.69;tie=0.016
olmo3-32b,dx-addressee-2-deictic,3shot_collide,all,0.372,180,0.287,0.036,,speaker_confusion=0.617;tie=0.011
olmo3-32b,dx-addressee-2-deictic,3shot_collide,free,0.421,126,0.286,0.044,,speaker_confusion=0.563;tie=0.016
olmo3-32b,dx-addressee-2-deictic,3shot,all,0.383,180,0.287,0.036,,speaker_confusion=0.617
olmo3-32b,dx-addressee-2-deictic,3shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.579
olmo3-32b,dx-addressee-2-third,0shot,all,0.728,180,0.287,0.033,,
olmo3-32b,dx-addressee-2-third,0shot,free,0.802,126,0.286,0.035,,
olmo3-32b,dx-addressee-2-third,3shot_collide,all,0.728,180,0.287,0.033,,
olmo3-32b,dx-addressee-2-third,3shot_collide,free,0.818,126,0.286,0.034,,
olmo3-32b,dx-addressee-2-third,3shot,all,0.717,180,0.287,0.034,,
olmo3-32b,dx-addressee-2-third,3shot,free,0.809,126,0.286,0.035,,
olmo3-32b,sb-k1,0shot,all,1.000,64,0.125,0.000,,
olmo3-32b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
olmo3-32b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
olmo3-32b,sb-k23,0shot,all,0.758,128,0.027,0.038,,
olmo3-32b,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
olmo3-32b,sb-k23,3shot,all,0.992,128,0.027,0.008,,
olmo3-32b,sb-split,0shot,all,0.578,128,0.027,0.044,,
olmo3-32b,sb-split,3shot_collide,all,0.898,128,0.027,0.027,,
olmo3-32b,sb-split,3shot,all,0.914,128,0.027,0.025,,
olmo3-32b,sb-dyn,0shot,all,0.625,256,0.027,0.030,,
olmo3-32b,sb-dyn,3shot_collide,all,0.555,256,0.027,0.031,,
olmo3-32b,sb-dyn,3shot,all,0.570,256,0.027,0.031,,
olmo3-32b,an-narr,0shot,all,0.667,640,0.250,0.019,,
olmo3-32b,an-narr,3shot_collide,all,0.964,640,0.250,0.007,,
olmo3-32b,an-narr,3shot,all,0.980,640,0.250,0.006,,
olmo3-32b,an-other-any,0shot,all,0.062,320,0.250,0.013,,
olmo3-32b,an-other-any,3shot_collide,all,0.791,320,0.250,0.023,,
olmo3-32b,an-other-any,3shot,all,0.744,320,0.250,0.024,,
olmo3-32b,an-me-any,0shot,all,0.713,160,0.250,0.036,,
olmo3-32b,an-me-any,3shot_collide,all,0.606,160,0.250,0.039,,
olmo3-32b,an-me-any,3shot,all,0.644,160,0.250,0.038,,
olmo3-32b,an-you-voc,0shot,all,0.487,80,0.250,0.056,,
olmo3-32b,an-you-voc,3shot_collide,all,0.662,80,0.250,0.053,,
olmo3-32b,an-you-voc,3shot,all,0.775,80,0.250,0.047,,
olmo3-32b,an-you-novoc,0shot,all,0.075,80,0.250,0.029,,
olmo3-32b,an-you-novoc,3shot_collide,all,0.425,80,0.250,0.055,,
olmo3-32b,an-you-novoc,3shot,all,0.475,80,0.250,0.056,,
olmo3-32b,adj-adj,0shot,all,0.977,260,0.333,0.009,,
olmo3-32b,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
olmo3-32b,adj-adj,3shot,all,0.996,260,0.333,0.004,,
olmo3-32b,adj-voc,0shot,all,0.996,260,0.333,0.004,,
olmo3-32b,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
olmo3-32b,adj-voc,3shot,all,0.996,260,0.333,0.004,,
olmo3-32b,adj-third,0shot,all,0.996,260,0.333,0.004,,
olmo3-32b,adj-third,3shot_collide,all,0.992,260,0.333,0.005,,
olmo3-32b,adj-third,3shot,all,0.981,260,0.333,0.009,,
olmo3-32b,cm-mutate,0shot,all,0.975,200,0.200,0.011,,
olmo3-32b,cm-mutate,3shot_collide,all,0.995,200,0.200,0.005,,
olmo3-32b,cm-mutate,3shot,all,0.965,200,0.200,0.013,,
olmo3-32b,cm-rebind,0shot,all,0.690,200,0.200,0.033,,
olmo3-32b,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
olmo3-32b,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
olmo3-32b,cm-copy,0shot,all,0.350,200,0.200,0.034,,
olmo3-32b,cm-copy,3shot_collide,all,0.980,200,0.200,0.010,,
olmo3-32b,cm-copy,3shot,all,0.995,200,0.200,0.005,,
olmo3-32b-stage1,attr,0shot,all,0.696,1210,0.168,0.013,,
olmo3-32b-stage1,attr,3shot_collide,all,0.998,1210,0.168,0.001,,
olmo3-32b-stage1,attr,3shot,all,0.999,1210,0.168,0.001,,
olmo3-32b-stage1,codevar,0shot,all,0.528,1200,0.267,0.014,,
olmo3-32b-stage1,codevar,3shot_collide,all,0.748,1200,0.267,0.013,,
olmo3-32b-stage1,codevar,3shot,all,0.728,1200,0.267,0.013,,
olmo3-32b-stage1,dialogue,0shot,all,0.853,1200,0.312,0.010,,
olmo3-32b-stage1,dialogue,3shot_collide,all,0.906,1200,0.312,0.008,,
olmo3-32b-stage1,dialogue,3shot,all,0.907,1200,0.312,0.008,,
olmo3-32b-stage1,dx-owner-3-deictic,0shot,all,0.575,120,0.333,0.045,,other_participant=0.383;speaker_confusion=0.025;tie=0.017
olmo3-32b-stage1,dx-owner-3-deictic,0shot,free,0.400,40,0.333,0.077,,other_participant=0.525;speaker_confusion=0.025;tie=0.05
olmo3-32b-stage1,dx-owner-3-deictic,3shot_collide,all,0.867,120,0.333,0.031,,other_participant=0.067;speaker_confusion=0.067
olmo3-32b-stage1,dx-owner-3-deictic,3shot_collide,free,0.825,40,0.333,0.060,,other_participant=0.075;speaker_confusion=0.1
olmo3-32b-stage1,dx-owner-3-deictic,3shot,all,0.825,120,0.333,0.035,,other_participant=0.083;speaker_confusion=0.092
olmo3-32b-stage1,dx-owner-3-deictic,3shot,free,0.775,40,0.333,0.066,,other_participant=0.075;speaker_confusion=0.15
olmo3-32b-stage1,dx-owner-3-third,0shot,all,0.675,120,0.333,0.043,,
olmo3-32b-stage1,dx-owner-3-third,0shot,free,0.600,40,0.333,0.077,,
olmo3-32b-stage1,dx-owner-3-third,3shot_collide,all,0.933,120,0.333,0.023,,
olmo3-32b-stage1,dx-owner-3-third,3shot_collide,free,0.925,40,0.333,0.042,,
olmo3-32b-stage1,dx-owner-3-third,3shot,all,0.900,120,0.333,0.027,,
olmo3-32b-stage1,dx-owner-3-third,3shot,free,0.925,40,0.333,0.042,,
olmo3-32b-stage1,dx-self-3-deictic,0shot,all,0.383,180,0.287,0.036,,other_object=0.244;previous_owner=0.061;speaker_confusion=0.294;tie=0.017
olmo3-32b-stage1,dx-self-3-deictic,0shot,free,0.436,126,0.286,0.044,,other_object=0.238;previous_owner=0.04;speaker_confusion=0.262;tie=0.024
olmo3-32b-stage1,dx-self-3-deictic,3shot_collide,all,0.522,180,0.287,0.037,,other_object=0.111;previous_owner=0.033;speaker_confusion=0.322;tie=0.011
olmo3-32b-stage1,dx-self-3-deictic,3shot_collide,free,0.627,126,0.286,0.043,,other_object=0.095;previous_owner=0.016;speaker_confusion=0.254;tie=0.008
olmo3-32b-stage1,dx-self-3-deictic,3shot,all,0.506,180,0.287,0.037,,other_object=0.111;previous_owner=0.061;speaker_confusion=0.322
olmo3-32b-stage1,dx-self-3-deictic,3shot,free,0.603,126,0.286,0.044,,other_object=0.095;previous_owner=0.048;speaker_confusion=0.254
olmo3-32b-stage1,dx-self-3-third,0shot,all,0.711,180,0.287,0.034,,
olmo3-32b-stage1,dx-self-3-third,0shot,free,0.754,126,0.286,0.038,,
olmo3-32b-stage1,dx-self-3-third,3shot_collide,all,0.800,180,0.287,0.030,,
olmo3-32b-stage1,dx-self-3-third,3shot_collide,free,0.818,126,0.286,0.034,,
olmo3-32b-stage1,dx-self-3-third,3shot,all,0.767,180,0.287,0.032,,
olmo3-32b-stage1,dx-self-3-third,3shot,free,0.770,126,0.286,0.037,,
olmo3-32b-stage1,dx-self-2-deictic,0shot,all,0.394,180,0.287,0.036,,speaker_confusion=0.606
olmo3-32b-stage1,dx-self-2-deictic,0shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
olmo3-32b-stage1,dx-self-2-deictic,3shot_collide,all,0.489,180,0.287,0.037,,speaker_confusion=0.506;tie=0.006
olmo3-32b-stage1,dx-self-2-deictic,3shot_collide,free,0.587,126,0.286,0.044,,speaker_confusion=0.405;tie=0.008
olmo3-32b-stage1,dx-self-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.511;tie=0.006
olmo3-32b-stage1,dx-self-2-deictic,3shot,free,0.579,126,0.286,0.044,,speaker_confusion=0.421
olmo3-32b-stage1,dx-self-2-third,0shot,all,0.717,180,0.287,0.034,,
olmo3-32b-stage1,dx-self-2-third,0shot,free,0.778,126,0.286,0.037,,
olmo3-32b-stage1,dx-self-2-third,3shot_collide,all,0.722,180,0.287,0.033,,
olmo3-32b-stage1,dx-self-2-third,3shot_collide,free,0.730,126,0.286,0.040,,
olmo3-32b-stage1,dx-self-2-third,3shot,all,0.683,180,0.287,0.035,,
olmo3-32b-stage1,dx-self-2-third,3shot,free,0.730,126,0.286,0.040,,
olmo3-32b-stage1,dx-addressee-3-deictic,0shot,all,0.306,180,0.287,0.034,,other_object=0.167;previous_owner=0.05;speaker_confusion=0.467;tie=0.011
olmo3-32b-stage1,dx-addressee-3-deictic,0shot,free,0.349,126,0.286,0.043,,other_object=0.151;previous_owner=0.056;speaker_confusion=0.437;tie=0.008
olmo3-32b-stage1,dx-addressee-3-deictic,3shot_collide,all,0.350,180,0.287,0.036,,other_object=0.167;previous_owner=0.094;speaker_confusion=0.389
olmo3-32b-stage1,dx-addressee-3-deictic,3shot_collide,free,0.405,126,0.286,0.044,,other_object=0.111;previous_owner=0.111;speaker_confusion=0.373
olmo3-32b-stage1,dx-addressee-3-deictic,3shot,all,0.389,180,0.287,0.036,,other_object=0.111;previous_owner=0.067;speaker_confusion=0.422;tie=0.011
olmo3-32b-stage1,dx-addressee-3-deictic,3shot,free,0.452,126,0.286,0.044,,other_object=0.087;previous_owner=0.095;speaker_confusion=0.365
olmo3-32b-stage1,dx-addressee-3-third,0shot,all,0.750,180,0.287,0.032,,
olmo3-32b-stage1,dx-addressee-3-third,0shot,free,0.738,126,0.286,0.039,,
olmo3-32b-stage1,dx-addressee-3-third,3shot_collide,all,0.728,180,0.287,0.033,,
olmo3-32b-stage1,dx-addressee-3-third,3shot_collide,free,0.762,126,0.286,0.038,,
olmo3-32b-stage1,dx-addressee-3-third,3shot,all,0.717,180,0.287,0.034,,
olmo3-32b-stage1,dx-addressee-3-third,3shot,free,0.722,126,0.286,0.040,,
olmo3-32b-stage1,dx-addressee-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.689;tie=0.006
olmo3-32b-stage1,dx-addressee-2-deictic,0shot,free,0.230,126,0.286,0.037,,speaker_confusion=0.762;tie=0.008
olmo3-32b-stage1,dx-addressee-2-deictic,3shot_collide,all,0.394,180,0.287,0.036,,speaker_confusion=0.6;tie=0.006
olmo3-32b-stage1,dx-addressee-2-deictic,3shot_collide,free,0.468,126,0.286,0.044,,speaker_confusion=0.524;tie=0.008
olmo3-32b-stage1,dx-addressee-2-deictic,3shot,all,0.361,180,0.287,0.036,,speaker_confusion=0.633;tie=0.006
olmo3-32b-stage1,dx-addressee-2-deictic,3shot,free,0.429,126,0.286,0.044,,speaker_confusion=0.571
olmo3-32b-stage1,dx-addressee-2-third,0shot,all,0.711,180,0.287,0.034,,
olmo3-32b-stage1,dx-addressee-2-third,0shot,free,0.762,126,0.286,0.038,,
olmo3-32b-stage1,dx-addressee-2-third,3shot_collide,all,0.711,180,0.287,0.034,,
olmo3-32b-stage1,dx-addressee-2-third,3shot_collide,free,0.802,126,0.286,0.035,,
olmo3-32b-stage1,dx-addressee-2-third,3shot,all,0.706,180,0.287,0.034,,
olmo3-32b-stage1,dx-addressee-2-third,3shot,free,0.770,126,0.286,0.037,,
olmo3-32b-stage1,sb-k1,0shot,all,0.859,64,0.125,0.043,,
olmo3-32b-stage1,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
olmo3-32b-stage1,sb-k1,3shot,all,1.000,64,0.125,0.000,,
olmo3-32b-stage1,sb-k23,0shot,all,0.445,128,0.027,0.044,,
olmo3-32b-stage1,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
olmo3-32b-stage1,sb-k23,3shot,all,0.992,128,0.027,0.008,,
olmo3-32b-stage1,sb-split,0shot,all,0.398,128,0.027,0.043,,
olmo3-32b-stage1,sb-split,3shot_collide,all,0.867,128,0.027,0.030,,
olmo3-32b-stage1,sb-split,3shot,all,0.828,128,0.027,0.033,,
olmo3-32b-stage1,sb-dyn,0shot,all,0.512,256,0.027,0.031,,
olmo3-32b-stage1,sb-dyn,3shot_collide,all,0.523,256,0.027,0.031,,
olmo3-32b-stage1,sb-dyn,3shot,all,0.457,256,0.027,0.031,,
olmo3-32b-stage1,an-narr,0shot,all,0.434,640,0.250,0.020,,
olmo3-32b-stage1,an-narr,3shot_collide,all,0.870,640,0.250,0.013,,
olmo3-32b-stage1,an-narr,3shot,all,0.863,640,0.250,0.014,,
olmo3-32b-stage1,an-other-any,0shot,all,0.284,320,0.250,0.025,,
olmo3-32b-stage1,an-other-any,3shot_collide,all,0.547,320,0.250,0.028,,
olmo3-32b-stage1,an-other-any,3shot,all,0.512,320,0.250,0.028,,
olmo3-32b-stage1,an-me-any,0shot,all,0.500,160,0.250,0.040,,
olmo3-32b-stage1,an-me-any,3shot_collide,all,0.312,160,0.250,0.037,,
olmo3-32b-stage1,an-me-any,3shot,all,0.319,160,0.250,0.037,,
olmo3-32b-stage1,an-you-voc,0shot,all,0.463,80,0.250,0.056,,
olmo3-32b-stage1,an-you-voc,3shot_collide,all,0.450,80,0.250,0.056,,
olmo3-32b-stage1,an-you-voc,3shot,all,0.613,80,0.250,0.054,,
olmo3-32b-stage1,an-you-novoc,0shot,all,0.087,80,0.250,0.032,,
olmo3-32b-stage1,an-you-novoc,3shot_collide,all,0.087,80,0.250,0.032,,
olmo3-32b-stage1,an-you-novoc,3shot,all,0.325,80,0.250,0.052,,
olmo3-32b-stage1,adj-adj,0shot,all,0.896,260,0.333,0.019,,
olmo3-32b-stage1,adj-adj,3shot_collide,all,0.962,260,0.333,0.012,,
olmo3-32b-stage1,adj-adj,3shot,all,0.977,260,0.333,0.009,,
olmo3-32b-stage1,adj-voc,0shot,all,0.946,260,0.333,0.014,,
olmo3-32b-stage1,adj-voc,3shot_collide,all,0.985,260,0.333,0.008,,
olmo3-32b-stage1,adj-voc,3shot,all,1.000,260,0.333,0.000,,
olmo3-32b-stage1,adj-third,0shot,all,0.977,260,0.333,0.009,,
olmo3-32b-stage1,adj-third,3shot_collide,all,0.989,260,0.333,0.007,,
olmo3-32b-stage1,adj-third,3shot,all,0.989,260,0.333,0.007,,
olmo3-32b-stage1,cm-mutate,0shot,all,0.860,200,0.200,0.025,,
olmo3-32b-stage1,cm-mutate,3shot_collide,all,0.920,200,0.200,0.019,,
olmo3-32b-stage1,cm-mutate,3shot,all,0.855,200,0.200,0.025,,
olmo3-32b-stage1,cm-rebind,0shot,all,0.145,200,0.200,0.025,,
olmo3-32b-stage1,cm-rebind,3shot_collide,all,0.990,200,0.200,0.007,,
olmo3-32b-stage1,cm-rebind,3shot,all,0.965,200,0.200,0.013,,
olmo3-32b-stage1,cm-copy,0shot,all,0.030,200,0.200,0.012,,
olmo3-32b-stage1,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
olmo3-32b-stage1,cm-copy,3shot,all,0.990,200,0.200,0.007,,
llama31-70b,attr,0shot,all,0.532,1210,0.168,0.014,,
llama31-70b,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
llama31-70b,attr,3shot,all,0.995,1210,0.168,0.002,,
llama31-70b,codevar,0shot,all,0.792,1200,0.267,0.012,,
llama31-70b,codevar,3shot_collide,all,0.922,1200,0.267,0.008,,
llama31-70b,codevar,3shot,all,0.895,1200,0.267,0.009,,
llama31-70b,dialogue,0shot,all,0.907,1200,0.312,0.008,,
llama31-70b,dialogue,3shot_collide,all,0.972,1200,0.312,0.005,,
llama31-70b,dialogue,3shot,all,0.966,1200,0.312,0.005,,
llama31-70b,dx-owner-3-deictic,0shot,all,0.675,120,0.333,0.043,,other_participant=0.308;tie=0.017
llama31-70b,dx-owner-3-deictic,0shot,free,0.550,40,0.333,0.079,,other_participant=0.425;tie=0.025
llama31-70b,dx-owner-3-deictic,3shot_collide,all,0.992,120,0.333,0.008,,other_participant=0.008
llama31-70b,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
llama31-70b,dx-owner-3-deictic,3shot,all,0.967,120,0.333,0.016,,other_participant=0.025;tie=0.008
llama31-70b,dx-owner-3-deictic,3shot,free,0.975,40,0.333,0.025,,other_participant=0.025
llama31-70b,dx-owner-3-third,0shot,all,0.683,120,0.333,0.043,,
llama31-70b,dx-owner-3-third,0shot,free,0.500,40,0.333,0.079,,
llama31-70b,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
llama31-70b,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
llama31-70b,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
llama31-70b,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
llama31-70b,dx-self-3-deictic,0shot,all,0.567,180,0.287,0.037,,other_object=0.15;previous_owner=0.028;speaker_confusion=0.239;tie=0.017
llama31-70b,dx-self-3-deictic,0shot,free,0.579,126,0.286,0.044,,other_object=0.19;previous_owner=0.008;speaker_confusion=0.206;tie=0.016
llama31-70b,dx-self-3-deictic,3shot_collide,all,0.822,180,0.287,0.029,,other_object=0.022;previous_owner=0.039;speaker_confusion=0.1;tie=0.017
llama31-70b,dx-self-3-deictic,3shot_collide,free,0.889,126,0.286,0.028,,other_object=0.016;previous_owner=0.024;speaker_confusion=0.056;tie=0.016
llama31-70b,dx-self-3-deictic,3shot,all,0.844,180,0.287,0.027,,other_object=0.033;previous_owner=0.056;speaker_confusion=0.056;tie=0.011
llama31-70b,dx-self-3-deictic,3shot,free,0.936,126,0.286,0.022,,other_object=0.024;previous_owner=0.016;speaker_confusion=0.024
llama31-70b,dx-self-3-third,0shot,all,0.806,180,0.287,0.029,,
llama31-70b,dx-self-3-third,0shot,free,0.802,126,0.286,0.035,,
llama31-70b,dx-self-3-third,3shot_collide,all,0.939,180,0.287,0.018,,
llama31-70b,dx-self-3-third,3shot_collide,free,0.968,126,0.286,0.016,,
llama31-70b,dx-self-3-third,3shot,all,0.939,180,0.287,0.018,,
llama31-70b,dx-self-3-third,3shot,free,0.952,126,0.286,0.019,,
llama31-70b,dx-self-2-deictic,0shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.622;tie=0.006
llama31-70b,dx-self-2-deictic,0shot,free,0.373,126,0.286,0.043,,speaker_confusion=0.619;tie=0.008
llama31-70b,dx-self-2-deictic,3shot_collide,all,0.606,180,0.287,0.036,,speaker_confusion=0.372;tie=0.022
llama31-70b,dx-self-2-deictic,3shot_collide,free,0.682,126,0.286,0.042,,speaker_confusion=0.302;tie=0.016
llama31-70b,dx-self-2-deictic,3shot,all,0.650,180,0.287,0.036,,speaker_confusion=0.333;tie=0.017
llama31-70b,dx-self-2-deictic,3shot,free,0.722,126,0.286,0.040,,speaker_confusion=0.27;tie=0.008
llama31-70b,dx-self-2-third,0shot,all,0.844,180,0.287,0.027,,
llama31-70b,dx-self-2-third,0shot,free,0.825,126,0.286,0.034,,
llama31-70b,dx-self-2-third,3shot_collide,all,0.944,180,0.287,0.017,,
llama31-70b,dx-self-2-third,3shot_collide,free,0.944,126,0.286,0.020,,
llama31-70b,dx-self-2-third,3shot,all,0.933,180,0.287,0.019,,
llama31-70b,dx-self-2-third,3shot,free,0.936,126,0.286,0.022,,
llama31-70b,dx-addressee-3-deictic,0shot,all,0.522,180,0.287,0.037,,other_object=0.044;previous_owner=0.078;speaker_confusion=0.339;tie=0.017
llama31-70b,dx-addressee-3-deictic,0shot,free,0.500,126,0.286,0.044,,other_object=0.04;previous_owner=0.095;speaker_confusion=0.357;tie=0.008
llama31-70b,dx-addressee-3-deictic,3shot_collide,all,0.517,180,0.287,0.037,,other_object=0.161;previous_owner=0.039;speaker_confusion=0.272;tie=0.011
llama31-70b,dx-addressee-3-deictic,3shot_collide,free,0.619,126,0.286,0.043,,other_object=0.127;previous_owner=0.032;speaker_confusion=0.222
llama31-70b,dx-addressee-3-deictic,3shot,all,0.506,180,0.287,0.037,,other_object=0.15;previous_owner=0.033;speaker_confusion=0.3;tie=0.011
llama31-70b,dx-addressee-3-deictic,3shot,free,0.635,126,0.286,0.043,,other_object=0.111;previous_owner=0.04;speaker_confusion=0.198;tie=0.016
llama31-70b,dx-addressee-3-third,0shot,all,0.800,180,0.287,0.030,,
llama31-70b,dx-addressee-3-third,0shot,free,0.730,126,0.286,0.040,,
llama31-70b,dx-addressee-3-third,3shot_collide,all,0.928,180,0.287,0.019,,
llama31-70b,dx-addressee-3-third,3shot_collide,free,0.929,126,0.286,0.023,,
llama31-70b,dx-addressee-3-third,3shot,all,0.933,180,0.287,0.019,,
llama31-70b,dx-addressee-3-third,3shot,free,0.952,126,0.286,0.019,,
llama31-70b,dx-addressee-2-deictic,0shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.644;tie=0.028
llama31-70b,dx-addressee-2-deictic,0shot,free,0.246,126,0.286,0.038,,speaker_confusion=0.722;tie=0.032
llama31-70b,dx-addressee-2-deictic,3shot_collide,all,0.350,180,0.287,0.036,,speaker_confusion=0.617;tie=0.033
llama31-70b,dx-addressee-2-deictic,3shot_collide,free,0.381,126,0.286,0.043,,speaker_confusion=0.595;tie=0.024
llama31-70b,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.611;tie=0.033
llama31-70b,dx-addressee-2-deictic,3shot,free,0.389,126,0.286,0.043,,speaker_confusion=0.579;tie=0.032
llama31-70b,dx-addressee-2-third,0shot,all,0.822,180,0.287,0.029,,
llama31-70b,dx-addressee-2-third,0shot,free,0.833,126,0.286,0.033,,
llama31-70b,dx-addressee-2-third,3shot_collide,all,0.917,180,0.287,0.021,,
llama31-70b,dx-addressee-2-third,3shot_collide,free,0.952,126,0.286,0.019,,
llama31-70b,dx-addressee-2-third,3shot,all,0.917,180,0.287,0.021,,
llama31-70b,dx-addressee-2-third,3shot,free,0.936,126,0.286,0.022,,
llama31-70b,sb-k1,0shot,all,0.312,64,0.125,0.058,,
llama31-70b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
llama31-70b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
llama31-70b,sb-k23,0shot,all,0.180,128,0.027,0.034,,
llama31-70b,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
llama31-70b,sb-k23,3shot,all,0.992,128,0.027,0.008,,
llama31-70b,sb-split,0shot,all,0.641,128,0.027,0.042,,
llama31-70b,sb-split,3shot_collide,all,0.922,128,0.027,0.024,,
llama31-70b,sb-split,3shot,all,0.922,128,0.027,0.024,,
llama31-70b,sb-dyn,0shot,all,0.594,256,0.027,0.031,,
llama31-70b,sb-dyn,3shot_collide,all,0.637,256,0.027,0.030,,
llama31-70b,sb-dyn,3shot,all,0.562,256,0.027,0.031,,
llama31-70b,an-narr,0shot,all,0.588,640,0.250,0.019,,
llama31-70b,an-narr,3shot_collide,all,0.995,640,0.250,0.003,,
llama31-70b,an-narr,3shot,all,0.994,640,0.250,0.003,,
llama31-70b,an-other-any,0shot,all,0.447,320,0.250,0.028,,
llama31-70b,an-other-any,3shot_collide,all,0.925,320,0.250,0.015,,
llama31-70b,an-other-any,3shot,all,0.828,320,0.250,0.021,,
llama31-70b,an-me-any,0shot,all,0.650,160,0.250,0.038,,
llama31-70b,an-me-any,3shot_collide,all,0.912,160,0.250,0.022,,
llama31-70b,an-me-any,3shot,all,0.894,160,0.250,0.024,,
llama31-70b,an-you-voc,0shot,all,0.375,80,0.250,0.054,,
llama31-70b,an-you-voc,3shot_collide,all,0.938,80,0.250,0.027,,
llama31-70b,an-you-voc,3shot,all,0.950,80,0.250,0.024,,
llama31-70b,an-you-novoc,0shot,all,0.425,80,0.250,0.055,,
llama31-70b,an-you-novoc,3shot_collide,all,0.613,80,0.250,0.054,,
llama31-70b,an-you-novoc,3shot,all,0.475,80,0.250,0.056,,
llama31-70b,adj-adj,0shot,all,0.992,260,0.333,0.005,,
llama31-70b,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-70b,adj-adj,3shot,all,1.000,260,0.333,0.000,,
llama31-70b,adj-voc,0shot,all,0.989,260,0.333,0.007,,
llama31-70b,adj-voc,3shot_collide,all,0.996,260,0.333,0.004,,
llama31-70b,adj-voc,3shot,all,1.000,260,0.333,0.000,,
llama31-70b,adj-third,0shot,all,0.992,260,0.333,0.005,,
llama31-70b,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-70b,adj-third,3shot,all,1.000,260,0.333,0.000,,
llama31-70b,cm-mutate,0shot,all,0.970,200,0.200,0.012,,
llama31-70b,cm-mutate,3shot_collide,all,0.975,200,0.200,0.011,,
llama31-70b,cm-mutate,3shot,all,0.955,200,0.200,0.015,,
llama31-70b,cm-rebind,0shot,all,0.970,200,0.200,0.012,,
llama31-70b,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
llama31-70b,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
llama31-70b,cm-copy,0shot,all,0.550,200,0.200,0.035,,
llama31-70b,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
llama31-70b,cm-copy,3shot,all,1.000,200,0.200,0.000,,
qwen25-72b,attr,0shot,all,0.952,1210,0.168,0.006,,
qwen25-72b,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
qwen25-72b,attr,3shot,all,1.000,1210,0.168,0.000,,
qwen25-72b,codevar,0shot,all,0.743,1200,0.267,0.013,,
qwen25-72b,codevar,3shot_collide,all,0.910,1200,0.267,0.008,,
qwen25-72b,codevar,3shot,all,0.884,1200,0.267,0.009,,
qwen25-72b,dialogue,0shot,all,0.936,1200,0.312,0.007,,
qwen25-72b,dialogue,3shot_collide,all,0.978,1200,0.312,0.004,,
qwen25-72b,dialogue,3shot,all,0.975,1200,0.312,0.004,,
qwen25-72b,dx-owner-3-deictic,0shot,all,0.675,120,0.333,0.043,,other_participant=0.308;speaker_confusion=0.017
qwen25-72b,dx-owner-3-deictic,0shot,free,0.600,40,0.333,0.077,,other_participant=0.4
qwen25-72b,dx-owner-3-deictic,3shot_collide,all,0.983,120,0.333,0.012,,other_participant=0.017
qwen25-72b,dx-owner-3-deictic,3shot_collide,free,0.975,40,0.333,0.025,,other_participant=0.025
qwen25-72b,dx-owner-3-deictic,3shot,all,0.983,120,0.333,0.012,,other_participant=0.017
qwen25-72b,dx-owner-3-deictic,3shot,free,0.975,40,0.333,0.025,,other_participant=0.025
qwen25-72b,dx-owner-3-third,0shot,all,0.783,120,0.333,0.038,,
qwen25-72b,dx-owner-3-third,0shot,free,0.725,40,0.333,0.071,,
qwen25-72b,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
qwen25-72b,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
qwen25-72b,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
qwen25-72b,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
qwen25-72b,dx-self-3-deictic,0shot,all,0.594,180,0.287,0.037,,other_object=0.15;previous_owner=0.044;speaker_confusion=0.2;tie=0.011
qwen25-72b,dx-self-3-deictic,0shot,free,0.603,126,0.286,0.044,,other_object=0.183;previous_owner=0.024;speaker_confusion=0.175;tie=0.016
qwen25-72b,dx-self-3-deictic,3shot_collide,all,0.867,180,0.287,0.025,,other_object=0.011;previous_owner=0.028;speaker_confusion=0.094
qwen25-72b,dx-self-3-deictic,3shot_collide,free,0.897,126,0.286,0.027,,previous_owner=0.024;speaker_confusion=0.079
qwen25-72b,dx-self-3-deictic,3shot,all,0.817,180,0.287,0.029,,other_object=0.033;previous_owner=0.044;speaker_confusion=0.1;tie=0.006
qwen25-72b,dx-self-3-deictic,3shot,free,0.865,126,0.286,0.030,,other_object=0.032;previous_owner=0.032;speaker_confusion=0.071
qwen25-72b,dx-self-3-third,0shot,all,0.894,180,0.287,0.023,,
qwen25-72b,dx-self-3-third,0shot,free,0.921,126,0.286,0.024,,
qwen25-72b,dx-self-3-third,3shot_collide,all,0.961,180,0.287,0.014,,
qwen25-72b,dx-self-3-third,3shot_collide,free,0.984,126,0.286,0.011,,
qwen25-72b,dx-self-3-third,3shot,all,0.972,180,0.287,0.012,,
qwen25-72b,dx-self-3-third,3shot,free,0.992,126,0.286,0.008,,
qwen25-72b,dx-self-2-deictic,0shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.6;tie=0.028
qwen25-72b,dx-self-2-deictic,0shot,free,0.373,126,0.286,0.043,,speaker_confusion=0.611;tie=0.016
qwen25-72b,dx-self-2-deictic,3shot_collide,all,0.611,180,0.287,0.036,,speaker_confusion=0.372;tie=0.017
qwen25-72b,dx-self-2-deictic,3shot_collide,free,0.667,126,0.286,0.042,,speaker_confusion=0.317;tie=0.016
qwen25-72b,dx-self-2-deictic,3shot,all,0.628,180,0.287,0.036,,speaker_confusion=0.367;tie=0.006
qwen25-72b,dx-self-2-deictic,3shot,free,0.691,126,0.286,0.041,,speaker_confusion=0.302;tie=0.008
qwen25-72b,dx-self-2-third,0shot,all,0.844,180,0.287,0.027,,
qwen25-72b,dx-self-2-third,0shot,free,0.873,126,0.286,0.030,,
qwen25-72b,dx-self-2-third,3shot_collide,all,0.956,180,0.287,0.015,,
qwen25-72b,dx-self-2-third,3shot_collide,free,0.936,126,0.286,0.022,,
qwen25-72b,dx-self-2-third,3shot,all,0.950,180,0.287,0.016,,
qwen25-72b,dx-self-2-third,3shot,free,0.952,126,0.286,0.019,,
qwen25-72b,dx-addressee-3-deictic,0shot,all,0.494,180,0.287,0.037,,other_object=0.078;previous_owner=0.083;speaker_confusion=0.328;tie=0.017
qwen25-72b,dx-addressee-3-deictic,0shot,free,0.516,126,0.286,0.044,,other_object=0.079;previous_owner=0.103;speaker_confusion=0.302
qwen25-72b,dx-addressee-3-deictic,3shot_collide,all,0.744,180,0.287,0.033,,other_object=0.033;previous_owner=0.028;speaker_confusion=0.178;tie=0.017
qwen25-72b,dx-addressee-3-deictic,3shot_collide,free,0.730,126,0.286,0.040,,other_object=0.024;previous_owner=0.024;speaker_confusion=0.222
qwen25-72b,dx-addressee-3-deictic,3shot,all,0.683,180,0.287,0.035,,other_object=0.039;previous_owner=0.05;speaker_confusion=0.222;tie=0.006
qwen25-72b,dx-addressee-3-deictic,3shot,free,0.675,126,0.286,0.042,,other_object=0.032;previous_owner=0.063;speaker_confusion=0.23
qwen25-72b,dx-addressee-3-third,0shot,all,0.900,180,0.287,0.022,,
qwen25-72b,dx-addressee-3-third,0shot,free,0.921,126,0.286,0.024,,
qwen25-72b,dx-addressee-3-third,3shot_collide,all,0.989,180,0.287,0.008,,
qwen25-72b,dx-addressee-3-third,3shot_collide,free,0.984,126,0.286,0.011,,
qwen25-72b,dx-addressee-3-third,3shot,all,0.967,180,0.287,0.013,,
qwen25-72b,dx-addressee-3-third,3shot,free,0.968,126,0.286,0.016,,
qwen25-72b,dx-addressee-2-deictic,0shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.661;tie=0.017
qwen25-72b,dx-addressee-2-deictic,0shot,free,0.270,126,0.286,0.040,,speaker_confusion=0.714;tie=0.016
qwen25-72b,dx-addressee-2-deictic,3shot_collide,all,0.422,180,0.287,0.037,,speaker_confusion=0.567;tie=0.011
qwen25-72b,dx-addressee-2-deictic,3shot_collide,free,0.429,126,0.286,0.044,,speaker_confusion=0.563;tie=0.008
qwen25-72b,dx-addressee-2-deictic,3shot,all,0.383,180,0.287,0.036,,speaker_confusion=0.611;tie=0.006
qwen25-72b,dx-addressee-2-deictic,3shot,free,0.381,126,0.286,0.043,,speaker_confusion=0.611;tie=0.008
qwen25-72b,dx-addressee-2-third,0shot,all,0.839,180,0.287,0.027,,
qwen25-72b,dx-addressee-2-third,0shot,free,0.881,126,0.286,0.029,,
qwen25-72b,dx-addressee-2-third,3shot_collide,all,0.933,180,0.287,0.019,,
qwen25-72b,dx-addressee-2-third,3shot_collide,free,0.936,126,0.286,0.022,,
qwen25-72b,dx-addressee-2-third,3shot,all,0.928,180,0.287,0.019,,
qwen25-72b,dx-addressee-2-third,3shot,free,0.936,126,0.286,0.022,,
qwen25-72b,sb-k1,0shot,all,1.000,64,0.125,0.000,,
qwen25-72b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
qwen25-72b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
qwen25-72b,sb-k23,0shot,all,0.773,128,0.027,0.037,,
qwen25-72b,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
qwen25-72b,sb-k23,3shot,all,1.000,128,0.027,0.000,,
qwen25-72b,sb-split,0shot,all,0.875,128,0.027,0.029,,
qwen25-72b,sb-split,3shot_collide,all,0.977,128,0.027,0.013,,
qwen25-72b,sb-split,3shot,all,0.992,128,0.027,0.008,,
qwen25-72b,sb-dyn,0shot,all,0.695,256,0.027,0.029,,
qwen25-72b,sb-dyn,3shot_collide,all,0.777,256,0.027,0.026,,
qwen25-72b,sb-dyn,3shot,all,0.750,256,0.027,0.027,,
qwen25-72b,an-narr,0shot,all,0.630,640,0.250,0.019,,
qwen25-72b,an-narr,3shot_collide,all,0.994,640,0.250,0.003,,
qwen25-72b,an-narr,3shot,all,0.989,640,0.250,0.004,,
qwen25-72b,an-other-any,0shot,all,0.297,320,0.250,0.025,,
qwen25-72b,an-other-any,3shot_collide,all,0.944,320,0.250,0.013,,
qwen25-72b,an-other-any,3shot,all,0.894,320,0.250,0.017,,
qwen25-72b,an-me-any,0shot,all,0.719,160,0.250,0.035,,
qwen25-72b,an-me-any,3shot_collide,all,0.969,160,0.250,0.014,,
qwen25-72b,an-me-any,3shot,all,0.919,160,0.250,0.022,,
qwen25-72b,an-you-voc,0shot,all,0.738,80,0.250,0.049,,
qwen25-72b,an-you-voc,3shot_collide,all,0.925,80,0.250,0.029,,
qwen25-72b,an-you-voc,3shot,all,0.988,80,0.250,0.012,,
qwen25-72b,an-you-novoc,0shot,all,0.537,80,0.250,0.056,,
qwen25-72b,an-you-novoc,3shot_collide,all,0.800,80,0.250,0.045,,
qwen25-72b,an-you-novoc,3shot,all,0.812,80,0.250,0.044,,
qwen25-72b,adj-adj,0shot,all,0.992,260,0.333,0.005,,
qwen25-72b,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
qwen25-72b,adj-adj,3shot,all,1.000,260,0.333,0.000,,
qwen25-72b,adj-voc,0shot,all,0.996,260,0.333,0.004,,
qwen25-72b,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
qwen25-72b,adj-voc,3shot,all,1.000,260,0.333,0.000,,
qwen25-72b,adj-third,0shot,all,0.996,260,0.333,0.004,,
qwen25-72b,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
qwen25-72b,adj-third,3shot,all,1.000,260,0.333,0.000,,
qwen25-72b,cm-mutate,0shot,all,0.990,200,0.200,0.007,,
qwen25-72b,cm-mutate,3shot_collide,all,0.995,200,0.200,0.005,,
qwen25-72b,cm-mutate,3shot,all,0.995,200,0.200,0.005,,
qwen25-72b,cm-rebind,0shot,all,0.825,200,0.200,0.027,,
qwen25-72b,cm-rebind,3shot_collide,all,0.975,200,0.200,0.011,,
qwen25-72b,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
qwen25-72b,cm-copy,0shot,all,0.690,200,0.200,0.033,,
qwen25-72b,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
qwen25-72b,cm-copy,3shot,all,0.995,200,0.200,0.005,,
dsv4-flash-base,attr,0shot,all,0.749,1210,0.168,0.013,,
dsv4-flash-base,attr,3shot_collide,all,0.996,1210,0.168,0.002,,
dsv4-flash-base,attr,3shot,all,0.999,1210,0.168,0.001,,
dsv4-flash-base,codevar,0shot,all,0.899,1200,0.267,0.009,,
dsv4-flash-base,codevar,3shot_collide,all,0.943,1200,0.267,0.007,,
dsv4-flash-base,codevar,3shot,all,0.920,1200,0.267,0.008,,
dsv4-flash-base,dialogue,0shot,all,0.916,1200,0.312,0.008,,
dsv4-flash-base,dialogue,3shot_collide,all,0.985,1200,0.312,0.004,,
dsv4-flash-base,dialogue,3shot,all,0.983,1200,0.312,0.004,,
dsv4-flash-base,dx-owner-3-deictic,0shot,all,0.750,120,0.333,0.040,,other_participant=0.208;speaker_confusion=0.017;tie=0.025
dsv4-flash-base,dx-owner-3-deictic,0shot,free,0.725,40,0.333,0.071,,other_participant=0.25;tie=0.025
dsv4-flash-base,dx-owner-3-deictic,3shot_collide,all,0.975,120,0.333,0.014,,other_participant=0.008;speaker_confusion=0.017
dsv4-flash-base,dx-owner-3-deictic,3shot_collide,free,0.975,40,0.333,0.025,,speaker_confusion=0.025
dsv4-flash-base,dx-owner-3-deictic,3shot,all,0.967,120,0.333,0.016,,other_participant=0.033
dsv4-flash-base,dx-owner-3-deictic,3shot,free,0.975,40,0.333,0.025,,other_participant=0.025
dsv4-flash-base,dx-owner-3-third,0shot,all,0.850,120,0.333,0.033,,
dsv4-flash-base,dx-owner-3-third,0shot,free,0.775,40,0.333,0.066,,
dsv4-flash-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
dsv4-flash-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
dsv4-flash-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
dsv4-flash-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
dsv4-flash-base,dx-self-3-deictic,0shot,all,0.650,180,0.287,0.036,,other_object=0.133;previous_owner=0.017;speaker_confusion=0.161;tie=0.039
dsv4-flash-base,dx-self-3-deictic,0shot,free,0.603,126,0.286,0.044,,other_object=0.175;previous_owner=0.008;speaker_confusion=0.167;tie=0.048
dsv4-flash-base,dx-self-3-deictic,3shot_collide,all,0.922,180,0.287,0.020,,other_object=0.006;previous_owner=0.011;speaker_confusion=0.061
dsv4-flash-base,dx-self-3-deictic,3shot_collide,free,0.929,126,0.286,0.023,,other_object=0.008;previous_owner=0.008;speaker_confusion=0.056
dsv4-flash-base,dx-self-3-deictic,3shot,all,0.900,180,0.287,0.022,,other_object=0.017;previous_owner=0.017;speaker_confusion=0.056;tie=0.011
dsv4-flash-base,dx-self-3-deictic,3shot,free,0.913,126,0.286,0.025,,other_object=0.024;previous_owner=0.016;speaker_confusion=0.04;tie=0.008
dsv4-flash-base,dx-self-3-third,0shot,all,0.883,180,0.287,0.024,,
dsv4-flash-base,dx-self-3-third,0shot,free,0.857,126,0.286,0.031,,
dsv4-flash-base,dx-self-3-third,3shot_collide,all,0.989,180,0.287,0.008,,
dsv4-flash-base,dx-self-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
dsv4-flash-base,dx-self-3-third,3shot,all,0.994,180,0.287,0.005,,
dsv4-flash-base,dx-self-3-third,3shot,free,1.000,126,0.286,0.000,,
dsv4-flash-base,dx-self-2-deictic,0shot,all,0.544,180,0.287,0.037,,speaker_confusion=0.422;tie=0.033
dsv4-flash-base,dx-self-2-deictic,0shot,free,0.516,126,0.286,0.044,,speaker_confusion=0.437;tie=0.048
dsv4-flash-base,dx-self-2-deictic,3shot_collide,all,0.711,180,0.287,0.034,,speaker_confusion=0.272;tie=0.017
dsv4-flash-base,dx-self-2-deictic,3shot_collide,free,0.730,126,0.286,0.040,,speaker_confusion=0.254;tie=0.016
dsv4-flash-base,dx-self-2-deictic,3shot,all,0.700,180,0.287,0.034,,speaker_confusion=0.283;tie=0.017
dsv4-flash-base,dx-self-2-deictic,3shot,free,0.730,126,0.286,0.040,,speaker_confusion=0.246;tie=0.024
dsv4-flash-base,dx-self-2-third,0shot,all,0.889,180,0.287,0.023,,
dsv4-flash-base,dx-self-2-third,0shot,free,0.889,126,0.286,0.028,,
dsv4-flash-base,dx-self-2-third,3shot_collide,all,0.961,180,0.287,0.014,,
dsv4-flash-base,dx-self-2-third,3shot_collide,free,0.984,126,0.286,0.011,,
dsv4-flash-base,dx-self-2-third,3shot,all,0.972,180,0.287,0.012,,
dsv4-flash-base,dx-self-2-third,3shot,free,0.984,126,0.286,0.011,,
dsv4-flash-base,dx-addressee-3-deictic,0shot,all,0.517,180,0.287,0.037,,other_object=0.128;previous_owner=0.028;speaker_confusion=0.306;tie=0.022
dsv4-flash-base,dx-addressee-3-deictic,0shot,free,0.508,126,0.286,0.044,,other_object=0.135;previous_owner=0.04;speaker_confusion=0.302;tie=0.016
dsv4-flash-base,dx-addressee-3-deictic,3shot_collide,all,0.700,180,0.287,0.034,,other_object=0.056;previous_owner=0.044;speaker_confusion=0.172;tie=0.028
dsv4-flash-base,dx-addressee-3-deictic,3shot_collide,free,0.738,126,0.286,0.039,,other_object=0.032;previous_owner=0.048;speaker_confusion=0.159;tie=0.024
dsv4-flash-base,dx-addressee-3-deictic,3shot,all,0.650,180,0.287,0.036,,other_object=0.05;previous_owner=0.05;speaker_confusion=0.228;tie=0.022
dsv4-flash-base,dx-addressee-3-deictic,3shot,free,0.659,126,0.286,0.042,,other_object=0.032;previous_owner=0.071;speaker_confusion=0.214;tie=0.024
dsv4-flash-base,dx-addressee-3-third,0shot,all,0.922,180,0.287,0.020,,
dsv4-flash-base,dx-addressee-3-third,0shot,free,0.897,126,0.286,0.027,,
dsv4-flash-base,dx-addressee-3-third,3shot_collide,all,0.978,180,0.287,0.011,,
dsv4-flash-base,dx-addressee-3-third,3shot_collide,free,0.984,126,0.286,0.011,,
dsv4-flash-base,dx-addressee-3-third,3shot,all,1.000,180,0.287,0.000,,
dsv4-flash-base,dx-addressee-3-third,3shot,free,1.000,126,0.286,0.000,,
dsv4-flash-base,dx-addressee-2-deictic,0shot,all,0.444,180,0.287,0.037,,speaker_confusion=0.539;tie=0.017
dsv4-flash-base,dx-addressee-2-deictic,0shot,free,0.286,126,0.286,0.040,,speaker_confusion=0.698;tie=0.016
dsv4-flash-base,dx-addressee-2-deictic,3shot_collide,all,0.417,180,0.287,0.037,,speaker_confusion=0.556;tie=0.028
dsv4-flash-base,dx-addressee-2-deictic,3shot_collide,free,0.397,126,0.286,0.044,,speaker_confusion=0.563;tie=0.04
dsv4-flash-base,dx-addressee-2-deictic,3shot,all,0.422,180,0.287,0.037,,speaker_confusion=0.556;tie=0.022
dsv4-flash-base,dx-addressee-2-deictic,3shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.571;tie=0.024
dsv4-flash-base,dx-addressee-2-third,0shot,all,0.900,180,0.287,0.022,,
dsv4-flash-base,dx-addressee-2-third,0shot,free,0.905,126,0.286,0.026,,
dsv4-flash-base,dx-addressee-2-third,3shot_collide,all,0.933,180,0.287,0.019,,
dsv4-flash-base,dx-addressee-2-third,3shot_collide,free,0.944,126,0.286,0.020,,
dsv4-flash-base,dx-addressee-2-third,3shot,all,0.933,180,0.287,0.019,,
dsv4-flash-base,dx-addressee-2-third,3shot,free,0.968,126,0.286,0.016,,
dsv4-flash-base,sb-k1,0shot,all,0.875,64,0.125,0.041,,
dsv4-flash-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
dsv4-flash-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
dsv4-flash-base,sb-k23,0shot,all,0.578,128,0.027,0.044,,
dsv4-flash-base,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
dsv4-flash-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
dsv4-flash-base,sb-split,0shot,all,0.820,128,0.027,0.034,,
dsv4-flash-base,sb-split,3shot_collide,all,0.984,128,0.027,0.011,,
dsv4-flash-base,sb-split,3shot,all,0.961,128,0.027,0.017,,
dsv4-flash-base,sb-dyn,0shot,all,0.809,256,0.027,0.025,,
dsv4-flash-base,sb-dyn,3shot_collide,all,0.894,256,0.027,0.019,,
dsv4-flash-base,sb-dyn,3shot,all,0.867,256,0.027,0.021,,
dsv4-flash-base,an-narr,0shot,all,0.694,640,0.250,0.018,,
dsv4-flash-base,an-narr,3shot_collide,all,0.989,640,0.250,0.004,,
dsv4-flash-base,an-narr,3shot,all,0.986,640,0.250,0.005,,
dsv4-flash-base,an-other-any,0shot,all,0.553,320,0.250,0.028,,
dsv4-flash-base,an-other-any,3shot_collide,all,0.925,320,0.250,0.015,,
dsv4-flash-base,an-other-any,3shot,all,0.884,320,0.250,0.018,,
dsv4-flash-base,an-me-any,0shot,all,0.675,160,0.250,0.037,,
dsv4-flash-base,an-me-any,3shot_collide,all,0.800,160,0.250,0.032,,
dsv4-flash-base,an-me-any,3shot,all,0.600,160,0.250,0.039,,
dsv4-flash-base,an-you-voc,0shot,all,0.263,80,0.250,0.049,,
dsv4-flash-base,an-you-voc,3shot_collide,all,0.900,80,0.250,0.034,,
dsv4-flash-base,an-you-voc,3shot,all,0.800,80,0.250,0.045,,
dsv4-flash-base,an-you-novoc,0shot,all,0.200,80,0.250,0.045,,
dsv4-flash-base,an-you-novoc,3shot_collide,all,0.650,80,0.250,0.053,,
dsv4-flash-base,an-you-novoc,3shot,all,0.588,80,0.250,0.055,,
dsv4-flash-base,adj-adj,0shot,all,0.981,260,0.333,0.009,,
dsv4-flash-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-voc,0shot,all,0.992,260,0.333,0.005,,
dsv4-flash-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-flash-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
dsv4-flash-base,cm-mutate,0shot,all,0.865,200,0.200,0.024,,
dsv4-flash-base,cm-mutate,3shot_collide,all,0.970,200,0.200,0.012,,
dsv4-flash-base,cm-mutate,3shot,all,0.950,200,0.200,0.015,,
dsv4-flash-base,cm-rebind,0shot,all,0.915,200,0.200,0.020,,
dsv4-flash-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
dsv4-flash-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
dsv4-flash-base,cm-copy,0shot,all,0.890,200,0.200,0.022,,
dsv4-flash-base,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
dsv4-flash-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
mimo-v25-base,attr,0shot,all,0.717,1210,0.168,0.013,,
mimo-v25-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
mimo-v25-base,attr,3shot,all,0.999,1210,0.168,0.001,,
mimo-v25-base,codevar,0shot,all,0.792,1200,0.267,0.012,,
mimo-v25-base,codevar,3shot_collide,all,0.908,1200,0.267,0.008,,
mimo-v25-base,codevar,3shot,all,0.921,1200,0.267,0.008,,
mimo-v25-base,dialogue,0shot,all,0.940,1200,0.312,0.007,,
mimo-v25-base,dialogue,3shot_collide,all,0.963,1200,0.312,0.005,,
mimo-v25-base,dialogue,3shot,all,0.963,1200,0.312,0.005,,
mimo-v25-base,dx-owner-3-deictic,0shot,all,0.625,120,0.333,0.044,,other_participant=0.342;speaker_confusion=0.008;tie=0.025
mimo-v25-base,dx-owner-3-deictic,0shot,free,0.475,40,0.333,0.079,,other_participant=0.5;tie=0.025
mimo-v25-base,dx-owner-3-deictic,3shot_collide,all,0.992,120,0.333,0.008,,other_participant=0.008
mimo-v25-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
mimo-v25-base,dx-owner-3-deictic,3shot,all,0.992,120,0.333,0.008,,other_participant=0.008
mimo-v25-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
mimo-v25-base,dx-owner-3-third,0shot,all,0.775,120,0.333,0.038,,
mimo-v25-base,dx-owner-3-third,0shot,free,0.575,40,0.333,0.078,,
mimo-v25-base,dx-owner-3-third,3shot_collide,all,0.992,120,0.333,0.008,,
mimo-v25-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
mimo-v25-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
mimo-v25-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
mimo-v25-base,dx-self-3-deictic,0shot,all,0.428,180,0.287,0.037,,other_object=0.189;previous_owner=0.072;speaker_confusion=0.283;tie=0.028
mimo-v25-base,dx-self-3-deictic,0shot,free,0.436,126,0.286,0.044,,other_object=0.214;previous_owner=0.04;speaker_confusion=0.286;tie=0.024
mimo-v25-base,dx-self-3-deictic,3shot_collide,all,0.772,180,0.287,0.031,,other_object=0.056;previous_owner=0.05;speaker_confusion=0.106;tie=0.017
mimo-v25-base,dx-self-3-deictic,3shot_collide,free,0.849,126,0.286,0.032,,other_object=0.04;previous_owner=0.032;speaker_confusion=0.071;tie=0.008
mimo-v25-base,dx-self-3-deictic,3shot,all,0.811,180,0.287,0.029,,other_object=0.044;previous_owner=0.033;speaker_confusion=0.106;tie=0.006
mimo-v25-base,dx-self-3-deictic,3shot,free,0.889,126,0.286,0.028,,other_object=0.032;previous_owner=0.008;speaker_confusion=0.071
mimo-v25-base,dx-self-3-third,0shot,all,0.778,180,0.287,0.031,,
mimo-v25-base,dx-self-3-third,0shot,free,0.794,126,0.286,0.036,,
mimo-v25-base,dx-self-3-third,3shot_collide,all,0.933,180,0.287,0.019,,
mimo-v25-base,dx-self-3-third,3shot_collide,free,0.929,126,0.286,0.023,,
mimo-v25-base,dx-self-3-third,3shot,all,0.939,180,0.287,0.018,,
mimo-v25-base,dx-self-3-third,3shot,free,0.929,126,0.286,0.023,,
mimo-v25-base,dx-self-2-deictic,0shot,all,0.428,180,0.287,0.037,,speaker_confusion=0.556;tie=0.017
mimo-v25-base,dx-self-2-deictic,0shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.5;tie=0.016
mimo-v25-base,dx-self-2-deictic,3shot_collide,all,0.639,180,0.287,0.036,,speaker_confusion=0.361
mimo-v25-base,dx-self-2-deictic,3shot_collide,free,0.730,126,0.286,0.040,,speaker_confusion=0.27
mimo-v25-base,dx-self-2-deictic,3shot,all,0.656,180,0.287,0.035,,speaker_confusion=0.333;tie=0.011
mimo-v25-base,dx-self-2-deictic,3shot,free,0.722,126,0.286,0.040,,speaker_confusion=0.27;tie=0.008
mimo-v25-base,dx-self-2-third,0shot,all,0.911,180,0.287,0.021,,
mimo-v25-base,dx-self-2-third,0shot,free,0.960,126,0.286,0.017,,
mimo-v25-base,dx-self-2-third,3shot_collide,all,0.917,180,0.287,0.021,,
mimo-v25-base,dx-self-2-third,3shot_collide,free,0.936,126,0.286,0.022,,
mimo-v25-base,dx-self-2-third,3shot,all,0.917,180,0.287,0.021,,
mimo-v25-base,dx-self-2-third,3shot,free,0.936,126,0.286,0.022,,
mimo-v25-base,dx-addressee-3-deictic,0shot,all,0.394,180,0.287,0.036,,other_object=0.167;previous_owner=0.056;speaker_confusion=0.35;tie=0.033
mimo-v25-base,dx-addressee-3-deictic,0shot,free,0.397,126,0.286,0.044,,other_object=0.159;previous_owner=0.071;speaker_confusion=0.341;tie=0.032
mimo-v25-base,dx-addressee-3-deictic,3shot_collide,all,0.556,180,0.287,0.037,,other_object=0.078;previous_owner=0.089;speaker_confusion=0.272;tie=0.006
mimo-v25-base,dx-addressee-3-deictic,3shot_collide,free,0.595,126,0.286,0.044,,other_object=0.032;previous_owner=0.111;speaker_confusion=0.254;tie=0.008
mimo-v25-base,dx-addressee-3-deictic,3shot,all,0.561,180,0.287,0.037,,other_object=0.061;previous_owner=0.094;speaker_confusion=0.261;tie=0.022
mimo-v25-base,dx-addressee-3-deictic,3shot,free,0.595,126,0.286,0.044,,other_object=0.032;previous_owner=0.111;speaker_confusion=0.238;tie=0.024
mimo-v25-base,dx-addressee-3-third,0shot,all,0.822,180,0.287,0.029,,
mimo-v25-base,dx-addressee-3-third,0shot,free,0.833,126,0.286,0.033,,
mimo-v25-base,dx-addressee-3-third,3shot_collide,all,0.950,180,0.287,0.016,,
mimo-v25-base,dx-addressee-3-third,3shot_collide,free,0.968,126,0.286,0.016,,
mimo-v25-base,dx-addressee-3-third,3shot,all,0.944,180,0.287,0.017,,
mimo-v25-base,dx-addressee-3-third,3shot,free,0.960,126,0.286,0.017,,
mimo-v25-base,dx-addressee-2-deictic,0shot,all,0.256,180,0.287,0.033,,speaker_confusion=0.683;tie=0.061
mimo-v25-base,dx-addressee-2-deictic,0shot,free,0.254,126,0.286,0.039,,speaker_confusion=0.667;tie=0.079
mimo-v25-base,dx-addressee-2-deictic,3shot_collide,all,0.361,180,0.287,0.036,,speaker_confusion=0.622;tie=0.017
mimo-v25-base,dx-addressee-2-deictic,3shot_collide,free,0.365,126,0.286,0.043,,speaker_confusion=0.611;tie=0.024
mimo-v25-base,dx-addressee-2-deictic,3shot,all,0.400,180,0.287,0.036,,speaker_confusion=0.572;tie=0.028
mimo-v25-base,dx-addressee-2-deictic,3shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.563;tie=0.016
mimo-v25-base,dx-addressee-2-third,0shot,all,0.894,180,0.287,0.023,,
mimo-v25-base,dx-addressee-2-third,0shot,free,0.913,126,0.286,0.025,,
mimo-v25-base,dx-addressee-2-third,3shot_collide,all,0.872,180,0.287,0.025,,
mimo-v25-base,dx-addressee-2-third,3shot_collide,free,0.936,126,0.286,0.022,,
mimo-v25-base,dx-addressee-2-third,3shot,all,0.889,180,0.287,0.023,,
mimo-v25-base,dx-addressee-2-third,3shot,free,0.936,126,0.286,0.022,,
mimo-v25-base,sb-k1,0shot,all,0.781,64,0.125,0.052,,
mimo-v25-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
mimo-v25-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
mimo-v25-base,sb-k23,0shot,all,0.414,128,0.027,0.043,,
mimo-v25-base,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
mimo-v25-base,sb-k23,3shot,all,0.969,128,0.027,0.015,,
mimo-v25-base,sb-split,0shot,all,0.438,128,0.027,0.044,,
mimo-v25-base,sb-split,3shot_collide,all,0.945,128,0.027,0.020,,
mimo-v25-base,sb-split,3shot,all,0.805,128,0.027,0.035,,
mimo-v25-base,sb-dyn,0shot,all,0.398,256,0.027,0.031,,
mimo-v25-base,sb-dyn,3shot_collide,all,0.598,256,0.027,0.031,,
mimo-v25-base,sb-dyn,3shot,all,0.606,256,0.027,0.030,,
mimo-v25-base,an-narr,0shot,all,0.539,640,0.250,0.020,,
mimo-v25-base,an-narr,3shot_collide,all,0.997,640,0.250,0.002,,
mimo-v25-base,an-narr,3shot,all,0.995,640,0.250,0.003,,
mimo-v25-base,an-other-any,0shot,all,0.300,320,0.250,0.026,,
mimo-v25-base,an-other-any,3shot_collide,all,0.884,320,0.250,0.018,,
mimo-v25-base,an-other-any,3shot,all,0.834,320,0.250,0.021,,
mimo-v25-base,an-me-any,0shot,all,0.688,160,0.250,0.037,,
mimo-v25-base,an-me-any,3shot_collide,all,0.731,160,0.250,0.035,,
mimo-v25-base,an-me-any,3shot,all,0.769,160,0.250,0.033,,
mimo-v25-base,an-you-voc,0shot,all,0.525,80,0.250,0.056,,
mimo-v25-base,an-you-voc,3shot_collide,all,0.900,80,0.250,0.034,,
mimo-v25-base,an-you-voc,3shot,all,0.975,80,0.250,0.018,,
mimo-v25-base,an-you-novoc,0shot,all,0.350,80,0.250,0.053,,
mimo-v25-base,an-you-novoc,3shot_collide,all,0.738,80,0.250,0.049,,
mimo-v25-base,an-you-novoc,3shot,all,0.800,80,0.250,0.045,,
mimo-v25-base,adj-adj,0shot,all,0.923,260,0.333,0.017,,
mimo-v25-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-voc,0shot,all,0.965,260,0.333,0.011,,
mimo-v25-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
mimo-v25-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-base,cm-mutate,0shot,all,0.885,200,0.200,0.023,,
mimo-v25-base,cm-mutate,3shot_collide,all,0.990,200,0.200,0.007,,
mimo-v25-base,cm-mutate,3shot,all,0.980,200,0.200,0.010,,
mimo-v25-base,cm-rebind,0shot,all,0.790,200,0.200,0.029,,
mimo-v25-base,cm-rebind,3shot_collide,all,0.980,200,0.200,0.010,,
mimo-v25-base,cm-rebind,3shot,all,0.995,200,0.200,0.005,,
mimo-v25-base,cm-copy,0shot,all,0.700,200,0.200,0.032,,
mimo-v25-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
mimo-v25-base,cm-copy,3shot,all,0.990,200,0.200,0.007,,
trinity-base,attr,0shot,all,0.737,1210,0.168,0.013,,
trinity-base,attr,3shot_collide,all,0.997,1210,0.168,0.002,,
trinity-base,attr,3shot,all,0.998,1210,0.168,0.001,,
trinity-base,codevar,0shot,all,0.833,1200,0.267,0.011,,
trinity-base,codevar,3shot_collide,all,0.901,1200,0.267,0.009,,
trinity-base,codevar,3shot,all,0.919,1200,0.267,0.008,,
trinity-base,dialogue,0shot,all,0.905,1200,0.312,0.009,,
trinity-base,dialogue,3shot_collide,all,0.961,1200,0.312,0.006,,
trinity-base,dialogue,3shot,all,0.949,1200,0.312,0.006,,
trinity-base,dx-owner-3-deictic,0shot,all,0.642,120,0.333,0.044,,other_participant=0.325;speaker_confusion=0.017;tie=0.017
trinity-base,dx-owner-3-deictic,0shot,free,0.400,40,0.333,0.077,,other_participant=0.525;speaker_confusion=0.025;tie=0.05
trinity-base,dx-owner-3-deictic,3shot_collide,all,0.950,120,0.333,0.020,,other_participant=0.025;speaker_confusion=0.025
trinity-base,dx-owner-3-deictic,3shot_collide,free,0.950,40,0.333,0.035,,other_participant=0.05
trinity-base,dx-owner-3-deictic,3shot,all,0.983,120,0.333,0.012,,other_participant=0.008;speaker_confusion=0.008
trinity-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
trinity-base,dx-owner-3-third,0shot,all,0.917,120,0.333,0.025,,
trinity-base,dx-owner-3-third,0shot,free,0.825,40,0.333,0.060,,
trinity-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
trinity-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
trinity-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
trinity-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
trinity-base,dx-self-3-deictic,0shot,all,0.344,180,0.287,0.035,,other_object=0.25;previous_owner=0.072;speaker_confusion=0.283;tie=0.05
trinity-base,dx-self-3-deictic,0shot,free,0.349,126,0.286,0.043,,other_object=0.262;previous_owner=0.032;speaker_confusion=0.286;tie=0.071
trinity-base,dx-self-3-deictic,3shot_collide,all,0.683,180,0.287,0.035,,other_object=0.044;previous_owner=0.044;speaker_confusion=0.228
trinity-base,dx-self-3-deictic,3shot_collide,free,0.698,126,0.286,0.041,,other_object=0.048;previous_owner=0.024;speaker_confusion=0.23
trinity-base,dx-self-3-deictic,3shot,all,0.728,180,0.287,0.033,,other_object=0.033;previous_owner=0.05;speaker_confusion=0.172;tie=0.017
trinity-base,dx-self-3-deictic,3shot,free,0.770,126,0.286,0.037,,other_object=0.04;previous_owner=0.024;speaker_confusion=0.167
trinity-base,dx-self-3-third,0shot,all,0.817,180,0.287,0.029,,
trinity-base,dx-self-3-third,0shot,free,0.873,126,0.286,0.030,,
trinity-base,dx-self-3-third,3shot_collide,all,0.922,180,0.287,0.020,,
trinity-base,dx-self-3-third,3shot_collide,free,0.952,126,0.286,0.019,,
trinity-base,dx-self-3-third,3shot,all,0.872,180,0.287,0.025,,
trinity-base,dx-self-3-third,3shot,free,0.921,126,0.286,0.024,,
trinity-base,dx-self-2-deictic,0shot,all,0.372,180,0.287,0.036,,speaker_confusion=0.594;tie=0.033
trinity-base,dx-self-2-deictic,0shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.548;tie=0.008
trinity-base,dx-self-2-deictic,3shot_collide,all,0.544,180,0.287,0.037,,speaker_confusion=0.439;tie=0.017
trinity-base,dx-self-2-deictic,3shot_collide,free,0.603,126,0.286,0.044,,speaker_confusion=0.373;tie=0.024
trinity-base,dx-self-2-deictic,3shot,all,0.544,180,0.287,0.037,,speaker_confusion=0.439;tie=0.017
trinity-base,dx-self-2-deictic,3shot,free,0.603,126,0.286,0.044,,speaker_confusion=0.381;tie=0.016
trinity-base,dx-self-2-third,0shot,all,0.806,180,0.287,0.029,,
trinity-base,dx-self-2-third,0shot,free,0.833,126,0.286,0.033,,
trinity-base,dx-self-2-third,3shot_collide,all,0.850,180,0.287,0.027,,
trinity-base,dx-self-2-third,3shot_collide,free,0.873,126,0.286,0.030,,
trinity-base,dx-self-2-third,3shot,all,0.833,180,0.287,0.028,,
trinity-base,dx-self-2-third,3shot,free,0.865,126,0.286,0.030,,
trinity-base,dx-addressee-3-deictic,0shot,all,0.339,180,0.287,0.035,,other_object=0.172;previous_owner=0.089;speaker_confusion=0.35;tie=0.05
trinity-base,dx-addressee-3-deictic,0shot,free,0.381,126,0.286,0.043,,other_object=0.143;previous_owner=0.103;speaker_confusion=0.31;tie=0.063
trinity-base,dx-addressee-3-deictic,3shot_collide,all,0.517,180,0.287,0.037,,other_object=0.078;previous_owner=0.089;speaker_confusion=0.311;tie=0.006
trinity-base,dx-addressee-3-deictic,3shot_collide,free,0.540,126,0.286,0.044,,other_object=0.032;previous_owner=0.111;speaker_confusion=0.31;tie=0.008
trinity-base,dx-addressee-3-deictic,3shot,all,0.589,180,0.287,0.037,,other_object=0.067;previous_owner=0.072;speaker_confusion=0.244;tie=0.028
trinity-base,dx-addressee-3-deictic,3shot,free,0.635,126,0.286,0.043,,other_object=0.008;previous_owner=0.095;speaker_confusion=0.222;tie=0.04
trinity-base,dx-addressee-3-third,0shot,all,0.883,180,0.287,0.024,,
trinity-base,dx-addressee-3-third,0shot,free,0.913,126,0.286,0.025,,
trinity-base,dx-addressee-3-third,3shot_collide,all,0.917,180,0.287,0.021,,
trinity-base,dx-addressee-3-third,3shot_collide,free,0.905,126,0.286,0.026,,
trinity-base,dx-addressee-3-third,3shot,all,0.928,180,0.287,0.019,,
trinity-base,dx-addressee-3-third,3shot,free,0.952,126,0.286,0.019,,
trinity-base,dx-addressee-2-deictic,0shot,all,0.178,180,0.287,0.029,,speaker_confusion=0.778;tie=0.044
trinity-base,dx-addressee-2-deictic,0shot,free,0.175,126,0.286,0.034,,speaker_confusion=0.786;tie=0.04
trinity-base,dx-addressee-2-deictic,3shot_collide,all,0.456,180,0.287,0.037,,speaker_confusion=0.511;tie=0.033
trinity-base,dx-addressee-2-deictic,3shot_collide,free,0.524,126,0.286,0.044,,speaker_confusion=0.444;tie=0.032
trinity-base,dx-addressee-2-deictic,3shot,all,0.494,180,0.287,0.037,,speaker_confusion=0.483;tie=0.022
trinity-base,dx-addressee-2-deictic,3shot,free,0.556,126,0.286,0.044,,speaker_confusion=0.413;tie=0.032
trinity-base,dx-addressee-2-third,0shot,all,0.789,180,0.287,0.030,,
trinity-base,dx-addressee-2-third,0shot,free,0.897,126,0.286,0.027,,
trinity-base,dx-addressee-2-third,3shot_collide,all,0.811,180,0.287,0.029,,
trinity-base,dx-addressee-2-third,3shot_collide,free,0.881,126,0.286,0.029,,
trinity-base,dx-addressee-2-third,3shot,all,0.822,180,0.287,0.029,,
trinity-base,dx-addressee-2-third,3shot,free,0.905,126,0.286,0.026,,
trinity-base,sb-k1,0shot,all,0.922,64,0.125,0.034,,
trinity-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
trinity-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
trinity-base,sb-k23,0shot,all,0.766,128,0.027,0.037,,
trinity-base,sb-k23,3shot_collide,all,0.992,128,0.027,0.008,,
trinity-base,sb-k23,3shot,all,0.992,128,0.027,0.008,,
trinity-base,sb-split,0shot,all,0.773,128,0.027,0.037,,
trinity-base,sb-split,3shot_collide,all,0.906,128,0.027,0.026,,
trinity-base,sb-split,3shot,all,0.812,128,0.027,0.035,,
trinity-base,sb-dyn,0shot,all,0.539,256,0.027,0.031,,
trinity-base,sb-dyn,3shot_collide,all,0.586,256,0.027,0.031,,
trinity-base,sb-dyn,3shot,all,0.574,256,0.027,0.031,,
trinity-base,an-narr,0shot,all,0.597,640,0.250,0.019,,
trinity-base,an-narr,3shot_collide,all,0.984,640,0.250,0.005,,
trinity-base,an-narr,3shot,all,0.978,640,0.250,0.006,,
trinity-base,an-other-any,0shot,all,0.428,320,0.250,0.028,,
trinity-base,an-other-any,3shot_collide,all,0.825,320,0.250,0.021,,
trinity-base,an-other-any,3shot,all,0.847,320,0.250,0.020,,
trinity-base,an-me-any,0shot,all,0.381,160,0.250,0.038,,
trinity-base,an-me-any,3shot_collide,all,0.887,160,0.250,0.025,,
trinity-base,an-me-any,3shot,all,0.938,160,0.250,0.019,,
trinity-base,an-you-voc,0shot,all,0.525,80,0.250,0.056,,
trinity-base,an-you-voc,3shot_collide,all,0.963,80,0.250,0.021,,
trinity-base,an-you-voc,3shot,all,1.000,80,0.250,0.000,,
trinity-base,an-you-novoc,0shot,all,0.237,80,0.250,0.048,,
trinity-base,an-you-novoc,3shot_collide,all,0.912,80,0.250,0.032,,
trinity-base,an-you-novoc,3shot,all,0.863,80,0.250,0.038,,
trinity-base,adj-adj,0shot,all,0.989,260,0.333,0.007,,
trinity-base,adj-adj,3shot_collide,all,0.992,260,0.333,0.005,,
trinity-base,adj-adj,3shot,all,0.996,260,0.333,0.004,,
trinity-base,adj-voc,0shot,all,0.992,260,0.333,0.005,,
trinity-base,adj-voc,3shot_collide,all,0.996,260,0.333,0.004,,
trinity-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
trinity-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
trinity-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
trinity-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
trinity-base,cm-mutate,0shot,all,0.980,200,0.200,0.010,,
trinity-base,cm-mutate,3shot_collide,all,0.995,200,0.200,0.005,,
trinity-base,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
trinity-base,cm-rebind,0shot,all,0.930,200,0.200,0.018,,
trinity-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
trinity-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
trinity-base,cm-copy,0shot,all,0.975,200,0.200,0.011,,
trinity-base,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
trinity-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
trinity-truebase,attr,0shot,all,0.790,1210,0.168,0.012,,
trinity-truebase,attr,3shot_collide,all,0.988,1210,0.168,0.003,,
trinity-truebase,attr,3shot,all,1.000,1210,0.168,0.000,,
trinity-truebase,codevar,0shot,all,0.873,1200,0.267,0.010,,
trinity-truebase,codevar,3shot_collide,all,0.882,1200,0.267,0.009,,
trinity-truebase,codevar,3shot,all,0.892,1200,0.267,0.009,,
trinity-truebase,dialogue,0shot,all,0.883,1200,0.312,0.009,,
trinity-truebase,dialogue,3shot_collide,all,0.898,1200,0.312,0.009,,
trinity-truebase,dialogue,3shot,all,0.915,1200,0.312,0.008,,
trinity-truebase,dx-owner-3-deictic,0shot,all,0.650,120,0.333,0.043,,other_participant=0.317;speaker_confusion=0.017;tie=0.017
trinity-truebase,dx-owner-3-deictic,0shot,free,0.650,40,0.333,0.075,,other_participant=0.35
trinity-truebase,dx-owner-3-deictic,3shot_collide,all,0.858,120,0.333,0.032,,other_participant=0.092;speaker_confusion=0.033;tie=0.017
trinity-truebase,dx-owner-3-deictic,3shot_collide,free,0.825,40,0.333,0.060,,other_participant=0.15;speaker_confusion=0.025
trinity-truebase,dx-owner-3-deictic,3shot,all,0.900,120,0.333,0.027,,other_participant=0.058;speaker_confusion=0.042
trinity-truebase,dx-owner-3-deictic,3shot,free,0.875,40,0.333,0.052,,other_participant=0.1;speaker_confusion=0.025
trinity-truebase,dx-owner-3-third,0shot,all,0.825,120,0.333,0.035,,
trinity-truebase,dx-owner-3-third,0shot,free,0.750,40,0.333,0.069,,
trinity-truebase,dx-owner-3-third,3shot_collide,all,0.925,120,0.333,0.024,,
trinity-truebase,dx-owner-3-third,3shot_collide,free,0.875,40,0.333,0.052,,
trinity-truebase,dx-owner-3-third,3shot,all,0.950,120,0.333,0.020,,
trinity-truebase,dx-owner-3-third,3shot,free,0.925,40,0.333,0.042,,
trinity-truebase,dx-self-3-deictic,0shot,all,0.389,180,0.287,0.036,,other_object=0.206;previous_owner=0.072;speaker_confusion=0.311;tie=0.022
trinity-truebase,dx-self-3-deictic,0shot,free,0.444,126,0.286,0.044,,other_object=0.222;previous_owner=0.04;speaker_confusion=0.27;tie=0.024
trinity-truebase,dx-self-3-deictic,3shot_collide,all,0.539,180,0.287,0.037,,other_object=0.139;previous_owner=0.022;speaker_confusion=0.272;tie=0.028
trinity-truebase,dx-self-3-deictic,3shot_collide,free,0.619,126,0.286,0.043,,other_object=0.127;previous_owner=0.024;speaker_confusion=0.206;tie=0.024
trinity-truebase,dx-self-3-deictic,3shot,all,0.567,180,0.287,0.037,,other_object=0.15;previous_owner=0.078;speaker_confusion=0.2;tie=0.006
trinity-truebase,dx-self-3-deictic,3shot,free,0.675,126,0.286,0.042,,other_object=0.111;previous_owner=0.048;speaker_confusion=0.159;tie=0.008
trinity-truebase,dx-self-3-third,0shot,all,0.772,180,0.287,0.031,,
trinity-truebase,dx-self-3-third,0shot,free,0.762,126,0.286,0.038,,
trinity-truebase,dx-self-3-third,3shot_collide,all,0.811,180,0.287,0.029,,
trinity-truebase,dx-self-3-third,3shot_collide,free,0.873,126,0.286,0.030,,
trinity-truebase,dx-self-3-third,3shot,all,0.828,180,0.287,0.028,,
trinity-truebase,dx-self-3-third,3shot,free,0.873,126,0.286,0.030,,
trinity-truebase,dx-self-2-deictic,0shot,all,0.383,180,0.287,0.036,,speaker_confusion=0.583;tie=0.033
trinity-truebase,dx-self-2-deictic,0shot,free,0.429,126,0.286,0.044,,speaker_confusion=0.54;tie=0.032
trinity-truebase,dx-self-2-deictic,3shot_collide,all,0.461,180,0.287,0.037,,speaker_confusion=0.528;tie=0.011
trinity-truebase,dx-self-2-deictic,3shot_collide,free,0.556,126,0.286,0.044,,speaker_confusion=0.429;tie=0.016
trinity-truebase,dx-self-2-deictic,3shot,all,0.472,180,0.287,0.037,,speaker_confusion=0.511;tie=0.017
trinity-truebase,dx-self-2-deictic,3shot,free,0.579,126,0.286,0.044,,speaker_confusion=0.413;tie=0.008
trinity-truebase,dx-self-2-third,0shot,all,0.767,180,0.287,0.032,,
trinity-truebase,dx-self-2-third,0shot,free,0.722,126,0.286,0.040,,
trinity-truebase,dx-self-2-third,3shot_collide,all,0.672,180,0.287,0.035,,
trinity-truebase,dx-self-2-third,3shot_collide,free,0.762,126,0.286,0.038,,
trinity-truebase,dx-self-2-third,3shot,all,0.706,180,0.287,0.034,,
trinity-truebase,dx-self-2-third,3shot,free,0.809,126,0.286,0.035,,
trinity-truebase,dx-addressee-3-deictic,0shot,all,0.367,180,0.287,0.036,,other_object=0.122;previous_owner=0.083;speaker_confusion=0.411;tie=0.017
trinity-truebase,dx-addressee-3-deictic,0shot,free,0.421,126,0.286,0.044,,other_object=0.111;previous_owner=0.087;speaker_confusion=0.357;tie=0.024
trinity-truebase,dx-addressee-3-deictic,3shot_collide,all,0.372,180,0.287,0.036,,other_object=0.172;previous_owner=0.078;speaker_confusion=0.333;tie=0.044
trinity-truebase,dx-addressee-3-deictic,3shot_collide,free,0.397,126,0.286,0.044,,other_object=0.127;previous_owner=0.095;speaker_confusion=0.325;tie=0.056
trinity-truebase,dx-addressee-3-deictic,3shot,all,0.428,180,0.287,0.037,,other_object=0.133;previous_owner=0.083;speaker_confusion=0.35;tie=0.006
trinity-truebase,dx-addressee-3-deictic,3shot,free,0.500,126,0.286,0.044,,other_object=0.095;previous_owner=0.087;speaker_confusion=0.31;tie=0.008
trinity-truebase,dx-addressee-3-third,0shot,all,0.811,180,0.287,0.029,,
trinity-truebase,dx-addressee-3-third,0shot,free,0.778,126,0.286,0.037,,
trinity-truebase,dx-addressee-3-third,3shot_collide,all,0.767,180,0.287,0.032,,
trinity-truebase,dx-addressee-3-third,3shot_collide,free,0.833,126,0.286,0.033,,
trinity-truebase,dx-addressee-3-third,3shot,all,0.811,180,0.287,0.029,,
trinity-truebase,dx-addressee-3-third,3shot,free,0.865,126,0.286,0.030,,
trinity-truebase,dx-addressee-2-deictic,0shot,all,0.256,180,0.287,0.033,,speaker_confusion=0.711;tie=0.033
trinity-truebase,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.754;tie=0.032
trinity-truebase,dx-addressee-2-deictic,3shot_collide,all,0.411,180,0.287,0.037,,speaker_confusion=0.567;tie=0.022
trinity-truebase,dx-addressee-2-deictic,3shot_collide,free,0.468,126,0.286,0.044,,speaker_confusion=0.5;tie=0.032
trinity-truebase,dx-addressee-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.594;tie=0.028
trinity-truebase,dx-addressee-2-deictic,3shot,free,0.444,126,0.286,0.044,,speaker_confusion=0.516;tie=0.04
trinity-truebase,dx-addressee-2-third,0shot,all,0.767,180,0.287,0.032,,
trinity-truebase,dx-addressee-2-third,0shot,free,0.722,126,0.286,0.040,,
trinity-truebase,dx-addressee-2-third,3shot_collide,all,0.711,180,0.287,0.034,,
trinity-truebase,dx-addressee-2-third,3shot_collide,free,0.825,126,0.286,0.034,,
trinity-truebase,dx-addressee-2-third,3shot,all,0.739,180,0.287,0.033,,
trinity-truebase,dx-addressee-2-third,3shot,free,0.841,126,0.286,0.033,,
trinity-truebase,sb-k1,0shot,all,0.938,64,0.125,0.030,,
trinity-truebase,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
trinity-truebase,sb-k1,3shot,all,1.000,64,0.125,0.000,,
trinity-truebase,sb-k23,0shot,all,0.648,128,0.027,0.042,,
trinity-truebase,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
trinity-truebase,sb-k23,3shot,all,0.977,128,0.027,0.013,,
trinity-truebase,sb-split,0shot,all,0.594,128,0.027,0.043,,
trinity-truebase,sb-split,3shot_collide,all,0.773,128,0.027,0.037,,
trinity-truebase,sb-split,3shot,all,0.727,128,0.027,0.039,,
trinity-truebase,sb-dyn,0shot,all,0.484,256,0.027,0.031,,
trinity-truebase,sb-dyn,3shot_collide,all,0.410,256,0.027,0.031,,
trinity-truebase,sb-dyn,3shot,all,0.422,256,0.027,0.031,,
trinity-truebase,an-narr,0shot,all,0.614,640,0.250,0.019,,
trinity-truebase,an-narr,3shot_collide,all,0.936,640,0.250,0.010,,
trinity-truebase,an-narr,3shot,all,0.936,640,0.250,0.010,,
trinity-truebase,an-other-any,0shot,all,0.144,320,0.250,0.020,,
trinity-truebase,an-other-any,3shot_collide,all,0.653,320,0.250,0.027,,
trinity-truebase,an-other-any,3shot,all,0.619,320,0.250,0.027,,
trinity-truebase,an-me-any,0shot,all,0.600,160,0.250,0.039,,
trinity-truebase,an-me-any,3shot_collide,all,0.562,160,0.250,0.039,,
trinity-truebase,an-me-any,3shot,all,0.600,160,0.250,0.039,,
trinity-truebase,an-you-voc,0shot,all,0.200,80,0.250,0.045,,
trinity-truebase,an-you-voc,3shot_collide,all,0.525,80,0.250,0.056,,
trinity-truebase,an-you-voc,3shot,all,0.537,80,0.250,0.056,,
trinity-truebase,an-you-novoc,0shot,all,0.487,80,0.250,0.056,,
trinity-truebase,an-you-novoc,3shot_collide,all,0.287,80,0.250,0.051,,
trinity-truebase,an-you-novoc,3shot,all,0.388,80,0.250,0.054,,
trinity-truebase,adj-adj,0shot,all,0.931,260,0.333,0.016,,
trinity-truebase,adj-adj,3shot_collide,all,0.992,260,0.333,0.005,,
trinity-truebase,adj-adj,3shot,all,0.981,260,0.333,0.009,,
trinity-truebase,adj-voc,0shot,all,0.954,260,0.333,0.013,,
trinity-truebase,adj-voc,3shot_collide,all,0.992,260,0.333,0.005,,
trinity-truebase,adj-voc,3shot,all,0.996,260,0.333,0.004,,
trinity-truebase,adj-third,0shot,all,0.977,260,0.333,0.009,,
trinity-truebase,adj-third,3shot_collide,all,0.996,260,0.333,0.004,,
trinity-truebase,adj-third,3shot,all,1.000,260,0.333,0.000,,
trinity-truebase,cm-mutate,0shot,all,0.985,200,0.200,0.009,,
trinity-truebase,cm-mutate,3shot_collide,all,0.990,200,0.200,0.007,,
trinity-truebase,cm-mutate,3shot,all,0.995,200,0.200,0.005,,
trinity-truebase,cm-rebind,0shot,all,0.775,200,0.200,0.029,,
trinity-truebase,cm-rebind,3shot_collide,all,0.990,200,0.200,0.007,,
trinity-truebase,cm-rebind,3shot,all,0.980,200,0.200,0.010,,
trinity-truebase,cm-copy,0shot,all,0.365,200,0.200,0.034,,
trinity-truebase,cm-copy,3shot_collide,all,0.995,200,0.200,0.005,,
trinity-truebase,cm-copy,3shot,all,1.000,200,0.200,0.000,,
llama31-405b,attr,0shot,all,0.791,1210,0.168,0.012,,
llama31-405b,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
llama31-405b,attr,3shot,all,1.000,1210,0.168,0.000,,
llama31-405b,codevar,0shot,all,0.844,1200,0.267,0.011,,
llama31-405b,codevar,3shot_collide,all,0.946,1200,0.267,0.006,,
llama31-405b,codevar,3shot,all,0.938,1200,0.267,0.007,,
llama31-405b,dialogue,0shot,all,0.946,1200,0.312,0.006,,
llama31-405b,dialogue,3shot_collide,all,0.995,1200,0.312,0.002,,
llama31-405b,dialogue,3shot,all,0.992,1200,0.312,0.003,,
llama31-405b,dx-owner-3-deictic,0shot,all,0.767,120,0.333,0.039,,other_participant=0.217;tie=0.017
llama31-405b,dx-owner-3-deictic,0shot,free,0.725,40,0.333,0.071,,other_participant=0.25;tie=0.025
llama31-405b,dx-owner-3-deictic,3shot_collide,all,1.000,120,0.333,0.000,,
llama31-405b,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
llama31-405b,dx-owner-3-deictic,3shot,all,1.000,120,0.333,0.000,,
llama31-405b,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
llama31-405b,dx-owner-3-third,0shot,all,0.900,120,0.333,0.027,,
llama31-405b,dx-owner-3-third,0shot,free,0.850,40,0.333,0.057,,
llama31-405b,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
llama31-405b,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
llama31-405b,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
llama31-405b,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
llama31-405b,dx-self-3-deictic,0shot,all,0.583,180,0.287,0.037,,other_object=0.044;previous_owner=0.072;speaker_confusion=0.261;tie=0.039
llama31-405b,dx-self-3-deictic,0shot,free,0.548,126,0.286,0.044,,other_object=0.063;previous_owner=0.04;speaker_confusion=0.317;tie=0.032
llama31-405b,dx-self-3-deictic,3shot_collide,all,0.983,180,0.287,0.009,,previous_owner=0.011;speaker_confusion=0.006
llama31-405b,dx-self-3-deictic,3shot_collide,free,1.000,126,0.286,0.000,,
llama31-405b,dx-self-3-deictic,3shot,all,0.961,180,0.287,0.014,,other_object=0.006;previous_owner=0.028;speaker_confusion=0.006
llama31-405b,dx-self-3-deictic,3shot,free,0.992,126,0.286,0.008,,other_object=0.008
llama31-405b,dx-self-3-third,0shot,all,0.956,180,0.287,0.015,,
llama31-405b,dx-self-3-third,0shot,free,0.992,126,0.286,0.008,,
llama31-405b,dx-self-3-third,3shot_collide,all,0.989,180,0.287,0.008,,
llama31-405b,dx-self-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
llama31-405b,dx-self-3-third,3shot,all,0.994,180,0.287,0.005,,
llama31-405b,dx-self-3-third,3shot,free,1.000,126,0.286,0.000,,
llama31-405b,dx-self-2-deictic,0shot,all,0.422,180,0.287,0.037,,speaker_confusion=0.567;tie=0.011
llama31-405b,dx-self-2-deictic,0shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.587;tie=0.008
llama31-405b,dx-self-2-deictic,3shot_collide,all,0.739,180,0.287,0.033,,speaker_confusion=0.244;tie=0.017
llama31-405b,dx-self-2-deictic,3shot_collide,free,0.770,126,0.286,0.037,,speaker_confusion=0.214;tie=0.016
llama31-405b,dx-self-2-deictic,3shot,all,0.706,180,0.287,0.034,,speaker_confusion=0.278;tie=0.017
llama31-405b,dx-self-2-deictic,3shot,free,0.762,126,0.286,0.038,,speaker_confusion=0.23;tie=0.008
llama31-405b,dx-self-2-third,0shot,all,0.956,180,0.287,0.015,,
llama31-405b,dx-self-2-third,0shot,free,0.992,126,0.286,0.008,,
llama31-405b,dx-self-2-third,3shot_collide,all,0.994,180,0.287,0.005,,
llama31-405b,dx-self-2-third,3shot_collide,free,0.992,126,0.286,0.008,,
llama31-405b,dx-self-2-third,3shot,all,0.989,180,0.287,0.008,,
llama31-405b,dx-self-2-third,3shot,free,0.984,126,0.286,0.011,,
llama31-405b,dx-addressee-3-deictic,0shot,all,0.444,180,0.287,0.037,,other_object=0.056;previous_owner=0.111;speaker_confusion=0.356;tie=0.033
llama31-405b,dx-addressee-3-deictic,0shot,free,0.381,126,0.286,0.043,,other_object=0.071;previous_owner=0.143;speaker_confusion=0.365;tie=0.04
llama31-405b,dx-addressee-3-deictic,3shot_collide,all,0.789,180,0.287,0.030,,other_object=0.039;previous_owner=0.039;speaker_confusion=0.133
llama31-405b,dx-addressee-3-deictic,3shot_collide,free,0.809,126,0.286,0.035,,other_object=0.032;previous_owner=0.04;speaker_confusion=0.119
llama31-405b,dx-addressee-3-deictic,3shot,all,0.767,180,0.287,0.032,,other_object=0.044;previous_owner=0.039;speaker_confusion=0.139;tie=0.011
llama31-405b,dx-addressee-3-deictic,3shot,free,0.802,126,0.286,0.035,,other_object=0.032;previous_owner=0.048;speaker_confusion=0.111;tie=0.008
llama31-405b,dx-addressee-3-third,0shot,all,0.972,180,0.287,0.012,,
llama31-405b,dx-addressee-3-third,0shot,free,0.976,126,0.286,0.014,,
llama31-405b,dx-addressee-3-third,3shot_collide,all,0.994,180,0.287,0.005,,
llama31-405b,dx-addressee-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
llama31-405b,dx-addressee-3-third,3shot,all,0.994,180,0.287,0.005,,
llama31-405b,dx-addressee-3-third,3shot,free,1.000,126,0.286,0.000,,
llama31-405b,dx-addressee-2-deictic,0shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.667;tie=0.011
llama31-405b,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.77;tie=0.016
llama31-405b,dx-addressee-2-deictic,3shot_collide,all,0.489,180,0.287,0.037,,speaker_confusion=0.489;tie=0.022
llama31-405b,dx-addressee-2-deictic,3shot_collide,free,0.484,126,0.286,0.044,,speaker_confusion=0.492;tie=0.024
llama31-405b,dx-addressee-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.506;tie=0.011
llama31-405b,dx-addressee-2-deictic,3shot,free,0.500,126,0.286,0.044,,speaker_confusion=0.492;tie=0.008
llama31-405b,dx-addressee-2-third,0shot,all,0.961,180,0.287,0.014,,
llama31-405b,dx-addressee-2-third,0shot,free,0.992,126,0.286,0.008,,
llama31-405b,dx-addressee-2-third,3shot_collide,all,0.994,180,0.287,0.005,,
llama31-405b,dx-addressee-2-third,3shot_collide,free,1.000,126,0.286,0.000,,
llama31-405b,dx-addressee-2-third,3shot,all,0.994,180,0.287,0.005,,
llama31-405b,dx-addressee-2-third,3shot,free,1.000,126,0.286,0.000,,
llama31-405b,sb-k1,0shot,all,0.875,64,0.125,0.041,,
llama31-405b,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
llama31-405b,sb-k1,3shot,all,1.000,64,0.125,0.000,,
llama31-405b,sb-k23,0shot,all,0.758,128,0.027,0.038,,
llama31-405b,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
llama31-405b,sb-k23,3shot,all,1.000,128,0.027,0.000,,
llama31-405b,sb-split,0shot,all,0.812,128,0.027,0.035,,
llama31-405b,sb-split,3shot_collide,all,0.992,128,0.027,0.008,,
llama31-405b,sb-split,3shot,all,0.977,128,0.027,0.013,,
llama31-405b,sb-dyn,0shot,all,0.840,256,0.027,0.023,,
llama31-405b,sb-dyn,3shot_collide,all,0.926,256,0.027,0.016,,
llama31-405b,sb-dyn,3shot,all,0.902,256,0.027,0.019,,
llama31-405b,an-narr,0shot,all,0.736,640,0.250,0.017,,
llama31-405b,an-narr,3shot_collide,all,1.000,640,0.250,0.000,,
llama31-405b,an-narr,3shot,all,1.000,640,0.250,0.000,,
llama31-405b,an-other-any,0shot,all,0.575,320,0.250,0.028,,
llama31-405b,an-other-any,3shot_collide,all,0.991,320,0.250,0.005,,
llama31-405b,an-other-any,3shot,all,0.984,320,0.250,0.007,,
llama31-405b,an-me-any,0shot,all,0.650,160,0.250,0.038,,
llama31-405b,an-me-any,3shot_collide,all,0.988,160,0.250,0.009,,
llama31-405b,an-me-any,3shot,all,0.988,160,0.250,0.009,,
llama31-405b,an-you-voc,0shot,all,0.138,80,0.250,0.038,,
llama31-405b,an-you-voc,3shot_collide,all,0.988,80,0.250,0.012,,
llama31-405b,an-you-voc,3shot,all,0.988,80,0.250,0.012,,
llama31-405b,an-you-novoc,0shot,all,0.200,80,0.250,0.045,,
llama31-405b,an-you-novoc,3shot_collide,all,0.863,80,0.250,0.038,,
llama31-405b,an-you-novoc,3shot,all,0.925,80,0.250,0.029,,
llama31-405b,adj-adj,0shot,all,1.000,260,0.333,0.000,,
llama31-405b,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-405b,adj-adj,3shot,all,1.000,260,0.333,0.000,,
llama31-405b,adj-voc,0shot,all,1.000,260,0.333,0.000,,
llama31-405b,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-405b,adj-voc,3shot,all,1.000,260,0.333,0.000,,
llama31-405b,adj-third,0shot,all,1.000,260,0.333,0.000,,
llama31-405b,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-405b,adj-third,3shot,all,1.000,260,0.333,0.000,,
llama31-405b,cm-mutate,0shot,all,0.990,200,0.200,0.007,,
llama31-405b,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
llama31-405b,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
llama31-405b,cm-rebind,0shot,all,0.805,200,0.200,0.028,,
llama31-405b,cm-rebind,3shot_collide,all,0.985,200,0.200,0.009,,
llama31-405b,cm-rebind,3shot,all,0.995,200,0.200,0.005,,
llama31-405b,cm-copy,0shot,all,0.830,200,0.200,0.027,,
llama31-405b,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
llama31-405b,cm-copy,3shot,all,1.000,200,0.200,0.000,,
dsv3-base,attr,0shot,all,0.678,1210,0.168,0.013,,
dsv3-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
dsv3-base,attr,3shot,all,1.000,1210,0.168,0.000,,
dsv3-base,codevar,0shot,all,0.876,1200,0.267,0.009,,
dsv3-base,codevar,3shot_collide,all,0.924,1200,0.267,0.008,,
dsv3-base,codevar,3shot,all,0.939,1200,0.267,0.007,,
dsv3-base,dialogue,0shot,all,0.958,1200,0.312,0.006,,
dsv3-base,dialogue,3shot_collide,all,0.991,1200,0.312,0.003,,
dsv3-base,dialogue,3shot,all,0.986,1200,0.312,0.003,,
dsv3-base,dx-owner-3-deictic,0shot,all,0.733,120,0.333,0.040,,other_participant=0.225;tie=0.042
dsv3-base,dx-owner-3-deictic,0shot,free,0.650,40,0.333,0.075,,other_participant=0.325;tie=0.025
dsv3-base,dx-owner-3-deictic,3shot_collide,all,0.983,120,0.333,0.012,,other_participant=0.017
dsv3-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
dsv3-base,dx-owner-3-deictic,3shot,all,0.992,120,0.333,0.008,,other_participant=0.008
dsv3-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
dsv3-base,dx-owner-3-third,0shot,all,0.800,120,0.333,0.036,,
dsv3-base,dx-owner-3-third,0shot,free,0.750,40,0.333,0.069,,
dsv3-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
dsv3-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
dsv3-base,dx-owner-3-third,3shot,all,0.992,120,0.333,0.008,,
dsv3-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
dsv3-base,dx-self-3-deictic,0shot,all,0.483,180,0.287,0.037,,other_object=0.072;previous_owner=0.072;speaker_confusion=0.344;tie=0.028
dsv3-base,dx-self-3-deictic,0shot,free,0.405,126,0.286,0.044,,other_object=0.079;previous_owner=0.063;speaker_confusion=0.421;tie=0.032
dsv3-base,dx-self-3-deictic,3shot_collide,all,0.806,180,0.287,0.029,,other_object=0.017;previous_owner=0.05;speaker_confusion=0.106;tie=0.022
dsv3-base,dx-self-3-deictic,3shot_collide,free,0.809,126,0.286,0.035,,other_object=0.024;previous_owner=0.032;speaker_confusion=0.111;tie=0.024
dsv3-base,dx-self-3-deictic,3shot,all,0.739,180,0.287,0.033,,other_object=0.033;previous_owner=0.067;speaker_confusion=0.139;tie=0.022
dsv3-base,dx-self-3-deictic,3shot,free,0.746,126,0.286,0.039,,other_object=0.048;previous_owner=0.056;speaker_confusion=0.135;tie=0.016
dsv3-base,dx-self-3-third,0shot,all,0.922,180,0.287,0.020,,
dsv3-base,dx-self-3-third,0shot,free,0.960,126,0.286,0.017,,
dsv3-base,dx-self-3-third,3shot_collide,all,0.989,180,0.287,0.008,,
dsv3-base,dx-self-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
dsv3-base,dx-self-3-third,3shot,all,0.994,180,0.287,0.005,,
dsv3-base,dx-self-3-third,3shot,free,1.000,126,0.286,0.000,,
dsv3-base,dx-self-2-deictic,0shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.639;tie=0.039
dsv3-base,dx-self-2-deictic,0shot,free,0.270,126,0.286,0.040,,speaker_confusion=0.69;tie=0.04
dsv3-base,dx-self-2-deictic,3shot_collide,all,0.656,180,0.287,0.035,,speaker_confusion=0.339;tie=0.006
dsv3-base,dx-self-2-deictic,3shot_collide,free,0.691,126,0.286,0.041,,speaker_confusion=0.31
dsv3-base,dx-self-2-deictic,3shot,all,0.606,180,0.287,0.036,,speaker_confusion=0.367;tie=0.028
dsv3-base,dx-self-2-deictic,3shot,free,0.659,126,0.286,0.042,,speaker_confusion=0.317;tie=0.024
dsv3-base,dx-self-2-third,0shot,all,0.956,180,0.287,0.015,,
dsv3-base,dx-self-2-third,0shot,free,0.968,126,0.286,0.016,,
dsv3-base,dx-self-2-third,3shot_collide,all,0.983,180,0.287,0.009,,
dsv3-base,dx-self-2-third,3shot_collide,free,0.992,126,0.286,0.008,,
dsv3-base,dx-self-2-third,3shot,all,0.972,180,0.287,0.012,,
dsv3-base,dx-self-2-third,3shot,free,0.976,126,0.286,0.014,,
dsv3-base,dx-addressee-3-deictic,0shot,all,0.539,180,0.287,0.037,,other_object=0.083;previous_owner=0.072;speaker_confusion=0.272;tie=0.033
dsv3-base,dx-addressee-3-deictic,0shot,free,0.476,126,0.286,0.044,,other_object=0.095;previous_owner=0.095;speaker_confusion=0.31;tie=0.024
dsv3-base,dx-addressee-3-deictic,3shot_collide,all,0.633,180,0.287,0.036,,other_object=0.05;previous_owner=0.094;speaker_confusion=0.206;tie=0.017
dsv3-base,dx-addressee-3-deictic,3shot_collide,free,0.619,126,0.286,0.043,,other_object=0.024;previous_owner=0.119;speaker_confusion=0.214;tie=0.024
dsv3-base,dx-addressee-3-deictic,3shot,all,0.600,180,0.287,0.036,,other_object=0.061;previous_owner=0.083;speaker_confusion=0.25;tie=0.006
dsv3-base,dx-addressee-3-deictic,3shot,free,0.595,126,0.286,0.044,,other_object=0.04;previous_owner=0.103;speaker_confusion=0.262
dsv3-base,dx-addressee-3-third,0shot,all,0.933,180,0.287,0.019,,
dsv3-base,dx-addressee-3-third,0shot,free,0.936,126,0.286,0.022,,
dsv3-base,dx-addressee-3-third,3shot_collide,all,0.994,180,0.287,0.005,,
dsv3-base,dx-addressee-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
dsv3-base,dx-addressee-3-third,3shot,all,0.994,180,0.287,0.005,,
dsv3-base,dx-addressee-3-third,3shot,free,0.992,126,0.286,0.008,,
dsv3-base,dx-addressee-2-deictic,0shot,all,0.311,180,0.287,0.035,,speaker_confusion=0.644;tie=0.044
dsv3-base,dx-addressee-2-deictic,0shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.754;tie=0.032
dsv3-base,dx-addressee-2-deictic,3shot_collide,all,0.356,180,0.287,0.036,,speaker_confusion=0.628;tie=0.017
dsv3-base,dx-addressee-2-deictic,3shot_collide,free,0.357,126,0.286,0.043,,speaker_confusion=0.627;tie=0.016
dsv3-base,dx-addressee-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.6;tie=0.022
dsv3-base,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.611;tie=0.024
dsv3-base,dx-addressee-2-third,0shot,all,0.906,180,0.287,0.022,,
dsv3-base,dx-addressee-2-third,0shot,free,0.952,126,0.286,0.019,,
dsv3-base,dx-addressee-2-third,3shot_collide,all,0.989,180,0.287,0.008,,
dsv3-base,dx-addressee-2-third,3shot_collide,free,0.984,126,0.286,0.011,,
dsv3-base,dx-addressee-2-third,3shot,all,0.983,180,0.287,0.009,,
dsv3-base,dx-addressee-2-third,3shot,free,0.992,126,0.286,0.008,,
dsv3-base,sb-k1,0shot,all,0.875,64,0.125,0.041,,
dsv3-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
dsv3-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
dsv3-base,sb-k23,0shot,all,0.703,128,0.027,0.040,,
dsv3-base,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
dsv3-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
dsv3-base,sb-split,0shot,all,0.680,128,0.027,0.041,,
dsv3-base,sb-split,3shot_collide,all,0.977,128,0.027,0.013,,
dsv3-base,sb-split,3shot,all,0.984,128,0.027,0.011,,
dsv3-base,sb-dyn,0shot,all,0.867,256,0.027,0.021,,
dsv3-base,sb-dyn,3shot_collide,all,0.836,256,0.027,0.023,,
dsv3-base,sb-dyn,3shot,all,0.836,256,0.027,0.023,,
dsv3-base,an-narr,0shot,all,0.698,640,0.250,0.018,,
dsv3-base,an-narr,3shot_collide,all,0.989,640,0.250,0.004,,
dsv3-base,an-narr,3shot,all,0.992,640,0.250,0.004,,
dsv3-base,an-other-any,0shot,all,0.594,320,0.250,0.028,,
dsv3-base,an-other-any,3shot_collide,all,0.928,320,0.250,0.014,,
dsv3-base,an-other-any,3shot,all,0.928,320,0.250,0.014,,
dsv3-base,an-me-any,0shot,all,0.581,160,0.250,0.039,,
dsv3-base,an-me-any,3shot_collide,all,0.850,160,0.250,0.028,,
dsv3-base,an-me-any,3shot,all,0.863,160,0.250,0.027,,
dsv3-base,an-you-voc,0shot,all,0.600,80,0.250,0.055,,
dsv3-base,an-you-voc,3shot_collide,all,0.912,80,0.250,0.032,,
dsv3-base,an-you-voc,3shot,all,0.938,80,0.250,0.027,,
dsv3-base,an-you-novoc,0shot,all,0.388,80,0.250,0.054,,
dsv3-base,an-you-novoc,3shot_collide,all,0.688,80,0.250,0.052,,
dsv3-base,an-you-novoc,3shot,all,0.787,80,0.250,0.046,,
dsv3-base,adj-adj,0shot,all,0.962,260,0.333,0.012,,
dsv3-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
dsv3-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
dsv3-base,adj-voc,0shot,all,0.969,260,0.333,0.011,,
dsv3-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
dsv3-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
dsv3-base,adj-third,0shot,all,0.992,260,0.333,0.005,,
dsv3-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
dsv3-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
dsv3-base,cm-mutate,0shot,all,0.955,200,0.200,0.015,,
dsv3-base,cm-mutate,3shot_collide,all,0.995,200,0.200,0.005,,
dsv3-base,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
dsv3-base,cm-rebind,0shot,all,0.940,200,0.200,0.017,,
dsv3-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
dsv3-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
dsv3-base,cm-copy,0shot,all,0.905,200,0.200,0.021,,
dsv3-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
dsv3-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
dsv4-pro-base,attr,0shot,all,0.676,1210,0.168,0.013,,
dsv4-pro-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
dsv4-pro-base,attr,3shot,all,0.998,1210,0.168,0.001,,
dsv4-pro-base,codevar,0shot,all,0.872,1200,0.267,0.010,,
dsv4-pro-base,codevar,3shot_collide,all,0.926,1200,0.267,0.008,,
dsv4-pro-base,codevar,3shot,all,0.912,1200,0.267,0.008,,
dsv4-pro-base,dialogue,0shot,all,0.956,1200,0.312,0.006,,
dsv4-pro-base,dialogue,3shot_collide,all,0.992,1200,0.312,0.003,,
dsv4-pro-base,dialogue,3shot,all,0.990,1200,0.312,0.003,,
dsv4-pro-base,dx-owner-3-deictic,0shot,all,0.642,120,0.333,0.044,,other_participant=0.35;tie=0.008
dsv4-pro-base,dx-owner-3-deictic,0shot,free,0.500,40,0.333,0.079,,other_participant=0.475;tie=0.025
dsv4-pro-base,dx-owner-3-deictic,3shot_collide,all,1.000,120,0.333,0.000,,
dsv4-pro-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
dsv4-pro-base,dx-owner-3-deictic,3shot,all,1.000,120,0.333,0.000,,
dsv4-pro-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
dsv4-pro-base,dx-owner-3-third,0shot,all,0.758,120,0.333,0.039,,
dsv4-pro-base,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
dsv4-pro-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
dsv4-pro-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
dsv4-pro-base,dx-owner-3-third,3shot,all,0.992,120,0.333,0.008,,
dsv4-pro-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
dsv4-pro-base,dx-self-3-deictic,0shot,all,0.372,180,0.287,0.036,,other_object=0.256;previous_owner=0.028;speaker_confusion=0.272;tie=0.072
dsv4-pro-base,dx-self-3-deictic,0shot,free,0.397,126,0.286,0.044,,other_object=0.278;previous_owner=0.016;speaker_confusion=0.222;tie=0.087
dsv4-pro-base,dx-self-3-deictic,3shot_collide,all,0.933,180,0.287,0.019,,previous_owner=0.017;speaker_confusion=0.05
dsv4-pro-base,dx-self-3-deictic,3shot_collide,free,0.921,126,0.286,0.024,,previous_owner=0.016;speaker_confusion=0.063
dsv4-pro-base,dx-self-3-deictic,3shot,all,0.894,180,0.287,0.023,,other_object=0.011;previous_owner=0.022;speaker_confusion=0.067;tie=0.006
dsv4-pro-base,dx-self-3-deictic,3shot,free,0.897,126,0.286,0.027,,other_object=0.016;previous_owner=0.016;speaker_confusion=0.071
dsv4-pro-base,dx-self-3-third,0shot,all,0.883,180,0.287,0.024,,
dsv4-pro-base,dx-self-3-third,0shot,free,0.889,126,0.286,0.028,,
dsv4-pro-base,dx-self-3-third,3shot_collide,all,1.000,180,0.287,0.000,,
dsv4-pro-base,dx-self-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
dsv4-pro-base,dx-self-3-third,3shot,all,0.983,180,0.287,0.009,,
dsv4-pro-base,dx-self-3-third,3shot,free,0.992,126,0.286,0.008,,
dsv4-pro-base,dx-self-2-deictic,0shot,all,0.444,180,0.287,0.037,,speaker_confusion=0.522;tie=0.033
dsv4-pro-base,dx-self-2-deictic,0shot,free,0.476,126,0.286,0.044,,speaker_confusion=0.508;tie=0.016
dsv4-pro-base,dx-self-2-deictic,3shot_collide,all,0.789,180,0.287,0.030,,speaker_confusion=0.194;tie=0.017
dsv4-pro-base,dx-self-2-deictic,3shot_collide,free,0.818,126,0.286,0.034,,speaker_confusion=0.159;tie=0.024
dsv4-pro-base,dx-self-2-deictic,3shot,all,0.767,180,0.287,0.032,,speaker_confusion=0.233
dsv4-pro-base,dx-self-2-deictic,3shot,free,0.778,126,0.286,0.037,,speaker_confusion=0.222
dsv4-pro-base,dx-self-2-third,0shot,all,0.956,180,0.287,0.015,,
dsv4-pro-base,dx-self-2-third,0shot,free,0.944,126,0.286,0.020,,
dsv4-pro-base,dx-self-2-third,3shot_collide,all,0.967,180,0.287,0.013,,
dsv4-pro-base,dx-self-2-third,3shot_collide,free,0.984,126,0.286,0.011,,
dsv4-pro-base,dx-self-2-third,3shot,all,0.978,180,0.287,0.011,,
dsv4-pro-base,dx-self-2-third,3shot,free,1.000,126,0.286,0.000,,
dsv4-pro-base,dx-addressee-3-deictic,0shot,all,0.233,180,0.287,0.032,,other_object=0.267;previous_owner=0.022;speaker_confusion=0.444;tie=0.033
dsv4-pro-base,dx-addressee-3-deictic,0shot,free,0.294,126,0.286,0.041,,other_object=0.27;previous_owner=0.008;speaker_confusion=0.397;tie=0.032
dsv4-pro-base,dx-addressee-3-deictic,3shot_collide,all,0.511,180,0.287,0.037,,other_object=0.072;previous_owner=0.078;speaker_confusion=0.339
dsv4-pro-base,dx-addressee-3-deictic,3shot_collide,free,0.484,126,0.286,0.044,,other_object=0.079;previous_owner=0.103;speaker_confusion=0.333
dsv4-pro-base,dx-addressee-3-deictic,3shot,all,0.511,180,0.287,0.037,,other_object=0.056;previous_owner=0.056;speaker_confusion=0.372;tie=0.006
dsv4-pro-base,dx-addressee-3-deictic,3shot,free,0.492,126,0.286,0.044,,other_object=0.048;previous_owner=0.079;speaker_confusion=0.373;tie=0.008
dsv4-pro-base,dx-addressee-3-third,0shot,all,0.911,180,0.287,0.021,,
dsv4-pro-base,dx-addressee-3-third,0shot,free,0.905,126,0.286,0.026,,
dsv4-pro-base,dx-addressee-3-third,3shot_collide,all,1.000,180,0.287,0.000,,
dsv4-pro-base,dx-addressee-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
dsv4-pro-base,dx-addressee-3-third,3shot,all,1.000,180,0.287,0.000,,
dsv4-pro-base,dx-addressee-3-third,3shot,free,1.000,126,0.286,0.000,,
dsv4-pro-base,dx-addressee-2-deictic,0shot,all,0.289,180,0.287,0.034,,speaker_confusion=0.644;tie=0.067
dsv4-pro-base,dx-addressee-2-deictic,0shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.603;tie=0.087
dsv4-pro-base,dx-addressee-2-deictic,3shot_collide,all,0.361,180,0.287,0.036,,speaker_confusion=0.611;tie=0.028
dsv4-pro-base,dx-addressee-2-deictic,3shot_collide,free,0.357,126,0.286,0.043,,speaker_confusion=0.619;tie=0.024
dsv4-pro-base,dx-addressee-2-deictic,3shot,all,0.411,180,0.287,0.037,,speaker_confusion=0.561;tie=0.028
dsv4-pro-base,dx-addressee-2-deictic,3shot,free,0.397,126,0.286,0.044,,speaker_confusion=0.571;tie=0.032
dsv4-pro-base,dx-addressee-2-third,0shot,all,0.967,180,0.287,0.013,,
dsv4-pro-base,dx-addressee-2-third,0shot,free,0.960,126,0.286,0.017,,
dsv4-pro-base,dx-addressee-2-third,3shot_collide,all,0.967,180,0.287,0.013,,
dsv4-pro-base,dx-addressee-2-third,3shot_collide,free,0.992,126,0.286,0.008,,
dsv4-pro-base,dx-addressee-2-third,3shot,all,0.967,180,0.287,0.013,,
dsv4-pro-base,dx-addressee-2-third,3shot,free,0.984,126,0.286,0.011,,
dsv4-pro-base,sb-k1,0shot,all,0.938,64,0.125,0.030,,
dsv4-pro-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
dsv4-pro-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
dsv4-pro-base,sb-k23,0shot,all,0.711,128,0.027,0.040,,
dsv4-pro-base,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
dsv4-pro-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
dsv4-pro-base,sb-split,0shot,all,0.898,128,0.027,0.027,,
dsv4-pro-base,sb-split,3shot_collide,all,0.914,128,0.027,0.025,,
dsv4-pro-base,sb-split,3shot,all,0.914,128,0.027,0.025,,
dsv4-pro-base,sb-dyn,0shot,all,0.883,256,0.027,0.020,,
dsv4-pro-base,sb-dyn,3shot_collide,all,0.883,256,0.027,0.020,,
dsv4-pro-base,sb-dyn,3shot,all,0.863,256,0.027,0.021,,
dsv4-pro-base,an-narr,0shot,all,0.494,640,0.250,0.020,,
dsv4-pro-base,an-narr,3shot_collide,all,1.000,640,0.250,0.000,,
dsv4-pro-base,an-narr,3shot,all,1.000,640,0.250,0.000,,
dsv4-pro-base,an-other-any,0shot,all,0.175,320,0.250,0.021,,
dsv4-pro-base,an-other-any,3shot_collide,all,0.969,320,0.250,0.010,,
dsv4-pro-base,an-other-any,3shot,all,0.959,320,0.250,0.011,,
dsv4-pro-base,an-me-any,0shot,all,0.400,160,0.250,0.039,,
dsv4-pro-base,an-me-any,3shot_collide,all,0.981,160,0.250,0.011,,
dsv4-pro-base,an-me-any,3shot,all,0.988,160,0.250,0.009,,
dsv4-pro-base,an-you-voc,0shot,all,0.375,80,0.250,0.054,,
dsv4-pro-base,an-you-voc,3shot_collide,all,0.938,80,0.250,0.027,,
dsv4-pro-base,an-you-voc,3shot,all,0.887,80,0.250,0.035,,
dsv4-pro-base,an-you-novoc,0shot,all,0.838,80,0.250,0.041,,
dsv4-pro-base,an-you-novoc,3shot_collide,all,0.613,80,0.250,0.054,,
dsv4-pro-base,an-you-novoc,3shot,all,0.525,80,0.250,0.056,,
dsv4-pro-base,adj-adj,0shot,all,0.977,260,0.333,0.009,,
dsv4-pro-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-voc,0shot,all,0.996,260,0.333,0.004,,
dsv4-pro-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
dsv4-pro-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
dsv4-pro-base,cm-mutate,0shot,all,0.995,200,0.200,0.005,,
dsv4-pro-base,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
dsv4-pro-base,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
dsv4-pro-base,cm-rebind,0shot,all,0.885,200,0.200,0.023,,
dsv4-pro-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
dsv4-pro-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
dsv4-pro-base,cm-copy,0shot,all,0.980,200,0.200,0.010,,
dsv4-pro-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
dsv4-pro-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
kimi-k2-base,attr,0shot,all,0.973,1210,0.168,0.005,,
kimi-k2-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
kimi-k2-base,attr,3shot,all,1.000,1210,0.168,0.000,,
kimi-k2-base,codevar,0shot,all,0.891,1200,0.267,0.009,,
kimi-k2-base,codevar,3shot_collide,all,0.954,1200,0.267,0.006,,
kimi-k2-base,codevar,3shot,all,0.927,1200,0.267,0.007,,
kimi-k2-base,dialogue,0shot,all,0.975,1200,0.312,0.004,,
kimi-k2-base,dialogue,3shot_collide,all,0.990,1200,0.312,0.003,,
kimi-k2-base,dialogue,3shot,all,0.989,1200,0.312,0.003,,
kimi-k2-base,dx-owner-3-deictic,0shot,all,0.758,120,0.333,0.039,,other_participant=0.2;speaker_confusion=0.033;tie=0.008
kimi-k2-base,dx-owner-3-deictic,0shot,free,0.675,40,0.333,0.074,,other_participant=0.325
kimi-k2-base,dx-owner-3-deictic,3shot_collide,all,0.992,120,0.333,0.008,,other_participant=0.008
kimi-k2-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
kimi-k2-base,dx-owner-3-deictic,3shot,all,1.000,120,0.333,0.000,,
kimi-k2-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,0shot,all,1.000,120,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,0shot,free,1.000,40,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
kimi-k2-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
kimi-k2-base,dx-self-3-deictic,0shot,all,0.406,180,0.287,0.037,,other_object=0.1;previous_owner=0.078;speaker_confusion=0.378;tie=0.039
kimi-k2-base,dx-self-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.103;previous_owner=0.04;speaker_confusion=0.429;tie=0.04
kimi-k2-base,dx-self-3-deictic,3shot_collide,all,0.900,180,0.287,0.022,,other_object=0.028;previous_owner=0.017;speaker_confusion=0.05;tie=0.006
kimi-k2-base,dx-self-3-deictic,3shot_collide,free,0.905,126,0.286,0.026,,other_object=0.04;previous_owner=0.016;speaker_confusion=0.04
kimi-k2-base,dx-self-3-deictic,3shot,all,0.922,180,0.287,0.020,,other_object=0.011;previous_owner=0.022;speaker_confusion=0.039;tie=0.006
kimi-k2-base,dx-self-3-deictic,3shot,free,0.905,126,0.286,0.026,,other_object=0.016;previous_owner=0.032;speaker_confusion=0.04;tie=0.008
kimi-k2-base,dx-self-3-third,0shot,all,0.906,180,0.287,0.022,,
kimi-k2-base,dx-self-3-third,0shot,free,0.952,126,0.286,0.019,,
kimi-k2-base,dx-self-3-third,3shot_collide,all,0.983,180,0.287,0.009,,
kimi-k2-base,dx-self-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
kimi-k2-base,dx-self-3-third,3shot,all,0.978,180,0.287,0.011,,
kimi-k2-base,dx-self-3-third,3shot,free,1.000,126,0.286,0.000,,
kimi-k2-base,dx-self-2-deictic,0shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.672;tie=0.011
kimi-k2-base,dx-self-2-deictic,0shot,free,0.325,126,0.286,0.042,,speaker_confusion=0.667;tie=0.008
kimi-k2-base,dx-self-2-deictic,3shot_collide,all,0.656,180,0.287,0.035,,speaker_confusion=0.344
kimi-k2-base,dx-self-2-deictic,3shot_collide,free,0.675,126,0.286,0.042,,speaker_confusion=0.325
kimi-k2-base,dx-self-2-deictic,3shot,all,0.678,180,0.287,0.035,,speaker_confusion=0.3;tie=0.022
kimi-k2-base,dx-self-2-deictic,3shot,free,0.682,126,0.286,0.042,,speaker_confusion=0.294;tie=0.024
kimi-k2-base,dx-self-2-third,0shot,all,0.900,180,0.287,0.022,,
kimi-k2-base,dx-self-2-third,0shot,free,0.936,126,0.286,0.022,,
kimi-k2-base,dx-self-2-third,3shot_collide,all,0.967,180,0.287,0.013,,
kimi-k2-base,dx-self-2-third,3shot_collide,free,0.968,126,0.286,0.016,,
kimi-k2-base,dx-self-2-third,3shot,all,0.972,180,0.287,0.012,,
kimi-k2-base,dx-self-2-third,3shot,free,0.968,126,0.286,0.016,,
kimi-k2-base,dx-addressee-3-deictic,0shot,all,0.428,180,0.287,0.037,,other_object=0.072;previous_owner=0.161;speaker_confusion=0.328;tie=0.011
kimi-k2-base,dx-addressee-3-deictic,0shot,free,0.389,126,0.286,0.043,,other_object=0.04;previous_owner=0.19;speaker_confusion=0.373;tie=0.008
kimi-k2-base,dx-addressee-3-deictic,3shot_collide,all,0.700,180,0.287,0.034,,other_object=0.011;previous_owner=0.05;speaker_confusion=0.233;tie=0.006
kimi-k2-base,dx-addressee-3-deictic,3shot_collide,free,0.682,126,0.286,0.042,,other_object=0.008;previous_owner=0.056;speaker_confusion=0.246;tie=0.008
kimi-k2-base,dx-addressee-3-deictic,3shot,all,0.733,180,0.287,0.033,,other_object=0.011;previous_owner=0.044;speaker_confusion=0.189;tie=0.022
kimi-k2-base,dx-addressee-3-deictic,3shot,free,0.738,126,0.286,0.039,,other_object=0.008;previous_owner=0.056;speaker_confusion=0.19;tie=0.008
kimi-k2-base,dx-addressee-3-third,0shot,all,0.944,180,0.287,0.017,,
kimi-k2-base,dx-addressee-3-third,0shot,free,0.968,126,0.286,0.016,,
kimi-k2-base,dx-addressee-3-third,3shot_collide,all,0.983,180,0.287,0.009,,
kimi-k2-base,dx-addressee-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
kimi-k2-base,dx-addressee-3-third,3shot,all,1.000,180,0.287,0.000,,
kimi-k2-base,dx-addressee-3-third,3shot,free,1.000,126,0.286,0.000,,
kimi-k2-base,dx-addressee-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.656;tie=0.039
kimi-k2-base,dx-addressee-2-deictic,0shot,free,0.254,126,0.286,0.039,,speaker_confusion=0.698;tie=0.048
kimi-k2-base,dx-addressee-2-deictic,3shot_collide,all,0.383,180,0.287,0.036,,speaker_confusion=0.572;tie=0.044
kimi-k2-base,dx-addressee-2-deictic,3shot_collide,free,0.397,126,0.286,0.044,,speaker_confusion=0.548;tie=0.056
kimi-k2-base,dx-addressee-2-deictic,3shot,all,0.422,180,0.287,0.037,,speaker_confusion=0.556;tie=0.022
kimi-k2-base,dx-addressee-2-deictic,3shot,free,0.436,126,0.286,0.044,,speaker_confusion=0.532;tie=0.032
kimi-k2-base,dx-addressee-2-third,0shot,all,0.883,180,0.287,0.024,,
kimi-k2-base,dx-addressee-2-third,0shot,free,0.929,126,0.286,0.023,,
kimi-k2-base,dx-addressee-2-third,3shot_collide,all,0.961,180,0.287,0.014,,
kimi-k2-base,dx-addressee-2-third,3shot_collide,free,0.976,126,0.286,0.014,,
kimi-k2-base,dx-addressee-2-third,3shot,all,0.978,180,0.287,0.011,,
kimi-k2-base,dx-addressee-2-third,3shot,free,0.984,126,0.286,0.011,,
kimi-k2-base,sb-k1,0shot,all,1.000,64,0.125,0.000,,
kimi-k2-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
kimi-k2-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
kimi-k2-base,sb-k23,0shot,all,0.930,128,0.027,0.023,,
kimi-k2-base,sb-k23,3shot_collide,all,1.000,128,0.027,0.000,,
kimi-k2-base,sb-k23,3shot,all,1.000,128,0.027,0.000,,
kimi-k2-base,sb-split,0shot,all,0.828,128,0.027,0.033,,
kimi-k2-base,sb-split,3shot_collide,all,1.000,128,0.027,0.000,,
kimi-k2-base,sb-split,3shot,all,0.992,128,0.027,0.008,,
kimi-k2-base,sb-dyn,0shot,all,0.742,256,0.027,0.027,,
kimi-k2-base,sb-dyn,3shot_collide,all,0.879,256,0.027,0.020,,
kimi-k2-base,sb-dyn,3shot,all,0.840,256,0.027,0.023,,
kimi-k2-base,an-narr,0shot,all,0.727,640,0.250,0.018,,
kimi-k2-base,an-narr,3shot_collide,all,0.998,640,0.250,0.002,,
kimi-k2-base,an-narr,3shot,all,0.998,640,0.250,0.002,,
kimi-k2-base,an-other-any,0shot,all,0.247,320,0.250,0.024,,
kimi-k2-base,an-other-any,3shot_collide,all,0.984,320,0.250,0.007,,
kimi-k2-base,an-other-any,3shot,all,0.969,320,0.250,0.010,,
kimi-k2-base,an-me-any,0shot,all,0.525,160,0.250,0.040,,
kimi-k2-base,an-me-any,3shot_collide,all,0.906,160,0.250,0.023,,
kimi-k2-base,an-me-any,3shot,all,0.938,160,0.250,0.019,,
kimi-k2-base,an-you-voc,0shot,all,0.838,80,0.250,0.041,,
kimi-k2-base,an-you-voc,3shot_collide,all,0.988,80,0.250,0.012,,
kimi-k2-base,an-you-voc,3shot,all,0.975,80,0.250,0.018,,
kimi-k2-base,an-you-novoc,0shot,all,0.925,80,0.250,0.029,,
kimi-k2-base,an-you-novoc,3shot_collide,all,0.938,80,0.250,0.027,,
kimi-k2-base,an-you-novoc,3shot,all,0.912,80,0.250,0.032,,
kimi-k2-base,adj-adj,0shot,all,0.942,260,0.333,0.015,,
kimi-k2-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-voc,0shot,all,0.977,260,0.333,0.009,,
kimi-k2-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
kimi-k2-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
kimi-k2-base,cm-mutate,0shot,all,0.975,200,0.200,0.011,,
kimi-k2-base,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
kimi-k2-base,cm-mutate,3shot,all,1.000,200,0.200,0.000,,
kimi-k2-base,cm-rebind,0shot,all,0.970,200,0.200,0.012,,
kimi-k2-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
kimi-k2-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
kimi-k2-base,cm-copy,0shot,all,0.900,200,0.200,0.021,,
kimi-k2-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
kimi-k2-base,cm-copy,3shot,all,1.000,200,0.200,0.000,,
mimo-v25-pro-base,attr,0shot,all,0.692,1210,0.168,0.013,,
mimo-v25-pro-base,attr,3shot_collide,all,1.000,1210,0.168,0.000,,
mimo-v25-pro-base,attr,3shot,all,1.000,1210,0.168,0.000,,
mimo-v25-pro-base,codevar,0shot,all,0.766,1200,0.267,0.012,,
mimo-v25-pro-base,codevar,3shot_collide,all,0.908,1200,0.267,0.008,,
mimo-v25-pro-base,codevar,3shot,all,0.893,1200,0.267,0.009,,
mimo-v25-pro-base,dialogue,0shot,all,0.916,1200,0.312,0.008,,
mimo-v25-pro-base,dialogue,3shot_collide,all,0.985,1200,0.312,0.004,,
mimo-v25-pro-base,dialogue,3shot,all,0.978,1200,0.312,0.004,,
mimo-v25-pro-base,dx-owner-3-deictic,0shot,all,0.542,120,0.333,0.045,,other_participant=0.325;speaker_confusion=0.033;tie=0.1
mimo-v25-pro-base,dx-owner-3-deictic,0shot,free,0.450,40,0.333,0.079,,other_participant=0.425;speaker_confusion=0.05;tie=0.075
mimo-v25-pro-base,dx-owner-3-deictic,3shot_collide,all,0.967,120,0.333,0.016,,other_participant=0.025;speaker_confusion=0.008
mimo-v25-pro-base,dx-owner-3-deictic,3shot_collide,free,0.975,40,0.333,0.025,,other_participant=0.025
mimo-v25-pro-base,dx-owner-3-deictic,3shot,all,0.992,120,0.333,0.008,,other_participant=0.008
mimo-v25-pro-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
mimo-v25-pro-base,dx-owner-3-third,0shot,all,0.833,120,0.333,0.034,,
mimo-v25-pro-base,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
mimo-v25-pro-base,dx-owner-3-third,3shot_collide,all,0.992,120,0.333,0.008,,
mimo-v25-pro-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
mimo-v25-pro-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
mimo-v25-pro-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
mimo-v25-pro-base,dx-self-3-deictic,0shot,all,0.217,180,0.287,0.031,,other_object=0.261;previous_owner=0.067;speaker_confusion=0.306;tie=0.15
mimo-v25-pro-base,dx-self-3-deictic,0shot,free,0.246,126,0.286,0.038,,other_object=0.31;previous_owner=0.063;speaker_confusion=0.23;tie=0.151
mimo-v25-pro-base,dx-self-3-deictic,3shot_collide,all,0.828,180,0.287,0.028,,other_object=0.017;previous_owner=0.028;speaker_confusion=0.1;tie=0.028
mimo-v25-pro-base,dx-self-3-deictic,3shot_collide,free,0.889,126,0.286,0.028,,other_object=0.024;previous_owner=0.008;speaker_confusion=0.063;tie=0.016
mimo-v25-pro-base,dx-self-3-deictic,3shot,all,0.850,180,0.287,0.027,,other_object=0.033;previous_owner=0.028;speaker_confusion=0.061;tie=0.028
mimo-v25-pro-base,dx-self-3-deictic,3shot,free,0.857,126,0.286,0.031,,other_object=0.04;previous_owner=0.024;speaker_confusion=0.056;tie=0.024
mimo-v25-pro-base,dx-self-3-third,0shot,all,0.844,180,0.287,0.027,,
mimo-v25-pro-base,dx-self-3-third,0shot,free,0.881,126,0.286,0.029,,
mimo-v25-pro-base,dx-self-3-third,3shot_collide,all,0.972,180,0.287,0.012,,
mimo-v25-pro-base,dx-self-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
mimo-v25-pro-base,dx-self-3-third,3shot,all,0.978,180,0.287,0.011,,
mimo-v25-pro-base,dx-self-3-third,3shot,free,0.992,126,0.286,0.008,,
mimo-v25-pro-base,dx-self-2-deictic,0shot,all,0.172,180,0.287,0.028,,speaker_confusion=0.611;tie=0.217
mimo-v25-pro-base,dx-self-2-deictic,0shot,free,0.206,126,0.286,0.036,,speaker_confusion=0.563;tie=0.23
mimo-v25-pro-base,dx-self-2-deictic,3shot_collide,all,0.567,180,0.287,0.037,,speaker_confusion=0.322;tie=0.111
mimo-v25-pro-base,dx-self-2-deictic,3shot_collide,free,0.635,126,0.286,0.043,,speaker_confusion=0.262;tie=0.103
mimo-v25-pro-base,dx-self-2-deictic,3shot,all,0.606,180,0.287,0.036,,speaker_confusion=0.294;tie=0.1
mimo-v25-pro-base,dx-self-2-deictic,3shot,free,0.643,126,0.286,0.043,,speaker_confusion=0.262;tie=0.095
mimo-v25-pro-base,dx-self-2-third,0shot,all,0.844,180,0.287,0.027,,
mimo-v25-pro-base,dx-self-2-third,0shot,free,0.881,126,0.286,0.029,,
mimo-v25-pro-base,dx-self-2-third,3shot_collide,all,0.956,180,0.287,0.015,,
mimo-v25-pro-base,dx-self-2-third,3shot_collide,free,0.976,126,0.286,0.014,,
mimo-v25-pro-base,dx-self-2-third,3shot,all,0.956,180,0.287,0.015,,
mimo-v25-pro-base,dx-self-2-third,3shot,free,0.968,126,0.286,0.016,,
mimo-v25-pro-base,dx-addressee-3-deictic,0shot,all,0.233,180,0.287,0.032,,other_object=0.233;previous_owner=0.056;speaker_confusion=0.406;tie=0.072
mimo-v25-pro-base,dx-addressee-3-deictic,0shot,free,0.262,126,0.286,0.039,,other_object=0.214;previous_owner=0.071;speaker_confusion=0.373;tie=0.079
mimo-v25-pro-base,dx-addressee-3-deictic,3shot_collide,all,0.778,180,0.287,0.031,,other_object=0.033;previous_owner=0.022;speaker_confusion=0.133;tie=0.033
mimo-v25-pro-base,dx-addressee-3-deictic,3shot_collide,free,0.825,126,0.286,0.034,,other_object=0.008;previous_owner=0.016;speaker_confusion=0.127;tie=0.024
mimo-v25-pro-base,dx-addressee-3-deictic,3shot,all,0.733,180,0.287,0.033,,other_object=0.033;previous_owner=0.044;speaker_confusion=0.15;tie=0.039
mimo-v25-pro-base,dx-addressee-3-deictic,3shot,free,0.722,126,0.286,0.040,,other_object=0.04;previous_owner=0.048;speaker_confusion=0.151;tie=0.04
mimo-v25-pro-base,dx-addressee-3-third,0shot,all,0.850,180,0.287,0.027,,
mimo-v25-pro-base,dx-addressee-3-third,0shot,free,0.881,126,0.286,0.029,,
mimo-v25-pro-base,dx-addressee-3-third,3shot_collide,all,0.994,180,0.287,0.005,,
mimo-v25-pro-base,dx-addressee-3-third,3shot_collide,free,1.000,126,0.286,0.000,,
mimo-v25-pro-base,dx-addressee-3-third,3shot,all,0.994,180,0.287,0.005,,
mimo-v25-pro-base,dx-addressee-3-third,3shot,free,1.000,126,0.286,0.000,,
mimo-v25-pro-base,dx-addressee-2-deictic,0shot,all,0.167,180,0.287,0.028,,speaker_confusion=0.628;tie=0.206
mimo-v25-pro-base,dx-addressee-2-deictic,0shot,free,0.175,126,0.286,0.034,,speaker_confusion=0.643;tie=0.183
mimo-v25-pro-base,dx-addressee-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.561;tie=0.111
mimo-v25-pro-base,dx-addressee-2-deictic,3shot_collide,free,0.349,126,0.286,0.043,,speaker_confusion=0.532;tie=0.119
mimo-v25-pro-base,dx-addressee-2-deictic,3shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.533;tie=0.139
mimo-v25-pro-base,dx-addressee-2-deictic,3shot,free,0.357,126,0.286,0.043,,speaker_confusion=0.492;tie=0.151
mimo-v25-pro-base,dx-addressee-2-third,0shot,all,0.889,180,0.287,0.023,,
mimo-v25-pro-base,dx-addressee-2-third,0shot,free,0.905,126,0.286,0.026,,
mimo-v25-pro-base,dx-addressee-2-third,3shot_collide,all,0.928,180,0.287,0.019,,
mimo-v25-pro-base,dx-addressee-2-third,3shot_collide,free,0.944,126,0.286,0.020,,
mimo-v25-pro-base,dx-addressee-2-third,3shot,all,0.939,180,0.287,0.018,,
mimo-v25-pro-base,dx-addressee-2-third,3shot,free,0.968,126,0.286,0.016,,
mimo-v25-pro-base,sb-k1,0shot,all,0.969,64,0.125,0.022,,
mimo-v25-pro-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
mimo-v25-pro-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
mimo-v25-pro-base,sb-k23,0shot,all,0.539,128,0.027,0.044,,
mimo-v25-pro-base,sb-k23,3shot_collide,all,0.984,128,0.027,0.011,,
mimo-v25-pro-base,sb-k23,3shot,all,0.992,128,0.027,0.008,,
mimo-v25-pro-base,sb-split,0shot,all,0.508,128,0.027,0.044,,
mimo-v25-pro-base,sb-split,3shot_collide,all,0.930,128,0.027,0.023,,
mimo-v25-pro-base,sb-split,3shot,all,0.875,128,0.027,0.029,,
mimo-v25-pro-base,sb-dyn,0shot,all,0.551,256,0.027,0.031,,
mimo-v25-pro-base,sb-dyn,3shot_collide,all,0.812,256,0.027,0.024,,
mimo-v25-pro-base,sb-dyn,3shot,all,0.797,256,0.027,0.025,,
mimo-v25-pro-base,an-narr,0shot,all,0.439,640,0.250,0.020,,
mimo-v25-pro-base,an-narr,3shot_collide,all,0.992,640,0.250,0.004,,
mimo-v25-pro-base,an-narr,3shot,all,0.994,640,0.250,0.003,,
mimo-v25-pro-base,an-other-any,0shot,all,0.175,320,0.250,0.021,,
mimo-v25-pro-base,an-other-any,3shot_collide,all,0.994,320,0.250,0.004,,
mimo-v25-pro-base,an-other-any,3shot,all,0.950,320,0.250,0.012,,
mimo-v25-pro-base,an-me-any,0shot,all,0.662,160,0.250,0.037,,
mimo-v25-pro-base,an-me-any,3shot_collide,all,0.956,160,0.250,0.016,,
mimo-v25-pro-base,an-me-any,3shot,all,0.925,160,0.250,0.021,,
mimo-v25-pro-base,an-you-voc,0shot,all,0.300,80,0.250,0.051,,
mimo-v25-pro-base,an-you-voc,3shot_collide,all,0.988,80,0.250,0.012,,
mimo-v25-pro-base,an-you-voc,3shot,all,1.000,80,0.250,0.000,,
mimo-v25-pro-base,an-you-novoc,0shot,all,0.450,80,0.250,0.056,,
mimo-v25-pro-base,an-you-novoc,3shot_collide,all,0.850,80,0.250,0.040,,
mimo-v25-pro-base,an-you-novoc,3shot,all,0.838,80,0.250,0.041,,
mimo-v25-pro-base,adj-adj,0shot,all,0.877,260,0.333,0.020,,
mimo-v25-pro-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
mimo-v25-pro-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-pro-base,adj-voc,0shot,all,0.969,260,0.333,0.011,,
mimo-v25-pro-base,adj-voc,3shot_collide,all,0.996,260,0.333,0.004,,
mimo-v25-pro-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-pro-base,adj-third,0shot,all,0.989,260,0.333,0.007,,
mimo-v25-pro-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
mimo-v25-pro-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
mimo-v25-pro-base,cm-mutate,0shot,all,0.980,200,0.200,0.010,,
mimo-v25-pro-base,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
mimo-v25-pro-base,cm-mutate,3shot,all,0.995,200,0.200,0.005,,
mimo-v25-pro-base,cm-rebind,0shot,all,0.895,200,0.200,0.022,,
mimo-v25-pro-base,cm-rebind,3shot_collide,all,0.995,200,0.200,0.005,,
mimo-v25-pro-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
mimo-v25-pro-base,cm-copy,0shot,all,0.885,200,0.200,0.023,,
mimo-v25-pro-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
mimo-v25-pro-base,cm-copy,3shot,all,0.995,200,0.200,0.005,,
olmo2-7b-pt2928B,dx-owner-3-deictic,0shot,all,0.517,120,0.333,0.046,,other_participant=0.417;speaker_confusion=0.067
olmo2-7b-pt2928B,dx-owner-3-deictic,0shot,free,0.275,40,0.333,0.071,,other_participant=0.6;speaker_confusion=0.125
olmo2-7b-pt2928B,dx-owner-3-deictic,3shot_collide,all,0.583,120,0.333,0.045,,other_participant=0.183;speaker_confusion=0.225;tie=0.008
olmo2-7b-pt2928B,dx-owner-3-deictic,3shot_collide,free,0.700,40,0.333,0.072,,other_participant=0.125;speaker_confusion=0.175
olmo2-7b-pt2928B,dx-owner-3-deictic,3shot,all,0.550,120,0.333,0.045,,other_participant=0.275;speaker_confusion=0.175
olmo2-7b-pt2928B,dx-owner-3-deictic,3shot,free,0.700,40,0.333,0.072,,other_participant=0.175;speaker_confusion=0.125
olmo2-7b-pt2928B,dx-owner-3-third,0shot,all,0.508,120,0.333,0.046,,
olmo2-7b-pt2928B,dx-owner-3-third,0shot,free,0.300,40,0.333,0.072,,
olmo2-7b-pt2928B,dx-owner-3-third,3shot_collide,all,0.558,120,0.333,0.045,,
olmo2-7b-pt2928B,dx-owner-3-third,3shot_collide,free,0.700,40,0.333,0.072,,
olmo2-7b-pt2928B,dx-owner-3-third,3shot,all,0.600,120,0.333,0.045,,
olmo2-7b-pt2928B,dx-owner-3-third,3shot,free,0.875,40,0.333,0.052,,
olmo2-7b-pt2928B,dx-self-3-deictic,0shot,all,0.350,180,0.287,0.036,,other_object=0.3;previous_owner=0.05;speaker_confusion=0.3
olmo2-7b-pt2928B,dx-self-3-deictic,0shot,free,0.373,126,0.286,0.043,,other_object=0.325;previous_owner=0.024;speaker_confusion=0.278
olmo2-7b-pt2928B,dx-self-3-deictic,3shot_collide,all,0.394,180,0.287,0.036,,other_object=0.3;previous_owner=0.028;speaker_confusion=0.278
olmo2-7b-pt2928B,dx-self-3-deictic,3shot_collide,free,0.413,126,0.286,0.044,,other_object=0.333;previous_owner=0.024;speaker_confusion=0.23
olmo2-7b-pt2928B,dx-self-3-deictic,3shot,all,0.367,180,0.287,0.036,,other_object=0.272;previous_owner=0.022;speaker_confusion=0.339
olmo2-7b-pt2928B,dx-self-3-deictic,3shot,free,0.373,126,0.286,0.043,,other_object=0.294;previous_owner=0.024;speaker_confusion=0.31
olmo2-7b-pt2928B,dx-self-3-third,0shot,all,0.528,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-self-3-third,0shot,free,0.452,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-self-3-third,3shot_collide,all,0.506,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-self-3-third,3shot_collide,free,0.500,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-self-3-third,3shot,all,0.567,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-self-3-third,3shot,free,0.564,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-self-2-deictic,0shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.694
olmo2-7b-pt2928B,dx-self-2-deictic,0shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.69
olmo2-7b-pt2928B,dx-self-2-deictic,3shot_collide,all,0.389,180,0.287,0.036,,speaker_confusion=0.611
olmo2-7b-pt2928B,dx-self-2-deictic,3shot_collide,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
olmo2-7b-pt2928B,dx-self-2-deictic,3shot,all,0.422,180,0.287,0.037,,speaker_confusion=0.578
olmo2-7b-pt2928B,dx-self-2-deictic,3shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
olmo2-7b-pt2928B,dx-self-2-third,0shot,all,0.517,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-self-2-third,0shot,free,0.516,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-self-2-third,3shot_collide,all,0.550,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-self-2-third,3shot_collide,free,0.587,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-self-2-third,3shot,all,0.611,180,0.287,0.036,,
olmo2-7b-pt2928B,dx-self-2-third,3shot,free,0.675,126,0.286,0.042,,
olmo2-7b-pt2928B,dx-addressee-3-deictic,0shot,all,0.267,180,0.287,0.033,,other_object=0.294;previous_owner=0.056;speaker_confusion=0.383
olmo2-7b-pt2928B,dx-addressee-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.222;previous_owner=0.071;speaker_confusion=0.397
olmo2-7b-pt2928B,dx-addressee-3-deictic,3shot_collide,all,0.300,180,0.287,0.034,,other_object=0.222;previous_owner=0.072;speaker_confusion=0.4;tie=0.006
olmo2-7b-pt2928B,dx-addressee-3-deictic,3shot_collide,free,0.302,126,0.286,0.041,,other_object=0.206;previous_owner=0.087;speaker_confusion=0.397;tie=0.008
olmo2-7b-pt2928B,dx-addressee-3-deictic,3shot,all,0.328,180,0.287,0.035,,other_object=0.222;previous_owner=0.078;speaker_confusion=0.372
olmo2-7b-pt2928B,dx-addressee-3-deictic,3shot,free,0.294,126,0.286,0.041,,other_object=0.214;previous_owner=0.111;speaker_confusion=0.381
olmo2-7b-pt2928B,dx-addressee-3-third,0shot,all,0.539,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-addressee-3-third,0shot,free,0.436,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-addressee-3-third,3shot_collide,all,0.578,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-addressee-3-third,3shot_collide,free,0.595,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-addressee-3-third,3shot,all,0.600,180,0.287,0.036,,
olmo2-7b-pt2928B,dx-addressee-3-third,3shot,free,0.571,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-addressee-2-deictic,0shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.717
olmo2-7b-pt2928B,dx-addressee-2-deictic,0shot,free,0.270,126,0.286,0.040,,speaker_confusion=0.73
olmo2-7b-pt2928B,dx-addressee-2-deictic,3shot_collide,all,0.289,180,0.287,0.034,,speaker_confusion=0.711
olmo2-7b-pt2928B,dx-addressee-2-deictic,3shot_collide,free,0.309,126,0.286,0.041,,speaker_confusion=0.69
olmo2-7b-pt2928B,dx-addressee-2-deictic,3shot,all,0.289,180,0.287,0.034,,speaker_confusion=0.711
olmo2-7b-pt2928B,dx-addressee-2-deictic,3shot,free,0.318,126,0.286,0.042,,speaker_confusion=0.683
olmo2-7b-pt2928B,dx-addressee-2-third,0shot,all,0.494,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-addressee-2-third,0shot,free,0.468,126,0.286,0.044,,
olmo2-7b-pt2928B,dx-addressee-2-third,3shot_collide,all,0.567,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-addressee-2-third,3shot_collide,free,0.627,126,0.286,0.043,,
olmo2-7b-pt2928B,dx-addressee-2-third,3shot,all,0.594,180,0.287,0.037,,
olmo2-7b-pt2928B,dx-addressee-2-third,3shot,free,0.643,126,0.286,0.043,,
olmo2-7b-pt3896B,dx-owner-3-deictic,0shot,all,0.550,120,0.333,0.045,,other_participant=0.392;speaker_confusion=0.058
olmo2-7b-pt3896B,dx-owner-3-deictic,0shot,free,0.450,40,0.333,0.079,,other_participant=0.475;speaker_confusion=0.075
olmo2-7b-pt3896B,dx-owner-3-deictic,3shot_collide,all,0.633,120,0.333,0.044,,other_participant=0.15;speaker_confusion=0.217
olmo2-7b-pt3896B,dx-owner-3-deictic,3shot_collide,free,0.725,40,0.333,0.071,,other_participant=0.125;speaker_confusion=0.15
olmo2-7b-pt3896B,dx-owner-3-deictic,3shot,all,0.600,120,0.333,0.045,,other_participant=0.183;speaker_confusion=0.217
olmo2-7b-pt3896B,dx-owner-3-deictic,3shot,free,0.700,40,0.333,0.072,,other_participant=0.175;speaker_confusion=0.125
olmo2-7b-pt3896B,dx-owner-3-third,0shot,all,0.583,120,0.333,0.045,,
olmo2-7b-pt3896B,dx-owner-3-third,0shot,free,0.550,40,0.333,0.079,,
olmo2-7b-pt3896B,dx-owner-3-third,3shot_collide,all,0.533,120,0.333,0.045,,
olmo2-7b-pt3896B,dx-owner-3-third,3shot_collide,free,0.650,40,0.333,0.075,,
olmo2-7b-pt3896B,dx-owner-3-third,3shot,all,0.633,120,0.333,0.044,,
olmo2-7b-pt3896B,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
olmo2-7b-pt3896B,dx-self-3-deictic,0shot,all,0.283,180,0.287,0.034,,other_object=0.322;previous_owner=0.067;speaker_confusion=0.328
olmo2-7b-pt3896B,dx-self-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.341;previous_owner=0.016;speaker_confusion=0.333
olmo2-7b-pt3896B,dx-self-3-deictic,3shot_collide,all,0.372,180,0.287,0.036,,other_object=0.25;previous_owner=0.061;speaker_confusion=0.317
olmo2-7b-pt3896B,dx-self-3-deictic,3shot_collide,free,0.405,126,0.286,0.044,,other_object=0.286;previous_owner=0.048;speaker_confusion=0.262
olmo2-7b-pt3896B,dx-self-3-deictic,3shot,all,0.356,180,0.287,0.036,,other_object=0.283;previous_owner=0.067;speaker_confusion=0.294
olmo2-7b-pt3896B,dx-self-3-deictic,3shot,free,0.349,126,0.286,0.043,,other_object=0.333;previous_owner=0.04;speaker_confusion=0.278
olmo2-7b-pt3896B,dx-self-3-third,0shot,all,0.444,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-3-third,0shot,free,0.397,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-self-3-third,3shot_collide,all,0.439,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-3-third,3shot_collide,free,0.429,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-self-3-third,3shot,all,0.511,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-3-third,3shot,free,0.532,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-self-2-deictic,0shot,all,0.394,180,0.287,0.036,,speaker_confusion=0.606
olmo2-7b-pt3896B,dx-self-2-deictic,0shot,free,0.397,126,0.286,0.044,,speaker_confusion=0.603
olmo2-7b-pt3896B,dx-self-2-deictic,3shot_collide,all,0.383,180,0.287,0.036,,speaker_confusion=0.617
olmo2-7b-pt3896B,dx-self-2-deictic,3shot_collide,free,0.452,126,0.286,0.044,,speaker_confusion=0.548
olmo2-7b-pt3896B,dx-self-2-deictic,3shot,all,0.394,180,0.287,0.036,,speaker_confusion=0.606
olmo2-7b-pt3896B,dx-self-2-deictic,3shot,free,0.436,126,0.286,0.044,,speaker_confusion=0.563
olmo2-7b-pt3896B,dx-self-2-third,0shot,all,0.511,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-2-third,0shot,free,0.532,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-self-2-third,3shot_collide,all,0.517,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-2-third,3shot_collide,free,0.548,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-self-2-third,3shot,all,0.550,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-self-2-third,3shot,free,0.595,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-3-deictic,0shot,all,0.322,180,0.287,0.035,,other_object=0.206;previous_owner=0.061;speaker_confusion=0.411
olmo2-7b-pt3896B,dx-addressee-3-deictic,0shot,free,0.318,126,0.286,0.042,,other_object=0.19;previous_owner=0.087;speaker_confusion=0.405
olmo2-7b-pt3896B,dx-addressee-3-deictic,3shot_collide,all,0.306,180,0.287,0.034,,other_object=0.183;previous_owner=0.139;speaker_confusion=0.372
olmo2-7b-pt3896B,dx-addressee-3-deictic,3shot_collide,free,0.309,126,0.286,0.041,,other_object=0.143;previous_owner=0.175;speaker_confusion=0.373
olmo2-7b-pt3896B,dx-addressee-3-deictic,3shot,all,0.317,180,0.287,0.035,,other_object=0.183;previous_owner=0.144;speaker_confusion=0.356
olmo2-7b-pt3896B,dx-addressee-3-deictic,3shot,free,0.302,126,0.286,0.041,,other_object=0.135;previous_owner=0.183;speaker_confusion=0.381
olmo2-7b-pt3896B,dx-addressee-3-third,0shot,all,0.528,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-3-third,0shot,free,0.484,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-3-third,3shot_collide,all,0.483,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-3-third,3shot_collide,free,0.476,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-3-third,3shot,all,0.550,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-3-third,3shot,free,0.508,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-2-deictic,0shot,all,0.228,180,0.287,0.031,,speaker_confusion=0.772
olmo2-7b-pt3896B,dx-addressee-2-deictic,0shot,free,0.167,126,0.286,0.033,,speaker_confusion=0.833
olmo2-7b-pt3896B,dx-addressee-2-deictic,3shot_collide,all,0.344,180,0.287,0.035,,speaker_confusion=0.656
olmo2-7b-pt3896B,dx-addressee-2-deictic,3shot_collide,free,0.381,126,0.286,0.043,,speaker_confusion=0.619
olmo2-7b-pt3896B,dx-addressee-2-deictic,3shot,all,0.317,180,0.287,0.035,,speaker_confusion=0.683
olmo2-7b-pt3896B,dx-addressee-2-deictic,3shot,free,0.341,126,0.286,0.042,,speaker_confusion=0.659
olmo2-7b-pt3896B,dx-addressee-2-third,0shot,all,0.483,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-2-third,0shot,free,0.500,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-2-third,3shot_collide,all,0.500,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-2-third,3shot_collide,free,0.524,126,0.286,0.044,,
olmo2-7b-pt3896B,dx-addressee-2-third,3shot,all,0.556,180,0.287,0.037,,
olmo2-7b-pt3896B,dx-addressee-2-third,3shot,free,0.579,126,0.286,0.044,,
olmo2-7b-mid5B,dx-owner-3-deictic,0shot,all,0.533,120,0.333,0.045,,other_participant=0.358;speaker_confusion=0.108
olmo2-7b-mid5B,dx-owner-3-deictic,0shot,free,0.325,40,0.333,0.074,,other_participant=0.575;speaker_confusion=0.1
olmo2-7b-mid5B,dx-owner-3-deictic,3shot_collide,all,0.775,120,0.333,0.038,,other_participant=0.158;speaker_confusion=0.067
olmo2-7b-mid5B,dx-owner-3-deictic,3shot_collide,free,0.700,40,0.333,0.072,,other_participant=0.225;speaker_confusion=0.075
olmo2-7b-mid5B,dx-owner-3-deictic,3shot,all,0.708,120,0.333,0.042,,other_participant=0.217;speaker_confusion=0.075
olmo2-7b-mid5B,dx-owner-3-deictic,3shot,free,0.725,40,0.333,0.071,,other_participant=0.25;speaker_confusion=0.025
olmo2-7b-mid5B,dx-owner-3-third,0shot,all,0.625,120,0.333,0.044,,
olmo2-7b-mid5B,dx-owner-3-third,0shot,free,0.450,40,0.333,0.079,,
olmo2-7b-mid5B,dx-owner-3-third,3shot_collide,all,0.683,120,0.333,0.043,,
olmo2-7b-mid5B,dx-owner-3-third,3shot_collide,free,0.550,40,0.333,0.079,,
olmo2-7b-mid5B,dx-owner-3-third,3shot,all,0.742,120,0.333,0.040,,
olmo2-7b-mid5B,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
olmo2-7b-mid5B,dx-self-3-deictic,0shot,all,0.389,180,0.287,0.036,,other_object=0.289;previous_owner=0.039;speaker_confusion=0.283
olmo2-7b-mid5B,dx-self-3-deictic,0shot,free,0.397,126,0.286,0.044,,other_object=0.286;previous_owner=0.016;speaker_confusion=0.302
olmo2-7b-mid5B,dx-self-3-deictic,3shot_collide,all,0.383,180,0.287,0.036,,other_object=0.25;previous_owner=0.072;speaker_confusion=0.294
olmo2-7b-mid5B,dx-self-3-deictic,3shot_collide,free,0.421,126,0.286,0.044,,other_object=0.278;previous_owner=0.056;speaker_confusion=0.246
olmo2-7b-mid5B,dx-self-3-deictic,3shot,all,0.394,180,0.287,0.036,,other_object=0.283;previous_owner=0.039;speaker_confusion=0.283
olmo2-7b-mid5B,dx-self-3-deictic,3shot,free,0.381,126,0.286,0.043,,other_object=0.333;previous_owner=0.032;speaker_confusion=0.254
olmo2-7b-mid5B,dx-self-3-third,0shot,all,0.467,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-3-third,0shot,free,0.389,126,0.286,0.043,,
olmo2-7b-mid5B,dx-self-3-third,3shot_collide,all,0.450,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-3-third,3shot_collide,free,0.413,126,0.286,0.044,,
olmo2-7b-mid5B,dx-self-3-third,3shot,all,0.561,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-3-third,3shot,free,0.516,126,0.286,0.044,,
olmo2-7b-mid5B,dx-self-2-deictic,0shot,all,0.333,180,0.287,0.035,,speaker_confusion=0.667
olmo2-7b-mid5B,dx-self-2-deictic,0shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.69
olmo2-7b-mid5B,dx-self-2-deictic,3shot_collide,all,0.411,180,0.287,0.037,,speaker_confusion=0.583;tie=0.006
olmo2-7b-mid5B,dx-self-2-deictic,3shot_collide,free,0.476,126,0.286,0.044,,speaker_confusion=0.524
olmo2-7b-mid5B,dx-self-2-deictic,3shot,all,0.439,180,0.287,0.037,,speaker_confusion=0.561
olmo2-7b-mid5B,dx-self-2-deictic,3shot,free,0.476,126,0.286,0.044,,speaker_confusion=0.524
olmo2-7b-mid5B,dx-self-2-third,0shot,all,0.417,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-2-third,0shot,free,0.389,126,0.286,0.043,,
olmo2-7b-mid5B,dx-self-2-third,3shot_collide,all,0.511,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-2-third,3shot_collide,free,0.500,126,0.286,0.044,,
olmo2-7b-mid5B,dx-self-2-third,3shot,all,0.517,180,0.287,0.037,,
olmo2-7b-mid5B,dx-self-2-third,3shot,free,0.524,126,0.286,0.044,,
olmo2-7b-mid5B,dx-addressee-3-deictic,0shot,all,0.333,180,0.287,0.035,,other_object=0.222;previous_owner=0.033;speaker_confusion=0.411
olmo2-7b-mid5B,dx-addressee-3-deictic,0shot,free,0.357,126,0.286,0.043,,other_object=0.198;previous_owner=0.048;speaker_confusion=0.397
olmo2-7b-mid5B,dx-addressee-3-deictic,3shot_collide,all,0.311,180,0.287,0.035,,other_object=0.183;previous_owner=0.1;speaker_confusion=0.406
olmo2-7b-mid5B,dx-addressee-3-deictic,3shot_collide,free,0.309,126,0.286,0.041,,other_object=0.151;previous_owner=0.135;speaker_confusion=0.405
olmo2-7b-mid5B,dx-addressee-3-deictic,3shot,all,0.333,180,0.287,0.035,,other_object=0.172;previous_owner=0.094;speaker_confusion=0.4
olmo2-7b-mid5B,dx-addressee-3-deictic,3shot,free,0.270,126,0.286,0.040,,other_object=0.175;previous_owner=0.127;speaker_confusion=0.429
olmo2-7b-mid5B,dx-addressee-3-third,0shot,all,0.461,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-3-third,0shot,free,0.349,126,0.286,0.043,,
olmo2-7b-mid5B,dx-addressee-3-third,3shot_collide,all,0.406,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-3-third,3shot_collide,free,0.381,126,0.286,0.043,,
olmo2-7b-mid5B,dx-addressee-3-third,3shot,all,0.500,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-3-third,3shot,free,0.452,126,0.286,0.044,,
olmo2-7b-mid5B,dx-addressee-2-deictic,0shot,all,0.311,180,0.287,0.035,,speaker_confusion=0.689
olmo2-7b-mid5B,dx-addressee-2-deictic,0shot,free,0.278,126,0.286,0.040,,speaker_confusion=0.722
olmo2-7b-mid5B,dx-addressee-2-deictic,3shot_collide,all,0.356,180,0.287,0.036,,speaker_confusion=0.644
olmo2-7b-mid5B,dx-addressee-2-deictic,3shot_collide,free,0.373,126,0.286,0.043,,speaker_confusion=0.627
olmo2-7b-mid5B,dx-addressee-2-deictic,3shot,all,0.306,180,0.287,0.034,,speaker_confusion=0.694
olmo2-7b-mid5B,dx-addressee-2-deictic,3shot,free,0.302,126,0.286,0.041,,speaker_confusion=0.698
olmo2-7b-mid5B,dx-addressee-2-third,0shot,all,0.444,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-2-third,0shot,free,0.460,126,0.286,0.044,,
olmo2-7b-mid5B,dx-addressee-2-third,3shot_collide,all,0.528,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-2-third,3shot_collide,free,0.516,126,0.286,0.044,,
olmo2-7b-mid5B,dx-addressee-2-third,3shot,all,0.567,180,0.287,0.037,,
olmo2-7b-mid5B,dx-addressee-2-third,3shot,free,0.548,126,0.286,0.044,,
olmo2-7b-mid17B,dx-owner-3-deictic,0shot,all,0.533,120,0.333,0.045,,other_participant=0.317;speaker_confusion=0.15
olmo2-7b-mid17B,dx-owner-3-deictic,0shot,free,0.325,40,0.333,0.074,,other_participant=0.55;speaker_confusion=0.125
olmo2-7b-mid17B,dx-owner-3-deictic,3shot_collide,all,0.675,120,0.333,0.043,,other_participant=0.275;speaker_confusion=0.05
olmo2-7b-mid17B,dx-owner-3-deictic,3shot_collide,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.05
olmo2-7b-mid17B,dx-owner-3-deictic,3shot,all,0.658,120,0.333,0.043,,other_participant=0.267;speaker_confusion=0.075
olmo2-7b-mid17B,dx-owner-3-deictic,3shot,free,0.575,40,0.333,0.078,,other_participant=0.4;speaker_confusion=0.025
olmo2-7b-mid17B,dx-owner-3-third,0shot,all,0.675,120,0.333,0.043,,
olmo2-7b-mid17B,dx-owner-3-third,0shot,free,0.525,40,0.333,0.079,,
olmo2-7b-mid17B,dx-owner-3-third,3shot_collide,all,0.683,120,0.333,0.043,,
olmo2-7b-mid17B,dx-owner-3-third,3shot_collide,free,0.575,40,0.333,0.078,,
olmo2-7b-mid17B,dx-owner-3-third,3shot,all,0.733,120,0.333,0.040,,
olmo2-7b-mid17B,dx-owner-3-third,3shot,free,0.650,40,0.333,0.075,,
olmo2-7b-mid17B,dx-self-3-deictic,0shot,all,0.361,180,0.287,0.036,,other_object=0.289;previous_owner=0.044;speaker_confusion=0.3;tie=0.006
olmo2-7b-mid17B,dx-self-3-deictic,0shot,free,0.349,126,0.286,0.043,,other_object=0.31;previous_owner=0.016;speaker_confusion=0.317;tie=0.008
olmo2-7b-mid17B,dx-self-3-deictic,3shot_collide,all,0.417,180,0.287,0.037,,other_object=0.267;previous_owner=0.044;speaker_confusion=0.272
olmo2-7b-mid17B,dx-self-3-deictic,3shot_collide,free,0.452,126,0.286,0.044,,other_object=0.278;previous_owner=0.024;speaker_confusion=0.246
olmo2-7b-mid17B,dx-self-3-deictic,3shot,all,0.422,180,0.287,0.037,,other_object=0.267;previous_owner=0.039;speaker_confusion=0.272
olmo2-7b-mid17B,dx-self-3-deictic,3shot,free,0.444,126,0.286,0.044,,other_object=0.286;previous_owner=0.024;speaker_confusion=0.246
olmo2-7b-mid17B,dx-self-3-third,0shot,all,0.450,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-3-third,0shot,free,0.349,126,0.286,0.043,,
olmo2-7b-mid17B,dx-self-3-third,3shot_collide,all,0.483,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
olmo2-7b-mid17B,dx-self-3-third,3shot,all,0.561,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-3-third,3shot,free,0.492,126,0.286,0.044,,
olmo2-7b-mid17B,dx-self-2-deictic,0shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.672
olmo2-7b-mid17B,dx-self-2-deictic,0shot,free,0.294,126,0.286,0.041,,speaker_confusion=0.706
olmo2-7b-mid17B,dx-self-2-deictic,3shot_collide,all,0.428,180,0.287,0.037,,speaker_confusion=0.572
olmo2-7b-mid17B,dx-self-2-deictic,3shot_collide,free,0.508,126,0.286,0.044,,speaker_confusion=0.492
olmo2-7b-mid17B,dx-self-2-deictic,3shot,all,0.378,180,0.287,0.036,,speaker_confusion=0.622
olmo2-7b-mid17B,dx-self-2-deictic,3shot,free,0.421,126,0.286,0.044,,speaker_confusion=0.579
olmo2-7b-mid17B,dx-self-2-third,0shot,all,0.439,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-2-third,0shot,free,0.389,126,0.286,0.043,,
olmo2-7b-mid17B,dx-self-2-third,3shot_collide,all,0.561,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-2-third,3shot_collide,free,0.540,126,0.286,0.044,,
olmo2-7b-mid17B,dx-self-2-third,3shot,all,0.556,180,0.287,0.037,,
olmo2-7b-mid17B,dx-self-2-third,3shot,free,0.564,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-3-deictic,0shot,all,0.300,180,0.287,0.034,,other_object=0.239;previous_owner=0.083;speaker_confusion=0.378
olmo2-7b-mid17B,dx-addressee-3-deictic,0shot,free,0.309,126,0.286,0.041,,other_object=0.214;previous_owner=0.111;speaker_confusion=0.365
olmo2-7b-mid17B,dx-addressee-3-deictic,3shot_collide,all,0.350,180,0.287,0.036,,other_object=0.2;previous_owner=0.1;speaker_confusion=0.35
olmo2-7b-mid17B,dx-addressee-3-deictic,3shot_collide,free,0.365,126,0.286,0.043,,other_object=0.151;previous_owner=0.127;speaker_confusion=0.357
olmo2-7b-mid17B,dx-addressee-3-deictic,3shot,all,0.306,180,0.287,0.034,,other_object=0.211;previous_owner=0.083;speaker_confusion=0.394;tie=0.006
olmo2-7b-mid17B,dx-addressee-3-deictic,3shot,free,0.302,126,0.286,0.041,,other_object=0.183;previous_owner=0.095;speaker_confusion=0.413;tie=0.008
olmo2-7b-mid17B,dx-addressee-3-third,0shot,all,0.494,180,0.287,0.037,,
olmo2-7b-mid17B,dx-addressee-3-third,0shot,free,0.405,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-3-third,3shot_collide,all,0.533,180,0.287,0.037,,
olmo2-7b-mid17B,dx-addressee-3-third,3shot_collide,free,0.484,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-3-third,3shot,all,0.606,180,0.287,0.036,,
olmo2-7b-mid17B,dx-addressee-3-third,3shot,free,0.571,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-2-deictic,0shot,all,0.244,180,0.287,0.032,,speaker_confusion=0.756
olmo2-7b-mid17B,dx-addressee-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.81
olmo2-7b-mid17B,dx-addressee-2-deictic,3shot_collide,all,0.356,180,0.287,0.036,,speaker_confusion=0.639;tie=0.006
olmo2-7b-mid17B,dx-addressee-2-deictic,3shot_collide,free,0.365,126,0.286,0.043,,speaker_confusion=0.627;tie=0.008
olmo2-7b-mid17B,dx-addressee-2-deictic,3shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.667;tie=0.006
olmo2-7b-mid17B,dx-addressee-2-deictic,3shot,free,0.341,126,0.286,0.042,,speaker_confusion=0.651;tie=0.008
olmo2-7b-mid17B,dx-addressee-2-third,0shot,all,0.439,180,0.287,0.037,,
olmo2-7b-mid17B,dx-addressee-2-third,0shot,free,0.421,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-2-third,3shot_collide,all,0.589,180,0.287,0.037,,
olmo2-7b-mid17B,dx-addressee-2-third,3shot_collide,free,0.571,126,0.286,0.044,,
olmo2-7b-mid17B,dx-addressee-2-third,3shot,all,0.633,180,0.287,0.036,,
olmo2-7b-mid17B,dx-addressee-2-third,3shot,free,0.603,126,0.286,0.044,,
olmo2-7b-mid50B,dx-owner-3-deictic,0shot,all,0.567,120,0.333,0.045,,other_participant=0.308;speaker_confusion=0.125
olmo2-7b-mid50B,dx-owner-3-deictic,0shot,free,0.350,40,0.333,0.075,,other_participant=0.55;speaker_confusion=0.1
olmo2-7b-mid50B,dx-owner-3-deictic,3shot_collide,all,0.692,120,0.333,0.042,,other_participant=0.275;speaker_confusion=0.033
olmo2-7b-mid50B,dx-owner-3-deictic,3shot_collide,free,0.650,40,0.333,0.075,,other_participant=0.325;speaker_confusion=0.025
olmo2-7b-mid50B,dx-owner-3-deictic,3shot,all,0.675,120,0.333,0.043,,other_participant=0.292;speaker_confusion=0.033
olmo2-7b-mid50B,dx-owner-3-deictic,3shot,free,0.550,40,0.333,0.079,,other_participant=0.45
olmo2-7b-mid50B,dx-owner-3-third,0shot,all,0.717,120,0.333,0.041,,
olmo2-7b-mid50B,dx-owner-3-third,0shot,free,0.600,40,0.333,0.077,,
olmo2-7b-mid50B,dx-owner-3-third,3shot_collide,all,0.608,120,0.333,0.045,,
olmo2-7b-mid50B,dx-owner-3-third,3shot_collide,free,0.525,40,0.333,0.079,,
olmo2-7b-mid50B,dx-owner-3-third,3shot,all,0.750,120,0.333,0.040,,
olmo2-7b-mid50B,dx-owner-3-third,3shot,free,0.725,40,0.333,0.071,,
olmo2-7b-mid50B,dx-self-3-deictic,0shot,all,0.411,180,0.287,0.037,,other_object=0.3;previous_owner=0.056;speaker_confusion=0.233
olmo2-7b-mid50B,dx-self-3-deictic,0shot,free,0.397,126,0.286,0.044,,other_object=0.333;previous_owner=0.024;speaker_confusion=0.246
olmo2-7b-mid50B,dx-self-3-deictic,3shot_collide,all,0.411,180,0.287,0.037,,other_object=0.294;previous_owner=0.05;speaker_confusion=0.244
olmo2-7b-mid50B,dx-self-3-deictic,3shot_collide,free,0.429,126,0.286,0.044,,other_object=0.325;previous_owner=0.04;speaker_confusion=0.206
olmo2-7b-mid50B,dx-self-3-deictic,3shot,all,0.406,180,0.287,0.037,,other_object=0.289;previous_owner=0.033;speaker_confusion=0.272
olmo2-7b-mid50B,dx-self-3-deictic,3shot,free,0.421,126,0.286,0.044,,other_object=0.317;previous_owner=0.024;speaker_confusion=0.238
olmo2-7b-mid50B,dx-self-3-third,0shot,all,0.461,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-3-third,0shot,free,0.365,126,0.286,0.043,,
olmo2-7b-mid50B,dx-self-3-third,3shot_collide,all,0.461,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
olmo2-7b-mid50B,dx-self-3-third,3shot,all,0.528,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-3-third,3shot,free,0.492,126,0.286,0.044,,
olmo2-7b-mid50B,dx-self-2-deictic,0shot,all,0.328,180,0.287,0.035,,speaker_confusion=0.672
olmo2-7b-mid50B,dx-self-2-deictic,0shot,free,0.294,126,0.286,0.041,,speaker_confusion=0.706
olmo2-7b-mid50B,dx-self-2-deictic,3shot_collide,all,0.444,180,0.287,0.037,,speaker_confusion=0.556
olmo2-7b-mid50B,dx-self-2-deictic,3shot_collide,free,0.532,126,0.286,0.044,,speaker_confusion=0.468
olmo2-7b-mid50B,dx-self-2-deictic,3shot,all,0.389,180,0.287,0.036,,speaker_confusion=0.606;tie=0.006
olmo2-7b-mid50B,dx-self-2-deictic,3shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.54;tie=0.008
olmo2-7b-mid50B,dx-self-2-third,0shot,all,0.467,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-2-third,0shot,free,0.468,126,0.286,0.044,,
olmo2-7b-mid50B,dx-self-2-third,3shot_collide,all,0.517,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-2-third,3shot_collide,free,0.516,126,0.286,0.044,,
olmo2-7b-mid50B,dx-self-2-third,3shot,all,0.517,180,0.287,0.037,,
olmo2-7b-mid50B,dx-self-2-third,3shot,free,0.516,126,0.286,0.044,,
olmo2-7b-mid50B,dx-addressee-3-deictic,0shot,all,0.317,180,0.287,0.035,,other_object=0.222;previous_owner=0.1;speaker_confusion=0.361
olmo2-7b-mid50B,dx-addressee-3-deictic,0shot,free,0.333,126,0.286,0.042,,other_object=0.175;previous_owner=0.119;speaker_confusion=0.373
olmo2-7b-mid50B,dx-addressee-3-deictic,3shot_collide,all,0.300,180,0.287,0.034,,other_object=0.217;previous_owner=0.1;speaker_confusion=0.383
olmo2-7b-mid50B,dx-addressee-3-deictic,3shot_collide,free,0.333,126,0.286,0.042,,other_object=0.159;previous_owner=0.119;speaker_confusion=0.389
olmo2-7b-mid50B,dx-addressee-3-deictic,3shot,all,0.311,180,0.287,0.035,,other_object=0.217;previous_owner=0.111;speaker_confusion=0.361
olmo2-7b-mid50B,dx-addressee-3-deictic,3shot,free,0.286,126,0.286,0.040,,other_object=0.175;previous_owner=0.127;speaker_confusion=0.413
olmo2-7b-mid50B,dx-addressee-3-third,0shot,all,0.467,180,0.287,0.037,,
olmo2-7b-mid50B,dx-addressee-3-third,0shot,free,0.373,126,0.286,0.043,,
olmo2-7b-mid50B,dx-addressee-3-third,3shot_collide,all,0.478,180,0.287,0.037,,
olmo2-7b-mid50B,dx-addressee-3-third,3shot_collide,free,0.389,126,0.286,0.043,,
olmo2-7b-mid50B,dx-addressee-3-third,3shot,all,0.511,180,0.287,0.037,,
olmo2-7b-mid50B,dx-addressee-3-third,3shot,free,0.436,126,0.286,0.044,,
olmo2-7b-mid50B,dx-addressee-2-deictic,0shot,all,0.239,180,0.287,0.032,,speaker_confusion=0.761
olmo2-7b-mid50B,dx-addressee-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.81
olmo2-7b-mid50B,dx-addressee-2-deictic,3shot_collide,all,0.350,180,0.287,0.036,,speaker_confusion=0.65
olmo2-7b-mid50B,dx-addressee-2-deictic,3shot_collide,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
olmo2-7b-mid50B,dx-addressee-2-deictic,3shot,all,0.356,180,0.287,0.036,,speaker_confusion=0.644
olmo2-7b-mid50B,dx-addressee-2-deictic,3shot,free,0.373,126,0.286,0.043,,speaker_confusion=0.627
olmo2-7b-mid50B,dx-addressee-2-third,0shot,all,0.428,180,0.287,0.037,,
olmo2-7b-mid50B,dx-addressee-2-third,0shot,free,0.436,126,0.286,0.044,,
olmo2-7b-mid50B,dx-addressee-2-third,3shot_collide,all,0.556,180,0.287,0.037,,
olmo2-7b-mid50B,dx-addressee-2-third,3shot_collide,free,0.564,126,0.286,0.044,,
olmo2-7b-mid50B,dx-addressee-2-third,3shot,all,0.622,180,0.287,0.036,,
olmo2-7b-mid50B,dx-addressee-2-third,3shot,free,0.603,126,0.286,0.044,,
olmo2-7b-sft,dx-owner-3-deictic,0shot,all,0.567,120,0.333,0.045,,other_participant=0.308;speaker_confusion=0.125
olmo2-7b-sft,dx-owner-3-deictic,0shot,free,0.425,40,0.333,0.078,,other_participant=0.525;speaker_confusion=0.05
olmo2-7b-sft,dx-owner-3-deictic,3shot_collide,all,0.600,120,0.333,0.045,,other_participant=0.333;speaker_confusion=0.067
olmo2-7b-sft,dx-owner-3-deictic,3shot_collide,free,0.575,40,0.333,0.078,,other_participant=0.4;speaker_confusion=0.025
olmo2-7b-sft,dx-owner-3-deictic,3shot,all,0.658,120,0.333,0.043,,other_participant=0.267;speaker_confusion=0.067;tie=0.008
olmo2-7b-sft,dx-owner-3-deictic,3shot,free,0.750,40,0.333,0.069,,other_participant=0.225;speaker_confusion=0.025
olmo2-7b-sft,dx-owner-3-third,0shot,all,0.742,120,0.333,0.040,,
olmo2-7b-sft,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
olmo2-7b-sft,dx-owner-3-third,3shot_collide,all,0.492,120,0.333,0.046,,
olmo2-7b-sft,dx-owner-3-third,3shot_collide,free,0.400,40,0.333,0.077,,
olmo2-7b-sft,dx-owner-3-third,3shot,all,0.583,120,0.333,0.045,,
olmo2-7b-sft,dx-owner-3-third,3shot,free,0.700,40,0.333,0.072,,
olmo2-7b-sft,dx-self-3-deictic,0shot,all,0.428,180,0.287,0.037,,other_object=0.217;previous_owner=0.067;speaker_confusion=0.289
olmo2-7b-sft,dx-self-3-deictic,0shot,free,0.429,126,0.286,0.044,,other_object=0.222;previous_owner=0.024;speaker_confusion=0.325
olmo2-7b-sft,dx-self-3-deictic,3shot_collide,all,0.417,180,0.287,0.037,,other_object=0.306;previous_owner=0.028;speaker_confusion=0.25
olmo2-7b-sft,dx-self-3-deictic,3shot_collide,free,0.516,126,0.286,0.044,,other_object=0.325;previous_owner=0.016;speaker_confusion=0.143
olmo2-7b-sft,dx-self-3-deictic,3shot,all,0.406,180,0.287,0.037,,other_object=0.372;previous_owner=0.017;speaker_confusion=0.206
olmo2-7b-sft,dx-self-3-deictic,3shot,free,0.429,126,0.286,0.044,,other_object=0.429;speaker_confusion=0.143
olmo2-7b-sft,dx-self-3-third,0shot,all,0.456,180,0.287,0.037,,
olmo2-7b-sft,dx-self-3-third,0shot,free,0.429,126,0.286,0.044,,
olmo2-7b-sft,dx-self-3-third,3shot_collide,all,0.494,180,0.287,0.037,,
olmo2-7b-sft,dx-self-3-third,3shot_collide,free,0.476,126,0.286,0.044,,
olmo2-7b-sft,dx-self-3-third,3shot,all,0.572,180,0.287,0.037,,
olmo2-7b-sft,dx-self-3-third,3shot,free,0.532,126,0.286,0.044,,
olmo2-7b-sft,dx-self-2-deictic,0shot,all,0.406,180,0.287,0.037,,speaker_confusion=0.589;tie=0.006
olmo2-7b-sft,dx-self-2-deictic,0shot,free,0.405,126,0.286,0.044,,speaker_confusion=0.595
olmo2-7b-sft,dx-self-2-deictic,3shot_collide,all,0.467,180,0.287,0.037,,speaker_confusion=0.528;tie=0.006
olmo2-7b-sft,dx-self-2-deictic,3shot_collide,free,0.564,126,0.286,0.044,,speaker_confusion=0.437
olmo2-7b-sft,dx-self-2-deictic,3shot,all,0.422,180,0.287,0.037,,speaker_confusion=0.572;tie=0.006
olmo2-7b-sft,dx-self-2-deictic,3shot,free,0.484,126,0.286,0.044,,speaker_confusion=0.516
olmo2-7b-sft,dx-self-2-third,0shot,all,0.511,180,0.287,0.037,,
olmo2-7b-sft,dx-self-2-third,0shot,free,0.492,126,0.286,0.044,,
olmo2-7b-sft,dx-self-2-third,3shot_collide,all,0.489,180,0.287,0.037,,
olmo2-7b-sft,dx-self-2-third,3shot_collide,free,0.476,126,0.286,0.044,,
olmo2-7b-sft,dx-self-2-third,3shot,all,0.589,180,0.287,0.037,,
olmo2-7b-sft,dx-self-2-third,3shot,free,0.579,126,0.286,0.044,,
olmo2-7b-sft,dx-addressee-3-deictic,0shot,all,0.356,180,0.287,0.036,,other_object=0.194;previous_owner=0.078;speaker_confusion=0.372
olmo2-7b-sft,dx-addressee-3-deictic,0shot,free,0.373,126,0.286,0.043,,other_object=0.175;previous_owner=0.103;speaker_confusion=0.349
olmo2-7b-sft,dx-addressee-3-deictic,3shot_collide,all,0.272,180,0.287,0.033,,other_object=0.272;previous_owner=0.061;speaker_confusion=0.389;tie=0.006
olmo2-7b-sft,dx-addressee-3-deictic,3shot_collide,free,0.302,126,0.286,0.041,,other_object=0.214;previous_owner=0.071;speaker_confusion=0.413
olmo2-7b-sft,dx-addressee-3-deictic,3shot,all,0.289,180,0.287,0.034,,other_object=0.239;previous_owner=0.083;speaker_confusion=0.389
olmo2-7b-sft,dx-addressee-3-deictic,3shot,free,0.294,126,0.286,0.041,,other_object=0.198;previous_owner=0.095;speaker_confusion=0.413
olmo2-7b-sft,dx-addressee-3-third,0shot,all,0.472,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-3-third,0shot,free,0.318,126,0.286,0.042,,
olmo2-7b-sft,dx-addressee-3-third,3shot_collide,all,0.483,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-3-third,3shot_collide,free,0.421,126,0.286,0.044,,
olmo2-7b-sft,dx-addressee-3-third,3shot,all,0.583,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-3-third,3shot,free,0.500,126,0.286,0.044,,
olmo2-7b-sft,dx-addressee-2-deictic,0shot,all,0.211,180,0.287,0.030,,speaker_confusion=0.789
olmo2-7b-sft,dx-addressee-2-deictic,0shot,free,0.222,126,0.286,0.037,,speaker_confusion=0.778
olmo2-7b-sft,dx-addressee-2-deictic,3shot_collide,all,0.328,180,0.287,0.035,,speaker_confusion=0.672
olmo2-7b-sft,dx-addressee-2-deictic,3shot_collide,free,0.381,126,0.286,0.043,,speaker_confusion=0.619
olmo2-7b-sft,dx-addressee-2-deictic,3shot,all,0.322,180,0.287,0.035,,speaker_confusion=0.678
olmo2-7b-sft,dx-addressee-2-deictic,3shot,free,0.381,126,0.286,0.043,,speaker_confusion=0.619
olmo2-7b-sft,dx-addressee-2-third,0shot,all,0.428,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-2-third,0shot,free,0.397,126,0.286,0.044,,
olmo2-7b-sft,dx-addressee-2-third,3shot_collide,all,0.522,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-2-third,3shot_collide,free,0.548,126,0.286,0.044,,
olmo2-7b-sft,dx-addressee-2-third,3shot,all,0.594,180,0.287,0.037,,
olmo2-7b-sft,dx-addressee-2-third,3shot,free,0.571,126,0.286,0.044,,
olmo2-7b-sft,sb-k1,0shot,all,0.734,64,0.125,0.055,,
olmo2-7b-sft,sb-k1,3shot_collide,all,0.953,64,0.125,0.026,,
olmo2-7b-sft,sb-k1,3shot,all,0.984,64,0.125,0.015,,
olmo2-7b-sft,sb-k23,0shot,all,0.469,128,0.027,0.044,,
olmo2-7b-sft,sb-k23,3shot_collide,all,0.727,128,0.027,0.039,,
olmo2-7b-sft,sb-k23,3shot,all,0.820,128,0.027,0.034,,
olmo2-7b-sft,sb-split,0shot,all,0.211,128,0.027,0.036,,
olmo2-7b-sft,sb-split,3shot_collide,all,0.484,128,0.027,0.044,,
olmo2-7b-sft,sb-split,3shot,all,0.508,128,0.027,0.044,,
olmo2-7b-sft,sb-dyn,0shot,all,0.184,256,0.027,0.024,,
olmo2-7b-sft,sb-dyn,3shot_collide,all,0.098,256,0.027,0.019,,
olmo2-7b-sft,sb-dyn,3shot,all,0.144,256,0.027,0.022,,
olmo2-7b-sft,an-narr,0shot,all,0.525,640,0.250,0.020,,
olmo2-7b-sft,an-narr,3shot_collide,all,0.605,640,0.250,0.019,,
olmo2-7b-sft,an-narr,3shot,all,0.669,640,0.250,0.019,,
olmo2-7b-sft,an-other-any,0shot,all,0.022,320,0.250,0.008,,
olmo2-7b-sft,an-other-any,3shot_collide,all,0.534,320,0.250,0.028,,
olmo2-7b-sft,an-other-any,3shot,all,0.472,320,0.250,0.028,,
olmo2-7b-sft,an-me-any,0shot,all,0.469,160,0.250,0.040,,
olmo2-7b-sft,an-me-any,3shot_collide,all,0.194,160,0.250,0.031,,
olmo2-7b-sft,an-me-any,3shot,all,0.069,160,0.250,0.020,,
olmo2-7b-sft,an-you-voc,0shot,all,0.863,80,0.250,0.038,,
olmo2-7b-sft,an-you-voc,3shot_collide,all,0.425,80,0.250,0.055,,
olmo2-7b-sft,an-you-voc,3shot,all,0.438,80,0.250,0.056,,
olmo2-7b-sft,an-you-novoc,0shot,all,0.375,80,0.250,0.054,,
olmo2-7b-sft,an-you-novoc,3shot_collide,all,0.125,80,0.250,0.037,,
olmo2-7b-sft,an-you-novoc,3shot,all,0.075,80,0.250,0.029,,
olmo2-7b-sft,adj-adj,0shot,all,0.865,260,0.333,0.021,,
olmo2-7b-sft,adj-adj,3shot_collide,all,0.873,260,0.333,0.021,,
olmo2-7b-sft,adj-adj,3shot,all,0.885,260,0.333,0.020,,
olmo2-7b-sft,adj-voc,0shot,all,0.885,260,0.333,0.020,,
olmo2-7b-sft,adj-voc,3shot_collide,all,0.892,260,0.333,0.019,,
olmo2-7b-sft,adj-voc,3shot,all,0.923,260,0.333,0.017,,
olmo2-7b-sft,adj-third,0shot,all,0.835,260,0.333,0.023,,
olmo2-7b-sft,adj-third,3shot_collide,all,0.792,260,0.333,0.025,,
olmo2-7b-sft,adj-third,3shot,all,0.862,260,0.333,0.021,,
olmo2-7b-sft,cm-mutate,0shot,all,0.865,200,0.200,0.024,,
olmo2-7b-sft,cm-mutate,3shot_collide,all,0.735,200,0.200,0.031,,
olmo2-7b-sft,cm-mutate,3shot,all,0.740,200,0.200,0.031,,
olmo2-7b-sft,cm-rebind,0shot,all,0.100,200,0.200,0.021,,
olmo2-7b-sft,cm-rebind,3shot_collide,all,0.400,200,0.200,0.035,,
olmo2-7b-sft,cm-rebind,3shot,all,0.340,200,0.200,0.034,,
olmo2-7b-sft,cm-copy,0shot,all,0.040,200,0.200,0.014,,
olmo2-7b-sft,cm-copy,3shot_collide,all,0.375,200,0.200,0.034,,
olmo2-7b-sft,cm-copy,3shot,all,0.385,200,0.200,0.034,,
olmo2-7b-dpo,dx-owner-3-deictic,0shot,all,0.633,120,0.333,0.044,,other_participant=0.25;speaker_confusion=0.117
olmo2-7b-dpo,dx-owner-3-deictic,0shot,free,0.625,40,0.333,0.076,,other_participant=0.375
olmo2-7b-dpo,dx-owner-3-deictic,3shot_collide,all,0.658,120,0.333,0.043,,other_participant=0.292;speaker_confusion=0.042;tie=0.008
olmo2-7b-dpo,dx-owner-3-deictic,3shot_collide,free,0.725,40,0.333,0.071,,other_participant=0.25;speaker_confusion=0.025
olmo2-7b-dpo,dx-owner-3-deictic,3shot,all,0.667,120,0.333,0.043,,other_participant=0.283;speaker_confusion=0.05
olmo2-7b-dpo,dx-owner-3-deictic,3shot,free,0.850,40,0.333,0.057,,other_participant=0.15
olmo2-7b-dpo,dx-owner-3-third,0shot,all,0.742,120,0.333,0.040,,
olmo2-7b-dpo,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
olmo2-7b-dpo,dx-owner-3-third,3shot_collide,all,0.525,120,0.333,0.046,,
olmo2-7b-dpo,dx-owner-3-third,3shot_collide,free,0.475,40,0.333,0.079,,
olmo2-7b-dpo,dx-owner-3-third,3shot,all,0.625,120,0.333,0.044,,
olmo2-7b-dpo,dx-owner-3-third,3shot,free,0.750,40,0.333,0.069,,
olmo2-7b-dpo,dx-self-3-deictic,0shot,all,0.444,180,0.287,0.037,,other_object=0.206;previous_owner=0.083;speaker_confusion=0.261;tie=0.006
olmo2-7b-dpo,dx-self-3-deictic,0shot,free,0.492,126,0.286,0.044,,other_object=0.206;previous_owner=0.032;speaker_confusion=0.262;tie=0.008
olmo2-7b-dpo,dx-self-3-deictic,3shot_collide,all,0.428,180,0.287,0.037,,other_object=0.289;previous_owner=0.033;speaker_confusion=0.25
olmo2-7b-dpo,dx-self-3-deictic,3shot_collide,free,0.548,126,0.286,0.044,,other_object=0.286;previous_owner=0.032;speaker_confusion=0.135
olmo2-7b-dpo,dx-self-3-deictic,3shot,all,0.439,180,0.287,0.037,,other_object=0.317;previous_owner=0.011;speaker_confusion=0.233
olmo2-7b-dpo,dx-self-3-deictic,3shot,free,0.516,126,0.286,0.044,,other_object=0.333;speaker_confusion=0.151
olmo2-7b-dpo,dx-self-3-third,0shot,all,0.500,180,0.287,0.037,,
olmo2-7b-dpo,dx-self-3-third,0shot,free,0.476,126,0.286,0.044,,
olmo2-7b-dpo,dx-self-3-third,3shot_collide,all,0.544,180,0.287,0.037,,
olmo2-7b-dpo,dx-self-3-third,3shot_collide,free,0.532,126,0.286,0.044,,
olmo2-7b-dpo,dx-self-3-third,3shot,all,0.611,180,0.287,0.036,,
olmo2-7b-dpo,dx-self-3-third,3shot,free,0.571,126,0.286,0.044,,
olmo2-7b-dpo,dx-self-2-deictic,0shot,all,0.428,180,0.287,0.037,,speaker_confusion=0.572
olmo2-7b-dpo,dx-self-2-deictic,0shot,free,0.460,126,0.286,0.044,,speaker_confusion=0.54
olmo2-7b-dpo,dx-self-2-deictic,3shot_collide,all,0.478,180,0.287,0.037,,speaker_confusion=0.522
olmo2-7b-dpo,dx-self-2-deictic,3shot_collide,free,0.627,126,0.286,0.043,,speaker_confusion=0.373
olmo2-7b-dpo,dx-self-2-deictic,3shot,all,0.467,180,0.287,0.037,,speaker_confusion=0.528;tie=0.006
olmo2-7b-dpo,dx-self-2-deictic,3shot,free,0.564,126,0.286,0.044,,speaker_confusion=0.437
olmo2-7b-dpo,dx-self-2-third,0shot,all,0.556,180,0.287,0.037,,
olmo2-7b-dpo,dx-self-2-third,0shot,free,0.540,126,0.286,0.044,,
olmo2-7b-dpo,dx-self-2-third,3shot_collide,all,0.522,180,0.287,0.037,,
olmo2-7b-dpo,dx-self-2-third,3shot_collide,free,0.524,126,0.286,0.044,,
olmo2-7b-dpo,dx-self-2-third,3shot,all,0.600,180,0.287,0.036,,
olmo2-7b-dpo,dx-self-2-third,3shot,free,0.595,126,0.286,0.044,,
olmo2-7b-dpo,dx-addressee-3-deictic,0shot,all,0.350,180,0.287,0.036,,other_object=0.194;previous_owner=0.033;speaker_confusion=0.417;tie=0.006
olmo2-7b-dpo,dx-addressee-3-deictic,0shot,free,0.373,126,0.286,0.043,,other_object=0.183;previous_owner=0.032;speaker_confusion=0.405;tie=0.008
olmo2-7b-dpo,dx-addressee-3-deictic,3shot_collide,all,0.356,180,0.287,0.036,,other_object=0.261;previous_owner=0.039;speaker_confusion=0.344
olmo2-7b-dpo,dx-addressee-3-deictic,3shot_collide,free,0.429,126,0.286,0.044,,other_object=0.214;previous_owner=0.048;speaker_confusion=0.31
olmo2-7b-dpo,dx-addressee-3-deictic,3shot,all,0.328,180,0.287,0.035,,other_object=0.283;previous_owner=0.044;speaker_confusion=0.339;tie=0.006
olmo2-7b-dpo,dx-addressee-3-deictic,3shot,free,0.341,126,0.286,0.042,,other_object=0.262;previous_owner=0.048;speaker_confusion=0.341;tie=0.008
olmo2-7b-dpo,dx-addressee-3-third,0shot,all,0.506,180,0.287,0.037,,
olmo2-7b-dpo,dx-addressee-3-third,0shot,free,0.413,126,0.286,0.044,,
olmo2-7b-dpo,dx-addressee-3-third,3shot_collide,all,0.556,180,0.287,0.037,,
olmo2-7b-dpo,dx-addressee-3-third,3shot_collide,free,0.500,126,0.286,0.044,,
olmo2-7b-dpo,dx-addressee-3-third,3shot,all,0.678,180,0.287,0.035,,
olmo2-7b-dpo,dx-addressee-3-third,3shot,free,0.619,126,0.286,0.043,,
olmo2-7b-dpo,dx-addressee-2-deictic,0shot,all,0.250,180,0.287,0.032,,speaker_confusion=0.739;tie=0.011
olmo2-7b-dpo,dx-addressee-2-deictic,0shot,free,0.278,126,0.286,0.040,,speaker_confusion=0.714;tie=0.008
olmo2-7b-dpo,dx-addressee-2-deictic,3shot_collide,all,0.367,180,0.287,0.036,,speaker_confusion=0.633
olmo2-7b-dpo,dx-addressee-2-deictic,3shot_collide,free,0.460,126,0.286,0.044,,speaker_confusion=0.54
olmo2-7b-dpo,dx-addressee-2-deictic,3shot,all,0.300,180,0.287,0.034,,speaker_confusion=0.7
olmo2-7b-dpo,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
olmo2-7b-dpo,dx-addressee-2-third,0shot,all,0.439,180,0.287,0.037,,
olmo2-7b-dpo,dx-addressee-2-third,0shot,free,0.429,126,0.286,0.044,,
olmo2-7b-dpo,dx-addressee-2-third,3shot_collide,all,0.528,180,0.287,0.037,,
olmo2-7b-dpo,dx-addressee-2-third,3shot_collide,free,0.564,126,0.286,0.044,,
olmo2-7b-dpo,dx-addressee-2-third,3shot,all,0.622,180,0.287,0.036,,
olmo2-7b-dpo,dx-addressee-2-third,3shot,free,0.603,126,0.286,0.044,,
olmo2-7b-dpo,sb-k1,0shot,all,0.766,64,0.125,0.053,,
olmo2-7b-dpo,sb-k1,3shot_collide,all,0.953,64,0.125,0.026,,
olmo2-7b-dpo,sb-k1,3shot,all,0.969,64,0.125,0.022,,
olmo2-7b-dpo,sb-k23,0shot,all,0.570,128,0.027,0.044,,
olmo2-7b-dpo,sb-k23,3shot_collide,all,0.750,128,0.027,0.038,,
olmo2-7b-dpo,sb-k23,3shot,all,0.797,128,0.027,0.036,,
olmo2-7b-dpo,sb-split,0shot,all,0.266,128,0.027,0.039,,
olmo2-7b-dpo,sb-split,3shot_collide,all,0.547,128,0.027,0.044,,
olmo2-7b-dpo,sb-split,3shot,all,0.602,128,0.027,0.043,,
olmo2-7b-dpo,sb-dyn,0shot,all,0.223,256,0.027,0.026,,
olmo2-7b-dpo,sb-dyn,3shot_collide,all,0.113,256,0.027,0.020,,
olmo2-7b-dpo,sb-dyn,3shot,all,0.148,256,0.027,0.022,,
olmo2-7b-dpo,an-narr,0shot,all,0.531,640,0.250,0.020,,
olmo2-7b-dpo,an-narr,3shot_collide,all,0.706,640,0.250,0.018,,
olmo2-7b-dpo,an-narr,3shot,all,0.683,640,0.250,0.018,,
olmo2-7b-dpo,an-other-any,0shot,all,0.047,320,0.250,0.012,,
olmo2-7b-dpo,an-other-any,3shot_collide,all,0.559,320,0.250,0.028,,
olmo2-7b-dpo,an-other-any,3shot,all,0.503,320,0.250,0.028,,
olmo2-7b-dpo,an-me-any,0shot,all,0.375,160,0.250,0.038,,
olmo2-7b-dpo,an-me-any,3shot_collide,all,0.381,160,0.250,0.038,,
olmo2-7b-dpo,an-me-any,3shot,all,0.188,160,0.250,0.031,,
olmo2-7b-dpo,an-you-voc,0shot,all,0.912,80,0.250,0.032,,
olmo2-7b-dpo,an-you-voc,3shot_collide,all,0.525,80,0.250,0.056,,
olmo2-7b-dpo,an-you-voc,3shot,all,0.637,80,0.250,0.054,,
olmo2-7b-dpo,an-you-novoc,0shot,all,0.613,80,0.250,0.054,,
olmo2-7b-dpo,an-you-novoc,3shot_collide,all,0.138,80,0.250,0.038,,
olmo2-7b-dpo,an-you-novoc,3shot,all,0.113,80,0.250,0.035,,
olmo2-7b-dpo,adj-adj,0shot,all,0.885,260,0.333,0.020,,
olmo2-7b-dpo,adj-adj,3shot_collide,all,0.950,260,0.333,0.013,,
olmo2-7b-dpo,adj-adj,3shot,all,0.908,260,0.333,0.018,,
olmo2-7b-dpo,adj-voc,0shot,all,0.919,260,0.333,0.017,,
olmo2-7b-dpo,adj-voc,3shot_collide,all,0.958,260,0.333,0.013,,
olmo2-7b-dpo,adj-voc,3shot,all,0.939,260,0.333,0.015,,
olmo2-7b-dpo,adj-third,0shot,all,0.911,260,0.333,0.018,,
olmo2-7b-dpo,adj-third,3shot_collide,all,0.892,260,0.333,0.019,,
olmo2-7b-dpo,adj-third,3shot,all,0.885,260,0.333,0.020,,
olmo2-7b-dpo,cm-mutate,0shot,all,0.940,200,0.200,0.017,,
olmo2-7b-dpo,cm-mutate,3shot_collide,all,0.890,200,0.200,0.022,,
olmo2-7b-dpo,cm-mutate,3shot,all,0.840,200,0.200,0.026,,
olmo2-7b-dpo,cm-rebind,0shot,all,0.060,200,0.200,0.017,,
olmo2-7b-dpo,cm-rebind,3shot_collide,all,0.375,200,0.200,0.034,,
olmo2-7b-dpo,cm-rebind,3shot,all,0.295,200,0.200,0.032,,
olmo2-7b-dpo,cm-copy,0shot,all,0.005,200,0.200,0.005,,
olmo2-7b-dpo,cm-copy,3shot_collide,all,0.265,200,0.200,0.031,,
olmo2-7b-dpo,cm-copy,3shot,all,0.215,200,0.200,0.029,,
olmo2-7b-instruct,attr,0shot,all,0.647,1210,0.168,0.014,,
olmo2-7b-instruct,attr,3shot_collide,all,0.887,1210,0.168,0.009,,
olmo2-7b-instruct,attr,3shot,all,0.904,1210,0.168,0.009,,
olmo2-7b-instruct,codevar,0shot,all,0.418,1200,0.267,0.014,,
olmo2-7b-instruct,codevar,3shot_collide,all,0.480,1200,0.267,0.014,,
olmo2-7b-instruct,codevar,3shot,all,0.508,1200,0.267,0.014,,
olmo2-7b-instruct,dialogue,0shot,all,0.792,1200,0.312,0.012,,
olmo2-7b-instruct,dialogue,3shot_collide,all,0.793,1200,0.312,0.012,,
olmo2-7b-instruct,dialogue,3shot,all,0.806,1200,0.312,0.011,,
olmo2-7b-instruct,dx-owner-3-deictic,0shot,all,0.625,120,0.333,0.044,,other_participant=0.25;speaker_confusion=0.117;tie=0.008
olmo2-7b-instruct,dx-owner-3-deictic,0shot,free,0.600,40,0.333,0.077,,other_participant=0.375;tie=0.025
olmo2-7b-instruct,dx-owner-3-deictic,3shot_collide,all,0.642,120,0.333,0.044,,other_participant=0.3;speaker_confusion=0.042;tie=0.017
olmo2-7b-instruct,dx-owner-3-deictic,3shot_collide,free,0.700,40,0.333,0.072,,other_participant=0.225;speaker_confusion=0.025;tie=0.05
olmo2-7b-instruct,dx-owner-3-deictic,3shot,all,0.675,120,0.333,0.043,,other_participant=0.275;speaker_confusion=0.05
olmo2-7b-instruct,dx-owner-3-deictic,3shot,free,0.850,40,0.333,0.057,,other_participant=0.15
olmo2-7b-instruct,dx-owner-3-third,0shot,all,0.750,120,0.333,0.040,,
olmo2-7b-instruct,dx-owner-3-third,0shot,free,0.675,40,0.333,0.074,,
olmo2-7b-instruct,dx-owner-3-third,3shot_collide,all,0.525,120,0.333,0.046,,
olmo2-7b-instruct,dx-owner-3-third,3shot_collide,free,0.450,40,0.333,0.079,,
olmo2-7b-instruct,dx-owner-3-third,3shot,all,0.633,120,0.333,0.044,,
olmo2-7b-instruct,dx-owner-3-third,3shot,free,0.750,40,0.333,0.069,,
olmo2-7b-instruct,dx-self-3-deictic,0shot,all,0.456,180,0.287,0.037,,other_object=0.206;previous_owner=0.078;speaker_confusion=0.256;tie=0.006
olmo2-7b-instruct,dx-self-3-deictic,0shot,free,0.500,126,0.286,0.044,,other_object=0.206;previous_owner=0.032;speaker_confusion=0.254;tie=0.008
olmo2-7b-instruct,dx-self-3-deictic,3shot_collide,all,0.433,180,0.287,0.037,,other_object=0.278;previous_owner=0.033;speaker_confusion=0.256
olmo2-7b-instruct,dx-self-3-deictic,3shot_collide,free,0.564,126,0.286,0.044,,other_object=0.27;previous_owner=0.032;speaker_confusion=0.135
olmo2-7b-instruct,dx-self-3-deictic,3shot,all,0.439,180,0.287,0.037,,other_object=0.317;previous_owner=0.011;speaker_confusion=0.228;tie=0.006
olmo2-7b-instruct,dx-self-3-deictic,3shot,free,0.516,126,0.286,0.044,,other_object=0.333;speaker_confusion=0.143;tie=0.008
olmo2-7b-instruct,dx-self-3-third,0shot,all,0.522,180,0.287,0.037,,
olmo2-7b-instruct,dx-self-3-third,0shot,free,0.492,126,0.286,0.044,,
olmo2-7b-instruct,dx-self-3-third,3shot_collide,all,0.567,180,0.287,0.037,,
olmo2-7b-instruct,dx-self-3-third,3shot_collide,free,0.556,126,0.286,0.044,,
olmo2-7b-instruct,dx-self-3-third,3shot,all,0.611,180,0.287,0.036,,
olmo2-7b-instruct,dx-self-3-third,3shot,free,0.556,126,0.286,0.044,,
olmo2-7b-instruct,dx-self-2-deictic,0shot,all,0.433,180,0.287,0.037,,speaker_confusion=0.567
olmo2-7b-instruct,dx-self-2-deictic,0shot,free,0.460,126,0.286,0.044,,speaker_confusion=0.54
olmo2-7b-instruct,dx-self-2-deictic,3shot_collide,all,0.478,180,0.287,0.037,,speaker_confusion=0.522
olmo2-7b-instruct,dx-self-2-deictic,3shot_collide,free,0.627,126,0.286,0.043,,speaker_confusion=0.373
olmo2-7b-instruct,dx-self-2-deictic,3shot,all,0.483,180,0.287,0.037,,speaker_confusion=0.517
olmo2-7b-instruct,dx-self-2-deictic,3shot,free,0.571,126,0.286,0.044,,speaker_confusion=0.429
olmo2-7b-instruct,dx-self-2-third,0shot,all,0.578,180,0.287,0.037,,
olmo2-7b-instruct,dx-self-2-third,0shot,free,0.564,126,0.286,0.044,,
olmo2-7b-instruct,dx-self-2-third,3shot_collide,all,0.561,180,0.287,0.037,,
olmo2-7b-instruct,dx-self-2-third,3shot_collide,free,0.540,126,0.286,0.044,,
olmo2-7b-instruct,dx-self-2-third,3shot,all,0.633,180,0.287,0.036,,
olmo2-7b-instruct,dx-self-2-third,3shot,free,0.627,126,0.286,0.043,,
olmo2-7b-instruct,dx-addressee-3-deictic,0shot,all,0.344,180,0.287,0.035,,other_object=0.183;previous_owner=0.039;speaker_confusion=0.428;tie=0.006
olmo2-7b-instruct,dx-addressee-3-deictic,0shot,free,0.365,126,0.286,0.043,,other_object=0.167;previous_owner=0.04;speaker_confusion=0.429
olmo2-7b-instruct,dx-addressee-3-deictic,3shot_collide,all,0.356,180,0.287,0.036,,other_object=0.261;previous_owner=0.028;speaker_confusion=0.35;tie=0.006
olmo2-7b-instruct,dx-addressee-3-deictic,3shot_collide,free,0.429,126,0.286,0.044,,other_object=0.206;previous_owner=0.032;speaker_confusion=0.325;tie=0.008
olmo2-7b-instruct,dx-addressee-3-deictic,3shot,all,0.333,180,0.287,0.035,,other_object=0.289;previous_owner=0.039;speaker_confusion=0.339
olmo2-7b-instruct,dx-addressee-3-deictic,3shot,free,0.349,126,0.286,0.043,,other_object=0.262;previous_owner=0.04;speaker_confusion=0.349
olmo2-7b-instruct,dx-addressee-3-third,0shot,all,0.528,180,0.287,0.037,,
olmo2-7b-instruct,dx-addressee-3-third,0shot,free,0.421,126,0.286,0.044,,
olmo2-7b-instruct,dx-addressee-3-third,3shot_collide,all,0.561,180,0.287,0.037,,
olmo2-7b-instruct,dx-addressee-3-third,3shot_collide,free,0.508,126,0.286,0.044,,
olmo2-7b-instruct,dx-addressee-3-third,3shot,all,0.667,180,0.287,0.035,,
olmo2-7b-instruct,dx-addressee-3-third,3shot,free,0.611,126,0.286,0.043,,
olmo2-7b-instruct,dx-addressee-2-deictic,0shot,all,0.256,180,0.287,0.033,,speaker_confusion=0.744
olmo2-7b-instruct,dx-addressee-2-deictic,0shot,free,0.286,126,0.286,0.040,,speaker_confusion=0.714
olmo2-7b-instruct,dx-addressee-2-deictic,3shot_collide,all,0.350,180,0.287,0.036,,speaker_confusion=0.639;tie=0.011
olmo2-7b-instruct,dx-addressee-2-deictic,3shot_collide,free,0.436,126,0.286,0.044,,speaker_confusion=0.548;tie=0.016
olmo2-7b-instruct,dx-addressee-2-deictic,3shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.717
olmo2-7b-instruct,dx-addressee-2-deictic,3shot,free,0.349,126,0.286,0.043,,speaker_confusion=0.651
olmo2-7b-instruct,dx-addressee-2-third,0shot,all,0.444,180,0.287,0.037,,
olmo2-7b-instruct,dx-addressee-2-third,0shot,free,0.421,126,0.286,0.044,,
olmo2-7b-instruct,dx-addressee-2-third,3shot_collide,all,0.539,180,0.287,0.037,,
olmo2-7b-instruct,dx-addressee-2-third,3shot_collide,free,0.571,126,0.286,0.044,,
olmo2-7b-instruct,dx-addressee-2-third,3shot,all,0.633,180,0.287,0.036,,
olmo2-7b-instruct,dx-addressee-2-third,3shot,free,0.611,126,0.286,0.043,,
olmo2-7b-instruct,sb-k1,0shot,all,0.781,64,0.125,0.052,,
olmo2-7b-instruct,sb-k1,3shot_collide,all,0.953,64,0.125,0.026,,
olmo2-7b-instruct,sb-k1,3shot,all,0.953,64,0.125,0.026,,
olmo2-7b-instruct,sb-k23,0shot,all,0.586,128,0.027,0.043,,
olmo2-7b-instruct,sb-k23,3shot_collide,all,0.750,128,0.027,0.038,,
olmo2-7b-instruct,sb-k23,3shot,all,0.820,128,0.027,0.034,,
olmo2-7b-instruct,sb-split,0shot,all,0.305,128,0.027,0.041,,
olmo2-7b-instruct,sb-split,3shot_collide,all,0.562,128,0.027,0.044,,
olmo2-7b-instruct,sb-split,3shot,all,0.633,128,0.027,0.043,,
olmo2-7b-instruct,sb-dyn,0shot,all,0.231,256,0.027,0.026,,
olmo2-7b-instruct,sb-dyn,3shot_collide,all,0.121,256,0.027,0.020,,
olmo2-7b-instruct,sb-dyn,3shot,all,0.156,256,0.027,0.023,,
olmo2-7b-instruct,an-narr,0shot,all,0.547,640,0.250,0.020,,
olmo2-7b-instruct,an-narr,3shot_collide,all,0.706,640,0.250,0.018,,
olmo2-7b-instruct,an-narr,3shot,all,0.691,640,0.250,0.018,,
olmo2-7b-instruct,an-other-any,0shot,all,0.056,320,0.250,0.013,,
olmo2-7b-instruct,an-other-any,3shot_collide,all,0.544,320,0.250,0.028,,
olmo2-7b-instruct,an-other-any,3shot,all,0.506,320,0.250,0.028,,
olmo2-7b-instruct,an-me-any,0shot,all,0.394,160,0.250,0.039,,
olmo2-7b-instruct,an-me-any,3shot_collide,all,0.381,160,0.250,0.038,,
olmo2-7b-instruct,an-me-any,3shot,all,0.200,160,0.250,0.032,,
olmo2-7b-instruct,an-you-voc,0shot,all,0.925,80,0.250,0.029,,
olmo2-7b-instruct,an-you-voc,3shot_collide,all,0.537,80,0.250,0.056,,
olmo2-7b-instruct,an-you-voc,3shot,all,0.650,80,0.250,0.053,,
olmo2-7b-instruct,an-you-novoc,0shot,all,0.600,80,0.250,0.055,,
olmo2-7b-instruct,an-you-novoc,3shot_collide,all,0.175,80,0.250,0.043,,
olmo2-7b-instruct,an-you-novoc,3shot,all,0.113,80,0.250,0.035,,
olmo2-7b-instruct,adj-adj,0shot,all,0.888,260,0.333,0.019,,
olmo2-7b-instruct,adj-adj,3shot_collide,all,0.958,260,0.333,0.013,,
olmo2-7b-instruct,adj-adj,3shot,all,0.911,260,0.333,0.018,,
olmo2-7b-instruct,adj-voc,0shot,all,0.927,260,0.333,0.016,,
olmo2-7b-instruct,adj-voc,3shot_collide,all,0.962,260,0.333,0.012,,
olmo2-7b-instruct,adj-voc,3shot,all,0.942,260,0.333,0.015,,
olmo2-7b-instruct,adj-third,0shot,all,0.923,260,0.333,0.017,,
olmo2-7b-instruct,adj-third,3shot_collide,all,0.892,260,0.333,0.019,,
olmo2-7b-instruct,adj-third,3shot,all,0.881,260,0.333,0.020,,
olmo2-7b-instruct,cm-mutate,0shot,all,0.935,200,0.200,0.017,,
olmo2-7b-instruct,cm-mutate,3shot_collide,all,0.895,200,0.200,0.022,,
olmo2-7b-instruct,cm-mutate,3shot,all,0.870,200,0.200,0.024,,
olmo2-7b-instruct,cm-rebind,0shot,all,0.060,200,0.200,0.017,,
olmo2-7b-instruct,cm-rebind,3shot_collide,all,0.380,200,0.200,0.034,,
olmo2-7b-instruct,cm-rebind,3shot,all,0.300,200,0.200,0.032,,
olmo2-7b-instruct,cm-copy,0shot,all,0.005,200,0.200,0.005,,
olmo2-7b-instruct,cm-copy,3shot_collide,all,0.245,200,0.200,0.030,,
olmo2-7b-instruct,cm-copy,3shot,all,0.210,200,0.200,0.029,,
llama31-8b-instruct,dx-owner-3-deictic,0shot,all,0.692,120,0.333,0.042,,other_participant=0.292;tie=0.017
llama31-8b-instruct,dx-owner-3-deictic,0shot,free,0.525,40,0.333,0.079,,other_participant=0.425;tie=0.05
llama31-8b-instruct,dx-owner-3-deictic,3shot_collide,all,0.908,120,0.333,0.026,,other_participant=0.083;speaker_confusion=0.008
llama31-8b-instruct,dx-owner-3-deictic,3shot_collide,free,0.900,40,0.333,0.047,,other_participant=0.1
llama31-8b-instruct,dx-owner-3-deictic,3shot,all,0.875,120,0.333,0.030,,other_participant=0.1;speaker_confusion=0.017;tie=0.008
llama31-8b-instruct,dx-owner-3-deictic,3shot,free,0.900,40,0.333,0.047,,other_participant=0.075;tie=0.025
llama31-8b-instruct,dx-owner-3-third,0shot,all,0.758,120,0.333,0.039,,
llama31-8b-instruct,dx-owner-3-third,0shot,free,0.725,40,0.333,0.071,,
llama31-8b-instruct,dx-owner-3-third,3shot_collide,all,0.867,120,0.333,0.031,,
llama31-8b-instruct,dx-owner-3-third,3shot_collide,free,0.850,40,0.333,0.057,,
llama31-8b-instruct,dx-owner-3-third,3shot,all,0.917,120,0.333,0.025,,
llama31-8b-instruct,dx-owner-3-third,3shot,free,0.975,40,0.333,0.025,,
llama31-8b-instruct,dx-self-3-deictic,0shot,all,0.472,180,0.287,0.037,,other_object=0.167;previous_owner=0.067;speaker_confusion=0.289;tie=0.006
llama31-8b-instruct,dx-self-3-deictic,0shot,free,0.564,126,0.286,0.044,,other_object=0.135;previous_owner=0.032;speaker_confusion=0.262;tie=0.008
llama31-8b-instruct,dx-self-3-deictic,3shot_collide,all,0.661,180,0.287,0.035,,other_object=0.122;previous_owner=0.044;speaker_confusion=0.161;tie=0.011
llama31-8b-instruct,dx-self-3-deictic,3shot_collide,free,0.698,126,0.286,0.041,,other_object=0.151;previous_owner=0.04;speaker_confusion=0.095;tie=0.016
llama31-8b-instruct,dx-self-3-deictic,3shot,all,0.678,180,0.287,0.035,,other_object=0.089;previous_owner=0.056;speaker_confusion=0.178
llama31-8b-instruct,dx-self-3-deictic,3shot,free,0.754,126,0.286,0.038,,other_object=0.103;previous_owner=0.024;speaker_confusion=0.119
llama31-8b-instruct,dx-self-3-third,0shot,all,0.783,180,0.287,0.031,,
llama31-8b-instruct,dx-self-3-third,0shot,free,0.833,126,0.286,0.033,,
llama31-8b-instruct,dx-self-3-third,3shot_collide,all,0.733,180,0.287,0.033,,
llama31-8b-instruct,dx-self-3-third,3shot_collide,free,0.818,126,0.286,0.034,,
llama31-8b-instruct,dx-self-3-third,3shot,all,0.761,180,0.287,0.032,,
llama31-8b-instruct,dx-self-3-third,3shot,free,0.809,126,0.286,0.035,,
llama31-8b-instruct,dx-self-2-deictic,0shot,all,0.361,180,0.287,0.036,,speaker_confusion=0.622;tie=0.017
llama31-8b-instruct,dx-self-2-deictic,0shot,free,0.452,126,0.286,0.044,,speaker_confusion=0.532;tie=0.016
llama31-8b-instruct,dx-self-2-deictic,3shot_collide,all,0.522,180,0.287,0.037,,speaker_confusion=0.461;tie=0.017
llama31-8b-instruct,dx-self-2-deictic,3shot_collide,free,0.611,126,0.286,0.043,,speaker_confusion=0.373;tie=0.016
llama31-8b-instruct,dx-self-2-deictic,3shot,all,0.517,180,0.287,0.037,,speaker_confusion=0.467;tie=0.017
llama31-8b-instruct,dx-self-2-deictic,3shot,free,0.635,126,0.286,0.043,,speaker_confusion=0.349;tie=0.016
llama31-8b-instruct,dx-self-2-third,0shot,all,0.744,180,0.287,0.033,,
llama31-8b-instruct,dx-self-2-third,0shot,free,0.841,126,0.286,0.033,,
llama31-8b-instruct,dx-self-2-third,3shot_collide,all,0.783,180,0.287,0.031,,
llama31-8b-instruct,dx-self-2-third,3shot_collide,free,0.809,126,0.286,0.035,,
llama31-8b-instruct,dx-self-2-third,3shot,all,0.711,180,0.287,0.034,,
llama31-8b-instruct,dx-self-2-third,3shot,free,0.794,126,0.286,0.036,,
llama31-8b-instruct,dx-addressee-3-deictic,0shot,all,0.428,180,0.287,0.037,,other_object=0.117;previous_owner=0.133;speaker_confusion=0.311;tie=0.011
llama31-8b-instruct,dx-addressee-3-deictic,0shot,free,0.436,126,0.286,0.044,,other_object=0.071;previous_owner=0.151;speaker_confusion=0.325;tie=0.016
llama31-8b-instruct,dx-addressee-3-deictic,3shot_collide,all,0.517,180,0.287,0.037,,other_object=0.15;previous_owner=0.039;speaker_confusion=0.283;tie=0.011
llama31-8b-instruct,dx-addressee-3-deictic,3shot_collide,free,0.556,126,0.286,0.044,,other_object=0.111;previous_owner=0.032;speaker_confusion=0.286;tie=0.016
llama31-8b-instruct,dx-addressee-3-deictic,3shot,all,0.511,180,0.287,0.037,,other_object=0.128;previous_owner=0.033;speaker_confusion=0.322;tie=0.006
llama31-8b-instruct,dx-addressee-3-deictic,3shot,free,0.564,126,0.286,0.044,,other_object=0.087;previous_owner=0.024;speaker_confusion=0.325
llama31-8b-instruct,dx-addressee-3-third,0shot,all,0.778,180,0.287,0.031,,
llama31-8b-instruct,dx-addressee-3-third,0shot,free,0.778,126,0.286,0.037,,
llama31-8b-instruct,dx-addressee-3-third,3shot_collide,all,0.733,180,0.287,0.033,,
llama31-8b-instruct,dx-addressee-3-third,3shot_collide,free,0.802,126,0.286,0.035,,
llama31-8b-instruct,dx-addressee-3-third,3shot,all,0.733,180,0.287,0.033,,
llama31-8b-instruct,dx-addressee-3-third,3shot,free,0.786,126,0.286,0.037,,
llama31-8b-instruct,dx-addressee-2-deictic,0shot,all,0.206,180,0.287,0.030,,speaker_confusion=0.761;tie=0.033
llama31-8b-instruct,dx-addressee-2-deictic,0shot,free,0.182,126,0.286,0.034,,speaker_confusion=0.77;tie=0.048
llama31-8b-instruct,dx-addressee-2-deictic,3shot_collide,all,0.378,180,0.287,0.036,,speaker_confusion=0.617;tie=0.006
llama31-8b-instruct,dx-addressee-2-deictic,3shot_collide,free,0.397,126,0.286,0.044,,speaker_confusion=0.595;tie=0.008
llama31-8b-instruct,dx-addressee-2-deictic,3shot,all,0.350,180,0.287,0.036,,speaker_confusion=0.644;tie=0.006
llama31-8b-instruct,dx-addressee-2-deictic,3shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
llama31-8b-instruct,dx-addressee-2-third,0shot,all,0.744,180,0.287,0.033,,
llama31-8b-instruct,dx-addressee-2-third,0shot,free,0.809,126,0.286,0.035,,
llama31-8b-instruct,dx-addressee-2-third,3shot_collide,all,0.733,180,0.287,0.033,,
llama31-8b-instruct,dx-addressee-2-third,3shot_collide,free,0.802,126,0.286,0.035,,
llama31-8b-instruct,dx-addressee-2-third,3shot,all,0.728,180,0.287,0.033,,
llama31-8b-instruct,dx-addressee-2-third,3shot,free,0.825,126,0.286,0.034,,
llama31-8b-instruct,sb-k1,0shot,all,0.969,64,0.125,0.022,,
llama31-8b-instruct,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
llama31-8b-instruct,sb-k1,3shot,all,1.000,64,0.125,0.000,,
llama31-8b-instruct,sb-k23,0shot,all,0.719,128,0.027,0.040,,
llama31-8b-instruct,sb-k23,3shot_collide,all,0.977,128,0.027,0.013,,
llama31-8b-instruct,sb-k23,3shot,all,0.953,128,0.027,0.019,,
llama31-8b-instruct,sb-split,0shot,all,0.852,128,0.027,0.031,,
llama31-8b-instruct,sb-split,3shot_collide,all,0.977,128,0.027,0.013,,
llama31-8b-instruct,sb-split,3shot,all,0.922,128,0.027,0.024,,
llama31-8b-instruct,sb-dyn,0shot,all,0.629,256,0.027,0.030,,
llama31-8b-instruct,sb-dyn,3shot_collide,all,0.609,256,0.027,0.030,,
llama31-8b-instruct,sb-dyn,3shot,all,0.609,256,0.027,0.030,,
llama31-8b-instruct,an-narr,0shot,all,0.645,640,0.250,0.019,,
llama31-8b-instruct,an-narr,3shot_collide,all,0.930,640,0.250,0.010,,
llama31-8b-instruct,an-narr,3shot,all,0.914,640,0.250,0.011,,
llama31-8b-instruct,an-other-any,0shot,all,0.800,320,0.250,0.022,,
llama31-8b-instruct,an-other-any,3shot_collide,all,0.659,320,0.250,0.026,,
llama31-8b-instruct,an-other-any,3shot,all,0.591,320,0.250,0.028,,
llama31-8b-instruct,an-me-any,0shot,all,0.094,160,0.250,0.023,,
llama31-8b-instruct,an-me-any,3shot_collide,all,0.700,160,0.250,0.036,,
llama31-8b-instruct,an-me-any,3shot,all,0.600,160,0.250,0.039,,
llama31-8b-instruct,an-you-voc,0shot,all,0.450,80,0.250,0.056,,
llama31-8b-instruct,an-you-voc,3shot_collide,all,0.775,80,0.250,0.047,,
llama31-8b-instruct,an-you-voc,3shot,all,0.838,80,0.250,0.041,,
llama31-8b-instruct,an-you-novoc,0shot,all,0.762,80,0.250,0.048,,
llama31-8b-instruct,an-you-novoc,3shot_collide,all,0.537,80,0.250,0.056,,
llama31-8b-instruct,an-you-novoc,3shot,all,0.775,80,0.250,0.047,,
llama31-8b-instruct,adj-adj,0shot,all,0.969,260,0.333,0.011,,
llama31-8b-instruct,adj-adj,3shot_collide,all,0.996,260,0.333,0.004,,
llama31-8b-instruct,adj-adj,3shot,all,0.996,260,0.333,0.004,,
llama31-8b-instruct,adj-voc,0shot,all,0.989,260,0.333,0.007,,
llama31-8b-instruct,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
llama31-8b-instruct,adj-voc,3shot,all,1.000,260,0.333,0.000,,
llama31-8b-instruct,adj-third,0shot,all,0.969,260,0.333,0.011,,
llama31-8b-instruct,adj-third,3shot_collide,all,0.989,260,0.333,0.007,,
llama31-8b-instruct,adj-third,3shot,all,0.969,260,0.333,0.011,,
llama31-8b-instruct,cm-mutate,0shot,all,0.770,200,0.200,0.030,,
llama31-8b-instruct,cm-mutate,3shot_collide,all,0.980,200,0.200,0.010,,
llama31-8b-instruct,cm-mutate,3shot,all,0.985,200,0.200,0.009,,
llama31-8b-instruct,cm-rebind,0shot,all,0.435,200,0.200,0.035,,
llama31-8b-instruct,cm-rebind,3shot_collide,all,0.925,200,0.200,0.019,,
llama31-8b-instruct,cm-rebind,3shot,all,0.920,200,0.200,0.019,,
llama31-8b-instruct,cm-copy,0shot,all,0.535,200,0.200,0.035,,
llama31-8b-instruct,cm-copy,3shot_collide,all,0.860,200,0.200,0.025,,
llama31-8b-instruct,cm-copy,3shot,all,0.930,200,0.200,0.018,,
falcon-h1-0.5b,dx-owner-3-deictic,0shot,all,0.375,8,0.333,0.171,,other_participant=0.125;speaker_confusion=0.5
falcon-h1-0.5b,dx-owner-3-deictic,3shot,all,0.500,8,0.333,0.177,,other_participant=0.25;speaker_confusion=0.25
falcon-h1-0.5b,dx-owner-3-third,0shot,all,0.625,8,0.333,0.171,,
falcon-h1-0.5b,dx-owner-3-third,3shot,all,0.625,8,0.333,0.171,,
falcon-h1-0.5b,dx-self-3-deictic,0shot,all,0.226,31,0.304,0.075,,other_object=0.129;previous_owner=0.129;speaker_confusion=0.516
falcon-h1-0.5b,dx-self-3-deictic,0shot,free,0.294,17,0.314,0.111,,other_object=0.118;previous_owner=0.059;speaker_confusion=0.529
falcon-h1-0.5b,dx-self-3-deictic,3shot,all,0.194,31,0.304,0.071,,other_object=0.226;previous_owner=0.032;speaker_confusion=0.548
falcon-h1-0.5b,dx-self-3-deictic,3shot,free,0.353,17,0.314,0.116,,other_object=0.176;speaker_confusion=0.471
falcon-h1-0.5b,dx-self-3-third,0shot,all,0.290,31,0.304,0.082,,
falcon-h1-0.5b,dx-self-3-third,0shot,free,0.235,17,0.314,0.103,,
falcon-h1-0.5b,dx-self-3-third,3shot,all,0.355,31,0.304,0.086,,
falcon-h1-0.5b,dx-self-3-third,3shot,free,0.294,17,0.314,0.111,,
falcon-h1-0.5b,dx-self-2-deictic,0shot,all,0.286,42,0.304,0.070,,speaker_confusion=0.714
falcon-h1-0.5b,dx-self-2-deictic,0shot,free,0.269,26,0.301,0.087,,speaker_confusion=0.731
falcon-h1-0.5b,dx-self-2-deictic,3shot,all,0.357,42,0.304,0.074,,speaker_confusion=0.643
falcon-h1-0.5b,dx-self-2-deictic,3shot,free,0.500,26,0.301,0.098,,speaker_confusion=0.5
falcon-h1-0.5b,dx-self-2-third,0shot,all,0.309,42,0.304,0.071,,
falcon-h1-0.5b,dx-self-2-third,0shot,free,0.192,26,0.301,0.077,,
falcon-h1-0.5b,dx-self-2-third,3shot,all,0.452,42,0.304,0.077,,
falcon-h1-0.5b,dx-self-2-third,3shot,free,0.462,26,0.301,0.098,,
falcon-h1-0.5b,dx-addressee-3-deictic,0shot,all,0.421,38,0.298,0.080,,other_object=0.105;previous_owner=0.132;speaker_confusion=0.342
falcon-h1-0.5b,dx-addressee-3-deictic,0shot,free,0.538,26,0.295,0.098,,other_object=0.038;previous_owner=0.192;speaker_confusion=0.231
falcon-h1-0.5b,dx-addressee-3-deictic,3shot,all,0.316,38,0.298,0.075,,other_object=0.316;previous_owner=0.053;speaker_confusion=0.316
falcon-h1-0.5b,dx-addressee-3-deictic,3shot,free,0.462,26,0.295,0.098,,other_object=0.154;previous_owner=0.077;speaker_confusion=0.308
falcon-h1-0.5b,dx-addressee-3-third,0shot,all,0.342,38,0.298,0.077,,
falcon-h1-0.5b,dx-addressee-3-third,0shot,free,0.346,26,0.295,0.093,,
falcon-h1-0.5b,dx-addressee-3-third,3shot,all,0.316,38,0.298,0.075,,
falcon-h1-0.5b,dx-addressee-3-third,3shot,free,0.346,26,0.295,0.093,,
falcon-h1-0.5b,dx-addressee-2-deictic,0shot,all,0.225,49,0.296,0.060,,speaker_confusion=0.776
falcon-h1-0.5b,dx-addressee-2-deictic,0shot,free,0.258,31,0.290,0.079,,speaker_confusion=0.742
falcon-h1-0.5b,dx-addressee-2-deictic,3shot,all,0.367,49,0.296,0.069,,speaker_confusion=0.633
falcon-h1-0.5b,dx-addressee-2-deictic,3shot,free,0.484,31,0.290,0.090,,speaker_confusion=0.516
falcon-h1-0.5b,dx-addressee-2-third,0shot,all,0.510,49,0.296,0.071,,
falcon-h1-0.5b,dx-addressee-2-third,0shot,free,0.452,31,0.290,0.089,,
falcon-h1-0.5b,dx-addressee-2-third,3shot,all,0.388,49,0.296,0.070,,
falcon-h1-0.5b,dx-addressee-2-third,3shot,free,0.452,31,0.290,0.089,,
falcon-h1-1.5b,attr,0shot,all,0.836,415,0.193,0.018,,
falcon-h1-1.5b,attr,3shot_collide,all,0.908,415,0.193,0.014,,
falcon-h1-1.5b,attr,3shot,all,0.990,415,0.193,0.005,,
falcon-h1-1.5b,codevar,0shot,all,0.442,625,0.276,0.020,,
falcon-h1-1.5b,codevar,3shot_collide,all,0.613,625,0.276,0.019,,
falcon-h1-1.5b,codevar,3shot,all,0.610,625,0.276,0.019,,
falcon-h1-1.5b,dialogue,0shot,all,0.498,466,0.328,0.023,,
falcon-h1-1.5b,dialogue,3shot_collide,all,0.614,466,0.328,0.023,,
falcon-h1-1.5b,dialogue,3shot,all,0.646,466,0.328,0.022,,
falcon-h1-1.5b,dx-owner-3-deictic,0shot,all,0.731,26,0.333,0.087,,other_participant=0.231;speaker_confusion=0.038
falcon-h1-1.5b,dx-owner-3-deictic,3shot,all,0.538,26,0.333,0.098,,other_participant=0.269;speaker_confusion=0.192
falcon-h1-1.5b,dx-owner-3-third,0shot,all,0.654,26,0.333,0.093,,
falcon-h1-1.5b,dx-owner-3-third,3shot,all,0.577,26,0.333,0.097,,
falcon-h1-1.5b,dx-self-3-deictic,0shot,all,0.330,100,0.295,0.047,,other_object=0.24;previous_owner=0.06;speaker_confusion=0.37
falcon-h1-1.5b,dx-self-3-deictic,0shot,free,0.468,62,0.294,0.063,,other_object=0.194;speaker_confusion=0.339
falcon-h1-1.5b,dx-self-3-deictic,3shot,all,0.380,100,0.295,0.049,,other_object=0.21;previous_owner=0.07;speaker_confusion=0.34
falcon-h1-1.5b,dx-self-3-deictic,3shot,free,0.500,62,0.294,0.064,,other_object=0.194;previous_owner=0.032;speaker_confusion=0.274
falcon-h1-1.5b,dx-self-3-third,0shot,all,0.320,100,0.295,0.047,,
falcon-h1-1.5b,dx-self-3-third,0shot,free,0.419,62,0.294,0.063,,
falcon-h1-1.5b,dx-self-3-third,3shot,all,0.450,100,0.295,0.050,,
falcon-h1-1.5b,dx-self-3-third,3shot,free,0.613,62,0.294,0.062,,
falcon-h1-1.5b,dx-self-2-deictic,0shot,all,0.480,102,0.292,0.050,,speaker_confusion=0.52
falcon-h1-1.5b,dx-self-2-deictic,0shot,free,0.500,72,0.288,0.059,,speaker_confusion=0.5
falcon-h1-1.5b,dx-self-2-deictic,3shot,all,0.412,102,0.292,0.049,,speaker_confusion=0.588
falcon-h1-1.5b,dx-self-2-deictic,3shot,free,0.556,72,0.288,0.059,,speaker_confusion=0.444
falcon-h1-1.5b,dx-self-2-third,0shot,all,0.490,102,0.292,0.050,,
falcon-h1-1.5b,dx-self-2-third,0shot,free,0.556,72,0.288,0.059,,
falcon-h1-1.5b,dx-self-2-third,3shot,all,0.471,102,0.292,0.049,,
falcon-h1-1.5b,dx-self-2-third,3shot,free,0.611,72,0.288,0.058,,
falcon-h1-1.5b,dx-addressee-3-deictic,0shot,all,0.371,116,0.294,0.045,,other_object=0.207;previous_owner=0.095;speaker_confusion=0.328
falcon-h1-1.5b,dx-addressee-3-deictic,0shot,free,0.425,80,0.291,0.055,,other_object=0.175;previous_owner=0.138;speaker_confusion=0.263
falcon-h1-1.5b,dx-addressee-3-deictic,3shot,all,0.284,116,0.294,0.042,,other_object=0.164;previous_owner=0.112;speaker_confusion=0.44
falcon-h1-1.5b,dx-addressee-3-deictic,3shot,free,0.325,80,0.291,0.052,,other_object=0.15;previous_owner=0.138;speaker_confusion=0.388
falcon-h1-1.5b,dx-addressee-3-third,0shot,all,0.405,116,0.294,0.046,,
falcon-h1-1.5b,dx-addressee-3-third,0shot,free,0.450,80,0.291,0.056,,
falcon-h1-1.5b,dx-addressee-3-third,3shot,all,0.500,116,0.294,0.046,,
falcon-h1-1.5b,dx-addressee-3-third,3shot,free,0.600,80,0.291,0.055,,
falcon-h1-1.5b,dx-addressee-2-deictic,0shot,all,0.229,109,0.292,0.040,,speaker_confusion=0.771
falcon-h1-1.5b,dx-addressee-2-deictic,0shot,free,0.250,76,0.291,0.050,,speaker_confusion=0.75
falcon-h1-1.5b,dx-addressee-2-deictic,3shot,all,0.330,109,0.292,0.045,,speaker_confusion=0.67
falcon-h1-1.5b,dx-addressee-2-deictic,3shot,free,0.368,76,0.291,0.055,,speaker_confusion=0.632
falcon-h1-1.5b,dx-addressee-2-third,0shot,all,0.422,109,0.292,0.047,,
falcon-h1-1.5b,dx-addressee-2-third,0shot,free,0.526,76,0.291,0.057,,
falcon-h1-1.5b,dx-addressee-2-third,3shot,all,0.532,109,0.292,0.048,,
falcon-h1-1.5b,dx-addressee-2-third,3shot,free,0.671,76,0.291,0.054,,
falcon-h1-1.5b,sb-k1,0shot,all,0.882,17,0.125,0.078,,
falcon-h1-1.5b,sb-k1,3shot_collide,all,1.000,17,0.125,0.000,,
falcon-h1-1.5b,sb-k1,3shot,all,1.000,17,0.125,0.000,,
falcon-h1-1.5b,sb-k23,0shot,all,0.676,34,0.027,0.080,,
falcon-h1-1.5b,sb-k23,3shot_collide,all,0.853,34,0.027,0.061,,
falcon-h1-1.5b,sb-k23,3shot,all,0.882,34,0.027,0.055,,
falcon-h1-1.5b,sb-split,0shot,all,0.344,32,0.026,0.084,,
falcon-h1-1.5b,sb-split,3shot_collide,all,0.281,32,0.026,0.080,,
falcon-h1-1.5b,sb-split,3shot,all,0.500,32,0.026,0.088,,
falcon-h1-1.5b,sb-dyn,0shot,all,0.128,86,0.026,0.036,,
falcon-h1-1.5b,sb-dyn,3shot_collide,all,0.116,86,0.026,0.035,,
falcon-h1-1.5b,sb-dyn,3shot,all,0.209,86,0.026,0.044,,
falcon-h1-1.5b,an-narr,0shot,all,0.395,114,0.250,0.046,,
falcon-h1-1.5b,an-narr,3shot_collide,all,0.447,114,0.250,0.047,,
falcon-h1-1.5b,an-narr,3shot,all,0.456,114,0.250,0.047,,
falcon-h1-1.5b,an-other-any,0shot,all,0.305,128,0.250,0.041,,
falcon-h1-1.5b,an-other-any,3shot_collide,all,0.211,128,0.250,0.036,,
falcon-h1-1.5b,an-other-any,3shot,all,0.305,128,0.250,0.041,,
falcon-h1-1.5b,an-me-any,0shot,all,0.746,67,0.250,0.053,,
falcon-h1-1.5b,an-me-any,3shot_collide,all,0.090,67,0.250,0.035,,
falcon-h1-1.5b,an-me-any,3shot,all,0.000,67,0.250,0.000,,
falcon-h1-1.5b,an-you-voc,0shot,all,0.088,34,0.250,0.049,,
falcon-h1-1.5b,an-you-voc,3shot_collide,all,0.118,34,0.250,0.055,,
falcon-h1-1.5b,an-you-voc,3shot,all,0.059,34,0.250,0.040,,
falcon-h1-1.5b,an-you-novoc,0shot,all,0.024,42,0.250,0.024,,
falcon-h1-1.5b,an-you-novoc,3shot_collide,all,0.119,42,0.250,0.050,,
falcon-h1-1.5b,an-you-novoc,3shot,all,0.071,42,0.250,0.040,,
falcon-h1-1.5b,adj-adj,0shot,all,0.661,62,0.333,0.060,,
falcon-h1-1.5b,adj-adj,3shot_collide,all,0.548,62,0.333,0.063,,
falcon-h1-1.5b,adj-adj,3shot,all,0.548,62,0.333,0.063,,
falcon-h1-1.5b,adj-voc,0shot,all,0.726,62,0.333,0.057,,
falcon-h1-1.5b,adj-voc,3shot_collide,all,0.726,62,0.333,0.057,,
falcon-h1-1.5b,adj-voc,3shot,all,0.661,62,0.333,0.060,,
falcon-h1-1.5b,adj-third,0shot,all,0.694,62,0.333,0.059,,
falcon-h1-1.5b,adj-third,3shot_collide,all,0.710,62,0.333,0.058,,
falcon-h1-1.5b,adj-third,3shot,all,0.742,62,0.333,0.056,,
falcon-h1-1.5b,cm-mutate,0shot,all,0.947,94,0.200,0.023,,
falcon-h1-1.5b,cm-mutate,3shot_collide,all,0.766,94,0.200,0.044,,
falcon-h1-1.5b,cm-mutate,3shot,all,0.830,94,0.200,0.039,,
falcon-h1-1.5b,cm-rebind,0shot,all,0.110,82,0.200,0.035,,
falcon-h1-1.5b,cm-rebind,3shot_collide,all,0.817,82,0.200,0.043,,
falcon-h1-1.5b,cm-rebind,3shot,all,0.732,82,0.200,0.049,,
falcon-h1-1.5b,cm-copy,0shot,all,0.011,87,0.200,0.011,,
falcon-h1-1.5b,cm-copy,3shot_collide,all,0.943,87,0.200,0.025,,
falcon-h1-1.5b,cm-copy,3shot,all,0.919,87,0.200,0.029,,
falcon-h1-1.5b-deep,attr,0shot,all,0.793,415,0.193,0.020,,
falcon-h1-1.5b-deep,attr,3shot_collide,all,0.935,415,0.193,0.012,,
falcon-h1-1.5b-deep,attr,3shot,all,0.988,415,0.193,0.005,,
falcon-h1-1.5b-deep,codevar,0shot,all,0.509,625,0.276,0.020,,
falcon-h1-1.5b-deep,codevar,3shot_collide,all,0.685,625,0.276,0.019,,
falcon-h1-1.5b-deep,codevar,3shot,all,0.677,625,0.276,0.019,,
falcon-h1-1.5b-deep,dialogue,0shot,all,0.708,466,0.328,0.021,,
falcon-h1-1.5b-deep,dialogue,3shot_collide,all,0.766,466,0.328,0.020,,
falcon-h1-1.5b-deep,dialogue,3shot,all,0.751,466,0.328,0.020,,
falcon-h1-1.5b-deep,dx-owner-3-deictic,0shot,all,0.731,26,0.333,0.087,,other_participant=0.231;speaker_confusion=0.038
falcon-h1-1.5b-deep,dx-owner-3-deictic,3shot,all,0.769,26,0.333,0.083,,other_participant=0.192;tie=0.038
falcon-h1-1.5b-deep,dx-owner-3-third,0shot,all,0.692,26,0.333,0.090,,
falcon-h1-1.5b-deep,dx-owner-3-third,3shot,all,0.923,26,0.333,0.052,,
falcon-h1-1.5b-deep,dx-self-3-deictic,0shot,all,0.360,100,0.295,0.048,,other_object=0.21;previous_owner=0.05;speaker_confusion=0.38
falcon-h1-1.5b-deep,dx-self-3-deictic,0shot,free,0.355,62,0.294,0.061,,other_object=0.113;previous_owner=0.048;speaker_confusion=0.484
falcon-h1-1.5b-deep,dx-self-3-deictic,3shot,all,0.390,100,0.295,0.049,,other_object=0.14;previous_owner=0.06;speaker_confusion=0.41
falcon-h1-1.5b-deep,dx-self-3-deictic,3shot,free,0.468,62,0.294,0.063,,other_object=0.145;previous_owner=0.032;speaker_confusion=0.355
falcon-h1-1.5b-deep,dx-self-3-third,0shot,all,0.560,100,0.295,0.050,,
falcon-h1-1.5b-deep,dx-self-3-third,0shot,free,0.581,62,0.294,0.063,,
falcon-h1-1.5b-deep,dx-self-3-third,3shot,all,0.670,100,0.295,0.047,,
falcon-h1-1.5b-deep,dx-self-3-third,3shot,free,0.661,62,0.294,0.060,,
falcon-h1-1.5b-deep,dx-self-2-deictic,0shot,all,0.451,102,0.292,0.049,,speaker_confusion=0.549
falcon-h1-1.5b-deep,dx-self-2-deictic,0shot,free,0.417,72,0.288,0.058,,speaker_confusion=0.583
falcon-h1-1.5b-deep,dx-self-2-deictic,3shot,all,0.539,102,0.292,0.049,,speaker_confusion=0.461
falcon-h1-1.5b-deep,dx-self-2-deictic,3shot,free,0.556,72,0.288,0.059,,speaker_confusion=0.444
falcon-h1-1.5b-deep,dx-self-2-third,0shot,all,0.755,102,0.292,0.043,,
falcon-h1-1.5b-deep,dx-self-2-third,0shot,free,0.833,72,0.288,0.044,,
falcon-h1-1.5b-deep,dx-self-2-third,3shot,all,0.667,102,0.292,0.047,,
falcon-h1-1.5b-deep,dx-self-2-third,3shot,free,0.750,72,0.288,0.051,,
falcon-h1-1.5b-deep,dx-addressee-3-deictic,0shot,all,0.353,116,0.294,0.044,,other_object=0.078;previous_owner=0.164;speaker_confusion=0.405
falcon-h1-1.5b-deep,dx-addressee-3-deictic,0shot,free,0.325,80,0.291,0.052,,other_object=0.062;previous_owner=0.225;speaker_confusion=0.388
falcon-h1-1.5b-deep,dx-addressee-3-deictic,3shot,all,0.328,116,0.294,0.044,,other_object=0.155;previous_owner=0.121;speaker_confusion=0.397
falcon-h1-1.5b-deep,dx-addressee-3-deictic,3shot,free,0.350,80,0.291,0.053,,other_object=0.1;previous_owner=0.163;speaker_confusion=0.388
falcon-h1-1.5b-deep,dx-addressee-3-third,0shot,all,0.664,116,0.294,0.044,,
falcon-h1-1.5b-deep,dx-addressee-3-third,0shot,free,0.637,80,0.291,0.054,,
falcon-h1-1.5b-deep,dx-addressee-3-third,3shot,all,0.612,116,0.294,0.045,,
falcon-h1-1.5b-deep,dx-addressee-3-third,3shot,free,0.613,80,0.291,0.054,,
falcon-h1-1.5b-deep,dx-addressee-2-deictic,0shot,all,0.174,109,0.292,0.036,,speaker_confusion=0.826
falcon-h1-1.5b-deep,dx-addressee-2-deictic,0shot,free,0.118,76,0.291,0.037,,speaker_confusion=0.882
falcon-h1-1.5b-deep,dx-addressee-2-deictic,3shot,all,0.349,109,0.292,0.046,,speaker_confusion=0.651
falcon-h1-1.5b-deep,dx-addressee-2-deictic,3shot,free,0.395,76,0.291,0.056,,speaker_confusion=0.605
falcon-h1-1.5b-deep,dx-addressee-2-third,0shot,all,0.651,109,0.292,0.046,,
falcon-h1-1.5b-deep,dx-addressee-2-third,0shot,free,0.776,76,0.291,0.048,,
falcon-h1-1.5b-deep,dx-addressee-2-third,3shot,all,0.651,109,0.292,0.046,,
falcon-h1-1.5b-deep,dx-addressee-2-third,3shot,free,0.803,76,0.291,0.046,,
falcon-h1-1.5b-deep,sb-k1,0shot,all,0.941,17,0.125,0.057,,
falcon-h1-1.5b-deep,sb-k1,3shot_collide,all,1.000,17,0.125,0.000,,
falcon-h1-1.5b-deep,sb-k1,3shot,all,1.000,17,0.125,0.000,,
falcon-h1-1.5b-deep,sb-k23,0shot,all,0.676,34,0.027,0.080,,
falcon-h1-1.5b-deep,sb-k23,3shot_collide,all,0.853,34,0.027,0.061,,
falcon-h1-1.5b-deep,sb-k23,3shot,all,0.853,34,0.027,0.061,,
falcon-h1-1.5b-deep,sb-split,0shot,all,0.656,32,0.026,0.084,,
falcon-h1-1.5b-deep,sb-split,3shot_collide,all,0.625,32,0.026,0.086,,
falcon-h1-1.5b-deep,sb-split,3shot,all,0.625,32,0.026,0.086,,
falcon-h1-1.5b-deep,sb-dyn,0shot,all,0.337,86,0.026,0.051,,
falcon-h1-1.5b-deep,sb-dyn,3shot_collide,all,0.209,86,0.026,0.044,,
falcon-h1-1.5b-deep,sb-dyn,3shot,all,0.198,86,0.026,0.043,,
falcon-h1-1.5b-deep,an-narr,0shot,all,0.447,114,0.250,0.047,,
falcon-h1-1.5b-deep,an-narr,3shot_collide,all,0.658,114,0.250,0.044,,
falcon-h1-1.5b-deep,an-narr,3shot,all,0.684,114,0.250,0.043,,
falcon-h1-1.5b-deep,an-other-any,0shot,all,0.430,128,0.250,0.044,,
falcon-h1-1.5b-deep,an-other-any,3shot_collide,all,0.336,128,0.250,0.042,,
falcon-h1-1.5b-deep,an-other-any,3shot,all,0.328,128,0.250,0.042,,
falcon-h1-1.5b-deep,an-me-any,0shot,all,0.164,67,0.250,0.045,,
falcon-h1-1.5b-deep,an-me-any,3shot_collide,all,0.000,67,0.250,0.000,,
falcon-h1-1.5b-deep,an-me-any,3shot,all,0.015,67,0.250,0.015,,
falcon-h1-1.5b-deep,an-you-voc,0shot,all,0.471,34,0.250,0.086,,
falcon-h1-1.5b-deep,an-you-voc,3shot_collide,all,0.588,34,0.250,0.084,,
falcon-h1-1.5b-deep,an-you-voc,3shot,all,0.559,34,0.250,0.085,,
falcon-h1-1.5b-deep,an-you-novoc,0shot,all,0.548,42,0.250,0.077,,
falcon-h1-1.5b-deep,an-you-novoc,3shot_collide,all,0.738,42,0.250,0.068,,
falcon-h1-1.5b-deep,an-you-novoc,3shot,all,0.595,42,0.250,0.076,,
falcon-h1-1.5b-deep,adj-adj,0shot,all,0.758,62,0.333,0.054,,
falcon-h1-1.5b-deep,adj-adj,3shot_collide,all,0.645,62,0.333,0.061,,
falcon-h1-1.5b-deep,adj-adj,3shot,all,0.710,62,0.333,0.058,,
falcon-h1-1.5b-deep,adj-voc,0shot,all,0.774,62,0.333,0.053,,
falcon-h1-1.5b-deep,adj-voc,3shot_collide,all,0.952,62,0.333,0.027,,
falcon-h1-1.5b-deep,adj-voc,3shot,all,0.952,62,0.333,0.027,,
falcon-h1-1.5b-deep,adj-third,0shot,all,0.806,62,0.333,0.050,,
falcon-h1-1.5b-deep,adj-third,3shot_collide,all,0.968,62,0.333,0.022,,
falcon-h1-1.5b-deep,adj-third,3shot,all,0.935,62,0.333,0.031,,
falcon-h1-1.5b-deep,cm-mutate,0shot,all,0.840,94,0.200,0.038,,
falcon-h1-1.5b-deep,cm-mutate,3shot_collide,all,0.936,94,0.200,0.025,,
falcon-h1-1.5b-deep,cm-mutate,3shot,all,0.925,94,0.200,0.027,,
falcon-h1-1.5b-deep,cm-rebind,0shot,all,0.390,82,0.200,0.054,,
falcon-h1-1.5b-deep,cm-rebind,3shot_collide,all,0.890,82,0.200,0.035,,
falcon-h1-1.5b-deep,cm-rebind,3shot,all,0.915,82,0.200,0.031,,
falcon-h1-1.5b-deep,cm-copy,0shot,all,0.023,87,0.200,0.016,,
falcon-h1-1.5b-deep,cm-copy,3shot_collide,all,0.966,87,0.200,0.020,,
falcon-h1-1.5b-deep,cm-copy,3shot,all,0.966,87,0.200,0.020,,
falcon-h1-3b,attr,0shot,all,0.945,415,0.193,0.011,,
falcon-h1-3b,attr,3shot_collide,all,0.998,415,0.193,0.002,,
falcon-h1-3b,attr,3shot,all,0.998,415,0.193,0.002,,
falcon-h1-3b,codevar,0shot,all,0.549,625,0.276,0.020,,
falcon-h1-3b,codevar,3shot_collide,all,0.752,625,0.276,0.017,,
falcon-h1-3b,codevar,3shot,all,0.712,625,0.276,0.018,,
falcon-h1-3b,dialogue,0shot,all,0.734,466,0.328,0.021,,
falcon-h1-3b,dialogue,3shot_collide,all,0.807,466,0.328,0.018,,
falcon-h1-3b,dialogue,3shot,all,0.813,466,0.328,0.018,,
falcon-h1-3b,dx-owner-3-deictic,0shot,all,0.692,26,0.333,0.090,,other_participant=0.231;speaker_confusion=0.077
falcon-h1-3b,dx-owner-3-deictic,3shot,all,0.846,26,0.333,0.071,,other_participant=0.154
falcon-h1-3b,dx-owner-3-third,0shot,all,0.654,26,0.333,0.093,,
falcon-h1-3b,dx-owner-3-third,3shot,all,0.923,26,0.333,0.052,,
falcon-h1-3b,dx-self-3-deictic,0shot,all,0.330,100,0.295,0.047,,other_object=0.18;previous_owner=0.05;speaker_confusion=0.44
falcon-h1-3b,dx-self-3-deictic,0shot,free,0.258,62,0.294,0.056,,other_object=0.145;previous_owner=0.032;speaker_confusion=0.565
falcon-h1-3b,dx-self-3-deictic,3shot,all,0.490,100,0.295,0.050,,other_object=0.15;previous_owner=0.03;speaker_confusion=0.33
falcon-h1-3b,dx-self-3-deictic,3shot,free,0.661,62,0.294,0.060,,other_object=0.113;previous_owner=0.016;speaker_confusion=0.21
falcon-h1-3b,dx-self-3-third,0shot,all,0.510,100,0.295,0.050,,
falcon-h1-3b,dx-self-3-third,0shot,free,0.597,62,0.294,0.062,,
falcon-h1-3b,dx-self-3-third,3shot,all,0.470,100,0.295,0.050,,
falcon-h1-3b,dx-self-3-third,3shot,free,0.629,62,0.294,0.061,,
falcon-h1-3b,dx-self-2-deictic,0shot,all,0.353,102,0.292,0.047,,speaker_confusion=0.647
falcon-h1-3b,dx-self-2-deictic,0shot,free,0.347,72,0.288,0.056,,speaker_confusion=0.653
falcon-h1-3b,dx-self-2-deictic,3shot,all,0.529,102,0.292,0.049,,speaker_confusion=0.471
falcon-h1-3b,dx-self-2-deictic,3shot,free,0.681,72,0.288,0.055,,speaker_confusion=0.319
falcon-h1-3b,dx-self-2-third,0shot,all,0.647,102,0.292,0.047,,
falcon-h1-3b,dx-self-2-third,0shot,free,0.806,72,0.288,0.047,,
falcon-h1-3b,dx-self-2-third,3shot,all,0.686,102,0.292,0.046,,
falcon-h1-3b,dx-self-2-third,3shot,free,0.778,72,0.288,0.049,,
falcon-h1-3b,dx-addressee-3-deictic,0shot,all,0.362,116,0.294,0.045,,other_object=0.138;previous_owner=0.095;speaker_confusion=0.405
falcon-h1-3b,dx-addressee-3-deictic,0shot,free,0.338,80,0.291,0.053,,other_object=0.113;previous_owner=0.138;speaker_confusion=0.412
falcon-h1-3b,dx-addressee-3-deictic,3shot,all,0.353,116,0.294,0.044,,other_object=0.172;previous_owner=0.06;speaker_confusion=0.414
falcon-h1-3b,dx-addressee-3-deictic,3shot,free,0.438,80,0.291,0.056,,other_object=0.113;previous_owner=0.075;speaker_confusion=0.375
falcon-h1-3b,dx-addressee-3-third,0shot,all,0.578,116,0.294,0.046,,
falcon-h1-3b,dx-addressee-3-third,0shot,free,0.650,80,0.291,0.053,,
falcon-h1-3b,dx-addressee-3-third,3shot,all,0.612,116,0.294,0.045,,
falcon-h1-3b,dx-addressee-3-third,3shot,free,0.688,80,0.291,0.052,,
falcon-h1-3b,dx-addressee-2-deictic,0shot,all,0.303,109,0.292,0.044,,speaker_confusion=0.697
falcon-h1-3b,dx-addressee-2-deictic,0shot,free,0.250,76,0.291,0.050,,speaker_confusion=0.75
falcon-h1-3b,dx-addressee-2-deictic,3shot,all,0.367,109,0.292,0.046,,speaker_confusion=0.624;tie=0.009
falcon-h1-3b,dx-addressee-2-deictic,3shot,free,0.447,76,0.291,0.057,,speaker_confusion=0.539;tie=0.013
falcon-h1-3b,dx-addressee-2-third,0shot,all,0.615,109,0.292,0.047,,
falcon-h1-3b,dx-addressee-2-third,0shot,free,0.750,76,0.291,0.050,,
falcon-h1-3b,dx-addressee-2-third,3shot,all,0.642,109,0.292,0.046,,
falcon-h1-3b,dx-addressee-2-third,3shot,free,0.789,76,0.291,0.047,,
falcon-h1-7b,attr,0shot,all,0.875,972,0.172,0.011,,
falcon-h1-7b,attr,3shot_collide,all,0.994,972,0.172,0.003,,
falcon-h1-7b,attr,3shot,all,0.999,972,0.172,0.001,,
falcon-h1-7b,codevar,0shot,all,0.540,1040,0.270,0.015,,
falcon-h1-7b,codevar,3shot_collide,all,0.834,1040,0.270,0.011,,
falcon-h1-7b,codevar,3shot,all,0.808,1040,0.270,0.012,,
falcon-h1-7b,dialogue,0shot,all,0.904,1055,0.314,0.009,,
falcon-h1-7b,dialogue,3shot_collide,all,0.905,1055,0.314,0.009,,
falcon-h1-7b,dialogue,3shot,all,0.896,1055,0.314,0.009,,
falcon-h1-7b,dx-owner-3-deictic,0shot,all,0.667,99,0.333,0.047,,other_participant=0.303;speaker_confusion=0.03
falcon-h1-7b,dx-owner-3-deictic,0shot,free,0.636,33,0.333,0.084,,other_participant=0.364
falcon-h1-7b,dx-owner-3-deictic,3shot,all,0.899,99,0.333,0.030,,other_participant=0.081;speaker_confusion=0.02
falcon-h1-7b,dx-owner-3-deictic,3shot,free,0.909,33,0.333,0.050,,other_participant=0.091
falcon-h1-7b,dx-owner-3-third,0shot,all,0.889,99,0.333,0.032,,
falcon-h1-7b,dx-owner-3-third,0shot,free,0.909,33,0.333,0.050,,
falcon-h1-7b,dx-owner-3-third,3shot,all,0.950,99,0.333,0.022,,
falcon-h1-7b,dx-owner-3-third,3shot,free,1.000,33,0.333,0.000,,
falcon-h1-7b,dx-self-3-deictic,0shot,all,0.350,163,0.289,0.037,,other_object=0.184;previous_owner=0.061;speaker_confusion=0.399;tie=0.006
falcon-h1-7b,dx-self-3-deictic,0shot,free,0.414,111,0.287,0.047,,other_object=0.144;previous_owner=0.027;speaker_confusion=0.405;tie=0.009
falcon-h1-7b,dx-self-3-deictic,3shot,all,0.595,163,0.289,0.038,,other_object=0.086;previous_owner=0.055;speaker_confusion=0.264
falcon-h1-7b,dx-self-3-deictic,3shot,free,0.649,111,0.287,0.045,,other_object=0.081;previous_owner=0.045;speaker_confusion=0.225
falcon-h1-7b,dx-self-3-third,0shot,all,0.797,163,0.289,0.032,,
falcon-h1-7b,dx-self-3-third,0shot,free,0.856,111,0.287,0.033,,
falcon-h1-7b,dx-self-3-third,3shot,all,0.755,163,0.289,0.034,,
falcon-h1-7b,dx-self-3-third,3shot,free,0.865,111,0.287,0.032,,
falcon-h1-7b,dx-self-2-deictic,0shot,all,0.352,159,0.288,0.038,,speaker_confusion=0.642;tie=0.006
falcon-h1-7b,dx-self-2-deictic,0shot,free,0.404,109,0.287,0.047,,speaker_confusion=0.587;tie=0.009
falcon-h1-7b,dx-self-2-deictic,3shot,all,0.528,159,0.288,0.040,,speaker_confusion=0.472
falcon-h1-7b,dx-self-2-deictic,3shot,free,0.633,109,0.287,0.046,,speaker_confusion=0.367
falcon-h1-7b,dx-self-2-third,0shot,all,0.679,159,0.288,0.037,,
falcon-h1-7b,dx-self-2-third,0shot,free,0.780,109,0.287,0.040,,
falcon-h1-7b,dx-self-2-third,3shot,all,0.679,159,0.288,0.037,,
falcon-h1-7b,dx-self-2-third,3shot,free,0.771,109,0.287,0.040,,
falcon-h1-7b,dx-addressee-3-deictic,0shot,all,0.366,164,0.289,0.038,,other_object=0.195;previous_owner=0.061;speaker_confusion=0.372;tie=0.006
falcon-h1-7b,dx-addressee-3-deictic,0shot,free,0.383,115,0.288,0.045,,other_object=0.174;previous_owner=0.078;speaker_confusion=0.357;tie=0.009
falcon-h1-7b,dx-addressee-3-deictic,3shot,all,0.457,164,0.289,0.039,,other_object=0.091;previous_owner=0.116;speaker_confusion=0.335
falcon-h1-7b,dx-addressee-3-deictic,3shot,free,0.513,115,0.288,0.047,,other_object=0.043;previous_owner=0.148;speaker_confusion=0.296
falcon-h1-7b,dx-addressee-3-third,0shot,all,0.811,164,0.289,0.031,,
falcon-h1-7b,dx-addressee-3-third,0shot,free,0.844,115,0.288,0.034,,
falcon-h1-7b,dx-addressee-3-third,3shot,all,0.738,164,0.289,0.034,,
falcon-h1-7b,dx-addressee-3-third,3shot,free,0.826,115,0.288,0.035,,
falcon-h1-7b,dx-addressee-2-deictic,0shot,all,0.289,166,0.287,0.035,,speaker_confusion=0.711
falcon-h1-7b,dx-addressee-2-deictic,0shot,free,0.291,117,0.286,0.042,,speaker_confusion=0.709
falcon-h1-7b,dx-addressee-2-deictic,3shot,all,0.428,166,0.287,0.038,,speaker_confusion=0.572
falcon-h1-7b,dx-addressee-2-deictic,3shot,free,0.479,117,0.286,0.046,,speaker_confusion=0.521
falcon-h1-7b,dx-addressee-2-third,0shot,all,0.705,166,0.287,0.035,,
falcon-h1-7b,dx-addressee-2-third,0shot,free,0.821,117,0.286,0.035,,
falcon-h1-7b,dx-addressee-2-third,3shot,all,0.717,166,0.287,0.035,,
falcon-h1-7b,dx-addressee-2-third,3shot,free,0.803,117,0.286,0.037,,
falcon-h1-7b,sb-k1,0shot,all,0.978,45,0.125,0.022,,
falcon-h1-7b,sb-k1,3shot_collide,all,1.000,45,0.125,0.000,,
falcon-h1-7b,sb-k1,3shot,all,1.000,45,0.125,0.000,,
falcon-h1-7b,sb-k23,0shot,all,0.698,96,0.027,0.047,,
falcon-h1-7b,sb-k23,3shot_collide,all,0.938,96,0.027,0.025,,
falcon-h1-7b,sb-k23,3shot,all,0.938,96,0.027,0.025,,
falcon-h1-7b,sb-split,0shot,all,0.726,102,0.026,0.044,,
falcon-h1-7b,sb-split,3shot_collide,all,0.529,102,0.026,0.049,,
falcon-h1-7b,sb-split,3shot,all,0.539,102,0.026,0.049,,
falcon-h1-7b,sb-dyn,0shot,all,0.608,194,0.027,0.035,,
falcon-h1-7b,sb-dyn,3shot_collide,all,0.340,194,0.027,0.034,,
falcon-h1-7b,sb-dyn,3shot,all,0.387,194,0.027,0.035,,
falcon-h1-7b,an-narr,0shot,all,0.594,549,0.250,0.021,,
falcon-h1-7b,an-narr,3shot_collide,all,0.878,549,0.250,0.014,,
falcon-h1-7b,an-narr,3shot,all,0.920,549,0.250,0.012,,
falcon-h1-7b,an-other-any,0shot,all,0.492,297,0.250,0.029,,
falcon-h1-7b,an-other-any,3shot_collide,all,0.667,297,0.250,0.027,,
falcon-h1-7b,an-other-any,3shot,all,0.714,297,0.250,0.026,,
falcon-h1-7b,an-me-any,0shot,all,0.265,151,0.250,0.036,,
falcon-h1-7b,an-me-any,3shot_collide,all,0.199,151,0.250,0.033,,
falcon-h1-7b,an-me-any,3shot,all,0.212,151,0.250,0.033,,
falcon-h1-7b,an-you-voc,0shot,all,0.635,74,0.250,0.056,,
falcon-h1-7b,an-you-voc,3shot_collide,all,0.824,74,0.250,0.044,,
falcon-h1-7b,an-you-voc,3shot,all,0.878,74,0.250,0.038,,
falcon-h1-7b,an-you-novoc,0shot,all,0.416,77,0.250,0.056,,
falcon-h1-7b,an-you-novoc,3shot_collide,all,0.506,77,0.250,0.057,,
falcon-h1-7b,an-you-novoc,3shot,all,0.532,77,0.250,0.057,,
falcon-h1-7b,adj-adj,0shot,all,0.947,226,0.333,0.015,,
falcon-h1-7b,adj-adj,3shot_collide,all,0.969,226,0.333,0.011,,
falcon-h1-7b,adj-adj,3shot,all,0.978,226,0.333,0.010,,
falcon-h1-7b,adj-voc,0shot,all,0.969,226,0.333,0.011,,
falcon-h1-7b,adj-voc,3shot_collide,all,0.987,226,0.333,0.008,,
falcon-h1-7b,adj-voc,3shot,all,0.982,226,0.333,0.009,,
falcon-h1-7b,adj-third,0shot,all,0.996,226,0.333,0.004,,
falcon-h1-7b,adj-third,3shot_collide,all,0.996,226,0.333,0.004,,
falcon-h1-7b,adj-third,3shot,all,1.000,226,0.333,0.000,,
falcon-h1-7b,cm-mutate,0shot,all,0.982,169,0.200,0.010,,
falcon-h1-7b,cm-mutate,3shot_collide,all,0.994,169,0.200,0.006,,
falcon-h1-7b,cm-mutate,3shot,all,0.988,169,0.200,0.008,,
falcon-h1-7b,cm-rebind,0shot,all,0.095,169,0.200,0.022,,
falcon-h1-7b,cm-rebind,3shot_collide,all,0.988,169,0.200,0.008,,
falcon-h1-7b,cm-rebind,3shot,all,0.976,169,0.200,0.012,,
falcon-h1-7b,cm-copy,0shot,all,0.059,170,0.200,0.018,,
falcon-h1-7b,cm-copy,3shot_collide,all,1.000,170,0.200,0.000,,
falcon-h1-7b,cm-copy,3shot,all,0.994,170,0.200,0.006,,
falcon-mamba-7b,attr,0shot,all,0.398,1210,0.168,0.014,,
falcon-mamba-7b,attr,3shot_collide,all,0.774,1210,0.168,0.012,,
falcon-mamba-7b,attr,3shot,all,0.962,1210,0.168,0.005,,
falcon-mamba-7b,codevar,0shot,all,0.242,1200,0.267,0.012,,
falcon-mamba-7b,codevar,3shot_collide,all,0.327,1200,0.267,0.013,,
falcon-mamba-7b,codevar,3shot,all,0.314,1200,0.267,0.013,,
falcon-mamba-7b,dialogue,0shot,all,0.622,1200,0.312,0.014,,
falcon-mamba-7b,dialogue,3shot_collide,all,0.613,1200,0.312,0.014,,
falcon-mamba-7b,dialogue,3shot,all,0.580,1200,0.312,0.014,,
falcon-mamba-7b,dx-owner-3-deictic,0shot,all,0.250,120,0.333,0.040,,other_participant=0.467;speaker_confusion=0.092;tie=0.192
falcon-mamba-7b,dx-owner-3-deictic,0shot,free,0.050,40,0.333,0.035,,other_participant=0.65;speaker_confusion=0.075;tie=0.225
falcon-mamba-7b,dx-owner-3-deictic,3shot,all,0.267,120,0.333,0.040,,other_participant=0.383;speaker_confusion=0.2;tie=0.15
falcon-mamba-7b,dx-owner-3-deictic,3shot,free,0.250,40,0.333,0.069,,other_participant=0.475;speaker_confusion=0.15;tie=0.125
falcon-mamba-7b,dx-owner-3-third,0shot,all,0.458,120,0.333,0.045,,
falcon-mamba-7b,dx-owner-3-third,0shot,free,0.375,40,0.333,0.076,,
falcon-mamba-7b,dx-owner-3-third,3shot,all,0.308,120,0.333,0.042,,
falcon-mamba-7b,dx-owner-3-third,3shot,free,0.425,40,0.333,0.078,,
falcon-mamba-7b,dx-self-3-deictic,0shot,all,0.172,180,0.287,0.028,,other_object=0.278;previous_owner=0.061;speaker_confusion=0.317;tie=0.172
falcon-mamba-7b,dx-self-3-deictic,0shot,free,0.191,126,0.286,0.035,,other_object=0.302;previous_owner=0.048;speaker_confusion=0.294;tie=0.167
falcon-mamba-7b,dx-self-3-deictic,3shot,all,0.200,180,0.287,0.030,,other_object=0.317;previous_owner=0.039;speaker_confusion=0.261;tie=0.183
falcon-mamba-7b,dx-self-3-deictic,3shot,free,0.222,126,0.286,0.037,,other_object=0.349;previous_owner=0.024;speaker_confusion=0.206;tie=0.198
falcon-mamba-7b,dx-self-3-third,0shot,all,0.300,180,0.287,0.034,,
falcon-mamba-7b,dx-self-3-third,0shot,free,0.175,126,0.286,0.034,,
falcon-mamba-7b,dx-self-3-third,3shot,all,0.317,180,0.287,0.035,,
falcon-mamba-7b,dx-self-3-third,3shot,free,0.262,126,0.286,0.039,,
falcon-mamba-7b,dx-self-2-deictic,0shot,all,0.178,180,0.287,0.029,,speaker_confusion=0.622;tie=0.2
falcon-mamba-7b,dx-self-2-deictic,0shot,free,0.191,126,0.286,0.035,,speaker_confusion=0.611;tie=0.198
falcon-mamba-7b,dx-self-2-deictic,3shot,all,0.272,180,0.287,0.033,,speaker_confusion=0.567;tie=0.161
falcon-mamba-7b,dx-self-2-deictic,3shot,free,0.309,126,0.286,0.041,,speaker_confusion=0.5;tie=0.19
falcon-mamba-7b,dx-self-2-third,0shot,all,0.406,180,0.287,0.037,,
falcon-mamba-7b,dx-self-2-third,0shot,free,0.318,126,0.286,0.042,,
falcon-mamba-7b,dx-self-2-third,3shot,all,0.400,180,0.287,0.036,,
falcon-mamba-7b,dx-self-2-third,3shot,free,0.357,126,0.286,0.043,,
falcon-mamba-7b,dx-addressee-3-deictic,0shot,all,0.183,180,0.287,0.029,,other_object=0.228;previous_owner=0.078;speaker_confusion=0.394;tie=0.117
falcon-mamba-7b,dx-addressee-3-deictic,0shot,free,0.191,126,0.286,0.035,,other_object=0.206;previous_owner=0.095;speaker_confusion=0.397;tie=0.111
falcon-mamba-7b,dx-addressee-3-deictic,3shot,all,0.172,180,0.287,0.028,,other_object=0.25;previous_owner=0.106;speaker_confusion=0.361;tie=0.111
falcon-mamba-7b,dx-addressee-3-deictic,3shot,free,0.191,126,0.286,0.035,,other_object=0.206;previous_owner=0.127;speaker_confusion=0.357;tie=0.119
falcon-mamba-7b,dx-addressee-3-third,0shot,all,0.344,180,0.287,0.035,,
falcon-mamba-7b,dx-addressee-3-third,0shot,free,0.214,126,0.286,0.037,,
falcon-mamba-7b,dx-addressee-3-third,3shot,all,0.272,180,0.287,0.033,,
falcon-mamba-7b,dx-addressee-3-third,3shot,free,0.230,126,0.286,0.037,,
falcon-mamba-7b,dx-addressee-2-deictic,0shot,all,0.144,180,0.287,0.026,,speaker_confusion=0.694;tie=0.161
falcon-mamba-7b,dx-addressee-2-deictic,0shot,free,0.167,126,0.286,0.033,,speaker_confusion=0.675;tie=0.159
falcon-mamba-7b,dx-addressee-2-deictic,3shot,all,0.144,180,0.287,0.026,,speaker_confusion=0.744;tie=0.111
falcon-mamba-7b,dx-addressee-2-deictic,3shot,free,0.127,126,0.286,0.030,,speaker_confusion=0.73;tie=0.143
falcon-mamba-7b,dx-addressee-2-third,0shot,all,0.444,180,0.287,0.037,,
falcon-mamba-7b,dx-addressee-2-third,0shot,free,0.365,126,0.286,0.043,,
falcon-mamba-7b,dx-addressee-2-third,3shot,all,0.439,180,0.287,0.037,,
falcon-mamba-7b,dx-addressee-2-third,3shot,free,0.405,126,0.286,0.044,,
bamba-9b,attr,0shot,all,0.301,1210,0.168,0.013,,
bamba-9b,attr,3shot_collide,all,0.245,1210,0.168,0.012,,
bamba-9b,attr,3shot,all,0.237,1210,0.168,0.012,,
bamba-9b,codevar,0shot,all,0.314,1200,0.267,0.013,,
bamba-9b,codevar,3shot_collide,all,0.331,1200,0.267,0.014,,
bamba-9b,codevar,3shot,all,0.327,1200,0.267,0.013,,
bamba-9b,dialogue,0shot,all,0.540,1200,0.312,0.014,,
bamba-9b,dialogue,3shot_collide,all,0.435,1200,0.312,0.014,,
bamba-9b,dialogue,3shot,all,0.462,1200,0.312,0.014,,
bamba-9b,dx-owner-3-deictic,0shot,all,0.392,120,0.333,0.045,,other_participant=0.425;speaker_confusion=0.183
bamba-9b,dx-owner-3-deictic,0shot,free,0.250,40,0.333,0.069,,other_participant=0.575;speaker_confusion=0.175
bamba-9b,dx-owner-3-deictic,3shot,all,0.417,120,0.333,0.045,,other_participant=0.433;speaker_confusion=0.142;tie=0.008
bamba-9b,dx-owner-3-deictic,3shot,free,0.225,40,0.333,0.066,,other_participant=0.6;speaker_confusion=0.175
bamba-9b,dx-owner-3-third,0shot,all,0.408,120,0.333,0.045,,
bamba-9b,dx-owner-3-third,0shot,free,0.250,40,0.333,0.069,,
bamba-9b,dx-owner-3-third,3shot,all,0.400,120,0.333,0.045,,
bamba-9b,dx-owner-3-third,3shot,free,0.225,40,0.333,0.066,,
bamba-9b,dx-self-3-deictic,0shot,all,0.283,180,0.287,0.034,,other_object=0.217;previous_owner=0.111;speaker_confusion=0.383;tie=0.006
bamba-9b,dx-self-3-deictic,0shot,free,0.206,126,0.286,0.036,,other_object=0.246;previous_owner=0.063;speaker_confusion=0.484
bamba-9b,dx-self-3-deictic,3shot,all,0.272,180,0.287,0.033,,other_object=0.2;previous_owner=0.117;speaker_confusion=0.411
bamba-9b,dx-self-3-deictic,3shot,free,0.214,126,0.286,0.037,,other_object=0.206;previous_owner=0.087;speaker_confusion=0.492
bamba-9b,dx-self-3-third,0shot,all,0.250,180,0.287,0.032,,
bamba-9b,dx-self-3-third,0shot,free,0.111,126,0.286,0.028,,
bamba-9b,dx-self-3-third,3shot,all,0.244,180,0.287,0.032,,
bamba-9b,dx-self-3-third,3shot,free,0.167,126,0.286,0.033,,
bamba-9b,dx-self-2-deictic,0shot,all,0.228,180,0.287,0.031,,speaker_confusion=0.767;tie=0.006
bamba-9b,dx-self-2-deictic,0shot,free,0.135,126,0.286,0.030,,speaker_confusion=0.857;tie=0.008
bamba-9b,dx-self-2-deictic,3shot,all,0.233,180,0.287,0.032,,speaker_confusion=0.767
bamba-9b,dx-self-2-deictic,3shot,free,0.206,126,0.286,0.036,,speaker_confusion=0.794
bamba-9b,dx-self-2-third,0shot,all,0.339,180,0.287,0.035,,
bamba-9b,dx-self-2-third,0shot,free,0.191,126,0.286,0.035,,
bamba-9b,dx-self-2-third,3shot,all,0.278,180,0.287,0.033,,
bamba-9b,dx-self-2-third,3shot,free,0.198,126,0.286,0.035,,
bamba-9b,dx-addressee-3-deictic,0shot,all,0.367,180,0.287,0.036,,other_object=0.117;previous_owner=0.189;speaker_confusion=0.328
bamba-9b,dx-addressee-3-deictic,0shot,free,0.230,126,0.286,0.037,,other_object=0.111;previous_owner=0.254;speaker_confusion=0.405
bamba-9b,dx-addressee-3-deictic,3shot,all,0.278,180,0.287,0.033,,other_object=0.172;previous_owner=0.228;speaker_confusion=0.322
bamba-9b,dx-addressee-3-deictic,3shot,free,0.143,126,0.286,0.031,,other_object=0.167;previous_owner=0.294;speaker_confusion=0.397
bamba-9b,dx-addressee-3-third,0shot,all,0.339,180,0.287,0.035,,
bamba-9b,dx-addressee-3-third,0shot,free,0.151,126,0.286,0.032,,
bamba-9b,dx-addressee-3-third,3shot,all,0.294,180,0.287,0.034,,
bamba-9b,dx-addressee-3-third,3shot,free,0.198,126,0.286,0.035,,
bamba-9b,dx-addressee-2-deictic,0shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.717
bamba-9b,dx-addressee-2-deictic,0shot,free,0.167,126,0.286,0.033,,speaker_confusion=0.833
bamba-9b,dx-addressee-2-deictic,3shot,all,0.267,180,0.287,0.033,,speaker_confusion=0.733
bamba-9b,dx-addressee-2-deictic,3shot,free,0.159,126,0.286,0.033,,speaker_confusion=0.841
bamba-9b,dx-addressee-2-third,0shot,all,0.372,180,0.287,0.036,,
bamba-9b,dx-addressee-2-third,0shot,free,0.230,126,0.286,0.037,,
bamba-9b,dx-addressee-2-third,3shot,all,0.261,180,0.287,0.033,,
bamba-9b,dx-addressee-2-third,3shot,free,0.143,126,0.286,0.031,,
recurrentgemma-9b,dx-owner-3-deictic,0shot,all,0.433,120,0.333,0.045,,other_participant=0.433;speaker_confusion=0.1;tie=0.033
recurrentgemma-9b,dx-owner-3-deictic,0shot,free,0.425,40,0.333,0.078,,other_participant=0.45;speaker_confusion=0.075;tie=0.05
recurrentgemma-9b,dx-owner-3-deictic,3shot,all,0.208,120,0.333,0.037,,other_participant=0.408;speaker_confusion=0.367;tie=0.017
recurrentgemma-9b,dx-owner-3-deictic,3shot,free,0.175,40,0.333,0.060,,other_participant=0.575;speaker_confusion=0.225;tie=0.025
recurrentgemma-9b,dx-owner-3-third,0shot,all,0.500,120,0.333,0.046,,
recurrentgemma-9b,dx-owner-3-third,0shot,free,0.450,40,0.333,0.079,,
recurrentgemma-9b,dx-owner-3-third,3shot,all,0.333,120,0.333,0.043,,
recurrentgemma-9b,dx-owner-3-third,3shot,free,0.325,40,0.333,0.074,,
recurrentgemma-9b,dx-self-3-deictic,0shot,all,0.250,180,0.287,0.032,,other_object=0.106;previous_owner=0.117;speaker_confusion=0.517;tie=0.011
recurrentgemma-9b,dx-self-3-deictic,0shot,free,0.095,126,0.286,0.026,,other_object=0.135;previous_owner=0.087;speaker_confusion=0.675;tie=0.008
recurrentgemma-9b,dx-self-3-deictic,3shot,all,0.272,180,0.287,0.033,,other_object=0.233;previous_owner=0.089;speaker_confusion=0.389;tie=0.017
recurrentgemma-9b,dx-self-3-deictic,3shot,free,0.238,126,0.286,0.038,,other_object=0.246;previous_owner=0.071;speaker_confusion=0.421;tie=0.024
recurrentgemma-9b,dx-self-3-third,0shot,all,0.244,180,0.287,0.032,,
recurrentgemma-9b,dx-self-3-third,0shot,free,0.095,126,0.286,0.026,,
recurrentgemma-9b,dx-self-3-third,3shot,all,0.306,180,0.287,0.034,,
recurrentgemma-9b,dx-self-3-third,3shot,free,0.222,126,0.286,0.037,,
recurrentgemma-9b,dx-self-2-deictic,0shot,all,0.283,180,0.287,0.034,,speaker_confusion=0.711;tie=0.006
recurrentgemma-9b,dx-self-2-deictic,0shot,free,0.151,126,0.286,0.032,,speaker_confusion=0.841;tie=0.008
recurrentgemma-9b,dx-self-2-deictic,3shot,all,0.261,180,0.287,0.033,,speaker_confusion=0.722;tie=0.017
recurrentgemma-9b,dx-self-2-deictic,3shot,free,0.214,126,0.286,0.037,,speaker_confusion=0.778;tie=0.008
recurrentgemma-9b,dx-self-2-third,0shot,all,0.261,180,0.287,0.033,,
recurrentgemma-9b,dx-self-2-third,0shot,free,0.095,126,0.286,0.026,,
recurrentgemma-9b,dx-self-2-third,3shot,all,0.300,180,0.287,0.034,,
recurrentgemma-9b,dx-self-2-third,3shot,free,0.182,126,0.286,0.034,,
recurrentgemma-9b,dx-addressee-3-deictic,0shot,all,0.306,180,0.287,0.034,,other_object=0.083;previous_owner=0.244;speaker_confusion=0.344;tie=0.022
recurrentgemma-9b,dx-addressee-3-deictic,0shot,free,0.111,126,0.286,0.028,,other_object=0.095;previous_owner=0.333;speaker_confusion=0.429;tie=0.032
recurrentgemma-9b,dx-addressee-3-deictic,3shot,all,0.283,180,0.287,0.034,,other_object=0.156;previous_owner=0.228;speaker_confusion=0.322;tie=0.011
recurrentgemma-9b,dx-addressee-3-deictic,3shot,free,0.159,126,0.286,0.033,,other_object=0.151;previous_owner=0.302;speaker_confusion=0.381;tie=0.008
recurrentgemma-9b,dx-addressee-3-third,0shot,all,0.317,180,0.287,0.035,,
recurrentgemma-9b,dx-addressee-3-third,0shot,free,0.143,126,0.286,0.031,,
recurrentgemma-9b,dx-addressee-3-third,3shot,all,0.300,180,0.287,0.034,,
recurrentgemma-9b,dx-addressee-3-third,3shot,free,0.206,126,0.286,0.036,,
recurrentgemma-9b,dx-addressee-2-deictic,0shot,all,0.228,180,0.287,0.031,,speaker_confusion=0.772
recurrentgemma-9b,dx-addressee-2-deictic,0shot,free,0.048,126,0.286,0.019,,speaker_confusion=0.952
recurrentgemma-9b,dx-addressee-2-deictic,3shot,all,0.233,180,0.287,0.032,,speaker_confusion=0.767
recurrentgemma-9b,dx-addressee-2-deictic,3shot,free,0.127,126,0.286,0.030,,speaker_confusion=0.873
recurrentgemma-9b,dx-addressee-2-third,0shot,all,0.278,180,0.287,0.033,,
recurrentgemma-9b,dx-addressee-2-third,0shot,free,0.111,126,0.286,0.028,,
recurrentgemma-9b,dx-addressee-2-third,3shot,all,0.328,180,0.287,0.035,,
recurrentgemma-9b,dx-addressee-2-third,3shot,free,0.222,126,0.286,0.037,,
falcon-h1-34b-base,dx-owner-3-deictic,0shot,all,0.808,120,0.333,0.036,,other_participant=0.175;speaker_confusion=0.017
falcon-h1-34b-base,dx-owner-3-deictic,0shot,free,0.750,40,0.333,0.069,,other_participant=0.25
falcon-h1-34b-base,dx-owner-3-deictic,3shot_collide,all,0.983,120,0.333,0.012,,other_participant=0.017
falcon-h1-34b-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
falcon-h1-34b-base,dx-owner-3-deictic,3shot,all,0.983,120,0.333,0.012,,other_participant=0.017
falcon-h1-34b-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
falcon-h1-34b-base,dx-owner-3-third,0shot,all,0.800,120,0.333,0.036,,
falcon-h1-34b-base,dx-owner-3-third,0shot,free,0.650,40,0.333,0.075,,
falcon-h1-34b-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
falcon-h1-34b-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
falcon-h1-34b-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
falcon-h1-34b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
falcon-h1-34b-base,dx-self-3-deictic,0shot,all,0.544,180,0.287,0.037,,other_object=0.161;previous_owner=0.033;speaker_confusion=0.261
falcon-h1-34b-base,dx-self-3-deictic,0shot,free,0.611,126,0.286,0.043,,other_object=0.135;previous_owner=0.016;speaker_confusion=0.238
falcon-h1-34b-base,dx-self-3-deictic,3shot_collide,all,0.728,180,0.287,0.033,,other_object=0.067;previous_owner=0.039;speaker_confusion=0.156;tie=0.011
falcon-h1-34b-base,dx-self-3-deictic,3shot_collide,free,0.818,126,0.286,0.034,,other_object=0.079;previous_owner=0.016;speaker_confusion=0.079;tie=0.008
falcon-h1-34b-base,dx-self-3-deictic,3shot,all,0.839,180,0.287,0.027,,other_object=0.05;previous_owner=0.05;speaker_confusion=0.061
falcon-h1-34b-base,dx-self-3-deictic,3shot,free,0.913,126,0.286,0.025,,other_object=0.04;previous_owner=0.024;speaker_confusion=0.024
falcon-h1-34b-base,dx-self-3-third,0shot,all,0.839,180,0.287,0.027,,
falcon-h1-34b-base,dx-self-3-third,0shot,free,0.936,126,0.286,0.022,,
falcon-h1-34b-base,dx-self-3-third,3shot_collide,all,0.883,180,0.287,0.024,,
falcon-h1-34b-base,dx-self-3-third,3shot_collide,free,0.992,126,0.286,0.008,,
falcon-h1-34b-base,dx-self-3-third,3shot,all,0.861,180,0.287,0.026,,
falcon-h1-34b-base,dx-self-3-third,3shot,free,0.952,126,0.286,0.019,,
falcon-h1-34b-base,dx-self-2-deictic,0shot,all,0.461,180,0.287,0.037,,speaker_confusion=0.539
falcon-h1-34b-base,dx-self-2-deictic,0shot,free,0.524,126,0.286,0.044,,speaker_confusion=0.476
falcon-h1-34b-base,dx-self-2-deictic,3shot_collide,all,0.578,180,0.287,0.037,,speaker_confusion=0.422
falcon-h1-34b-base,dx-self-2-deictic,3shot_collide,free,0.643,126,0.286,0.043,,speaker_confusion=0.357
falcon-h1-34b-base,dx-self-2-deictic,3shot,all,0.639,180,0.287,0.036,,speaker_confusion=0.361
falcon-h1-34b-base,dx-self-2-deictic,3shot,free,0.714,126,0.286,0.040,,speaker_confusion=0.286
falcon-h1-34b-base,dx-self-2-third,0shot,all,0.722,180,0.287,0.033,,
falcon-h1-34b-base,dx-self-2-third,0shot,free,0.833,126,0.286,0.033,,
falcon-h1-34b-base,dx-self-2-third,3shot_collide,all,0.844,180,0.287,0.027,,
falcon-h1-34b-base,dx-self-2-third,3shot_collide,free,0.913,126,0.286,0.025,,
falcon-h1-34b-base,dx-self-2-third,3shot,all,0.789,180,0.287,0.030,,
falcon-h1-34b-base,dx-self-2-third,3shot,free,0.889,126,0.286,0.028,,
falcon-h1-34b-base,dx-addressee-3-deictic,0shot,all,0.494,180,0.287,0.037,,other_object=0.144;previous_owner=0.05;speaker_confusion=0.306;tie=0.006
falcon-h1-34b-base,dx-addressee-3-deictic,0shot,free,0.548,126,0.286,0.044,,other_object=0.135;previous_owner=0.056;speaker_confusion=0.254;tie=0.008
falcon-h1-34b-base,dx-addressee-3-deictic,3shot_collide,all,0.672,180,0.287,0.035,,other_object=0.067;previous_owner=0.033;speaker_confusion=0.217;tie=0.011
falcon-h1-34b-base,dx-addressee-3-deictic,3shot_collide,free,0.730,126,0.286,0.040,,other_object=0.04;previous_owner=0.048;speaker_confusion=0.183
falcon-h1-34b-base,dx-addressee-3-deictic,3shot,all,0.733,180,0.287,0.033,,other_object=0.056;previous_owner=0.033;speaker_confusion=0.178
falcon-h1-34b-base,dx-addressee-3-deictic,3shot,free,0.770,126,0.286,0.037,,other_object=0.032;previous_owner=0.048;speaker_confusion=0.151
falcon-h1-34b-base,dx-addressee-3-third,0shot,all,0.872,180,0.287,0.025,,
falcon-h1-34b-base,dx-addressee-3-third,0shot,free,0.913,126,0.286,0.025,,
falcon-h1-34b-base,dx-addressee-3-third,3shot_collide,all,0.911,180,0.287,0.021,,
falcon-h1-34b-base,dx-addressee-3-third,3shot_collide,free,0.952,126,0.286,0.019,,
falcon-h1-34b-base,dx-addressee-3-third,3shot,all,0.900,180,0.287,0.022,,
falcon-h1-34b-base,dx-addressee-3-third,3shot,free,0.960,126,0.286,0.017,,
falcon-h1-34b-base,dx-addressee-2-deictic,0shot,all,0.344,180,0.287,0.035,,speaker_confusion=0.656
falcon-h1-34b-base,dx-addressee-2-deictic,0shot,free,0.365,126,0.286,0.043,,speaker_confusion=0.635
falcon-h1-34b-base,dx-addressee-2-deictic,3shot_collide,all,0.433,180,0.287,0.037,,speaker_confusion=0.556;tie=0.011
falcon-h1-34b-base,dx-addressee-2-deictic,3shot_collide,free,0.452,126,0.286,0.044,,speaker_confusion=0.532;tie=0.016
falcon-h1-34b-base,dx-addressee-2-deictic,3shot,all,0.433,180,0.287,0.037,,speaker_confusion=0.567
falcon-h1-34b-base,dx-addressee-2-deictic,3shot,free,0.460,126,0.286,0.044,,speaker_confusion=0.54
falcon-h1-34b-base,dx-addressee-2-third,0shot,all,0.806,180,0.287,0.029,,
falcon-h1-34b-base,dx-addressee-2-third,0shot,free,0.913,126,0.286,0.025,,
falcon-h1-34b-base,dx-addressee-2-third,3shot_collide,all,0.817,180,0.287,0.029,,
falcon-h1-34b-base,dx-addressee-2-third,3shot_collide,free,0.913,126,0.286,0.025,,
falcon-h1-34b-base,dx-addressee-2-third,3shot,all,0.794,180,0.287,0.030,,
falcon-h1-34b-base,dx-addressee-2-third,3shot,free,0.913,126,0.286,0.025,,
falcon-h1-34b-base,sb-k1,0shot,all,0.984,64,0.125,0.015,,
falcon-h1-34b-base,sb-k1,3shot_collide,all,1.000,64,0.125,0.000,,
falcon-h1-34b-base,sb-k1,3shot,all,1.000,64,0.125,0.000,,
falcon-h1-34b-base,sb-k23,0shot,all,0.836,128,0.027,0.033,,
falcon-h1-34b-base,sb-k23,3shot_collide,all,0.992,128,0.027,0.008,,
falcon-h1-34b-base,sb-k23,3shot,all,0.992,128,0.027,0.008,,
falcon-h1-34b-base,sb-split,0shot,all,0.812,128,0.027,0.035,,
falcon-h1-34b-base,sb-split,3shot_collide,all,0.930,128,0.027,0.023,,
falcon-h1-34b-base,sb-split,3shot,all,0.945,128,0.027,0.020,,
falcon-h1-34b-base,sb-dyn,0shot,all,0.695,256,0.027,0.029,,
falcon-h1-34b-base,sb-dyn,3shot_collide,all,0.652,256,0.027,0.030,,
falcon-h1-34b-base,sb-dyn,3shot,all,0.656,256,0.027,0.030,,
falcon-h1-34b-base,an-narr,0shot,all,0.714,640,0.250,0.018,,
falcon-h1-34b-base,an-narr,3shot_collide,all,0.986,640,0.250,0.005,,
falcon-h1-34b-base,an-narr,3shot,all,0.992,640,0.250,0.004,,
falcon-h1-34b-base,an-other-any,0shot,all,0.356,320,0.250,0.027,,
falcon-h1-34b-base,an-other-any,3shot_collide,all,0.866,320,0.250,0.019,,
falcon-h1-34b-base,an-other-any,3shot,all,0.863,320,0.250,0.019,,
falcon-h1-34b-base,an-me-any,0shot,all,0.800,160,0.250,0.032,,
falcon-h1-34b-base,an-me-any,3shot_collide,all,0.787,160,0.250,0.032,,
falcon-h1-34b-base,an-me-any,3shot,all,0.831,160,0.250,0.030,,
falcon-h1-34b-base,an-you-voc,0shot,all,0.650,80,0.250,0.053,,
falcon-h1-34b-base,an-you-voc,3shot_collide,all,0.925,80,0.250,0.029,,
falcon-h1-34b-base,an-you-voc,3shot,all,0.887,80,0.250,0.035,,
falcon-h1-34b-base,an-you-novoc,0shot,all,0.188,80,0.250,0.044,,
falcon-h1-34b-base,an-you-novoc,3shot_collide,all,0.537,80,0.250,0.056,,
falcon-h1-34b-base,an-you-novoc,3shot,all,0.588,80,0.250,0.055,,
falcon-h1-34b-base,adj-adj,0shot,all,0.992,260,0.333,0.005,,
falcon-h1-34b-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-voc,0shot,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
falcon-h1-34b-base,cm-mutate,0shot,all,0.835,200,0.200,0.026,,
falcon-h1-34b-base,cm-mutate,3shot_collide,all,1.000,200,0.200,0.000,,
falcon-h1-34b-base,cm-mutate,3shot,all,0.990,200,0.200,0.007,,
falcon-h1-34b-base,cm-rebind,0shot,all,0.775,200,0.200,0.029,,
falcon-h1-34b-base,cm-rebind,3shot_collide,all,1.000,200,0.200,0.000,,
falcon-h1-34b-base,cm-rebind,3shot,all,1.000,200,0.200,0.000,,
falcon-h1-34b-base,cm-copy,0shot,all,0.780,200,0.200,0.029,,
falcon-h1-34b-base,cm-copy,3shot_collide,all,1.000,200,0.200,0.000,,
falcon-h1-34b-base,cm-copy,3shot,all,0.995,200,0.200,0.005,,
nemotron3-ultra-550b-base,attr,0shot,all,0.883,1067,0.170,0.010,,
nemotron3-ultra-550b-base,attr,3shot_collide,all,1.000,1067,0.170,0.000,,
nemotron3-ultra-550b-base,attr,3shot,all,1.000,1067,0.170,0.000,,
nemotron3-ultra-550b-base,codevar,0shot,all,0.841,1107,0.268,0.011,,
nemotron3-ultra-550b-base,codevar,3shot_collide,all,0.924,1107,0.268,0.008,,
nemotron3-ultra-550b-base,codevar,3shot,all,0.905,1107,0.268,0.009,,
nemotron3-ultra-550b-base,dialogue,0shot,all,0.964,1173,0.312,0.005,,
nemotron3-ultra-550b-base,dialogue,3shot_collide,all,0.994,1173,0.312,0.002,,
nemotron3-ultra-550b-base,dialogue,3shot,all,0.994,1173,0.312,0.002,,
nemotron3-ultra-550b-base,dx-owner-3-deictic,0shot,all,0.683,120,0.333,0.043,,other_participant=0.292;speaker_confusion=0.025
nemotron3-ultra-550b-base,dx-owner-3-deictic,0shot,free,0.575,40,0.333,0.078,,other_participant=0.375;speaker_confusion=0.05
nemotron3-ultra-550b-base,dx-owner-3-deictic,3shot_collide,all,1.000,120,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-deictic,3shot_collide,free,1.000,40,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-deictic,3shot,all,0.992,120,0.333,0.008,,other_participant=0.008
nemotron3-ultra-550b-base,dx-owner-3-deictic,3shot,free,1.000,40,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-third,0shot,all,0.900,120,0.333,0.027,,
nemotron3-ultra-550b-base,dx-owner-3-third,0shot,free,0.875,40,0.333,0.052,,
nemotron3-ultra-550b-base,dx-owner-3-third,3shot_collide,all,1.000,120,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-third,3shot_collide,free,1.000,40,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-third,3shot,all,1.000,120,0.333,0.000,,
nemotron3-ultra-550b-base,dx-owner-3-third,3shot,free,1.000,40,0.333,0.000,,
nemotron3-ultra-550b-base,dx-self-3-deictic,0shot,all,0.580,169,0.288,0.038,,other_object=0.083;previous_owner=0.059;speaker_confusion=0.231;tie=0.047
nemotron3-ultra-550b-base,dx-self-3-deictic,0shot,free,0.613,119,0.286,0.045,,other_object=0.084;previous_owner=0.017;speaker_confusion=0.244;tie=0.042
nemotron3-ultra-550b-base,dx-self-3-deictic,3shot_collide,all,0.888,169,0.288,0.024,,other_object=0.047;previous_owner=0.03;speaker_confusion=0.024;tie=0.012
nemotron3-ultra-550b-base,dx-self-3-deictic,3shot_collide,free,0.908,119,0.286,0.027,,other_object=0.067;speaker_confusion=0.025
nemotron3-ultra-550b-base,dx-self-3-deictic,3shot,all,0.888,169,0.288,0.024,,other_object=0.047;previous_owner=0.036;speaker_confusion=0.024;tie=0.006
nemotron3-ultra-550b-base,dx-self-3-deictic,3shot,free,0.908,119,0.286,0.027,,other_object=0.067;speaker_confusion=0.017;tie=0.008
nemotron3-ultra-550b-base,dx-self-3-third,0shot,all,0.876,169,0.288,0.025,,
nemotron3-ultra-550b-base,dx-self-3-third,0shot,free,0.924,119,0.286,0.024,,
nemotron3-ultra-550b-base,dx-self-3-third,3shot_collide,all,0.982,169,0.288,0.010,,
nemotron3-ultra-550b-base,dx-self-3-third,3shot_collide,free,0.983,119,0.286,0.012,,
nemotron3-ultra-550b-base,dx-self-3-third,3shot,all,0.982,169,0.288,0.010,,
nemotron3-ultra-550b-base,dx-self-3-third,3shot,free,0.983,119,0.286,0.012,,
nemotron3-ultra-550b-base,dx-self-2-deictic,0shot,all,0.388,170,0.288,0.037,,speaker_confusion=0.571;tie=0.041
nemotron3-ultra-550b-base,dx-self-2-deictic,0shot,free,0.442,120,0.286,0.045,,speaker_confusion=0.5;tie=0.058
nemotron3-ultra-550b-base,dx-self-2-deictic,3shot_collide,all,0.771,170,0.288,0.032,,speaker_confusion=0.224;tie=0.006
nemotron3-ultra-550b-base,dx-self-2-deictic,3shot_collide,free,0.808,120,0.286,0.036,,speaker_confusion=0.183;tie=0.008
nemotron3-ultra-550b-base,dx-self-2-deictic,3shot,all,0.782,170,0.288,0.032,,speaker_confusion=0.206;tie=0.012
nemotron3-ultra-550b-base,dx-self-2-deictic,3shot,free,0.842,120,0.286,0.033,,speaker_confusion=0.15;tie=0.008
nemotron3-ultra-550b-base,dx-self-2-third,0shot,all,0.871,170,0.288,0.026,,
nemotron3-ultra-550b-base,dx-self-2-third,0shot,free,0.850,120,0.286,0.033,,
nemotron3-ultra-550b-base,dx-self-2-third,3shot_collide,all,0.994,170,0.288,0.006,,
nemotron3-ultra-550b-base,dx-self-2-third,3shot_collide,free,1.000,120,0.286,0.000,,
nemotron3-ultra-550b-base,dx-self-2-third,3shot,all,0.988,170,0.288,0.008,,
nemotron3-ultra-550b-base,dx-self-2-third,3shot,free,0.992,120,0.286,0.008,,
nemotron3-ultra-550b-base,dx-addressee-3-deictic,0shot,all,0.404,171,0.288,0.037,,other_object=0.14;previous_owner=0.058;speaker_confusion=0.368;tie=0.029
nemotron3-ultra-550b-base,dx-addressee-3-deictic,0shot,free,0.432,118,0.286,0.046,,other_object=0.144;previous_owner=0.068;speaker_confusion=0.339;tie=0.017
nemotron3-ultra-550b-base,dx-addressee-3-deictic,3shot_collide,all,0.743,171,0.288,0.033,,other_object=0.058;previous_owner=0.023;speaker_confusion=0.17;tie=0.006
nemotron3-ultra-550b-base,dx-addressee-3-deictic,3shot_collide,free,0.729,118,0.286,0.041,,other_object=0.076;previous_owner=0.034;speaker_confusion=0.153;tie=0.008
nemotron3-ultra-550b-base,dx-addressee-3-deictic,3shot,all,0.737,171,0.288,0.034,,other_object=0.053;previous_owner=0.035;speaker_confusion=0.152;tie=0.023
nemotron3-ultra-550b-base,dx-addressee-3-deictic,3shot,free,0.712,118,0.286,0.042,,other_object=0.059;previous_owner=0.042;speaker_confusion=0.169;tie=0.017
nemotron3-ultra-550b-base,dx-addressee-3-third,0shot,all,0.895,171,0.288,0.024,,
nemotron3-ultra-550b-base,dx-addressee-3-third,0shot,free,0.907,118,0.286,0.027,,
nemotron3-ultra-550b-base,dx-addressee-3-third,3shot_collide,all,1.000,171,0.288,0.000,,
nemotron3-ultra-550b-base,dx-addressee-3-third,3shot_collide,free,1.000,118,0.286,0.000,,
nemotron3-ultra-550b-base,dx-addressee-3-third,3shot,all,1.000,171,0.288,0.000,,
nemotron3-ultra-550b-base,dx-addressee-3-third,3shot,free,1.000,118,0.286,0.000,,
nemotron3-ultra-550b-base,dx-addressee-2-deictic,0shot,all,0.244,176,0.288,0.032,,speaker_confusion=0.722;tie=0.034
nemotron3-ultra-550b-base,dx-addressee-2-deictic,0shot,free,0.192,125,0.285,0.035,,speaker_confusion=0.76;tie=0.048
nemotron3-ultra-550b-base,dx-addressee-2-deictic,3shot_collide,all,0.392,176,0.288,0.037,,speaker_confusion=0.557;tie=0.051
nemotron3-ultra-550b-base,dx-addressee-2-deictic,3shot_collide,free,0.416,125,0.285,0.044,,speaker_confusion=0.536;tie=0.048
nemotron3-ultra-550b-base,dx-addressee-2-deictic,3shot,all,0.398,176,0.288,0.037,,speaker_confusion=0.574;tie=0.028
nemotron3-ultra-550b-base,dx-addressee-2-deictic,3shot,free,0.440,125,0.285,0.044,,speaker_confusion=0.536;tie=0.024
nemotron3-ultra-550b-base,dx-addressee-2-third,0shot,all,0.920,176,0.288,0.020,,
nemotron3-ultra-550b-base,dx-addressee-2-third,0shot,free,0.928,125,0.285,0.023,,
nemotron3-ultra-550b-base,dx-addressee-2-third,3shot_collide,all,0.994,176,0.288,0.006,,
nemotron3-ultra-550b-base,dx-addressee-2-third,3shot_collide,free,1.000,125,0.285,0.000,,
nemotron3-ultra-550b-base,dx-addressee-2-third,3shot,all,0.989,176,0.288,0.008,,
nemotron3-ultra-550b-base,dx-addressee-2-third,3shot,free,1.000,125,0.285,0.000,,
nemotron3-ultra-550b-base,sb-k1,0shot,all,1.000,53,0.125,0.000,,
nemotron3-ultra-550b-base,sb-k1,3shot_collide,all,1.000,53,0.125,0.000,,
nemotron3-ultra-550b-base,sb-k1,3shot,all,1.000,53,0.125,0.000,,
nemotron3-ultra-550b-base,sb-k23,0shot,all,0.953,106,0.026,0.021,,
nemotron3-ultra-550b-base,sb-k23,3shot_collide,all,1.000,106,0.026,0.000,,
nemotron3-ultra-550b-base,sb-k23,3shot,all,1.000,106,0.026,0.000,,
nemotron3-ultra-550b-base,sb-split,0shot,all,0.972,106,0.027,0.016,,
nemotron3-ultra-550b-base,sb-split,3shot_collide,all,1.000,106,0.027,0.000,,
nemotron3-ultra-550b-base,sb-split,3shot,all,1.000,106,0.027,0.000,,
nemotron3-ultra-550b-base,sb-dyn,0shot,all,0.821,223,0.027,0.026,,
nemotron3-ultra-550b-base,sb-dyn,3shot_collide,all,0.955,223,0.027,0.014,,
nemotron3-ultra-550b-base,sb-dyn,3shot,all,0.919,223,0.027,0.018,,
nemotron3-ultra-550b-base,an-narr,0shot,all,0.559,640,0.250,0.020,,
nemotron3-ultra-550b-base,an-narr,3shot_collide,all,0.997,640,0.250,0.002,,
nemotron3-ultra-550b-base,an-narr,3shot,all,0.998,640,0.250,0.002,,
nemotron3-ultra-550b-base,an-other-any,0shot,all,0.456,320,0.250,0.028,,
nemotron3-ultra-550b-base,an-other-any,3shot_collide,all,0.956,320,0.250,0.011,,
nemotron3-ultra-550b-base,an-other-any,3shot,all,0.944,320,0.250,0.013,,
nemotron3-ultra-550b-base,an-me-any,0shot,all,0.838,160,0.250,0.029,,
nemotron3-ultra-550b-base,an-me-any,3shot_collide,all,0.988,160,0.250,0.009,,
nemotron3-ultra-550b-base,an-me-any,3shot,all,0.969,160,0.250,0.014,,
nemotron3-ultra-550b-base,an-you-voc,0shot,all,0.900,80,0.250,0.034,,
nemotron3-ultra-550b-base,an-you-voc,3shot_collide,all,0.988,80,0.250,0.012,,
nemotron3-ultra-550b-base,an-you-voc,3shot,all,1.000,80,0.250,0.000,,
nemotron3-ultra-550b-base,an-you-novoc,0shot,all,0.400,80,0.250,0.055,,
nemotron3-ultra-550b-base,an-you-novoc,3shot_collide,all,0.912,80,0.250,0.032,,
nemotron3-ultra-550b-base,an-you-novoc,3shot,all,0.925,80,0.250,0.029,,
nemotron3-ultra-550b-base,adj-adj,0shot,all,0.992,260,0.333,0.005,,
nemotron3-ultra-550b-base,adj-adj,3shot_collide,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-adj,3shot,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-voc,0shot,all,0.992,260,0.333,0.005,,
nemotron3-ultra-550b-base,adj-voc,3shot_collide,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-voc,3shot,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-third,0shot,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-third,3shot_collide,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,adj-third,3shot,all,1.000,260,0.333,0.000,,
nemotron3-ultra-550b-base,cm-mutate,0shot,all,1.000,186,0.200,0.000,,
nemotron3-ultra-550b-base,cm-mutate,3shot_collide,all,1.000,186,0.200,0.000,,
nemotron3-ultra-550b-base,cm-mutate,3shot,all,1.000,186,0.200,0.000,,
nemotron3-ultra-550b-base,cm-rebind,0shot,all,0.837,178,0.200,0.028,,
nemotron3-ultra-550b-base,cm-rebind,3shot_collide,all,1.000,178,0.200,0.000,,
nemotron3-ultra-550b-base,cm-rebind,3shot,all,1.000,178,0.200,0.000,,
nemotron3-ultra-550b-base,cm-copy,0shot,all,0.833,186,0.200,0.027,,
nemotron3-ultra-550b-base,cm-copy,3shot_collide,all,0.962,186,0.200,0.014,,
nemotron3-ultra-550b-base,cm-copy,3shot,all,1.000,186,0.200,0.000,,
```
