# Binding Atlas — agent view

Plain-text twin of the [Binding Atlas page](https://kotodama.animalabs.ai/binding-atlas). Same data, same build. Built 2026-10-30 22:24 UTC: 54 rows (models + one human), 24 columns, 2,029 cells, 1,303,620 scored items.

## Summary

This atlas scores 53 models and checkpoints, plus one human reader, on 24 tests of tracking who *I* and *you* refer to. "Example items per column" quotes the exact prompts that the models were given.

Base language models acquire the referent of *you* in five ordered stages. Every base of 0.6B parameters or more trained on trillions of tokens reaches the first three. The fifth stage, tracking *you* after objects change hands, is **never reached zero-shot** (with no worked examples) by a public model up to 8B, at any token count measured up to 36T; its swap shift is negative in every one of them. From 14B to 1T parameters, stage 5 is weak and nearly flat. In these models, parameter count and depth grow together. **kotodama-tsumugi**, a 1.4B model run as 72 effective layers (looped depth, boundary-fitted attention residuals), separates them: it reaches stage 3 by **33B tokens** (annealed; on the constant-LR trunk by 40B), the token cost of OLMo-2 13B, against about 420B for OLMo-2 1B. At 110B tokens its zero-shot stage-5 reading (−.021) is not inverted under the gauge rule, where every 1–2B public base whose static *you* binds, except Llama-3.2-1B, inverts. In a check of 20 bases, kotodama-tsumugi (.336) follows a hand-off told in the third person better than every Pythia model up to 6.9B (.275 to .294) and Llama-3.2-1B (.303). Most bases up to 1.5B barely follow one: 7 of those 10 score .26 to .34, against about .29 by chance. For every model of this size, that hand-off is the open step toward stage 5.

Swap shift = mean[p(gold | original) − p(gold | speaker swapped)] over the candidates: above +.02 a pronoun binds (its answer moves with the speaker), below −.02 it inverts (it resolves to the speaker).

## Part I · Where base models fail

### The staircase

| stage | name | what it means | example | where public models stand | OLMo-2 1B | kotodama-tsumugi 1.4B |
|---|---|---|---|---|---|---|
| 1 | I binds | "I" resolves to whoever is speaking. | `Jack: I'm holding the ___` | every base ≥ 0.5B | ~21–63B tokens | ~3–3.75B tokens |
| 2 | you = me | "you" is read as the speaker too: the pronoun-reversal phase young children go through. | `Jack: you're holding the ___  → Jack's object` | transient in every trajectory | ~63B → ~500B tokens | transient, ~4–20B |
| 3 | static you binds | "you" resolves to the addressee, when ownership is simply stated. | `Carol has the onion. … Jack: you're holding the ___` | every base ≥ 0.6B trained on trillions of tokens | ~420B tokens | by 33B tokens annealed (trunk: 40B); +.235 at 110B |
| 4 | you = me, again | once objects change hands in the scene, "you" collapses back onto the speaker. | `Erin gives Paul her cup. … Paul: now you're holding the ___` | where almost every base stops | from ~63B, still there at 4T | from ~4B; no longer inverted at 110B under the gauge rule (−.047 at 11B → −.021) |
| 5 | dynamic you binds | "you" tracked through transfers. In-context first (3-shot), zero-shot last. | `same scene, answered correctly without examples` | zero-shot: reached by no public base ≤ 8B at any token count measured; weak and nearly flat from 14B to 1T; open whether more tokens reach it | never (flat to 4T) | not reached by 110B |

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

No model at or below 8B binds zero-shot dynamic *you* at any token count measured (to 36T). The same holds with each chat rewritten as a scene (8 of 8 models tested at or below 8B; the highest is +.010, Qwen2.5-7B), so the result does not come from the chat format. In these models depth grows with size, so the table leaves two routes open for small models: more tokens, and more depth at a fixed size. No model at or below 8B has been trained far enough to rule out the first. From 14B to 1T parameters, stage 5 is weak and nearly flat. Ten of the 11 bases with a public size read +.01 to +.07 zero-shot. Static *you* reads +.27 to +.82 in the seven of them where it was measured.

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

Three layers: (1) possession transfer itself: every 1.4–1.5B model is at or near chance even narrated (kotodama-tsumugi through 110B, Pythia-1.4B, OLMo-2 1B at 4T); SmolLM2 1.7B at 11T and Qwen3 1.7B at 36T lift off chance; from 4B it is easy. (2) The transfer as a speech act is the mid-size wall: 4B–70B models that pass the narrated version stay at .29–.41 spoken; narrating only the hand-off recovers part of the gap. (3) Naming the addressee does not help: the vocative variant is at or below the spoken one for every public model.

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

By the release, attribution reads .986, level with the 7–8B bases under the same balanced examples (.981 to 1.00). Scored with one harness (the kotodama binding suite). Each item gets four worked examples, balanced so that none of them shows the item's own answer pattern; deixis-lite has no such condition and is read zero-shot. The bookkeeping probes rise with tokens: attribution from .71 to .99, set binding and dialogue tracking by .13 to .17. Aliased Python lists stays at chance through 33B and lifts only after 55B. The perspective probes stay within .15 of their 5B level. The two seeds differ at 5B (seed 7 forms set binding earlier) and agree within .05 on seven of the ten rows at 11B.

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

Zero-shot, the annealed kotodama-tsumugi checkpoints from 33B on (.455–.490) sit above every public checkpoint measured on this harness (best: SmolLM2 1.7B at 8.4T, .420). The zero-shot column pools narrative and screenplay prose, and the public pool is three 1–2B models. fixed3 = the same three exemplars for every item; fairK = K exemplars drawn per item. Fixed exemplars push accuracy well below zero-shot (a setup artifact). The penalty is not specific to these models. On a lighter riders slice, fixed exemplars lower prose accuracy for most of an eight-model panel (chat 8 of 8, screenplay 6 of 8, narrative 5 of 8) and raise code accuracy for all eight. Fair exemplars help every public model here and kotodama-tsumugi through 55B. The two 110B anneals score highest zero-shot (fair6: second anneal −.03, release −.045).

## How to read the full atlas

- **A cell** = one model (row) on one test (column) under one condition. `score` averages, over items, (credit − chance)/(1 − chance), with `chance` = 1/#candidates. `credit` is 1 when the gold is the model's top-ranked candidate, and an exact tie between candidates splits the credit evenly. 0 = guessing, 1 = always right, below 0 = reliably wrong. `lo`, `hi` = the bootstrap 95% interval (1,000 resamples of items).
- **State:** `above` when the interval is above 0, and `at` when it includes 0. `below` means the interval is below 0: the model picks a wrong answer more often than guessing would.
- **Conditions:** `fair` = four solved items first, balanced so that no answer can be copied from them (the reading); `0shot` = the item alone (the floor). Columns without worked examples (lookups in lists, box refills, the static gauges) have a `0shot` reading only. The rule-learning columns are read across their difficulty ladder (`ladder`: the mean chance-normalized accuracy over the ladder's rungs, clipped at 0 and 1).
- **Groups:** columns are grouped by what the questions look like. Each column is scored and read on its own.
- **Models:** every base model is scored by log-probability over the candidate set, with one scorer and one set of batteries. The **Human** row is one person reading items cold, with typed answers.
- **Parsing:** sections are `##` headings outside code fences (prompts inside code fences can contain lines that start with `#`). Two CSV blocks at the end: `rows` (id → label, group, size, tokens) and `cells` (every number).

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

## Scoring notes

- **3-shot (same type)** shows how much a model takes from examples of the test item's own type. On the aliased-list columns, those examples show the answer's role, so these cells read how well a model copies that role. In a check on eight models, the examples' answers were flipped to the wrong role. Every model that had gained 20 points or more from the examples then fell to near zero (Qwen2.5-7B: 98% → 2%). Balanced examples carry almost none of the role. The same check finds that code variables are partly shortcut (balanced examples keep 39% of the 3-shot gain) and static lookup mostly format (85%). Table 3, the battery across kotodama-tsumugi's annealed reads, uses four balanced examples.
- **Static lookup at 0-shot** is bare prose with no frame, so its cells mostly show whether a model infers the task's format. All eight models in a check panel gain 19 to 46 points from three worked examples of the item's type. Balanced examples keep most of that gain (28 of 33 points on average). For aliased lists, 0-shot sits close to the balanced reading (7.5 points below it on average) and is a fair floor.
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

| id | group | label | what it asks | readings | headline |
|---|---|---|---|---|---|
| `lookup` | Who has what? Nothing moves | Simple lookup | Several people each keep an object; name one person's object. | fair, 0shot | yes |
| `shortlist` | Who has what? Nothing moves | Short list lookup | A list pairs one to four names with tasks; name one person's task. | 0shot |  |
| `pairs` | Who has what? Nothing moves | Lookup among pairs | Some people hold two objects and some one; name what a one-object person holds. | fair, 0shot |  |
| `noise` | Who has what? Nothing moves | Lookup past noise | Boxes are filled, unrelated events follow; name what one box holds. | 0shot |  |
| `now` | Who has what? Nothing moves | Who holds it now | A scene says who holds what; nothing moves. Then: 'Now X is holding the'. (The query opens with 'Now', which implies a change: the largest models answer as if something moved.) | 0shot |  |
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

**0-shot example** (`attr_2-first_000`) — candidates: knife, ribbon, rope, soap · gold: **knife**

```text
Amy has the knife. A ribbon lies on the shelf. A rope lies on the shelf. Felix has the soap. So Amy has the
```

### `shortlist` — Who has what? Nothing moves · Short list lookup

A list pairs one to four names with tasks; name one person's task.

**0-shot example** (`cap1|K1|D0|s0|t0`) — candidates: laundry, dishes, garden, cooking, shopping, painting, windows, trash, mail, dog, car, plants, bills, lunch, dinner, breakfast, garage, kitchen, roof, fence, pool, lawn, taxes, report, budget · gold: **shopping**

```text
Tyler: shopping
The task for Tyler is:
```

### `pairs` — Who has what? Nothing moves · Lookup among pairs

Some people hold two objects and some one; name what a one-object person holds.

**0-shot example** (`setbind-v3_1-joint-0_000`) — candidates: funnel, magnet, pencil, towel, photo, chain, marble, bolt · gold: **bolt**

```text
Nina has the funnel and the magnet. Matt has the pencil. Linda has the towel and the photo. A chain lies on the shelf. A marble lies on the shelf. Greg has the bolt. So Greg has the
```

### `noise` — Who has what? Nothing moves · Lookup past noise

Boxes are filled, unrelated events follow; name what one box holds.

**0-shot example** (`formupdate-v3|K2|U1|prose|b00|A`) — candidates: hammer, sock, marble, ball, button · gold: **hammer**

```text
The drawer holds the hammer. The tin holds the ball. Then the kettle is switched on. Then the lights are switched off. In the end, the drawer holds the
```

### `now` — Who has what? Nothing moves · Who holds it now

A scene says who holds what; nothing moves. Then: 'Now X is holding the'.

**0-shot example** (`jl0000-third-orig`) — candidates: photo, onion, marble · gold: **onion**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and a marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Now Carol is holding the
```

### `handoffs` — Who has what after things move: told as a story · Hand-offs in a story

Objects pass between people in a short story; name who has an object at the end.

**0-shot example** (`dialogue_3-1-holder_000`) — candidates: Lily, Jerry, Julia · gold: **Lily**

```text
Lily, Jerry and Julia are sitting at a table. Jerry has the tape, the rope and the envelope. Jerry hands the tape to Lily. Now the tape is with
```

### `sets` — Who has what after things move: told as a story · Sets after hand-offs

Each person starts with a set of objects, then two gives; name the set one person ends with.

**0-shot example** (`setbind-v3_1-joint-2_000`) — candidates: screw, bolt, notebook, fork, battery, drum, banana, potato · gold: **fork**

```text
Matt has the screw. Neil has the bolt and the notebook. Eric has the fork and the battery. Oscar has the drum. A banana lies on the shelf. A potato lies on the shelf. Oscar gives the drum to Eric. Eric gives the fork to Oscar. So now Oscar has the
```

### `refilled` — Who has what after things move: told as a story · Refilled box

A box is emptied and refilled; name its new contents.

**0-shot example** (`formupdate-v3|K2|U1|prose|b00|C`) — candidates: hammer, sock, marble, ball, button · gold: **button**

```text
The drawer holds the hammer. The tin holds the ball. Then the drawer is emptied and the button is put in it. Then the lights are switched off. In the end, the drawer holds the
```

### `refills` — Who has what after things move: told as a story · Many refills

Boxes are emptied and refilled again and again; name one box's latest contents.

**0-shot example** (`fluid-v2-rebind|N1|RN|prose|s0|t0`) — candidates: stone, brush, button, ball, fork · gold: **button**

```text
The tin holds the fork. The crate holds the ball. The bag holds the hat. The drawer holds the brush. Then the drawer is emptied and the button is put in it. Then the bag is emptied and the stone is put in it. In the end, the drawer holds the
```

### `codevar` — Who has what after things move: written as code · Variable chains

Python variables are assigned, copied and reassigned; name what one prints.

**0-shot example** (`codevar_0-1_000`) — candidates: mirror, bulb, bowl, bolt · gold: **bolt**

```text
b = "mirror"
d = "bulb"
n = "bowl"
b = "bolt"
print(b)  # output:
```

### `sharedlists` — Who has what after things move: written as code · Shared lists

Two names point at one Python list that is changed, rebound or copied; name what a print shows.

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

### `box-narrative` — One box puzzle, four formats · Box puzzle: prose

Boxes are swapped, moved and refilled, written as plain sentences; name one box's contents.

**0-shot example** (`n2m0_131|narrative|0shot`) — candidates: fork, tape · gold: **fork**

```text
Box F contains the fork. Box G contains the tape. Box F now contains the
```

### `box-screenplay` — One box puzzle, four formats · Box puzzle: script

Boxes are swapped, moved and refilled, written as a two-speaker script; name one box's contents.

**0-shot example** (`n2m0_131|screenplay|0shot`) — candidates: fork, tape · gold: **fork**

```text
SAM: Box F contains the fork. Box G contains the tape.
RIVER: So Box F now contains the
```

### `box-code` — One box puzzle, four formats · Box puzzle: code

Boxes are swapped, moved and refilled, written as a Python dictionary; name one box's contents.

**0-shot example** (`n2m0_131|code|0shot`) — candidates: fork, tape · gold: **fork**

```text
state = {"F": "fork", "G": "tape"}
print(state["F"])  #
```

### `box-chat` — One box puzzle, four formats · Box puzzle: chat

Boxes are swapped, moved and refilled, written as chat turns; name one box's contents.

**0-shot example** (`n2m0_131|chat|0shot`) — candidates: fork, tape · gold: **fork**

```text
<|im_start|>user
Box F contains the fork. Box G contains the tape.<|im_end|>
<|im_start|>assistant
So Box F now contains the
```

### `you-static` — Who are I and you? · 'You', nothing moves

A narrator says who holds what; then a speaker says 'right now you're holding the'. Name the listener's object.

**0-shot example** (`jl0000-addressee-orig`) — candidates: photo, onion, marble · gold: **onion**

```text
A chat between Jack and Carol. Carol has the onion, Jack has the photo, and a marble is on the table.
Jack: The coffee here is good.
Jack: I had lunch already.
Jack: Right now you're holding the
```

### `i-moves` — Who are I and you? · 'I', after hand-offs

Two people hand objects over only through I / my / you lines; a speaker says 'right now I'm holding the'.

**0-shot example** (`deixis-v2_2-1-self_000_deictic`) — candidates: knife, switch, fan · gold: **switch**

```text
A chat between Olivia and Maya.
Olivia: I have the knife.
Olivia: The switch is mine.
Olivia: The fan is mine.
Olivia: I'm giving you my switch.
Maya: Right now I'm holding the
```

### `you-moves` — Who are I and you? · 'You', after hand-offs

Two people hand objects over only through I / my / you lines; a speaker says 'right now you're holding the'. Name the listener's object.

**0-shot example** (`deixis-v2_2-1-addressee_000_deictic`) — candidates: clock, lantern, whistle · gold: **clock**

```text
A chat between Judy and Diana.
Judy: The clock is mine.
Judy: I have the lantern.
Judy: I have the whistle.
Judy: My clock is yours now.
Judy: Right now you're holding the
```

### `you-named` — Who are I and you? · 'You' among three

Three people talk and hand objects over; the line naming the listener asks what they are holding.

**0-shot example** (`deixis-v2_3-1-addressee_000_deictic`) — candidates: magnet, sponge, lamp · gold: **sponge**

```text
A chat between Amy, Omar and Henry.
Amy: The magnet is mine.
Omar: The sponge is mine.
Omar: I have the lamp.
Omar: Here, Henry, take my sponge.
Amy: Henry, right now you're holding the
```

### `answer-self` — Who are I and you? · Answering as yourself

A speaker must answer with 'me', 'you' or a name, from where they stand in the conversation.

**0-shot example** (`answer-v3_1-novoc_000_qa`) — candidates: me, you, Hal, Jill · gold: **me**

```text
A chat between Dan, Hal, Jill and Mike.
Hal: I have the belt.
Hal: I have the photo.
Mike: I have the basket.
Hal: Dan, I'm giving you my photo.
Mike: Who has the photo now?
Dan: I think it's with
```

### `grammar` — Learning a rule from the examples · Made-up grammar

Examples show an invented word ending; apply it to a new invented word.

**0-shot example** (`fluid-v1-morph|suffix|n8|s0|t0`) — candidates: fo, di, fa, bi, ta · gold: **bi**

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

**0-shot example** (`fluid-v2-algebra|k1|d1|s0|t0`) — candidates: vap, viv, ker, piv, reb, sab · gold: **piv**

```text
Equations:
reb ⊙ vap = reb
vap ⊙ viv = vap
sab ⊙ reb = sab
piv ⊙ viv =
```

### `patterns` — Learning a rule from the examples · Pattern completion

Example lines repeat or mirror a pattern of nonsense words; continue a new line. A copy rule alone scores .26 here.

**0-shot example** (`fluid-v1-seqind|m2|g0|s0|t0`) — candidates: raz, bif, gaz, nex, sab, ker · gold: **gaz**

```text
Sequences:
raz vir vir raz
sab ker ker sab
nex vap vap nex
gaz bif bif
```

### `facts` — Learning a rule from the examples · Known facts (control)

Real facts in the same examples-then-question layout; this column separates the layout from the learning.

**0-shot example** (`cryst-v1|capital|shots6|s0|t0`) — candidates: Copenhagen, Washington, Warsaw, Brussels, Berlin, Paris · gold: **Brussels**

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

API rows (`or-…`) are all generation-scored; `-A` = reflex frame (single next word, reasoning off), `-B` = deliberate frame (think, then the word). Notes below list only what differs from that.

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
| `gemma-7b` | Gemma 7B | 3–10B | 8.538 | 8.538 | 6T |  |
| `gemma2-9b` | Gemma-2 9B | 3–10B | 9.242 | 9.242 | distilled 8T |  |
| `recurrentgemma-9b` | RecurrentGemma 9B | 3–10B | 9.629 | 9.629 | 2T |  |
| `bamba-9b` | Bamba 9B | 3–10B | 9.78 | 9.78 | ≈ 3.1T |  |
| `olmo2-13b` | OLMo-2 13B | 10B and up | 13.716 | 13.716 | 5.6T |  |
| `olmo3-32b` | OLMo-3 32B | 10B and up | 32.234 | 32.234 | 5.7T |  |
| `olmo3-32b-stage1` | OLMo-3 32B, before mid-training | 10B and up | 32.234 | 32.234 | stage 1 only 5.5T |  |
| `glm4-32b-base` | GLM-4 32B | 10B and up | 32.566 | 32.566 | 15T |  |
| `falcon-h1-34b` | Falcon-H1 34B | 10B and up | 33.643 | 33.643 | ≈ 18T |  |
| `llama3.1-70b` | Llama-3.1 70B | 10B and up | 70.554 | 70.554 | ≈ 15T |  |
| `qwen2.5-72b` | Qwen2.5 72B | 10B and up | 72.706 | 72.706 | ≤ 18T |  |
| `llama3.1-405b` | Llama-3.1 405B | 10B and up | 405.853 | 405.853 | 15.6T |  |
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
| Pythia 160M | +0.10 | +0.20° | +0.12 | +0.22° | +0.07° |
| SmolLM2 360M | +0.33 | +0.60° | +0.16 | +0.26° | +0.25° |
| Qwen2.5 0.5B | +0.75 | +0.46° | +0.23 | +0.49° | +0.47° |
| Falcon-H1 0.5B | +0.77 | +0.79° | +0.32 | +0.68° | +0.69° |
| Qwen3 0.6B | +0.92 | +0.48° | +0.38 | +0.73° | +0.65° |
| **1–3B** | | | | | |
| Pythia 1B | +0.39 | +0.55° | +0.18 | +0.50° | +0.38° |
| Llama-3.2 1B | +0.82 | +0.67° | +0.31 | +0.62° | +0.46° |
| OLMo 1B | +0.74 | +0.67° | +0.28 | +0.15° | +0.27° |
| kotodama-tsumugi 1.4B | +0.98 | +0.68° | +0.64 | +0.46° | +0.62° |
| OLMo-2 1B | +0.33 | +0.69° | +0.07 | +0.33° | +0.32° |
| Pythia 1.4B | +0.51 | +0.58° | +0.33 | +0.45° | +0.39° |
| Qwen2.5 1.5B | +0.98 | +0.62° | +0.49 | +0.68° | +0.60° |
| Falcon-H1 1.5B | +0.99 | +0.81° | +0.80 | +0.62° | +0.39° |
| Falcon-H1 1.5B-Deep | +0.99 | +0.92° | +0.86 | +0.75° | +0.63° |
| GPT-2 XL | +0.58 | +0.44° | +0.56 | +0.43° | +0.19° |
| SmolLM2 1.7B | +0.69 | +0.75° | +0.18 | +0.43° | +0.40° |
| Qwen3 1.7B | +1.00 | +0.89° | +0.81 | +0.89° | +0.79° |
| Gemma 2B | +0.96 | +0.84° | +0.45 | +0.48° | +0.18° |
| Gemma-2 2B | +1.00 | +0.82° | +0.81 | +0.58° | +0.29° |
| Pythia 2.8B | +0.83 | +0.54° | +0.47 | +0.53° | +0.59° |
| Pythia 2.8B (deduped) | +0.85 | +0.56° | +0.54 | +0.54° | +0.59° |
| **3–10B** | | | | | |
| Qwen2.5 3B | +0.99 | +0.72° | +0.89 | +0.66° | +0.74° |
| Falcon-H1 3B | +0.99 | +0.90° | +0.88 | +0.69° | +0.42° |
| Falcon3 3B | +0.95 | +0.56° | +0.57 | +0.61° | +0.65° |
| Qwen3 30B-A3B (MoE) | +1.00 | +0.76° | +0.97 | +0.77° | +0.13° |
| Qwen3 4B | +1.00 | +0.88° | +0.96 | +0.79° | +0.72° |
| Pythia 6.9B | +0.79 | +0.57° | +0.51 | +0.53° | +0.42° |
| Falcon-Mamba 7B | +0.98 | +0.74° | +0.12 | +0.57° | +0.39° |
| OLMo-2 7B | +0.98 | +0.88° | +0.77 | +0.56° | +0.32° |
| Falcon3 7B | +1.00 | +0.72° | +0.88 | +0.64° | +0.40° |
| Falcon-H1 7B | +1.00 | +0.97° | +1.00 | +0.79° | +0.00° |
| Qwen2.5 7B | +1.00 | +0.88° | +1.00 | +0.89° | +0.70° |
| Llama-3 8B | +1.00 | +0.99° | +0.96 | +0.89° | +0.05° |
| Llama-3.1 8B | +1.00 | +0.99° | +0.99 | +0.91° | +0.10° |
| Gemma 7B | +1.00 | +0.96° | +0.84 | +0.69° | -0.01° |
| Gemma-2 9B | +1.00 | +0.94° | +0.98 | +0.93° | +0.21° |
| RecurrentGemma 9B | +0.16 | +0.69° | +0.06 | +0.28° | +0.24° |
| Bamba 9B | +0.06 | +0.64° | -0.02 | +0.23° | +0.16° |
| **10B and up** | | | | | |
| OLMo-2 13B | +0.99 | +0.92° | +0.87 | +0.62° | -0.04° |
| OLMo-3 32B | +1.00 | +0.94° | +0.98 | +0.62° | -0.11°↓ |
| OLMo-3 32B, before mid-training | +1.00 | +0.88° | +1.00 | +0.61° | -0.28°↓ |
| GLM-4 32B | +1.00 | +0.99° | +1.00 | +0.76° | -0.27°↓ |
| Falcon-H1 34B | +1.00 | +0.99° | +1.00 | +0.73° | -0.27°↓ |
| Llama-3.1 70B | +1.00 | +0.98° | +1.00 | +0.86° | -0.15°↓ |
| Qwen2.5 72B | +1.00 | +0.98° | +1.00 | +0.90° | -0.27°↓ |
| Llama-3.1 405B | +1.00 | +0.99° | +1.00 | +0.88° | -0.31°↓ |
| **kotodama-tsumugi, by training tokens** | | | | | |
| kotodama-tsumugi · 5B tokens | +0.66 | +0.48° | +0.46 | +0.36° | +0.24° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.79 | +0.51° | +0.49 | +0.42° | +0.15° |
| kotodama-tsumugi · 11B tokens | +0.89 | +0.49° | +0.68 | +0.31° | +0.34° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.90 | +0.53° | +0.54 | +0.39° | +0.35° |
| kotodama-tsumugi · 33B tokens | +0.93 | +0.59° | +0.66 | +0.40° | +0.63° |
| kotodama-tsumugi · 55B tokens | +0.97 | +0.61° | +0.70 | +0.47° | +0.36° |
| kotodama-tsumugi · 88B tokens | +0.97 | +0.63° | +0.62 | +0.55° | +0.72° |

### Who has what after things move: told as a story

| model | Hand-offs in a story | Sets after hand-offs | Refilled box | Many refills |
|---|---|---|---|---|
| **Human** | | | | |
| Human reader (gauntlet) | +1.00 | +1.00 | — | — |
| **under 1B** | | | | |
| Pythia 160M | +0.09 | +0.06 | +0.03° | -0.01° |
| SmolLM2 360M | +0.34 | +0.10 | +0.26° | +0.10° |
| Qwen2.5 0.5B | +0.29 | +0.33 | +0.16° | +0.14° |
| Falcon-H1 0.5B | +0.50 | +0.22 | +0.29° | +0.14° |
| Qwen3 0.6B | +0.45 | +0.53 | +0.14° | +0.13° |
| **1–3B** | | | | |
| Pythia 1B | +0.35 | +0.25 | +0.16° | +0.13° |
| Llama-3.2 1B | +0.43 | +0.45 | +0.14° | +0.11° |
| OLMo 1B | +0.39 | +0.33 | +0.29° | +0.16° |
| kotodama-tsumugi 1.4B | +0.44 | +0.43 | +0.21° | +0.14° |
| OLMo-2 1B | +0.50 | +0.31 | +0.43° | +0.24° |
| Pythia 1.4B | +0.23 | +0.32 | +0.19° | +0.05° |
| Qwen2.5 1.5B | +0.50 | +0.52 | +0.49° | +0.25° |
| Falcon-H1 1.5B | +0.63 | +0.66 | +0.41° | +0.14° |
| Falcon-H1 1.5B-Deep | +0.72 | +0.57 | +0.44° | +0.28° |
| GPT-2 XL | +0.16 | +0.25 | +0.22° | -0.01° |
| SmolLM2 1.7B | +0.58 | +0.20 | +0.19° | +0.07° |
| Qwen3 1.7B | +0.68 | +0.69 | +0.53° | +0.20° |
| Gemma 2B | +0.51 | +0.51 | +0.28° | +0.31° |
| Gemma-2 2B | +0.67 | +0.52 | +0.41° | +0.36° |
| Pythia 2.8B | +0.39 | +0.34 | +0.15° | +0.03° |
| Pythia 2.8B (deduped) | +0.39 | +0.31 | +0.13° | +0.03° |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.64 | +0.52 | +0.30° | +0.13° |
| Falcon-H1 3B | +0.79 | +0.72 | +0.70° | +0.34° |
| Falcon3 3B | +0.66 | +0.28 | +0.31° | +0.19° |
| Qwen3 30B-A3B (MoE) | +0.86 | +0.81 | +0.37° | +0.33° |
| Qwen3 4B | +0.81 | +0.73 | +0.49° | +0.38° |
| Pythia 6.9B | +0.37 | +0.35 | +0.19° | +0.07° |
| Falcon-Mamba 7B | +0.55 | +0.18 | +0.34° | +0.11° |
| OLMo-2 7B | +0.71 | +0.69 | +0.47° | +0.25° |
| Falcon3 7B | +0.82 | +0.54 | +0.49° | +0.31° |
| Falcon-H1 7B | +0.88 | +0.69 | +0.63° | +0.54° |
| Qwen2.5 7B | +0.81 | +0.86 | +0.57° | +0.44° |
| Llama-3 8B | +0.79 | +0.77 | +0.31° | +0.29° |
| Llama-3.1 8B | +0.81 | +0.82 | +0.32° | +0.29° |
| Gemma 7B | +0.82 | +0.71 | +0.37° | +0.26° |
| Gemma-2 9B | +0.89 | +0.73 | +0.39° | +0.36° |
| RecurrentGemma 9B | +0.29 | +0.06 | +0.06° | +0.10° |
| Bamba 9B | +0.22 | +0.13 | +0.27° | +0.22° |
| **10B and up** | | | | |
| OLMo-2 13B | +0.87 | +0.86 | +0.68° | +0.48° |
| OLMo-3 32B | +0.85 | +0.77 | +0.59° | +0.46° |
| OLMo-3 32B, before mid-training | +0.88 | +0.84 | +0.65° | +0.47° |
| GLM-4 32B | +0.96 | +0.92 | +0.62° | +0.53° |
| Falcon-H1 34B | +0.93 | +0.86 | +0.78° | +0.66° |
| Llama-3.1 70B | +0.97 | +0.92 | +0.54° | +0.58° |
| Qwen2.5 72B | +0.98 | +0.93 | +0.61° | +0.60° |
| Llama-3.1 405B | +0.99 | +0.99 | +0.73° | +0.74° |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.26 | +0.26 | +0.04° | -0.00° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.30 | +0.27 | +0.05° | -0.04° |
| kotodama-tsumugi · 11B tokens | +0.33 | +0.31 | -0.05°↓ | -0.01° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.35 | +0.33 | +0.00° | +0.03° |
| kotodama-tsumugi · 33B tokens | +0.30 | +0.36 | +0.07° | +0.01° |
| kotodama-tsumugi · 55B tokens | +0.46 | +0.44 | +0.21° | +0.09° |
| kotodama-tsumugi · 88B tokens | +0.48 | +0.46 | +0.32° | +0.22° |

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
| Gemma 7B | +0.45 | +0.42 |
| Gemma-2 9B | +0.67 | +0.63 |
| RecurrentGemma 9B | +0.02 | +0.12 |
| Bamba 9B | +0.04 | -0.02 |
| **10B and up** | | |
| OLMo-2 13B | +0.35 | +0.44 |
| OLMo-3 32B | +0.78 | +0.74 |
| OLMo-3 32B, before mid-training | +0.60 | +0.58 |
| GLM-4 32B | +0.85 | +0.65 |
| Falcon-H1 34B | +0.84 | +0.79 |
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
| Pythia 160M | -0.00 | +0.02 | +0.01 | +0.02 |
| SmolLM2 360M | +0.22 | +0.21 | +0.40 | +0.25 |
| Qwen2.5 0.5B | +0.31 | +0.30 | +0.41 | +0.34 |
| Falcon-H1 0.5B | +0.13 | +0.26 | +0.49 | +0.08 |
| Qwen3 0.6B | +0.20 | +0.21 | +0.47 | +0.21 |
| **1–3B** | | | | |
| Pythia 1B | +0.10 | +0.09 | +0.31 | +0.05 |
| Llama-3.2 1B | +0.41 | +0.44 | +0.55 | +0.44 |
| OLMo 1B | +0.16 | +0.08 | +0.30 | +0.22 |
| kotodama-tsumugi 1.4B | +0.20 | +0.22 | +0.49 | +0.34 |
| OLMo-2 1B | +0.10 | +0.20 | +0.43 | +0.07 |
| Pythia 1.4B | +0.26 | +0.19 | +0.28 | +0.15 |
| Qwen2.5 1.5B | +0.22 | +0.21 | +0.60 | +0.29 |
| Falcon-H1 1.5B | +0.16 | +0.15 | +0.66 | +0.19 |
| Falcon-H1 1.5B-Deep | +0.07 | +0.25 | +0.76 | +0.43 |
| GPT-2 XL | +0.11 | +0.10 | +0.23 | +0.05 |
| SmolLM2 1.7B | +0.10 | +0.20 | +0.51 | -0.03 |
| Qwen3 1.7B | +0.32 | +0.33 | +0.57 | +0.39 |
| Gemma 2B | +0.24 | +0.21 | +0.50 | +0.46 |
| Gemma-2 2B | +0.27 | +0.33 | +0.60 | +0.45 |
| Pythia 2.8B | +0.38 | +0.35 | +0.45 | +0.38 |
| Pythia 2.8B (deduped) | +0.38 | +0.37 | +0.42 | +0.35 |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.47 | +0.52 | +0.53 | +0.58 |
| Falcon-H1 3B | +0.30 | +0.48 | +0.68 | +0.54 |
| Falcon3 3B | +0.23 | +0.39 | +0.62 | +0.58 |
| Qwen3 30B-A3B (MoE) | +0.32 | +0.44 | +0.80 | +0.57 |
| Qwen3 4B | +0.30 | +0.32 | +0.72 | +0.42 |
| Pythia 6.9B | +0.34 | +0.37 | +0.39 | +0.29 |
| Falcon-Mamba 7B | +0.10 | +0.17 | +0.48 | +0.23 |
| OLMo-2 7B | +0.21 | +0.20 | +0.48 | +0.17 |
| Falcon3 7B | +0.28 | +0.48 | +0.75 | +0.55 |
| Falcon-H1 7B | +0.37 | +0.55 | +0.80 | +0.56 |
| Qwen2.5 7B | +0.34 | +0.52 | +0.72 | +0.58 |
| Llama-3 8B | +0.33 | +0.45 | +0.60 | +0.60 |
| Llama-3.1 8B | +0.38 | +0.51 | +0.65 | +0.58 |
| Gemma 7B | +0.10 | +0.28 | +0.68 | +0.32 |
| Gemma-2 9B | +0.50 | +0.59 | +0.75 | +0.61 |
| RecurrentGemma 9B | +0.02 | +0.23 | +0.26 | +0.28 |
| Bamba 9B | -0.08↓ | -0.03 | -0.01 | +0.00 |
| **10B and up** | | | | |
| OLMo-2 13B | +0.23 | +0.33 | +0.60 | +0.44 |
| OLMo-3 32B | +0.34 | +0.56 | +0.78 | +0.50 |
| OLMo-3 32B, before mid-training | +0.35 | +0.48 | +0.76 | +0.56 |
| GLM-4 32B | +0.47 | +0.75 | +0.86 | +0.66 |
| Falcon-H1 34B | +0.65 | +0.84 | +0.88 | +0.78 |
| Llama-3.1 70B | +0.63 | +0.85 | +0.88 | +0.86 |
| Qwen2.5 72B | +0.54 | +0.81 | +0.80 | +0.66 |
| Llama-3.1 405B | +0.77 | +0.87 | +0.82 | +0.89 |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.06 | +0.13 | +0.20 | +0.12 |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.07 | +0.15 | +0.29 | +0.17 |
| kotodama-tsumugi · 11B tokens | +0.19 | +0.33 | +0.35 | +0.27 |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.30 | +0.29 | +0.42 | +0.29 |
| kotodama-tsumugi · 33B tokens | +0.34 | +0.30 | +0.48 | +0.33 |
| kotodama-tsumugi · 55B tokens | +0.24 | +0.32 | +0.51 | +0.38 |
| kotodama-tsumugi · 88B tokens | +0.23 | +0.29 | +0.50 | +0.35 |

### Who are I and you?

| model | 'You', nothing moves | 'I', after hand-offs | 'You', after hand-offs | 'You' among three | Answering as yourself |
|---|---|---|---|---|---|
| **Human** | | | | | |
| Human reader (gauntlet) | — | +1.00 | +1.00 | +1.00 | +1.00 |
| **under 1B** | | | | | |
| Pythia 160M | +0.10° | -0.01 | -0.05 | +0.05 | +0.02 |
| SmolLM2 360M | +0.17° | -0.09↓ | -0.03 | +0.02 | +0.04 |
| Qwen2.5 0.5B | +0.18° | +0.01 | +0.06 | +0.01 | +0.05 |
| Falcon-H1 0.5B | +0.34° | -0.01 | -0.06 | +0.00 | +0.06 |
| Qwen3 0.6B | +0.46° | +0.17 | +0.13 | -0.01 | +0.09 |
| **1–3B** | | | | | |
| Pythia 1B | +0.14° | -0.05 | -0.04 | -0.05 | +0.05 |
| Llama-3.2 1B | +0.32° | +0.07 | +0.04 | +0.07 | +0.04 |
| OLMo 1B | +0.22° | +0.01 | -0.06 | +0.02 | +0.06 |
| kotodama-tsumugi 1.4B | +0.70° | +0.08 | +0.03 | +0.09 | +0.11 |
| OLMo-2 1B | +0.36° | +0.06 | -0.02 | +0.03 | +0.14 |
| Pythia 1.4B | +0.17° | +0.01 | -0.00 | +0.06 | +0.06 |
| Qwen2.5 1.5B | +0.49° | +0.19 | -0.00 | -0.03 | +0.08 |
| Falcon-H1 1.5B | +0.76° | +0.20 | +0.02 | +0.01 | +0.10 |
| Falcon-H1 1.5B-Deep | +0.61° | +0.19 | -0.05 | +0.01 | +0.24 |
| GPT-2 XL | +0.07° | -0.01 | +0.05 | +0.05 | +0.01 |
| SmolLM2 1.7B | +0.42° | +0.13 | +0.08 | +0.04 | +0.04 |
| Qwen3 1.7B | +0.51° | +0.20 | +0.11 | +0.06 | +0.20 |
| Gemma 2B | +0.46° | +0.08 | -0.08 | +0.02 | +0.09 |
| Gemma-2 2B | +0.66° | +0.11 | -0.04 | +0.05 | +0.15 |
| Pythia 2.8B | +0.20° | +0.04 | +0.03 | +0.02 | +0.06 |
| Pythia 2.8B (deduped) | +0.17° | +0.06 | +0.02 | +0.01 | +0.05 |
| **3–10B** | | | | | |
| Qwen2.5 3B | +0.82° | +0.11 | -0.06 | -0.00 | +0.13 |
| Falcon-H1 3B | +0.91° | +0.25 | +0.00 | -0.02 | +0.24 |
| Falcon3 3B | +0.75° | +0.10 | +0.01 | +0.00 | +0.13 |
| Qwen3 30B-A3B (MoE) | +0.94° | +0.34 | +0.10 | +0.31 | +0.31 |
| Qwen3 4B | +0.68° | +0.35 | +0.20 | +0.10 | +0.42 |
| Pythia 6.9B | +0.25° | +0.03 | -0.02 | +0.12 | +0.06 |
| Falcon-Mamba 7B | +0.70° | +0.17 | -0.07 | -0.05 | +0.05 |
| OLMo-2 7B | +0.82° | +0.16 | +0.07 | +0.05 | +0.17 |
| Falcon3 7B | +0.84° | +0.12 | -0.05 | +0.20 | +0.27 |
| Falcon-H1 7B | +0.94° | +0.37 | +0.10 | +0.24 | +0.32 |
| Qwen2.5 7B | +0.75° | +0.31 | +0.13 | +0.11 | +0.29 |
| Llama-3 8B | +0.90° | +0.23 | +0.04 | +0.11 | +0.21 |
| Llama-3.1 8B | +0.87° | +0.23 | +0.07 | +0.12 | +0.21 |
| Gemma 7B | +0.93° | +0.23 | +0.06 | -0.00 | +0.17 |
| Gemma-2 9B | +0.84° | +0.32 | +0.16 | +0.19 | +0.35 |
| RecurrentGemma 9B | +0.40° | -0.04 | +0.02 | +0.04 | -0.01 |
| Bamba 9B | +0.12° | -0.04 | -0.01 | +0.01 | +0.01 |
| **10B and up** | | | | | |
| OLMo-2 13B | +0.96° | +0.39 | +0.25 | +0.33 | +0.26 |
| OLMo-3 32B | +0.98° | +0.41 | +0.22 | +0.30 | +0.37 |
| OLMo-3 32B, before mid-training | +0.94° | +0.29 | +0.17 | +0.23 | +0.29 |
| GLM-4 32B | +0.94° | +0.52 | +0.22 | +0.37 | +0.49 |
| Falcon-H1 34B | +0.92° | +0.50 | +0.35 | +0.56 | +0.54 |
| Llama-3.1 70B | +1.00° | +0.54 | +0.34 | +0.62 | +0.61 |
| Qwen2.5 72B | +0.90° | +0.67 | +0.42 | +0.69 | +0.50 |
| Llama-3.1 405B | +0.99° | +0.74 | +0.59 | +0.64 | +0.69 |
| **kotodama-tsumugi, by training tokens** | | | | | |
| kotodama-tsumugi · 5B tokens | +0.13° | -0.06 | -0.05 | -0.01 | +0.03 |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.10° | -0.07 | -0.08 | -0.04 | +0.04 |
| kotodama-tsumugi · 11B tokens | +0.13° | +0.10 | +0.03 | +0.02 | +0.07 |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.14° | +0.00 | +0.00 | +0.05 | +0.06 |
| kotodama-tsumugi · 33B tokens | +0.45° | +0.11 | +0.08 | +0.06 | +0.08 |
| kotodama-tsumugi · 55B tokens | +0.42° | +0.03 | +0.06 | +0.03 | +0.06 |
| kotodama-tsumugi · 88B tokens | +0.51° | +0.03 | +0.00 | +0.12 | +0.09 |

### Learning a rule from the examples

| model | Made-up grammar | Made-up operators | Pattern completion | Known facts (control) |
|---|---|---|---|---|
| **under 1B** | | | | |
| Pythia 160M | +0.37° | +0.11° | +0.13° | +0.44° |
| SmolLM2 360M | +0.75° | +0.26° | +0.68° | +0.97° |
| Qwen2.5 0.5B | +0.71° | +0.20° | +0.64° | +0.92° |
| Falcon-H1 0.5B | +0.70° | +0.22° | +0.51° | +0.90° |
| Qwen3 0.6B | +0.73° | +0.25° | +0.77° | +0.94° |
| **1–3B** | | | | |
| Pythia 1B | +0.67° | +0.23° | +0.27° | +0.90° |
| Llama-3.2 1B | +0.80° | +0.23° | +0.67° | +1.00° |
| OLMo 1B | +0.77° | +0.25° | +0.56° | +0.98° |
| kotodama-tsumugi 1.4B | +0.79° | +0.28° | +0.80° | +0.95° |
| OLMo-2 1B | +0.70° | +0.27° | +0.61° | +0.99° |
| Pythia 1.4B | +0.70° | +0.23° | +0.38° | +0.95° |
| Qwen2.5 1.5B | +0.78° | +0.20° | +0.79° | +1.00° |
| Falcon-H1 1.5B | +0.75° | +0.23° | +0.65° | +0.97° |
| Falcon-H1 1.5B-Deep | +0.81° | +0.27° | +0.68° | +0.98° |
| GPT-2 XL | +0.72° | +0.22° | +0.59° | +0.92° |
| SmolLM2 1.7B | +0.76° | +0.25° | +0.71° | +1.00° |
| Qwen3 1.7B | +0.82° | +0.27° | +0.83° | +0.99° |
| Gemma 2B | +0.73° | +0.20° | +0.75° | +1.00° |
| Gemma-2 2B | +0.80° | +0.27° | +0.72° | +1.00° |
| Pythia 2.8B | +0.80° | +0.25° | +0.58° | +0.98° |
| Pythia 2.8B (deduped) | +0.80° | +0.27° | +0.61° | +0.99° |
| **3–10B** | | | | |
| Qwen2.5 3B | +0.79° | +0.18° | +0.90° | +1.00° |
| Falcon-H1 3B | +0.84° | +0.26° | +0.77° | +0.98° |
| Falcon3 3B | +0.85° | +0.24° | +0.79° | +0.97° |
| Qwen3 30B-A3B (MoE) | +0.84° | +0.21° | +0.99° | +1.00° |
| Qwen3 4B | +0.85° | +0.22° | +0.95° | +1.00° |
| Pythia 6.9B | +0.78° | +0.21° | +0.50° | +0.99° |
| Falcon-Mamba 7B | +0.82° | +0.28° | +0.80° | +1.00° |
| OLMo-2 7B | +0.86° | +0.29° | +0.81° | +1.00° |
| Falcon3 7B | +0.83° | +0.23° | +0.83° | +1.00° |
| Falcon-H1 7B | +0.87° | +0.28° | +0.92° | +1.00° |
| Qwen2.5 7B | +0.83° | +0.22° | +0.94° | +1.00° |
| Llama-3 8B | +0.85° | +0.26° | +0.93° | +1.00° |
| Llama-3.1 8B | +0.84° | +0.27° | +0.94° | +1.00° |
| Gemma 7B | +0.85° | +0.24° | +0.90° | +1.00° |
| Gemma-2 9B | +0.86° | +0.30° | +0.90° | +1.00° |
| RecurrentGemma 9B | +0.63° | +0.13° | +0.34° | +1.00° |
| Bamba 9B | +0.57° | +0.14° | +0.43° | +1.00° |
| **10B and up** | | | | |
| OLMo-2 13B | +0.84° | +0.24° | +0.82° | +1.00° |
| OLMo-3 32B | +0.85° | +0.21° | +0.90° | +1.00° |
| OLMo-3 32B, before mid-training | +0.83° | +0.28° | +0.92° | +1.00° |
| GLM-4 32B | +0.86° | +0.23° | +0.96° | +1.00° |
| Falcon-H1 34B | +0.87° | +0.24° | +0.94° | +1.00° |
| Llama-3.1 70B | +0.85° | +0.32° | +0.96° | +1.00° |
| Qwen2.5 72B | +0.87° | +0.22° | +0.99° | +1.00° |
| Llama-3.1 405B | +0.90° | +0.21° | +0.98° | +1.00° |
| **kotodama-tsumugi, by training tokens** | | | | |
| kotodama-tsumugi · 5B tokens | +0.66° | +0.25° | +0.56° | +0.64° |
| kotodama-tsumugi, seed 7 · 5B tokens | +0.65° | +0.22° | +0.49° | +0.65° |
| kotodama-tsumugi · 11B tokens | +0.76° | +0.27° | +0.56° | +0.71° |
| kotodama-tsumugi, seed 7 · 11B tokens | +0.77° | +0.23° | +0.67° | +0.75° |
| kotodama-tsumugi · 33B tokens | +0.79° | +0.28° | +0.74° | +0.89° |
| kotodama-tsumugi · 55B tokens | +0.78° | +0.27° | +0.75° | +0.90° |
| kotodama-tsumugi · 88B tokens | +0.82° | +0.29° | +0.80° | +0.91° |

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
gemma-7b,"Gemma 7B","3–10B",8.538,8.538,6000000000000.0,logprob
gemma2-9b,"Gemma-2 9B","3–10B",9.242,9.242,8000000000000.0,logprob
recurrentgemma-9b,"RecurrentGemma 9B","3–10B",9.629,9.629,2000000000000.0,logprob
bamba-9b,"Bamba 9B","3–10B",9.78,9.78,3100000000000.0,logprob
olmo2-13b,"OLMo-2 13B","10B and up",13.716,13.716,5600000000000.0,logprob
olmo3-32b,"OLMo-3 32B","10B and up",32.234,32.234,5700000000000.0,logprob
olmo3-32b-stage1,"OLMo-3 32B, before mid-training","10B and up",32.234,32.234,5500000000000.0,logprob
glm4-32b-base,"GLM-4 32B","10B and up",32.566,32.566,15000000000000.0,logprob
falcon-h1-34b,"Falcon-H1 34B","10B and up",33.643,33.643,18000000000000.0,logprob
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
pythia-160m,lookup,0shot,0.222,1210,0.168,0.069,0.055,0.083,above
pythia-160m,lookup,fair,0.251,1210,0.168,0.103,0.082,0.124,above
pythia-160m,shortlist,0shot,0.230,960,0.040,0.198,0.174,0.221,above
pythia-160m,pairs,0shot,0.169,64,0.125,0.051,0.019,0.092,above
pythia-160m,pairs,fair,0.231,64,0.125,0.121,0.035,0.216,above
pythia-160m,noise,0shot,0.373,720,0.200,0.217,0.185,0.246,above
pythia-160m,now,0shot,0.339,300,0.292,0.066,0.016,0.118,above
pythia-160m,handoffs,0shot,0.359,1200,0.312,0.068,0.047,0.091,above
pythia-160m,handoffs,fair,0.374,1200,0.312,0.089,0.059,0.117,above
pythia-160m,sets,0shot,0.352,320,0.275,0.112,0.070,0.153,above
pythia-160m,sets,fair,0.320,320,0.275,0.062,0.017,0.115,above
pythia-160m,refilled,0shot,0.224,720,0.200,0.030,0.008,0.053,above
pythia-160m,refills,0shot,0.189,600,0.200,-0.014,-0.037,0.013,at
pythia-160m,codevar,0shot,0.288,1200,0.267,0.030,0.007,0.051,above
pythia-160m,codevar,fair,0.278,1200,0.267,0.014,-0.009,0.039,at
pythia-160m,sharedlists,0shot,0.238,600,0.200,0.047,0.016,0.081,above
pythia-160m,sharedlists,fair,0.190,600,0.200,-0.013,-0.040,0.016,at
pythia-160m,box-narrative,0shot,0.241,200,0.253,-0.017,-0.072,0.034,at
pythia-160m,box-narrative,fair,0.255,200,0.253,-0.001,-0.053,0.050,at
pythia-160m,box-screenplay,0shot,0.272,200,0.253,0.029,-0.023,0.077,at
pythia-160m,box-screenplay,fair,0.258,200,0.253,0.016,-0.036,0.068,at
pythia-160m,box-code,0shot,0.283,200,0.253,0.059,-0.002,0.119,at
pythia-160m,box-code,fair,0.261,200,0.253,0.012,-0.052,0.077,at
pythia-160m,box-chat,0shot,0.294,200,0.253,0.057,0.000,0.114,above
pythia-160m,box-chat,fair,0.263,200,0.253,0.020,-0.033,0.071,at
pythia-160m,you-static,0shot,0.366,300,0.292,0.104,0.055,0.155,above
pythia-160m,i-moves,0shot,0.298,180,0.287,0.011,-0.038,0.061,at
pythia-160m,i-moves,fair,0.281,180,0.287,-0.011,-0.089,0.061,at
pythia-160m,you-moves,0shot,0.281,180,0.287,-0.009,-0.069,0.051,at
pythia-160m,you-moves,fair,0.252,180,0.287,-0.051,-0.109,0.012,at
pythia-160m,you-named,0shot,0.307,180,0.287,0.029,-0.027,0.095,at
pythia-160m,you-named,fair,0.325,180,0.287,0.053,-0.011,0.128,at
pythia-160m,answer-self,0shot,0.264,2560,0.250,0.018,0.004,0.031,above
pythia-160m,answer-self,fair,0.267,2560,0.250,0.023,0.005,0.040,above
pythia-160m,grammar,ladder,,1080,,0.374,0.355,0.394,above
pythia-160m,operators,ladder,,1080,,0.106,0.084,0.126,above
pythia-160m,patterns,ladder,,1080,,0.135,0.114,0.164,above
pythia-160m,facts,ladder,,768,,0.437,0.401,0.466,above
smollm2-360m,lookup,0shot,0.462,1210,0.168,0.348,0.313,0.381,above
smollm2-360m,lookup,fair,0.445,1210,0.168,0.331,0.297,0.364,above
smollm2-360m,shortlist,0shot,0.617,960,0.040,0.601,0.569,0.631,above
smollm2-360m,pairs,0shot,0.352,64,0.125,0.259,0.134,0.402,above
smollm2-360m,pairs,fair,0.266,64,0.125,0.161,0.036,0.286,above
smollm2-360m,noise,0shot,0.411,720,0.200,0.264,0.222,0.306,above
smollm2-360m,now,0shot,0.465,300,0.292,0.245,0.172,0.320,above
smollm2-360m,handoffs,0shot,0.588,1200,0.312,0.397,0.361,0.438,above
smollm2-360m,handoffs,fair,0.545,1200,0.312,0.336,0.292,0.375,above
smollm2-360m,sets,0shot,0.358,320,0.275,0.118,0.052,0.189,above
smollm2-360m,sets,fair,0.345,320,0.275,0.097,0.025,0.169,above
smollm2-360m,refilled,0shot,0.409,720,0.200,0.261,0.216,0.310,above
smollm2-360m,refills,0shot,0.278,600,0.200,0.097,0.054,0.141,above
smollm2-360m,codevar,0shot,0.316,1200,0.267,0.058,0.022,0.097,above
smollm2-360m,codevar,fair,0.311,1200,0.267,0.052,0.016,0.090,above
smollm2-360m,sharedlists,0shot,0.210,600,0.200,0.013,-0.026,0.054,at
smollm2-360m,sharedlists,fair,0.184,600,0.200,-0.020,-0.057,0.019,at
smollm2-360m,box-narrative,0shot,0.306,200,0.253,0.052,-0.036,0.143,at
smollm2-360m,box-narrative,fair,0.417,200,0.253,0.223,0.121,0.316,above
smollm2-360m,box-screenplay,0shot,0.364,200,0.253,0.144,0.051,0.238,above
smollm2-360m,box-screenplay,fair,0.407,200,0.253,0.207,0.112,0.303,above
smollm2-360m,box-code,0shot,0.343,200,0.253,0.120,0.025,0.210,above
smollm2-360m,box-code,fair,0.560,200,0.253,0.396,0.299,0.489,above
smollm2-360m,box-chat,0shot,0.458,200,0.253,0.274,0.178,0.359,above
smollm2-360m,box-chat,fair,0.440,200,0.253,0.253,0.156,0.345,above
smollm2-360m,you-static,0shot,0.412,300,0.292,0.166,0.093,0.239,above
smollm2-360m,i-moves,0shot,0.314,180,0.287,0.029,-0.063,0.123,at
smollm2-360m,i-moves,fair,0.222,180,0.287,-0.095,-0.182,-0.005,below
smollm2-360m,you-moves,0shot,0.256,180,0.287,-0.046,-0.133,0.040,at
smollm2-360m,you-moves,fair,0.269,180,0.287,-0.028,-0.119,0.058,at
smollm2-360m,you-named,0shot,0.222,180,0.287,-0.096,-0.176,-0.010,below
smollm2-360m,you-named,fair,0.303,180,0.287,0.017,-0.073,0.105,at
smollm2-360m,answer-self,0shot,0.299,2560,0.250,0.066,0.042,0.090,above
smollm2-360m,answer-self,fair,0.280,2560,0.250,0.040,0.018,0.063,above
smollm2-360m,grammar,ladder,,1080,,0.750,0.726,0.776,above
smollm2-360m,operators,ladder,,1080,,0.261,0.229,0.291,above
smollm2-360m,patterns,ladder,,1080,,0.678,0.649,0.708,above
smollm2-360m,facts,ladder,,768,,0.973,0.960,0.984,above
qwen2.5-0.5b,lookup,0shot,0.397,1210,0.168,0.267,0.231,0.301,above
qwen2.5-0.5b,lookup,fair,0.792,1210,0.168,0.751,0.723,0.778,above
qwen2.5-0.5b,shortlist,0shot,0.486,960,0.040,0.465,0.433,0.498,above
qwen2.5-0.5b,pairs,0shot,0.391,64,0.125,0.304,0.179,0.446,above
qwen2.5-0.5b,pairs,fair,0.328,64,0.125,0.232,0.098,0.357,above
qwen2.5-0.5b,noise,0shot,0.594,720,0.200,0.492,0.445,0.533,above
qwen2.5-0.5b,now,0shot,0.627,300,0.292,0.466,0.392,0.538,above
qwen2.5-0.5b,handoffs,0shot,0.462,1200,0.312,0.217,0.177,0.257,above
qwen2.5-0.5b,handoffs,fair,0.513,1200,0.312,0.290,0.250,0.333,above
qwen2.5-0.5b,sets,0shot,0.372,320,0.275,0.143,0.074,0.218,above
qwen2.5-0.5b,sets,fair,0.508,320,0.275,0.334,0.261,0.405,above
qwen2.5-0.5b,refilled,0shot,0.328,720,0.200,0.160,0.118,0.202,above
qwen2.5-0.5b,refills,0shot,0.311,600,0.200,0.139,0.093,0.188,above
qwen2.5-0.5b,codevar,0shot,0.329,1200,0.267,0.083,0.047,0.119,above
qwen2.5-0.5b,codevar,fair,0.311,1200,0.267,0.057,0.022,0.093,above
qwen2.5-0.5b,sharedlists,0shot,0.227,600,0.200,0.033,-0.009,0.076,at
qwen2.5-0.5b,sharedlists,fair,0.332,600,0.200,0.165,0.120,0.210,above
qwen2.5-0.5b,box-narrative,0shot,0.338,200,0.253,0.080,-0.026,0.173,at
qwen2.5-0.5b,box-narrative,fair,0.492,200,0.253,0.312,0.214,0.408,above
qwen2.5-0.5b,box-screenplay,0shot,0.427,200,0.253,0.191,0.095,0.292,above
qwen2.5-0.5b,box-screenplay,fair,0.507,200,0.253,0.305,0.204,0.398,above
qwen2.5-0.5b,box-code,0shot,0.362,200,0.253,0.164,0.074,0.255,above
qwen2.5-0.5b,box-code,fair,0.575,200,0.253,0.407,0.306,0.502,above
qwen2.5-0.5b,box-chat,0shot,0.502,200,0.253,0.288,0.184,0.391,above
qwen2.5-0.5b,box-chat,fair,0.525,200,0.253,0.342,0.239,0.427,above
qwen2.5-0.5b,you-static,0shot,0.428,300,0.292,0.177,0.100,0.253,above
qwen2.5-0.5b,i-moves,0shot,0.289,180,0.287,-0.004,-0.093,0.082,at
qwen2.5-0.5b,i-moves,fair,0.294,180,0.287,0.005,-0.092,0.101,at
qwen2.5-0.5b,you-moves,0shot,0.242,180,0.287,-0.066,-0.153,0.018,at
qwen2.5-0.5b,you-moves,fair,0.331,180,0.287,0.063,-0.034,0.164,at
qwen2.5-0.5b,you-named,0shot,0.317,180,0.287,0.038,-0.051,0.132,at
qwen2.5-0.5b,you-named,fair,0.294,180,0.287,0.010,-0.081,0.100,at
qwen2.5-0.5b,answer-self,0shot,0.304,2560,0.250,0.072,0.048,0.095,above
qwen2.5-0.5b,answer-self,fair,0.288,2560,0.250,0.051,0.028,0.073,above
qwen2.5-0.5b,grammar,ladder,,1080,,0.707,0.681,0.735,above
qwen2.5-0.5b,operators,ladder,,1080,,0.205,0.178,0.235,above
qwen2.5-0.5b,patterns,ladder,,1080,,0.644,0.612,0.679,above
qwen2.5-0.5b,facts,ladder,,768,,0.919,0.895,0.941,above
falcon-h1-0.5b,lookup,0shot,0.610,1210,0.168,0.518,0.484,0.553,above
falcon-h1-0.5b,lookup,fair,0.809,1210,0.168,0.766,0.740,0.793,above
falcon-h1-0.5b,shortlist,0shot,0.794,960,0.040,0.785,0.759,0.813,above
falcon-h1-0.5b,pairs,0shot,0.766,64,0.125,0.732,0.607,0.839,above
falcon-h1-0.5b,pairs,fair,0.406,64,0.125,0.321,0.179,0.464,above
falcon-h1-0.5b,noise,0shot,0.747,720,0.200,0.684,0.645,0.722,above
falcon-h1-0.5b,now,0shot,0.780,300,0.292,0.685,0.612,0.752,above
falcon-h1-0.5b,handoffs,0shot,0.606,1200,0.312,0.419,0.376,0.458,above
falcon-h1-0.5b,handoffs,fair,0.658,1200,0.312,0.497,0.458,0.536,above
falcon-h1-0.5b,sets,0shot,0.589,320,0.275,0.435,0.361,0.507,above
falcon-h1-0.5b,sets,fair,0.436,320,0.275,0.225,0.155,0.301,above
falcon-h1-0.5b,refilled,0shot,0.428,720,0.200,0.286,0.240,0.335,above
falcon-h1-0.5b,refills,0shot,0.313,600,0.200,0.142,0.097,0.190,above
falcon-h1-0.5b,codevar,0shot,0.329,1200,0.267,0.079,0.040,0.120,above
falcon-h1-0.5b,codevar,fair,0.330,1200,0.267,0.083,0.044,0.119,above
falcon-h1-0.5b,sharedlists,0shot,0.212,600,0.200,0.016,-0.025,0.056,at
falcon-h1-0.5b,sharedlists,fair,0.302,600,0.200,0.127,0.080,0.176,above
falcon-h1-0.5b,box-narrative,0shot,0.450,200,0.253,0.265,0.166,0.360,above
falcon-h1-0.5b,box-narrative,fair,0.367,200,0.253,0.134,0.041,0.226,above
falcon-h1-0.5b,box-screenplay,0shot,0.568,200,0.253,0.430,0.338,0.518,above
falcon-h1-0.5b,box-screenplay,fair,0.445,200,0.253,0.260,0.162,0.350,above
falcon-h1-0.5b,box-code,0shot,0.398,200,0.253,0.216,0.124,0.302,above
falcon-h1-0.5b,box-code,fair,0.635,200,0.253,0.492,0.398,0.581,above
falcon-h1-0.5b,box-chat,0shot,0.590,200,0.253,0.454,0.361,0.544,above
falcon-h1-0.5b,box-chat,fair,0.333,200,0.253,0.081,-0.022,0.172,at
falcon-h1-0.5b,you-static,0shot,0.547,300,0.292,0.339,0.262,0.424,above
falcon-h1-0.5b,i-moves,0shot,0.350,180,0.287,0.082,-0.018,0.184,at
falcon-h1-0.5b,i-moves,fair,0.286,180,0.287,-0.006,-0.095,0.093,at
falcon-h1-0.5b,you-moves,0shot,0.211,180,0.287,-0.115,-0.194,-0.026,below
falcon-h1-0.5b,you-moves,fair,0.247,180,0.287,-0.057,-0.151,0.033,at
falcon-h1-0.5b,you-named,0shot,0.333,180,0.287,0.061,-0.036,0.163,at
falcon-h1-0.5b,you-named,fair,0.294,180,0.287,0.005,-0.084,0.103,at
falcon-h1-0.5b,answer-self,0shot,0.318,2560,0.250,0.090,0.067,0.115,above
falcon-h1-0.5b,answer-self,fair,0.295,2560,0.250,0.060,0.035,0.085,above
falcon-h1-0.5b,grammar,ladder,,1080,,0.700,0.673,0.733,above
falcon-h1-0.5b,operators,ladder,,1080,,0.225,0.202,0.257,above
falcon-h1-0.5b,patterns,ladder,,1080,,0.507,0.476,0.539,above
falcon-h1-0.5b,facts,ladder,,768,,0.898,0.875,0.922,above
qwen3-0.6b,lookup,0shot,0.642,1210,0.168,0.559,0.527,0.592,above
qwen3-0.6b,lookup,fair,0.935,1210,0.168,0.921,0.905,0.938,above
qwen3-0.6b,shortlist,0shot,0.504,960,0.040,0.483,0.451,0.515,above
qwen3-0.6b,pairs,0shot,0.523,64,0.125,0.455,0.330,0.580,above
qwen3-0.6b,pairs,fair,0.458,64,0.125,0.381,0.238,0.518,above
qwen3-0.6b,noise,0shot,0.785,720,0.200,0.731,0.693,0.768,above
qwen3-0.6b,now,0shot,0.756,300,0.292,0.649,0.578,0.716,above
qwen3-0.6b,handoffs,0shot,0.569,1200,0.312,0.369,0.328,0.409,above
qwen3-0.6b,handoffs,fair,0.626,1200,0.312,0.450,0.409,0.489,above
qwen3-0.6b,sets,0shot,0.545,320,0.275,0.379,0.309,0.454,above
qwen3-0.6b,sets,fair,0.653,320,0.275,0.530,0.454,0.598,above
qwen3-0.6b,refilled,0shot,0.314,720,0.200,0.142,0.103,0.188,above
qwen3-0.6b,refills,0shot,0.301,600,0.200,0.127,0.081,0.174,above
qwen3-0.6b,codevar,0shot,0.409,1200,0.267,0.192,0.152,0.229,above
qwen3-0.6b,codevar,fair,0.467,1200,0.267,0.266,0.228,0.303,above
qwen3-0.6b,sharedlists,0shot,0.294,600,0.200,0.118,0.071,0.163,above
qwen3-0.6b,sharedlists,fair,0.278,600,0.200,0.097,0.055,0.140,above
qwen3-0.6b,box-narrative,0shot,0.362,200,0.253,0.141,0.057,0.222,above
qwen3-0.6b,box-narrative,fair,0.432,200,0.253,0.199,0.099,0.293,above
qwen3-0.6b,box-screenplay,0shot,0.422,200,0.253,0.206,0.108,0.303,above
qwen3-0.6b,box-screenplay,fair,0.430,200,0.253,0.210,0.113,0.300,above
qwen3-0.6b,box-code,0shot,0.525,200,0.253,0.349,0.249,0.448,above
qwen3-0.6b,box-code,fair,0.623,200,0.253,0.467,0.370,0.562,above
qwen3-0.6b,box-chat,0shot,0.467,200,0.253,0.277,0.182,0.373,above
qwen3-0.6b,box-chat,fair,0.432,200,0.253,0.209,0.110,0.304,above
qwen3-0.6b,you-static,0shot,0.622,300,0.292,0.456,0.379,0.533,above
qwen3-0.6b,i-moves,0shot,0.378,180,0.287,0.126,0.030,0.230,above
qwen3-0.6b,i-moves,fair,0.408,180,0.287,0.170,0.075,0.273,above
qwen3-0.6b,you-moves,0shot,0.286,180,0.287,-0.007,-0.099,0.082,at
qwen3-0.6b,you-moves,fair,0.378,180,0.287,0.127,0.028,0.232,above
qwen3-0.6b,you-named,0shot,0.286,180,0.287,-0.005,-0.096,0.093,at
qwen3-0.6b,you-named,fair,0.283,180,0.287,-0.010,-0.100,0.087,at
qwen3-0.6b,answer-self,0shot,0.318,2560,0.250,0.091,0.068,0.115,above
qwen3-0.6b,answer-self,fair,0.317,2560,0.250,0.089,0.068,0.113,above
qwen3-0.6b,grammar,ladder,,1080,,0.728,0.701,0.754,above
qwen3-0.6b,operators,ladder,,1080,,0.247,0.221,0.278,above
qwen3-0.6b,patterns,ladder,,1080,,0.767,0.742,0.793,above
qwen3-0.6b,facts,ladder,,768,,0.943,0.922,0.960,above
pythia-1b,lookup,0shot,0.468,1210,0.168,0.359,0.325,0.392,above
pythia-1b,lookup,fair,0.493,1210,0.168,0.389,0.354,0.423,above
pythia-1b,shortlist,0shot,0.570,960,0.040,0.552,0.520,0.584,above
pythia-1b,pairs,0shot,0.281,64,0.125,0.179,0.062,0.304,above
pythia-1b,pairs,fair,0.286,64,0.125,0.184,0.062,0.312,above
pythia-1b,noise,0shot,0.604,720,0.200,0.504,0.460,0.549,above
pythia-1b,now,0shot,0.560,300,0.292,0.376,0.299,0.458,above
pythia-1b,handoffs,0shot,0.527,1200,0.312,0.306,0.268,0.346,above
pythia-1b,handoffs,fair,0.554,1200,0.312,0.346,0.307,0.386,above
pythia-1b,sets,0shot,0.434,320,0.275,0.227,0.151,0.300,above
pythia-1b,sets,fair,0.453,320,0.275,0.250,0.178,0.324,above
pythia-1b,refilled,0shot,0.326,720,0.200,0.157,0.117,0.199,above
pythia-1b,refills,0shot,0.302,600,0.200,0.128,0.084,0.173,above
pythia-1b,codevar,0shot,0.272,1200,0.267,0.007,-0.029,0.040,at
pythia-1b,codevar,fair,0.269,1200,0.267,0.001,-0.031,0.035,at
pythia-1b,sharedlists,0shot,0.228,600,0.200,0.035,-0.003,0.076,at
pythia-1b,sharedlists,fair,0.254,600,0.200,0.068,0.026,0.116,above
pythia-1b,box-narrative,0shot,0.327,200,0.253,0.105,0.018,0.192,above
pythia-1b,box-narrative,fair,0.325,200,0.253,0.100,0.011,0.200,above
pythia-1b,box-screenplay,0shot,0.307,200,0.253,0.056,-0.033,0.148,at
pythia-1b,box-screenplay,fair,0.323,200,0.253,0.093,0.008,0.189,above
pythia-1b,box-code,0shot,0.367,200,0.253,0.162,0.076,0.248,above
pythia-1b,box-code,fair,0.497,200,0.253,0.311,0.213,0.401,above
pythia-1b,box-chat,0shot,0.345,200,0.253,0.134,0.044,0.226,above
pythia-1b,box-chat,fair,0.305,200,0.253,0.053,-0.044,0.146,at
pythia-1b,you-static,0shot,0.398,300,0.292,0.142,0.066,0.226,above
pythia-1b,i-moves,0shot,0.317,180,0.287,0.037,-0.051,0.136,at
pythia-1b,i-moves,fair,0.253,180,0.287,-0.050,-0.133,0.043,at
pythia-1b,you-moves,0shot,0.197,180,0.287,-0.127,-0.206,-0.045,below
pythia-1b,you-moves,fair,0.258,180,0.287,-0.045,-0.133,0.047,at
pythia-1b,you-named,0shot,0.322,180,0.287,0.048,-0.053,0.141,at
pythia-1b,you-named,fair,0.253,180,0.287,-0.048,-0.131,0.047,at
pythia-1b,answer-self,0shot,0.296,2560,0.250,0.061,0.038,0.085,above
pythia-1b,answer-self,fair,0.286,2560,0.250,0.049,0.027,0.070,above
pythia-1b,grammar,ladder,,1080,,0.674,0.650,0.701,above
pythia-1b,operators,ladder,,1080,,0.229,0.201,0.265,above
pythia-1b,patterns,ladder,,1080,,0.273,0.233,0.307,above
pythia-1b,facts,ladder,,768,,0.903,0.878,0.925,above
llama3.2-1b,lookup,0shot,0.580,1210,0.168,0.480,0.445,0.514,above
llama3.2-1b,lookup,fair,0.857,1210,0.168,0.824,0.798,0.846,above
llama3.2-1b,shortlist,0shot,0.681,960,0.040,0.668,0.636,0.698,above
llama3.2-1b,pairs,0shot,0.490,64,0.125,0.417,0.274,0.545,above
llama3.2-1b,pairs,fair,0.398,64,0.125,0.312,0.179,0.438,above
llama3.2-1b,noise,0shot,0.692,720,0.200,0.616,0.572,0.658,above
llama3.2-1b,now,0shot,0.618,300,0.292,0.459,0.379,0.531,above
llama3.2-1b,handoffs,0shot,0.552,1200,0.312,0.341,0.301,0.383,above
llama3.2-1b,handoffs,fair,0.616,1200,0.312,0.435,0.396,0.473,above
llama3.2-1b,sets,0shot,0.409,320,0.275,0.185,0.116,0.262,above
llama3.2-1b,sets,fair,0.592,320,0.275,0.453,0.378,0.521,above
llama3.2-1b,refilled,0shot,0.316,720,0.200,0.145,0.103,0.188,above
llama3.2-1b,refills,0shot,0.290,600,0.200,0.113,0.065,0.156,above
llama3.2-1b,codevar,0shot,0.313,1200,0.267,0.059,0.023,0.096,above
llama3.2-1b,codevar,fair,0.364,1200,0.267,0.123,0.088,0.161,above
llama3.2-1b,sharedlists,0shot,0.341,600,0.200,0.176,0.129,0.224,above
llama3.2-1b,sharedlists,fair,0.303,600,0.200,0.129,0.085,0.173,above
llama3.2-1b,box-narrative,0shot,0.522,200,0.253,0.334,0.234,0.430,above
llama3.2-1b,box-narrative,fair,0.555,200,0.253,0.411,0.316,0.494,above
llama3.2-1b,box-screenplay,0shot,0.618,200,0.253,0.484,0.396,0.574,above
llama3.2-1b,box-screenplay,fair,0.585,200,0.253,0.444,0.354,0.526,above
llama3.2-1b,box-code,0shot,0.603,200,0.253,0.442,0.342,0.538,above
llama3.2-1b,box-code,fair,0.682,200,0.253,0.550,0.452,0.640,above
llama3.2-1b,box-chat,0shot,0.594,200,0.253,0.447,0.355,0.532,above
llama3.2-1b,box-chat,fair,0.593,200,0.253,0.442,0.344,0.529,above
llama3.2-1b,you-static,0shot,0.526,300,0.292,0.319,0.238,0.403,above
llama3.2-1b,i-moves,0shot,0.304,180,0.287,0.019,-0.067,0.109,at
llama3.2-1b,i-moves,fair,0.342,180,0.287,0.074,-0.032,0.170,at
llama3.2-1b,you-moves,0shot,0.255,180,0.287,-0.050,-0.141,0.034,at
llama3.2-1b,you-moves,fair,0.311,180,0.287,0.037,-0.053,0.126,at
llama3.2-1b,you-named,0shot,0.281,180,0.287,-0.014,-0.106,0.077,at
llama3.2-1b,you-named,fair,0.339,180,0.287,0.069,-0.029,0.168,at
llama3.2-1b,answer-self,0shot,0.315,2560,0.250,0.087,0.063,0.112,above
llama3.2-1b,answer-self,fair,0.280,2560,0.250,0.041,0.019,0.063,above
llama3.2-1b,grammar,ladder,,1080,,0.797,0.777,0.823,above
llama3.2-1b,operators,ladder,,1080,,0.234,0.210,0.265,above
llama3.2-1b,patterns,ladder,,1080,,0.669,0.641,0.694,above
llama3.2-1b,facts,ladder,,768,,0.997,0.992,1.000,above
olmo1-1b,lookup,0shot,0.436,1210,0.168,0.308,0.275,0.343,above
olmo1-1b,lookup,fair,0.784,1210,0.168,0.741,0.713,0.769,above
olmo1-1b,shortlist,0shot,0.685,960,0.040,0.672,0.640,0.701,above
olmo1-1b,pairs,0shot,0.336,64,0.125,0.241,0.116,0.375,above
olmo1-1b,pairs,fair,0.367,64,0.125,0.277,0.152,0.402,above
olmo1-1b,noise,0shot,0.317,720,0.200,0.146,0.108,0.189,above
olmo1-1b,now,0shot,0.485,300,0.292,0.271,0.195,0.353,above
olmo1-1b,handoffs,0shot,0.501,1200,0.312,0.268,0.227,0.307,above
olmo1-1b,handoffs,fair,0.584,1200,0.312,0.390,0.352,0.432,above
olmo1-1b,sets,0shot,0.470,320,0.275,0.266,0.196,0.337,above
olmo1-1b,sets,fair,0.503,320,0.275,0.328,0.253,0.399,above
olmo1-1b,refilled,0shot,0.431,720,0.200,0.288,0.244,0.335,above
olmo1-1b,refills,0shot,0.327,600,0.200,0.158,0.114,0.205,above
olmo1-1b,codevar,0shot,0.321,1200,0.267,0.070,0.035,0.107,above
olmo1-1b,codevar,fair,0.277,1200,0.267,0.008,-0.028,0.042,at
olmo1-1b,sharedlists,0shot,0.229,600,0.200,0.036,-0.005,0.078,at
olmo1-1b,sharedlists,fair,0.261,600,0.200,0.077,0.034,0.119,above
olmo1-1b,box-narrative,0shot,0.387,200,0.253,0.166,0.070,0.259,above
olmo1-1b,box-narrative,fair,0.380,200,0.253,0.163,0.072,0.260,above
olmo1-1b,box-screenplay,0shot,0.484,200,0.253,0.292,0.191,0.384,above
olmo1-1b,box-screenplay,fair,0.327,200,0.253,0.079,-0.010,0.166,at
olmo1-1b,box-code,0shot,0.537,200,0.253,0.367,0.264,0.465,above
olmo1-1b,box-code,fair,0.490,200,0.253,0.298,0.198,0.398,above
olmo1-1b,box-chat,0shot,0.510,200,0.253,0.340,0.241,0.441,above
olmo1-1b,box-chat,fair,0.430,200,0.253,0.224,0.128,0.324,above
olmo1-1b,you-static,0shot,0.450,300,0.292,0.222,0.146,0.297,above
olmo1-1b,i-moves,0shot,0.403,180,0.287,0.161,0.059,0.261,above
olmo1-1b,i-moves,fair,0.294,180,0.287,0.010,-0.082,0.103,at
olmo1-1b,you-moves,0shot,0.206,180,0.287,-0.113,-0.198,-0.031,below
olmo1-1b,you-moves,fair,0.247,180,0.287,-0.055,-0.139,0.030,at
olmo1-1b,you-named,0shot,0.267,180,0.287,-0.033,-0.123,0.057,at
olmo1-1b,you-named,fair,0.300,180,0.287,0.018,-0.073,0.110,at
olmo1-1b,answer-self,0shot,0.300,2560,0.250,0.066,0.042,0.090,above
olmo1-1b,answer-self,fair,0.294,2560,0.250,0.059,0.037,0.081,above
olmo1-1b,grammar,ladder,,1080,,0.773,0.749,0.803,above
olmo1-1b,operators,ladder,,1080,,0.254,0.222,0.286,above
olmo1-1b,patterns,ladder,,1080,,0.556,0.520,0.587,above
olmo1-1b,facts,ladder,,768,,0.981,0.969,0.991,above
tsumugi,lookup,0shot,0.555,1210,0.168,0.453,0.420,0.487,above
tsumugi,lookup,fair,0.986,1210,0.168,0.983,0.975,0.991,above
tsumugi,shortlist,0shot,0.693,960,0.040,0.680,0.648,0.710,above
tsumugi,pairs,0shot,0.508,64,0.125,0.438,0.304,0.580,above
tsumugi,pairs,fair,0.688,64,0.125,0.643,0.518,0.768,above
tsumugi,noise,0shot,0.569,720,0.200,0.461,0.416,0.506,above
tsumugi,now,0shot,0.737,300,0.292,0.621,0.547,0.692,above
tsumugi,handoffs,0shot,0.604,1200,0.312,0.417,0.381,0.458,above
tsumugi,handoffs,fair,0.621,1200,0.312,0.445,0.404,0.485,above
tsumugi,sets,0shot,0.478,320,0.275,0.287,0.219,0.360,above
tsumugi,sets,fair,0.578,320,0.275,0.426,0.349,0.502,above
tsumugi,refilled,0shot,0.371,720,0.200,0.213,0.170,0.260,above
tsumugi,refills,0shot,0.309,600,0.200,0.137,0.092,0.184,above
tsumugi,codevar,0shot,0.323,1200,0.267,0.072,0.035,0.109,above
tsumugi,codevar,fair,0.301,1200,0.267,0.042,0.009,0.078,above
tsumugi,sharedlists,0shot,0.254,600,0.200,0.067,0.025,0.114,above
tsumugi,sharedlists,fair,0.319,600,0.200,0.149,0.104,0.192,above
tsumugi,box-narrative,0shot,0.420,200,0.253,0.192,0.089,0.284,above
tsumugi,box-narrative,fair,0.422,200,0.253,0.196,0.099,0.297,above
tsumugi,box-screenplay,0shot,0.542,200,0.253,0.367,0.266,0.467,above
tsumugi,box-screenplay,fair,0.440,200,0.253,0.220,0.117,0.313,above
tsumugi,box-code,0shot,0.468,200,0.253,0.292,0.192,0.384,above
tsumugi,box-code,fair,0.640,200,0.253,0.489,0.392,0.583,above
tsumugi,box-chat,0shot,0.390,200,0.253,0.164,0.067,0.254,above
tsumugi,box-chat,fair,0.520,200,0.253,0.343,0.246,0.438,above
tsumugi,you-static,0shot,0.790,300,0.292,0.697,0.629,0.759,above
tsumugi,i-moves,0shot,0.317,180,0.287,0.035,-0.065,0.129,at
tsumugi,i-moves,fair,0.344,180,0.287,0.081,-0.016,0.177,at
tsumugi,you-moves,0shot,0.253,180,0.287,-0.052,-0.141,0.031,at
tsumugi,you-moves,fair,0.311,180,0.287,0.033,-0.055,0.132,at
tsumugi,you-named,0shot,0.331,180,0.287,0.056,-0.042,0.149,at
tsumugi,you-named,fair,0.353,180,0.287,0.088,-0.013,0.190,at
tsumugi,answer-self,0shot,0.357,2560,0.250,0.143,0.118,0.166,above
tsumugi,answer-self,fair,0.332,2560,0.250,0.110,0.087,0.132,above
tsumugi,grammar,ladder,,1080,,0.791,0.770,0.815,above
tsumugi,operators,ladder,,1080,,0.284,0.254,0.320,above
tsumugi,patterns,ladder,,1080,,0.796,0.769,0.820,above
tsumugi,facts,ladder,,768,,0.949,0.931,0.966,above
olmo2-1b,lookup,0shot,0.329,1210,0.168,0.186,0.152,0.220,above
olmo2-1b,lookup,fair,0.450,1210,0.168,0.333,0.298,0.365,above
olmo2-1b,shortlist,0shot,0.702,960,0.040,0.689,0.657,0.718,above
olmo2-1b,pairs,0shot,0.406,64,0.125,0.321,0.196,0.464,above
olmo2-1b,pairs,fair,0.188,64,0.125,0.071,-0.036,0.179,at
olmo2-1b,noise,0shot,0.463,720,0.200,0.329,0.286,0.375,above
olmo2-1b,now,0shot,0.527,300,0.292,0.325,0.245,0.406,above
olmo2-1b,handoffs,0shot,0.602,1200,0.312,0.414,0.371,0.452,above
olmo2-1b,handoffs,fair,0.661,1200,0.312,0.503,0.464,0.539,above
olmo2-1b,sets,0shot,0.541,320,0.275,0.363,0.293,0.442,above
olmo2-1b,sets,fair,0.500,320,0.275,0.309,0.228,0.385,above
olmo2-1b,refilled,0shot,0.544,720,0.200,0.431,0.382,0.476,above
olmo2-1b,refills,0shot,0.393,600,0.200,0.242,0.196,0.291,above
olmo2-1b,codevar,0shot,0.349,1200,0.267,0.110,0.073,0.147,above
olmo2-1b,codevar,fair,0.336,1200,0.267,0.088,0.052,0.125,above
olmo2-1b,sharedlists,0shot,0.216,600,0.200,0.020,-0.020,0.065,at
olmo2-1b,sharedlists,fair,0.251,600,0.200,0.064,0.022,0.108,above
olmo2-1b,box-narrative,0shot,0.412,200,0.253,0.216,0.122,0.314,above
olmo2-1b,box-narrative,fair,0.347,200,0.253,0.104,0.011,0.189,above
olmo2-1b,box-screenplay,0shot,0.438,200,0.253,0.247,0.159,0.342,above
olmo2-1b,box-screenplay,fair,0.410,200,0.253,0.201,0.106,0.291,above
olmo2-1b,box-code,0shot,0.410,200,0.253,0.225,0.134,0.320,above
olmo2-1b,box-code,fair,0.600,200,0.253,0.426,0.324,0.527,above
olmo2-1b,box-chat,0shot,0.522,200,0.253,0.352,0.257,0.440,above
olmo2-1b,box-chat,fair,0.325,200,0.253,0.074,-0.023,0.171,at
olmo2-1b,you-static,0shot,0.553,300,0.292,0.359,0.279,0.435,above
olmo2-1b,i-moves,0shot,0.381,180,0.287,0.126,0.027,0.236,above
olmo2-1b,i-moves,fair,0.328,180,0.287,0.057,-0.037,0.157,at
olmo2-1b,you-moves,0shot,0.192,180,0.287,-0.138,-0.214,-0.052,below
olmo2-1b,you-moves,fair,0.272,180,0.287,-0.022,-0.109,0.072,at
olmo2-1b,you-named,0shot,0.322,180,0.287,0.047,-0.047,0.138,at
olmo2-1b,you-named,fair,0.308,180,0.287,0.030,-0.067,0.124,at
olmo2-1b,answer-self,0shot,0.348,2560,0.250,0.131,0.108,0.156,above
olmo2-1b,answer-self,fair,0.354,2560,0.250,0.139,0.113,0.163,above
olmo2-1b,grammar,ladder,,1080,,0.699,0.672,0.726,above
olmo2-1b,operators,ladder,,1080,,0.266,0.234,0.303,above
olmo2-1b,patterns,ladder,,1080,,0.606,0.576,0.637,above
olmo2-1b,facts,ladder,,768,,0.990,0.979,0.997,above
pythia-1.4b,lookup,0shot,0.526,1210,0.168,0.425,0.390,0.458,above
pythia-1.4b,lookup,fair,0.601,1210,0.168,0.515,0.481,0.549,above
pythia-1.4b,shortlist,0shot,0.596,960,0.040,0.580,0.546,0.613,above
pythia-1.4b,pairs,0shot,0.469,64,0.125,0.393,0.250,0.536,above
pythia-1.4b,pairs,fair,0.414,64,0.125,0.330,0.205,0.473,above
pythia-1.4b,noise,0shot,0.563,720,0.200,0.454,0.409,0.496,above
pythia-1.4b,now,0shot,0.568,300,0.292,0.388,0.310,0.468,above
pythia-1.4b,handoffs,0shot,0.491,1200,0.312,0.254,0.213,0.296,above
pythia-1.4b,handoffs,fair,0.473,1200,0.312,0.229,0.188,0.267,above
pythia-1.4b,sets,0shot,0.537,320,0.275,0.363,0.283,0.435,above
pythia-1.4b,sets,fair,0.495,320,0.275,0.315,0.241,0.393,above
pythia-1.4b,refilled,0shot,0.353,720,0.200,0.192,0.146,0.238,above
pythia-1.4b,refills,0shot,0.242,600,0.200,0.052,0.009,0.093,above
pythia-1.4b,codevar,0shot,0.245,1200,0.267,-0.031,-0.064,0.001,at
pythia-1.4b,codevar,fair,0.247,1200,0.267,-0.030,-0.064,0.002,at
pythia-1.4b,sharedlists,0shot,0.212,600,0.200,0.016,-0.024,0.057,at
pythia-1.4b,sharedlists,fair,0.195,600,0.200,-0.006,-0.044,0.035,at
pythia-1.4b,box-narrative,0shot,0.420,200,0.253,0.214,0.125,0.306,above
pythia-1.4b,box-narrative,fair,0.458,200,0.253,0.257,0.157,0.353,above
pythia-1.4b,box-screenplay,0shot,0.357,200,0.253,0.119,0.023,0.213,above
pythia-1.4b,box-screenplay,fair,0.415,200,0.253,0.193,0.093,0.294,above
pythia-1.4b,box-code,0shot,0.395,200,0.253,0.197,0.104,0.285,above
pythia-1.4b,box-code,fair,0.475,200,0.253,0.280,0.182,0.381,above
pythia-1.4b,box-chat,0shot,0.415,200,0.253,0.201,0.109,0.295,above
pythia-1.4b,box-chat,fair,0.390,200,0.253,0.148,0.052,0.241,above
pythia-1.4b,you-static,0shot,0.423,300,0.292,0.174,0.094,0.253,above
pythia-1.4b,i-moves,0shot,0.344,180,0.287,0.077,-0.018,0.181,at
pythia-1.4b,i-moves,fair,0.294,180,0.287,0.013,-0.073,0.101,at
pythia-1.4b,you-moves,0shot,0.214,180,0.287,-0.105,-0.187,-0.022,below
pythia-1.4b,you-moves,fair,0.286,180,0.287,-0.003,-0.092,0.091,at
pythia-1.4b,you-named,0shot,0.353,180,0.287,0.093,0.006,0.190,above
pythia-1.4b,you-named,fair,0.331,180,0.287,0.064,-0.033,0.163,at
pythia-1.4b,answer-self,0shot,0.309,2560,0.250,0.079,0.058,0.102,above
pythia-1.4b,answer-self,fair,0.292,2560,0.250,0.056,0.034,0.079,above
pythia-1.4b,grammar,ladder,,1080,,0.697,0.672,0.721,above
pythia-1.4b,operators,ladder,,1080,,0.228,0.201,0.261,above
pythia-1.4b,patterns,ladder,,1080,,0.376,0.345,0.408,above
pythia-1.4b,facts,ladder,,768,,0.955,0.936,0.972,above
qwen2.5-1.5b,lookup,0shot,0.722,1210,0.168,0.656,0.625,0.687,above
qwen2.5-1.5b,lookup,fair,0.984,1210,0.168,0.980,0.970,0.988,above
qwen2.5-1.5b,shortlist,0shot,0.638,960,0.040,0.623,0.588,0.656,above
qwen2.5-1.5b,pairs,0shot,0.805,64,0.125,0.777,0.661,0.875,above
qwen2.5-1.5b,pairs,fair,0.555,64,0.125,0.491,0.348,0.616,above
qwen2.5-1.5b,noise,0shot,0.744,720,0.200,0.680,0.641,0.717,above
qwen2.5-1.5b,now,0shot,0.716,300,0.292,0.596,0.526,0.667,above
qwen2.5-1.5b,handoffs,0shot,0.649,1200,0.312,0.483,0.449,0.523,above
qwen2.5-1.5b,handoffs,fair,0.661,1200,0.312,0.500,0.462,0.535,above
qwen2.5-1.5b,sets,0shot,0.625,320,0.275,0.487,0.417,0.558,above
qwen2.5-1.5b,sets,fair,0.640,320,0.275,0.523,0.457,0.585,above
qwen2.5-1.5b,refilled,0shot,0.593,720,0.200,0.491,0.443,0.538,above
qwen2.5-1.5b,refills,0shot,0.397,600,0.200,0.246,0.197,0.293,above
qwen2.5-1.5b,codevar,0shot,0.399,1200,0.267,0.171,0.134,0.207,above
qwen2.5-1.5b,codevar,fair,0.433,1200,0.267,0.219,0.182,0.256,above
qwen2.5-1.5b,sharedlists,0shot,0.278,600,0.200,0.097,0.056,0.142,above
qwen2.5-1.5b,sharedlists,fair,0.315,600,0.200,0.143,0.101,0.189,above
qwen2.5-1.5b,box-narrative,0shot,0.332,200,0.253,0.066,-0.028,0.167,at
qwen2.5-1.5b,box-narrative,fair,0.427,200,0.253,0.223,0.135,0.325,above
qwen2.5-1.5b,box-screenplay,0shot,0.432,200,0.253,0.207,0.102,0.305,above
qwen2.5-1.5b,box-screenplay,fair,0.424,200,0.253,0.208,0.109,0.308,above
qwen2.5-1.5b,box-code,0shot,0.650,200,0.253,0.512,0.416,0.599,above
qwen2.5-1.5b,box-code,fair,0.718,200,0.253,0.605,0.514,0.691,above
qwen2.5-1.5b,box-chat,0shot,0.495,200,0.253,0.308,0.203,0.408,above
qwen2.5-1.5b,box-chat,fair,0.485,200,0.253,0.288,0.191,0.389,above
qwen2.5-1.5b,you-static,0shot,0.640,300,0.292,0.486,0.407,0.560,above
qwen2.5-1.5b,i-moves,0shot,0.397,180,0.287,0.152,0.053,0.253,above
qwen2.5-1.5b,i-moves,fair,0.422,180,0.287,0.190,0.093,0.293,above
qwen2.5-1.5b,you-moves,0shot,0.286,180,0.287,-0.003,-0.098,0.090,at
qwen2.5-1.5b,you-moves,fair,0.286,180,0.287,-0.002,-0.097,0.090,at
qwen2.5-1.5b,you-named,0shot,0.329,180,0.287,0.057,-0.042,0.149,at
qwen2.5-1.5b,you-named,fair,0.263,180,0.287,-0.035,-0.124,0.059,at
qwen2.5-1.5b,answer-self,0shot,0.316,2560,0.250,0.088,0.066,0.113,above
qwen2.5-1.5b,answer-self,fair,0.313,2560,0.250,0.084,0.061,0.107,above
qwen2.5-1.5b,grammar,ladder,,1080,,0.779,0.762,0.804,above
qwen2.5-1.5b,operators,ladder,,1080,,0.196,0.172,0.229,above
qwen2.5-1.5b,patterns,ladder,,1080,,0.793,0.766,0.815,above
qwen2.5-1.5b,facts,ladder,,768,,0.995,0.989,1.000,above
falcon-h1-1.5b,lookup,0shot,0.844,1210,0.168,0.804,0.777,0.831,above
falcon-h1-1.5b,lookup,fair,0.989,1210,0.168,0.986,0.978,0.993,above
falcon-h1-1.5b,shortlist,0shot,0.817,960,0.040,0.809,0.783,0.835,above
falcon-h1-1.5b,pairs,0shot,0.891,64,0.125,0.875,0.786,0.946,above
falcon-h1-1.5b,pairs,fair,0.828,64,0.125,0.804,0.696,0.911,above
falcon-h1-1.5b,noise,0shot,0.693,720,0.200,0.616,0.573,0.660,above
falcon-h1-1.5b,now,0shot,0.575,300,0.292,0.395,0.316,0.473,above
falcon-h1-1.5b,handoffs,0shot,0.593,1200,0.312,0.397,0.357,0.440,above
falcon-h1-1.5b,handoffs,fair,0.747,1200,0.312,0.626,0.589,0.660,above
falcon-h1-1.5b,sets,0shot,0.537,320,0.275,0.366,0.295,0.443,above
falcon-h1-1.5b,sets,fair,0.753,320,0.275,0.664,0.594,0.724,above
falcon-h1-1.5b,refilled,0shot,0.526,720,0.200,0.408,0.364,0.453,above
falcon-h1-1.5b,refills,0shot,0.310,600,0.200,0.138,0.091,0.183,above
falcon-h1-1.5b,codevar,0shot,0.432,1200,0.267,0.222,0.184,0.260,above
falcon-h1-1.5b,codevar,fair,0.518,1200,0.267,0.339,0.300,0.376,above
falcon-h1-1.5b,sharedlists,0shot,0.350,600,0.200,0.188,0.142,0.233,above
falcon-h1-1.5b,sharedlists,fair,0.500,600,0.200,0.375,0.323,0.429,above
falcon-h1-1.5b,box-narrative,0shot,0.393,200,0.253,0.177,0.084,0.267,above
falcon-h1-1.5b,box-narrative,fair,0.380,200,0.253,0.159,0.062,0.253,above
falcon-h1-1.5b,box-screenplay,0shot,0.485,200,0.253,0.326,0.230,0.417,above
falcon-h1-1.5b,box-screenplay,fair,0.365,200,0.253,0.147,0.053,0.242,above
falcon-h1-1.5b,box-code,0shot,0.575,200,0.253,0.409,0.315,0.499,above
falcon-h1-1.5b,box-code,fair,0.755,200,0.253,0.663,0.578,0.743,above
falcon-h1-1.5b,box-chat,0shot,0.545,200,0.253,0.385,0.287,0.481,above
falcon-h1-1.5b,box-chat,fair,0.395,200,0.253,0.191,0.095,0.289,above
falcon-h1-1.5b,you-static,0shot,0.832,300,0.292,0.756,0.690,0.815,above
falcon-h1-1.5b,i-moves,0shot,0.417,180,0.287,0.180,0.074,0.284,above
falcon-h1-1.5b,i-moves,fair,0.428,180,0.287,0.202,0.106,0.299,above
falcon-h1-1.5b,you-moves,0shot,0.211,180,0.287,-0.109,-0.196,-0.022,below
falcon-h1-1.5b,you-moves,fair,0.294,180,0.287,0.016,-0.070,0.109,at
falcon-h1-1.5b,you-named,0shot,0.311,180,0.287,0.029,-0.066,0.137,at
falcon-h1-1.5b,you-named,fair,0.297,180,0.287,0.013,-0.084,0.111,at
falcon-h1-1.5b,answer-self,0shot,0.330,2560,0.250,0.107,0.083,0.132,above
falcon-h1-1.5b,answer-self,fair,0.325,2560,0.250,0.101,0.077,0.124,above
falcon-h1-1.5b,grammar,ladder,,1080,,0.752,0.726,0.774,above
falcon-h1-1.5b,operators,ladder,,1080,,0.226,0.199,0.255,above
falcon-h1-1.5b,patterns,ladder,,1080,,0.653,0.623,0.682,above
falcon-h1-1.5b,facts,ladder,,768,,0.970,0.957,0.982,above
falcon-h1-1.5b-deep,lookup,0shot,0.809,1210,0.168,0.760,0.732,0.787,above
falcon-h1-1.5b-deep,lookup,fair,0.989,1210,0.168,0.987,0.981,0.994,above
falcon-h1-1.5b-deep,shortlist,0shot,0.924,960,0.040,0.921,0.905,0.938,above
falcon-h1-1.5b-deep,pairs,0shot,0.891,64,0.125,0.875,0.786,0.964,above
falcon-h1-1.5b-deep,pairs,fair,0.875,64,0.125,0.857,0.768,0.946,above
falcon-h1-1.5b-deep,noise,0shot,0.800,720,0.200,0.750,0.714,0.787,above
falcon-h1-1.5b-deep,now,0shot,0.738,300,0.292,0.627,0.554,0.692,above
falcon-h1-1.5b-deep,handoffs,0shot,0.761,1200,0.312,0.647,0.612,0.683,above
falcon-h1-1.5b-deep,handoffs,fair,0.810,1200,0.312,0.720,0.686,0.752,above
falcon-h1-1.5b-deep,sets,0shot,0.734,320,0.275,0.634,0.569,0.702,above
falcon-h1-1.5b-deep,sets,fair,0.683,320,0.275,0.566,0.492,0.635,above
falcon-h1-1.5b-deep,refilled,0shot,0.556,720,0.200,0.444,0.401,0.496,above
falcon-h1-1.5b-deep,refills,0shot,0.421,600,0.200,0.276,0.227,0.323,above
falcon-h1-1.5b-deep,codevar,0shot,0.493,1200,0.267,0.305,0.264,0.346,above
falcon-h1-1.5b-deep,codevar,fair,0.618,1200,0.267,0.474,0.435,0.511,above
falcon-h1-1.5b-deep,sharedlists,0shot,0.418,600,0.200,0.273,0.225,0.321,above
falcon-h1-1.5b-deep,sharedlists,fair,0.527,600,0.200,0.408,0.358,0.456,above
falcon-h1-1.5b-deep,box-narrative,0shot,0.395,200,0.253,0.161,0.067,0.265,above
falcon-h1-1.5b-deep,box-narrative,fair,0.320,200,0.253,0.065,-0.034,0.159,at
falcon-h1-1.5b-deep,box-screenplay,0shot,0.500,200,0.253,0.325,0.235,0.417,above
falcon-h1-1.5b-deep,box-screenplay,fair,0.445,200,0.253,0.252,0.161,0.352,above
falcon-h1-1.5b-deep,box-code,0shot,0.542,200,0.253,0.379,0.281,0.471,above
falcon-h1-1.5b-deep,box-code,fair,0.830,200,0.253,0.763,0.690,0.830,above
falcon-h1-1.5b-deep,box-chat,0shot,0.623,200,0.253,0.500,0.415,0.595,above
falcon-h1-1.5b-deep,box-chat,fair,0.575,200,0.253,0.429,0.328,0.525,above
falcon-h1-1.5b-deep,you-static,0shot,0.733,300,0.292,0.615,0.542,0.681,above
falcon-h1-1.5b-deep,i-moves,0shot,0.439,180,0.287,0.209,0.106,0.308,above
falcon-h1-1.5b-deep,i-moves,fair,0.422,180,0.287,0.191,0.092,0.305,above
falcon-h1-1.5b-deep,you-moves,0shot,0.194,180,0.287,-0.131,-0.206,-0.049,below
falcon-h1-1.5b-deep,you-moves,fair,0.250,180,0.287,-0.050,-0.140,0.038,at
falcon-h1-1.5b-deep,you-named,0shot,0.319,180,0.287,0.038,-0.059,0.137,at
falcon-h1-1.5b-deep,you-named,fair,0.300,180,0.287,0.012,-0.079,0.104,at
falcon-h1-1.5b-deep,answer-self,0shot,0.442,2560,0.250,0.257,0.231,0.283,above
falcon-h1-1.5b-deep,answer-self,fair,0.433,2560,0.250,0.243,0.219,0.269,above
falcon-h1-1.5b-deep,grammar,ladder,,1080,,0.812,0.789,0.839,above
falcon-h1-1.5b-deep,operators,ladder,,1080,,0.269,0.237,0.301,above
falcon-h1-1.5b-deep,patterns,ladder,,1080,,0.684,0.655,0.712,above
falcon-h1-1.5b-deep,facts,ladder,,768,,0.984,0.974,0.991,above
gpt2-xl,lookup,0shot,0.571,1210,0.168,0.478,0.443,0.510,above
gpt2-xl,lookup,fair,0.655,1210,0.168,0.584,0.552,0.617,above
gpt2-xl,shortlist,0shot,0.460,960,0.040,0.438,0.404,0.472,above
gpt2-xl,pairs,0shot,0.516,64,0.125,0.446,0.304,0.589,above
gpt2-xl,pairs,fair,0.617,64,0.125,0.562,0.420,0.696,above
gpt2-xl,noise,0shot,0.542,720,0.200,0.428,0.382,0.469,above
gpt2-xl,now,0shot,0.428,300,0.292,0.189,0.113,0.270,above
gpt2-xl,handoffs,0shot,0.398,1200,0.312,0.125,0.085,0.164,above
gpt2-xl,handoffs,fair,0.424,1200,0.312,0.162,0.122,0.200,above
gpt2-xl,sets,0shot,0.442,320,0.275,0.238,0.164,0.313,above
gpt2-xl,sets,fair,0.448,320,0.275,0.248,0.174,0.324,above
gpt2-xl,refilled,0shot,0.379,720,0.200,0.223,0.178,0.272,above
gpt2-xl,refills,0shot,0.195,600,0.200,-0.006,-0.048,0.035,at
gpt2-xl,codevar,0shot,0.253,1200,0.267,-0.016,-0.050,0.018,at
gpt2-xl,codevar,fair,0.223,1200,0.267,-0.062,-0.094,-0.029,below
gpt2-xl,sharedlists,0shot,0.274,600,0.200,0.093,0.050,0.134,above
gpt2-xl,sharedlists,fair,0.209,600,0.200,0.011,-0.030,0.052,at
gpt2-xl,box-narrative,0shot,0.367,200,0.253,0.147,0.056,0.239,above
gpt2-xl,box-narrative,fair,0.330,200,0.253,0.112,0.021,0.201,above
gpt2-xl,box-screenplay,0shot,0.410,200,0.253,0.211,0.115,0.302,above
gpt2-xl,box-screenplay,fair,0.328,200,0.253,0.102,0.009,0.194,above
gpt2-xl,box-code,0shot,0.307,200,0.253,0.066,-0.020,0.156,at
gpt2-xl,box-code,fair,0.420,200,0.253,0.234,0.144,0.323,above
gpt2-xl,box-chat,0shot,0.407,200,0.253,0.209,0.124,0.291,above
gpt2-xl,box-chat,fair,0.292,200,0.253,0.050,-0.040,0.140,at
gpt2-xl,you-static,0shot,0.346,300,0.292,0.073,-0.006,0.148,at
gpt2-xl,i-moves,0shot,0.302,180,0.287,0.015,-0.075,0.111,at
gpt2-xl,i-moves,fair,0.283,180,0.287,-0.009,-0.096,0.088,at
gpt2-xl,you-moves,0shot,0.272,180,0.287,-0.021,-0.113,0.065,at
gpt2-xl,you-moves,fair,0.328,180,0.287,0.054,-0.042,0.146,at
gpt2-xl,you-named,0shot,0.339,180,0.287,0.070,-0.020,0.163,at
gpt2-xl,you-named,fair,0.324,180,0.287,0.051,-0.040,0.150,at
gpt2-xl,answer-self,0shot,0.288,2560,0.250,0.051,0.028,0.074,above
gpt2-xl,answer-self,fair,0.261,2560,0.250,0.015,-0.007,0.036,at
gpt2-xl,grammar,ladder,,1080,,0.725,0.694,0.752,above
gpt2-xl,operators,ladder,,1080,,0.218,0.194,0.246,above
gpt2-xl,patterns,ladder,,1080,,0.587,0.551,0.619,above
gpt2-xl,facts,ladder,,768,,0.917,0.894,0.936,above
smollm2-1.7b,lookup,0shot,0.419,1210,0.168,0.290,0.258,0.324,above
smollm2-1.7b,lookup,fair,0.744,1210,0.168,0.695,0.665,0.722,above
smollm2-1.7b,shortlist,0shot,0.765,960,0.040,0.755,0.729,0.780,above
smollm2-1.7b,pairs,0shot,0.328,64,0.125,0.232,0.107,0.357,above
smollm2-1.7b,pairs,fair,0.281,64,0.125,0.179,0.054,0.304,above
smollm2-1.7b,noise,0shot,0.544,720,0.200,0.430,0.385,0.473,above
smollm2-1.7b,now,0shot,0.568,300,0.292,0.396,0.319,0.470,above
smollm2-1.7b,handoffs,0shot,0.612,1200,0.312,0.430,0.391,0.469,above
smollm2-1.7b,handoffs,fair,0.714,1200,0.312,0.580,0.543,0.615,above
smollm2-1.7b,sets,0shot,0.395,320,0.275,0.166,0.095,0.242,above
smollm2-1.7b,sets,fair,0.416,320,0.275,0.201,0.129,0.271,above
smollm2-1.7b,refilled,0shot,0.349,720,0.200,0.186,0.144,0.230,above
smollm2-1.7b,refills,0shot,0.253,600,0.200,0.066,0.024,0.115,above
smollm2-1.7b,codevar,0shot,0.361,1200,0.267,0.124,0.088,0.163,above
smollm2-1.7b,codevar,fair,0.409,1200,0.267,0.187,0.151,0.226,above
smollm2-1.7b,sharedlists,0shot,0.287,600,0.200,0.109,0.065,0.150,above
smollm2-1.7b,sharedlists,fair,0.332,600,0.200,0.165,0.118,0.210,above
smollm2-1.7b,box-narrative,0shot,0.307,200,0.253,0.048,-0.037,0.138,at
smollm2-1.7b,box-narrative,fair,0.343,200,0.253,0.104,0.010,0.198,above
smollm2-1.7b,box-screenplay,0shot,0.412,200,0.253,0.211,0.117,0.302,above
smollm2-1.7b,box-screenplay,fair,0.410,200,0.253,0.199,0.106,0.297,above
smollm2-1.7b,box-code,0shot,0.573,200,0.253,0.417,0.321,0.510,above
smollm2-1.7b,box-code,fair,0.647,200,0.253,0.507,0.406,0.600,above
smollm2-1.7b,box-chat,0shot,0.427,200,0.253,0.242,0.152,0.334,above
smollm2-1.7b,box-chat,fair,0.228,200,0.253,-0.034,-0.119,0.048,at
smollm2-1.7b,you-static,0shot,0.593,300,0.292,0.425,0.344,0.503,above
smollm2-1.7b,i-moves,0shot,0.375,180,0.287,0.122,0.022,0.226,above
smollm2-1.7b,i-moves,fair,0.378,180,0.287,0.127,0.027,0.218,above
smollm2-1.7b,you-moves,0shot,0.232,180,0.287,-0.081,-0.168,0.005,at
smollm2-1.7b,you-moves,fair,0.344,180,0.287,0.082,-0.010,0.175,at
smollm2-1.7b,you-named,0shot,0.286,180,0.287,-0.009,-0.097,0.090,at
smollm2-1.7b,you-named,fair,0.319,180,0.287,0.040,-0.059,0.134,at
smollm2-1.7b,answer-self,0shot,0.286,2560,0.250,0.048,0.026,0.071,above
smollm2-1.7b,answer-self,fair,0.277,2560,0.250,0.036,0.013,0.059,above
smollm2-1.7b,grammar,ladder,,1080,,0.764,0.741,0.791,above
smollm2-1.7b,operators,ladder,,1080,,0.247,0.216,0.278,above
smollm2-1.7b,patterns,ladder,,1080,,0.714,0.689,0.741,above
smollm2-1.7b,facts,ladder,,768,,1.000,1.000,1.000,above
qwen3-1.7b,lookup,0shot,0.731,1210,0.168,0.665,0.636,0.694,above
qwen3-1.7b,lookup,fair,0.998,1210,0.168,0.997,0.994,1.000,above
qwen3-1.7b,shortlist,0shot,0.897,960,0.040,0.893,0.874,0.913,above
qwen3-1.7b,pairs,0shot,0.766,64,0.125,0.732,0.607,0.848,above
qwen3-1.7b,pairs,fair,0.836,64,0.125,0.812,0.705,0.911,above
qwen3-1.7b,noise,0shot,0.910,720,0.200,0.888,0.862,0.913,above
qwen3-1.7b,now,0shot,0.852,300,0.292,0.787,0.728,0.838,above
qwen3-1.7b,handoffs,0shot,0.730,1200,0.312,0.604,0.565,0.640,above
qwen3-1.7b,handoffs,fair,0.785,1200,0.312,0.683,0.650,0.715,above
qwen3-1.7b,sets,0shot,0.788,320,0.275,0.714,0.655,0.770,above
qwen3-1.7b,sets,fair,0.767,320,0.275,0.689,0.628,0.751,above
qwen3-1.7b,refilled,0shot,0.626,720,0.200,0.533,0.490,0.576,above
qwen3-1.7b,refills,0shot,0.363,600,0.200,0.204,0.156,0.251,above
qwen3-1.7b,codevar,0shot,0.475,1200,0.267,0.280,0.244,0.321,above
qwen3-1.7b,codevar,fair,0.535,1200,0.267,0.361,0.322,0.405,above
qwen3-1.7b,sharedlists,0shot,0.372,600,0.200,0.215,0.167,0.258,above
qwen3-1.7b,sharedlists,fair,0.501,600,0.200,0.376,0.329,0.425,above
qwen3-1.7b,box-narrative,0shot,0.459,200,0.253,0.258,0.165,0.355,above
qwen3-1.7b,box-narrative,fair,0.515,200,0.253,0.324,0.229,0.416,above
qwen3-1.7b,box-screenplay,0shot,0.605,200,0.253,0.475,0.386,0.564,above
qwen3-1.7b,box-screenplay,fair,0.517,200,0.253,0.332,0.232,0.419,above
qwen3-1.7b,box-code,0shot,0.600,200,0.253,0.445,0.347,0.536,above
qwen3-1.7b,box-code,fair,0.695,200,0.253,0.567,0.474,0.651,above
qwen3-1.7b,box-chat,0shot,0.650,200,0.253,0.530,0.435,0.623,above
qwen3-1.7b,box-chat,fair,0.555,200,0.253,0.393,0.295,0.488,above
qwen3-1.7b,you-static,0shot,0.667,300,0.292,0.512,0.434,0.586,above
qwen3-1.7b,i-moves,0shot,0.450,180,0.287,0.228,0.131,0.344,above
qwen3-1.7b,i-moves,fair,0.428,180,0.287,0.197,0.093,0.295,above
qwen3-1.7b,you-moves,0shot,0.239,180,0.287,-0.072,-0.157,0.005,at
qwen3-1.7b,you-moves,fair,0.360,180,0.287,0.106,0.009,0.208,above
qwen3-1.7b,you-named,0shot,0.306,180,0.287,0.025,-0.064,0.121,at
qwen3-1.7b,you-named,fair,0.332,180,0.287,0.062,-0.029,0.153,at
qwen3-1.7b,answer-self,0shot,0.372,2560,0.250,0.163,0.140,0.186,above
qwen3-1.7b,answer-self,fair,0.401,2560,0.250,0.202,0.176,0.226,above
qwen3-1.7b,grammar,ladder,,1080,,0.819,0.800,0.839,above
qwen3-1.7b,operators,ladder,,1080,,0.271,0.233,0.299,above
qwen3-1.7b,patterns,ladder,,1080,,0.833,0.812,0.852,above
qwen3-1.7b,facts,ladder,,768,,0.994,0.986,0.998,above
gemma-2b,lookup,0shot,0.570,1210,0.168,0.470,0.434,0.503,above
gemma-2b,lookup,fair,0.967,1210,0.168,0.960,0.946,0.972,above
gemma-2b,shortlist,0shot,0.844,960,0.040,0.837,0.812,0.859,above
gemma-2b,pairs,0shot,0.562,64,0.125,0.500,0.375,0.634,above
gemma-2b,pairs,fair,0.516,64,0.125,0.446,0.304,0.571,above
gemma-2b,noise,0shot,0.583,720,0.200,0.478,0.436,0.521,above
gemma-2b,now,0shot,0.422,300,0.292,0.181,0.101,0.260,above
gemma-2b,handoffs,0shot,0.658,1200,0.312,0.497,0.458,0.535,above
gemma-2b,handoffs,fair,0.668,1200,0.312,0.513,0.473,0.553,above
gemma-2b,sets,0shot,0.506,320,0.275,0.316,0.247,0.394,above
gemma-2b,sets,fair,0.642,320,0.275,0.510,0.438,0.584,above
gemma-2b,refilled,0shot,0.422,720,0.200,0.278,0.233,0.324,above
gemma-2b,refills,0shot,0.447,600,0.200,0.309,0.260,0.357,above
gemma-2b,codevar,0shot,0.403,1200,0.267,0.183,0.146,0.220,above
gemma-2b,codevar,fair,0.416,1200,0.267,0.197,0.159,0.236,above
gemma-2b,sharedlists,0shot,0.330,600,0.200,0.163,0.117,0.207,above
gemma-2b,sharedlists,fair,0.417,600,0.200,0.272,0.222,0.321,above
gemma-2b,box-narrative,0shot,0.378,200,0.253,0.151,0.060,0.245,above
gemma-2b,box-narrative,fair,0.458,200,0.253,0.245,0.141,0.339,above
gemma-2b,box-screenplay,0shot,0.500,200,0.253,0.311,0.214,0.407,above
gemma-2b,box-screenplay,fair,0.443,200,0.253,0.212,0.113,0.308,above
gemma-2b,box-code,0shot,0.542,200,0.253,0.374,0.271,0.465,above
gemma-2b,box-code,fair,0.647,200,0.253,0.503,0.407,0.595,above
gemma-2b,box-chat,0shot,0.555,200,0.253,0.401,0.308,0.494,above
gemma-2b,box-chat,fair,0.608,200,0.253,0.463,0.370,0.554,above
gemma-2b,you-static,0shot,0.622,300,0.292,0.462,0.387,0.539,above
gemma-2b,i-moves,0shot,0.361,180,0.287,0.098,0.004,0.200,above
gemma-2b,i-moves,fair,0.344,180,0.287,0.078,-0.019,0.182,at
gemma-2b,you-moves,0shot,0.236,180,0.287,-0.071,-0.155,0.023,at
gemma-2b,you-moves,fair,0.231,180,0.287,-0.081,-0.163,0.009,at
gemma-2b,you-named,0shot,0.303,180,0.287,0.020,-0.069,0.113,at
gemma-2b,you-named,fair,0.303,180,0.287,0.020,-0.070,0.115,at
gemma-2b,answer-self,0shot,0.328,2560,0.250,0.104,0.080,0.126,above
gemma-2b,answer-self,fair,0.316,2560,0.250,0.087,0.064,0.110,above
gemma-2b,grammar,ladder,,1080,,0.727,0.702,0.755,above
gemma-2b,operators,ladder,,1080,,0.197,0.169,0.224,above
gemma-2b,patterns,ladder,,1080,,0.753,0.728,0.776,above
gemma-2b,facts,ladder,,768,,0.997,0.992,1.000,above
gemma2-2b,lookup,0shot,0.520,1210,0.168,0.409,0.374,0.444,above
gemma2-2b,lookup,fair,0.996,1210,0.168,0.995,0.991,0.999,above
gemma2-2b,shortlist,0shot,0.826,960,0.040,0.819,0.795,0.842,above
gemma2-2b,pairs,0shot,0.435,64,0.125,0.354,0.220,0.491,above
gemma2-2b,pairs,fair,0.836,64,0.125,0.812,0.705,0.911,above
gemma2-2b,noise,0shot,0.665,720,0.200,0.581,0.538,0.625,above
gemma2-2b,now,0shot,0.498,300,0.292,0.288,0.217,0.362,above
gemma2-2b,handoffs,0shot,0.689,1200,0.312,0.543,0.507,0.581,above
gemma2-2b,handoffs,fair,0.778,1200,0.312,0.673,0.640,0.707,above
gemma2-2b,sets,0shot,0.455,320,0.275,0.242,0.170,0.322,above
gemma2-2b,sets,fair,0.647,320,0.275,0.520,0.453,0.591,above
gemma2-2b,refilled,0shot,0.529,720,0.200,0.412,0.365,0.458,above
gemma2-2b,refills,0shot,0.491,600,0.200,0.364,0.312,0.414,above
gemma2-2b,codevar,0shot,0.431,1200,0.267,0.221,0.183,0.260,above
gemma2-2b,codevar,fair,0.414,1200,0.267,0.195,0.159,0.231,above
gemma2-2b,sharedlists,0shot,0.415,600,0.200,0.268,0.223,0.319,above
gemma2-2b,sharedlists,fair,0.489,600,0.200,0.361,0.315,0.406,above
gemma2-2b,box-narrative,0shot,0.470,200,0.253,0.280,0.186,0.369,above
gemma2-2b,box-narrative,fair,0.463,200,0.253,0.266,0.168,0.367,above
gemma2-2b,box-screenplay,0shot,0.512,200,0.253,0.344,0.255,0.435,above
gemma2-2b,box-screenplay,fair,0.515,200,0.253,0.330,0.231,0.422,above
gemma2-2b,box-code,0shot,0.627,200,0.253,0.480,0.379,0.577,above
gemma2-2b,box-code,fair,0.715,200,0.253,0.603,0.507,0.691,above
gemma2-2b,box-chat,0shot,0.587,200,0.253,0.441,0.352,0.533,above
gemma2-2b,box-chat,fair,0.597,200,0.253,0.453,0.357,0.537,above
gemma2-2b,you-static,0shot,0.763,300,0.292,0.660,0.591,0.720,above
gemma2-2b,i-moves,0shot,0.431,180,0.287,0.202,0.105,0.297,above
gemma2-2b,i-moves,fair,0.364,180,0.287,0.109,0.016,0.206,above
gemma2-2b,you-moves,0shot,0.242,180,0.287,-0.061,-0.143,0.026,at
gemma2-2b,you-moves,fair,0.257,180,0.287,-0.043,-0.124,0.042,at
gemma2-2b,you-named,0shot,0.394,180,0.287,0.146,0.047,0.249,above
gemma2-2b,you-named,fair,0.323,180,0.287,0.045,-0.046,0.139,at
gemma2-2b,answer-self,0shot,0.305,2560,0.250,0.073,0.050,0.098,above
gemma2-2b,answer-self,fair,0.361,2560,0.250,0.148,0.125,0.172,above
gemma2-2b,grammar,ladder,,1080,,0.801,0.775,0.828,above
gemma2-2b,operators,ladder,,1080,,0.274,0.241,0.306,above
gemma2-2b,patterns,ladder,,1080,,0.719,0.694,0.746,above
gemma2-2b,facts,ladder,,768,,1.000,1.000,1.000,above
pythia-2.8b,lookup,0shot,0.619,1210,0.168,0.534,0.500,0.568,above
pythia-2.8b,lookup,fair,0.861,1210,0.168,0.828,0.805,0.851,above
pythia-2.8b,shortlist,0shot,0.559,960,0.040,0.541,0.508,0.573,above
pythia-2.8b,pairs,0shot,0.664,64,0.125,0.616,0.482,0.732,above
pythia-2.8b,pairs,fair,0.539,64,0.125,0.473,0.339,0.616,above
pythia-2.8b,noise,0shot,0.621,720,0.200,0.526,0.483,0.569,above
pythia-2.8b,now,0shot,0.713,300,0.292,0.592,0.522,0.658,above
pythia-2.8b,handoffs,0shot,0.578,1200,0.312,0.380,0.339,0.420,above
pythia-2.8b,handoffs,fair,0.587,1200,0.312,0.394,0.354,0.435,above
pythia-2.8b,sets,0shot,0.450,320,0.275,0.247,0.170,0.320,above
pythia-2.8b,sets,fair,0.512,320,0.275,0.337,0.255,0.407,above
pythia-2.8b,refilled,0shot,0.319,720,0.200,0.148,0.107,0.191,above
pythia-2.8b,refills,0shot,0.228,600,0.200,0.034,-0.008,0.076,at
pythia-2.8b,codevar,0shot,0.332,1200,0.267,0.084,0.047,0.120,above
pythia-2.8b,codevar,fair,0.285,1200,0.267,0.022,-0.014,0.057,at
pythia-2.8b,sharedlists,0shot,0.309,600,0.200,0.137,0.094,0.185,above
pythia-2.8b,sharedlists,fair,0.287,600,0.200,0.108,0.064,0.152,above
pythia-2.8b,box-narrative,0shot,0.450,200,0.253,0.242,0.148,0.340,above
pythia-2.8b,box-narrative,fair,0.532,200,0.253,0.376,0.286,0.467,above
pythia-2.8b,box-screenplay,0shot,0.407,200,0.253,0.216,0.131,0.310,above
pythia-2.8b,box-screenplay,fair,0.520,200,0.253,0.352,0.262,0.445,above
pythia-2.8b,box-code,0shot,0.573,200,0.253,0.422,0.327,0.513,above
pythia-2.8b,box-code,fair,0.605,200,0.253,0.455,0.357,0.550,above
pythia-2.8b,box-chat,0shot,0.463,200,0.253,0.269,0.171,0.363,above
pythia-2.8b,box-chat,fair,0.537,200,0.253,0.378,0.284,0.476,above
pythia-2.8b,you-static,0shot,0.438,300,0.292,0.203,0.126,0.282,above
pythia-2.8b,i-moves,0shot,0.264,180,0.287,-0.038,-0.129,0.049,at
pythia-2.8b,i-moves,fair,0.319,180,0.287,0.041,-0.049,0.135,at
pythia-2.8b,you-moves,0shot,0.206,180,0.287,-0.114,-0.192,-0.032,below
pythia-2.8b,you-moves,fair,0.310,180,0.287,0.032,-0.055,0.129,at
pythia-2.8b,you-named,0shot,0.331,180,0.287,0.062,-0.025,0.163,at
pythia-2.8b,you-named,fair,0.303,180,0.287,0.019,-0.069,0.114,at
pythia-2.8b,answer-self,0shot,0.332,2560,0.250,0.109,0.085,0.133,above
pythia-2.8b,answer-self,fair,0.298,2560,0.250,0.064,0.041,0.086,above
pythia-2.8b,grammar,ladder,,1080,,0.798,0.772,0.824,above
pythia-2.8b,operators,ladder,,1080,,0.252,0.221,0.284,above
pythia-2.8b,patterns,ladder,,1080,,0.583,0.553,0.615,above
pythia-2.8b,facts,ladder,,768,,0.984,0.972,0.992,above
pythia-2.8b-deduped,lookup,0shot,0.625,1210,0.168,0.543,0.510,0.576,above
pythia-2.8b-deduped,lookup,fair,0.883,1210,0.168,0.855,0.833,0.876,above
pythia-2.8b-deduped,shortlist,0shot,0.578,960,0.040,0.560,0.528,0.590,above
pythia-2.8b-deduped,pairs,0shot,0.648,64,0.125,0.598,0.464,0.723,above
pythia-2.8b-deduped,pairs,fair,0.594,64,0.125,0.536,0.393,0.661,above
pythia-2.8b-deduped,noise,0shot,0.634,720,0.200,0.542,0.499,0.585,above
pythia-2.8b-deduped,now,0shot,0.713,300,0.292,0.592,0.521,0.663,above
pythia-2.8b-deduped,handoffs,0shot,0.574,1200,0.312,0.374,0.335,0.415,above
pythia-2.8b-deduped,handoffs,fair,0.586,1200,0.312,0.393,0.352,0.434,above
pythia-2.8b-deduped,sets,0shot,0.467,320,0.275,0.273,0.191,0.349,above
pythia-2.8b-deduped,sets,fair,0.496,320,0.275,0.309,0.231,0.384,above
pythia-2.8b-deduped,refilled,0shot,0.306,720,0.200,0.133,0.092,0.175,above
pythia-2.8b-deduped,refills,0shot,0.223,600,0.200,0.028,-0.011,0.069,at
pythia-2.8b-deduped,codevar,0shot,0.347,1200,0.267,0.104,0.068,0.141,above
pythia-2.8b-deduped,codevar,fair,0.267,1200,0.267,-0.003,-0.036,0.032,at
pythia-2.8b-deduped,sharedlists,0shot,0.279,600,0.200,0.099,0.052,0.144,above
pythia-2.8b-deduped,sharedlists,fair,0.291,600,0.200,0.114,0.069,0.156,above
pythia-2.8b-deduped,box-narrative,0shot,0.430,200,0.253,0.213,0.120,0.307,above
pythia-2.8b-deduped,box-narrative,fair,0.535,200,0.253,0.378,0.284,0.464,above
pythia-2.8b-deduped,box-screenplay,0shot,0.440,200,0.253,0.254,0.163,0.345,above
pythia-2.8b-deduped,box-screenplay,fair,0.525,200,0.253,0.367,0.278,0.454,above
pythia-2.8b-deduped,box-code,0shot,0.550,200,0.253,0.393,0.296,0.486,above
pythia-2.8b-deduped,box-code,fair,0.578,200,0.253,0.418,0.325,0.512,above
pythia-2.8b-deduped,box-chat,0shot,0.470,200,0.253,0.277,0.185,0.382,above
pythia-2.8b-deduped,box-chat,fair,0.522,200,0.253,0.354,0.259,0.450,above
pythia-2.8b-deduped,you-static,0shot,0.415,300,0.292,0.170,0.090,0.244,above
pythia-2.8b-deduped,i-moves,0shot,0.275,180,0.287,-0.025,-0.115,0.069,at
pythia-2.8b-deduped,i-moves,fair,0.327,180,0.287,0.056,-0.032,0.147,at
pythia-2.8b-deduped,you-moves,0shot,0.247,180,0.287,-0.057,-0.139,0.028,at
pythia-2.8b-deduped,you-moves,fair,0.303,180,0.287,0.020,-0.069,0.121,at
pythia-2.8b-deduped,you-named,0shot,0.311,180,0.287,0.033,-0.057,0.128,at
pythia-2.8b-deduped,you-named,fair,0.297,180,0.287,0.010,-0.088,0.107,at
pythia-2.8b-deduped,answer-self,0shot,0.327,2560,0.250,0.102,0.078,0.124,above
pythia-2.8b-deduped,answer-self,fair,0.287,2560,0.250,0.050,0.028,0.074,above
pythia-2.8b-deduped,grammar,ladder,,1080,,0.800,0.777,0.824,above
pythia-2.8b-deduped,operators,ladder,,1080,,0.273,0.240,0.303,above
pythia-2.8b-deduped,patterns,ladder,,1080,,0.609,0.577,0.644,above
pythia-2.8b-deduped,facts,ladder,,768,,0.987,0.977,0.994,above
qwen2.5-3b,lookup,0shot,0.708,1210,0.168,0.632,0.599,0.662,above
qwen2.5-3b,lookup,fair,0.995,1210,0.168,0.994,0.989,0.998,above
qwen2.5-3b,shortlist,0shot,0.731,960,0.040,0.720,0.690,0.750,above
qwen2.5-3b,pairs,0shot,0.766,64,0.125,0.732,0.616,0.848,above
qwen2.5-3b,pairs,fair,0.906,64,0.125,0.893,0.804,0.964,above
qwen2.5-3b,noise,0shot,0.726,720,0.200,0.657,0.617,0.696,above
qwen2.5-3b,now,0shot,0.820,300,0.292,0.745,0.685,0.802,above
qwen2.5-3b,handoffs,0shot,0.680,1200,0.312,0.533,0.493,0.570,above
qwen2.5-3b,handoffs,fair,0.757,1200,0.312,0.645,0.611,0.678,above
qwen2.5-3b,sets,0shot,0.754,320,0.275,0.656,0.593,0.725,above
qwen2.5-3b,sets,fair,0.644,320,0.275,0.516,0.445,0.587,above
qwen2.5-3b,refilled,0shot,0.438,720,0.200,0.297,0.251,0.341,above
qwen2.5-3b,refills,0shot,0.303,600,0.200,0.129,0.085,0.173,above
qwen2.5-3b,codevar,0shot,0.446,1200,0.267,0.239,0.203,0.277,above
qwen2.5-3b,codevar,fair,0.469,1200,0.267,0.267,0.229,0.308,above
qwen2.5-3b,sharedlists,0shot,0.371,600,0.200,0.214,0.169,0.260,above
qwen2.5-3b,sharedlists,fair,0.438,600,0.200,0.297,0.250,0.345,above
qwen2.5-3b,box-narrative,0shot,0.349,200,0.253,0.093,0.001,0.186,above
qwen2.5-3b,box-narrative,fair,0.618,200,0.253,0.472,0.372,0.567,above
qwen2.5-3b,box-screenplay,0shot,0.472,200,0.253,0.267,0.171,0.366,above
qwen2.5-3b,box-screenplay,fair,0.652,200,0.253,0.517,0.426,0.608,above
qwen2.5-3b,box-code,0shot,0.623,200,0.253,0.466,0.356,0.559,above
qwen2.5-3b,box-code,fair,0.665,200,0.253,0.526,0.433,0.616,above
qwen2.5-3b,box-chat,0shot,0.482,200,0.253,0.281,0.191,0.381,above
qwen2.5-3b,box-chat,fair,0.685,200,0.253,0.577,0.488,0.658,above
qwen2.5-3b,you-static,0shot,0.878,300,0.292,0.824,0.772,0.875,above
qwen2.5-3b,i-moves,0shot,0.333,180,0.287,0.068,-0.028,0.170,at
qwen2.5-3b,i-moves,fair,0.358,180,0.287,0.108,0.015,0.209,above
qwen2.5-3b,you-moves,0shot,0.251,180,0.287,-0.051,-0.136,0.038,at
qwen2.5-3b,you-moves,fair,0.247,180,0.287,-0.056,-0.138,0.038,at
qwen2.5-3b,you-named,0shot,0.239,180,0.287,-0.071,-0.156,0.014,at
qwen2.5-3b,you-named,fair,0.288,180,0.287,-0.003,-0.099,0.093,at
qwen2.5-3b,answer-self,0shot,0.352,2560,0.250,0.135,0.112,0.158,above
qwen2.5-3b,answer-self,fair,0.344,2560,0.250,0.125,0.102,0.150,above
qwen2.5-3b,grammar,ladder,,1080,,0.792,0.770,0.818,above
qwen2.5-3b,operators,ladder,,1080,,0.183,0.157,0.218,above
qwen2.5-3b,patterns,ladder,,1080,,0.897,0.879,0.914,above
qwen2.5-3b,facts,ladder,,768,,0.998,0.995,1.000,above
falcon-h1-3b,lookup,0shot,0.950,1210,0.168,0.937,0.920,0.952,above
falcon-h1-3b,lookup,fair,0.993,1210,0.168,0.990,0.984,0.996,above
falcon-h1-3b,shortlist,0shot,0.904,960,0.040,0.900,0.882,0.919,above
falcon-h1-3b,pairs,0shot,1.000,64,0.125,1.000,1.000,1.000,above
falcon-h1-3b,pairs,fair,0.891,64,0.125,0.875,0.786,0.946,above
falcon-h1-3b,noise,0shot,0.749,720,0.200,0.686,0.648,0.724,above
falcon-h1-3b,now,0shot,0.593,300,0.292,0.421,0.344,0.498,above
falcon-h1-3b,handoffs,0shot,0.752,1200,0.312,0.637,0.605,0.673,above
falcon-h1-3b,handoffs,fair,0.860,1200,0.312,0.794,0.767,0.820,above
falcon-h1-3b,sets,0shot,0.927,320,0.275,0.901,0.861,0.939,above
falcon-h1-3b,sets,fair,0.791,320,0.275,0.721,0.661,0.778,above
falcon-h1-3b,refilled,0shot,0.763,720,0.200,0.704,0.661,0.744,above
falcon-h1-3b,refills,0shot,0.470,600,0.200,0.338,0.291,0.391,above
falcon-h1-3b,codevar,0shot,0.548,1200,0.267,0.376,0.338,0.416,above
falcon-h1-3b,codevar,fair,0.683,1200,0.267,0.562,0.524,0.599,above
falcon-h1-3b,sharedlists,0shot,0.548,600,0.200,0.435,0.383,0.483,above
falcon-h1-3b,sharedlists,fair,0.551,600,0.200,0.439,0.391,0.485,above
falcon-h1-3b,box-narrative,0shot,0.400,200,0.253,0.141,0.044,0.240,above
falcon-h1-3b,box-narrative,fair,0.505,200,0.253,0.299,0.188,0.394,above
falcon-h1-3b,box-screenplay,0shot,0.547,200,0.253,0.367,0.272,0.470,above
falcon-h1-3b,box-screenplay,fair,0.620,200,0.253,0.478,0.381,0.577,above
falcon-h1-3b,box-code,0shot,0.545,200,0.253,0.389,0.294,0.485,above
falcon-h1-3b,box-code,fair,0.765,200,0.253,0.679,0.594,0.756,above
falcon-h1-3b,box-chat,0shot,0.645,200,0.253,0.516,0.426,0.611,above
falcon-h1-3b,box-chat,fair,0.655,200,0.253,0.540,0.453,0.634,above
falcon-h1-3b,you-static,0shot,0.940,300,0.292,0.913,0.874,0.952,above
falcon-h1-3b,i-moves,0shot,0.333,180,0.287,0.060,-0.040,0.163,at
falcon-h1-3b,i-moves,fair,0.464,180,0.287,0.251,0.149,0.361,above
falcon-h1-3b,you-moves,0shot,0.261,180,0.287,-0.032,-0.116,0.057,at
falcon-h1-3b,you-moves,fair,0.289,180,0.287,0.003,-0.087,0.095,at
falcon-h1-3b,you-named,0shot,0.294,180,0.287,0.005,-0.082,0.102,at
falcon-h1-3b,you-named,fair,0.278,180,0.287,-0.018,-0.108,0.074,at
falcon-h1-3b,answer-self,0shot,0.388,2560,0.250,0.184,0.157,0.208,above
falcon-h1-3b,answer-self,fair,0.430,2560,0.250,0.241,0.216,0.267,above
falcon-h1-3b,grammar,ladder,,1080,,0.836,0.814,0.857,above
falcon-h1-3b,operators,ladder,,1080,,0.262,0.230,0.299,above
falcon-h1-3b,patterns,ladder,,1080,,0.768,0.746,0.792,above
falcon-h1-3b,facts,ladder,,768,,0.984,0.977,0.991,above
falcon3-3b,lookup,0shot,0.558,1210,0.168,0.460,0.425,0.492,above
falcon3-3b,lookup,fair,0.962,1210,0.168,0.953,0.939,0.965,above
falcon3-3b,shortlist,0shot,0.574,960,0.040,0.556,0.526,0.588,above
falcon3-3b,pairs,0shot,0.492,64,0.125,0.420,0.295,0.545,above
falcon3-3b,pairs,fair,0.625,64,0.125,0.571,0.429,0.696,above
falcon3-3b,noise,0shot,0.685,720,0.200,0.606,0.562,0.652,above
falcon3-3b,now,0shot,0.753,300,0.292,0.649,0.579,0.718,above
falcon3-3b,handoffs,0shot,0.725,1200,0.312,0.594,0.559,0.632,above
falcon3-3b,handoffs,fair,0.771,1200,0.312,0.664,0.630,0.697,above
falcon3-3b,sets,0shot,0.439,320,0.275,0.230,0.164,0.301,above
falcon3-3b,sets,fair,0.470,320,0.275,0.280,0.210,0.356,above
falcon3-3b,refilled,0shot,0.446,720,0.200,0.307,0.260,0.354,above
falcon3-3b,refills,0shot,0.350,600,0.200,0.187,0.142,0.234,above
falcon3-3b,codevar,0shot,0.436,1200,0.267,0.222,0.184,0.260,above
falcon3-3b,codevar,fair,0.578,1200,0.267,0.420,0.381,0.456,above
falcon3-3b,sharedlists,0shot,0.357,600,0.200,0.196,0.147,0.242,above
falcon3-3b,sharedlists,fair,0.578,600,0.200,0.472,0.422,0.523,above
falcon3-3b,box-narrative,0shot,0.400,200,0.253,0.197,0.104,0.292,above
falcon3-3b,box-narrative,fair,0.439,200,0.253,0.230,0.136,0.327,above
falcon3-3b,box-screenplay,0shot,0.540,200,0.253,0.386,0.286,0.479,above
falcon3-3b,box-screenplay,fair,0.547,200,0.253,0.387,0.283,0.474,above
falcon3-3b,box-code,0shot,0.415,200,0.253,0.226,0.131,0.317,above
falcon3-3b,box-code,fair,0.735,200,0.253,0.623,0.531,0.707,above
falcon3-3b,box-chat,0shot,0.570,200,0.253,0.421,0.328,0.520,above
falcon3-3b,box-chat,fair,0.682,200,0.253,0.585,0.489,0.668,above
falcon3-3b,you-static,0shot,0.825,300,0.292,0.748,0.685,0.803,above
falcon3-3b,i-moves,0shot,0.274,180,0.287,-0.027,-0.114,0.068,at
falcon3-3b,i-moves,fair,0.360,180,0.287,0.103,0.007,0.206,above
falcon3-3b,you-moves,0shot,0.256,180,0.287,-0.045,-0.135,0.047,at
falcon3-3b,you-moves,fair,0.296,180,0.287,0.014,-0.079,0.113,at
falcon3-3b,you-named,0shot,0.314,180,0.287,0.038,-0.043,0.131,at
falcon3-3b,you-named,fair,0.291,180,0.287,0.001,-0.086,0.088,at
falcon3-3b,answer-self,0shot,0.334,2560,0.250,0.112,0.088,0.135,above
falcon3-3b,answer-self,fair,0.349,2560,0.250,0.132,0.108,0.156,above
falcon3-3b,grammar,ladder,,1080,,0.851,0.833,0.875,above
falcon3-3b,operators,ladder,,1080,,0.237,0.206,0.267,above
falcon3-3b,patterns,ladder,,1080,,0.787,0.764,0.810,above
falcon3-3b,facts,ladder,,768,,0.968,0.952,0.984,above
qwen3-30b-a3b,lookup,0shot,0.842,1210,0.168,0.800,0.775,0.825,above
qwen3-30b-a3b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
qwen3-30b-a3b,shortlist,0shot,0.766,960,0.040,0.756,0.729,0.784,above
qwen3-30b-a3b,pairs,0shot,0.906,64,0.125,0.893,0.804,0.964,above
qwen3-30b-a3b,pairs,fair,0.977,64,0.125,0.973,0.929,1.000,above
qwen3-30b-a3b,noise,0shot,0.812,720,0.200,0.766,0.730,0.802,above
qwen3-30b-a3b,now,0shot,0.387,300,0.292,0.132,0.056,0.214,above
qwen3-30b-a3b,handoffs,0shot,0.821,1200,0.312,0.736,0.704,0.766,above
qwen3-30b-a3b,handoffs,fair,0.904,1200,0.312,0.858,0.833,0.881,above
qwen3-30b-a3b,sets,0shot,0.828,320,0.275,0.754,0.696,0.814,above
qwen3-30b-a3b,sets,fair,0.866,320,0.275,0.815,0.760,0.867,above
qwen3-30b-a3b,refilled,0shot,0.493,720,0.200,0.366,0.324,0.411,above
qwen3-30b-a3b,refills,0shot,0.461,600,0.200,0.327,0.278,0.376,above
qwen3-30b-a3b,codevar,0shot,0.757,1200,0.267,0.665,0.632,0.697,above
qwen3-30b-a3b,codevar,fair,0.832,1200,0.267,0.765,0.735,0.794,above
qwen3-30b-a3b,sharedlists,0shot,0.775,600,0.200,0.719,0.676,0.759,above
qwen3-30b-a3b,sharedlists,fair,0.794,600,0.200,0.743,0.702,0.781,above
qwen3-30b-a3b,box-narrative,0shot,0.367,200,0.253,0.128,0.032,0.226,above
qwen3-30b-a3b,box-narrative,fair,0.515,200,0.253,0.324,0.226,0.426,above
qwen3-30b-a3b,box-screenplay,0shot,0.635,200,0.253,0.501,0.409,0.587,above
qwen3-30b-a3b,box-screenplay,fair,0.597,200,0.253,0.439,0.344,0.526,above
qwen3-30b-a3b,box-code,0shot,0.733,200,0.253,0.624,0.527,0.707,above
qwen3-30b-a3b,box-code,fair,0.855,200,0.253,0.803,0.735,0.863,above
qwen3-30b-a3b,box-chat,0shot,0.728,200,0.253,0.624,0.541,0.709,above
qwen3-30b-a3b,box-chat,fair,0.688,200,0.253,0.574,0.487,0.661,above
qwen3-30b-a3b,you-static,0shot,0.963,300,0.292,0.945,0.910,0.975,above
qwen3-30b-a3b,i-moves,0shot,0.378,180,0.287,0.124,0.028,0.222,above
qwen3-30b-a3b,i-moves,fair,0.528,180,0.287,0.340,0.235,0.446,above
qwen3-30b-a3b,you-moves,0shot,0.317,180,0.287,0.044,-0.038,0.138,at
qwen3-30b-a3b,you-moves,fair,0.357,180,0.287,0.101,0.009,0.204,above
qwen3-30b-a3b,you-named,0shot,0.352,180,0.287,0.087,-0.001,0.185,at
qwen3-30b-a3b,you-named,fair,0.511,180,0.287,0.307,0.206,0.413,above
qwen3-30b-a3b,answer-self,0shot,0.440,2560,0.250,0.253,0.228,0.278,above
qwen3-30b-a3b,answer-self,fair,0.486,2560,0.250,0.314,0.289,0.340,above
qwen3-30b-a3b,grammar,ladder,,1080,,0.844,0.821,0.863,above
qwen3-30b-a3b,operators,ladder,,1080,,0.208,0.181,0.240,above
qwen3-30b-a3b,patterns,ladder,,1080,,0.990,0.983,0.997,above
qwen3-30b-a3b,facts,ladder,,768,,0.995,0.989,1.000,above
qwen3-4b,lookup,0shot,0.975,1210,0.168,0.966,0.954,0.977,above
qwen3-4b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
qwen3-4b,shortlist,0shot,0.886,960,0.040,0.881,0.862,0.901,above
qwen3-4b,pairs,0shot,1.000,64,0.125,1.000,1.000,1.000,above
qwen3-4b,pairs,fair,0.969,64,0.125,0.964,0.911,1.000,above
qwen3-4b,noise,0shot,0.830,720,0.200,0.788,0.754,0.820,above
qwen3-4b,now,0shot,0.807,300,0.292,0.720,0.658,0.782,above
qwen3-4b,handoffs,0shot,0.877,1200,0.312,0.818,0.790,0.845,above
qwen3-4b,handoffs,fair,0.870,1200,0.312,0.807,0.781,0.834,above
qwen3-4b,sets,0shot,0.884,320,0.275,0.838,0.787,0.886,above
qwen3-4b,sets,fair,0.798,320,0.275,0.727,0.667,0.785,above
qwen3-4b,refilled,0shot,0.592,720,0.200,0.490,0.443,0.535,above
qwen3-4b,refills,0shot,0.505,600,0.200,0.382,0.333,0.429,above
qwen3-4b,codevar,0shot,0.619,1200,0.267,0.477,0.438,0.515,above
qwen3-4b,codevar,fair,0.635,1200,0.267,0.496,0.457,0.533,above
qwen3-4b,sharedlists,0shot,0.529,600,0.200,0.411,0.363,0.458,above
qwen3-4b,sharedlists,fair,0.694,600,0.200,0.618,0.572,0.660,above
qwen3-4b,box-narrative,0shot,0.315,200,0.253,0.055,-0.037,0.152,at
qwen3-4b,box-narrative,fair,0.497,200,0.253,0.301,0.211,0.398,above
qwen3-4b,box-screenplay,0shot,0.582,200,0.253,0.425,0.333,0.521,above
qwen3-4b,box-screenplay,fair,0.507,200,0.253,0.324,0.229,0.417,above
qwen3-4b,box-code,0shot,0.723,200,0.253,0.601,0.513,0.691,above
qwen3-4b,box-code,fair,0.802,200,0.253,0.720,0.637,0.793,above
qwen3-4b,box-chat,0shot,0.623,200,0.253,0.482,0.388,0.580,above
qwen3-4b,box-chat,fair,0.570,200,0.253,0.417,0.324,0.511,above
qwen3-4b,you-static,0shot,0.784,300,0.292,0.680,0.608,0.744,above
qwen3-4b,i-moves,0shot,0.450,180,0.287,0.226,0.124,0.330,above
qwen3-4b,i-moves,fair,0.533,180,0.287,0.347,0.250,0.448,above
qwen3-4b,you-moves,0shot,0.335,180,0.287,0.063,-0.030,0.160,at
qwen3-4b,you-moves,fair,0.431,180,0.287,0.199,0.096,0.302,above
qwen3-4b,you-named,0shot,0.400,180,0.287,0.152,0.055,0.256,above
qwen3-4b,you-named,fair,0.360,180,0.287,0.098,0.003,0.206,above
qwen3-4b,answer-self,0shot,0.460,2560,0.250,0.279,0.254,0.306,above
qwen3-4b,answer-self,fair,0.564,2560,0.250,0.418,0.393,0.443,above
qwen3-4b,grammar,ladder,,1080,,0.846,0.826,0.869,above
qwen3-4b,operators,ladder,,1080,,0.218,0.191,0.254,above
qwen3-4b,patterns,ladder,,1080,,0.947,0.934,0.962,above
qwen3-4b,facts,ladder,,768,,0.998,0.995,1.000,above
pythia-6.9b,lookup,0shot,0.520,1210,0.168,0.413,0.378,0.448,above
pythia-6.9b,lookup,fair,0.827,1210,0.168,0.793,0.767,0.818,above
pythia-6.9b,shortlist,0shot,0.588,960,0.040,0.571,0.538,0.603,above
pythia-6.9b,pairs,0shot,0.492,64,0.125,0.420,0.286,0.562,above
pythia-6.9b,pairs,fair,0.570,64,0.125,0.509,0.375,0.643,above
pythia-6.9b,noise,0shot,0.624,720,0.200,0.529,0.488,0.571,above
pythia-6.9b,now,0shot,0.590,300,0.292,0.425,0.348,0.502,above
pythia-6.9b,handoffs,0shot,0.579,1200,0.312,0.383,0.344,0.423,above
pythia-6.9b,handoffs,fair,0.571,1200,0.312,0.372,0.332,0.411,above
pythia-6.9b,sets,0shot,0.486,320,0.275,0.297,0.228,0.376,above
pythia-6.9b,sets,fair,0.523,320,0.275,0.346,0.278,0.420,above
pythia-6.9b,refilled,0shot,0.350,720,0.200,0.188,0.143,0.228,above
pythia-6.9b,refills,0shot,0.258,600,0.200,0.073,0.031,0.118,above
pythia-6.9b,codevar,0shot,0.360,1200,0.267,0.120,0.084,0.157,above
pythia-6.9b,codevar,fair,0.320,1200,0.267,0.065,0.031,0.101,above
pythia-6.9b,sharedlists,0shot,0.279,600,0.200,0.099,0.056,0.146,above
pythia-6.9b,sharedlists,fair,0.212,600,0.200,0.016,-0.023,0.053,at
pythia-6.9b,box-narrative,0shot,0.463,200,0.253,0.267,0.169,0.360,above
pythia-6.9b,box-narrative,fair,0.507,200,0.253,0.343,0.242,0.434,above
pythia-6.9b,box-screenplay,0shot,0.537,200,0.253,0.381,0.276,0.467,above
pythia-6.9b,box-screenplay,fair,0.527,200,0.253,0.369,0.272,0.459,above
pythia-6.9b,box-code,0shot,0.458,200,0.253,0.275,0.178,0.355,above
pythia-6.9b,box-code,fair,0.560,200,0.253,0.390,0.294,0.484,above
pythia-6.9b,box-chat,0shot,0.515,200,0.253,0.338,0.241,0.433,above
pythia-6.9b,box-chat,fair,0.482,200,0.253,0.292,0.194,0.387,above
pythia-6.9b,you-static,0shot,0.470,300,0.292,0.248,0.169,0.328,above
pythia-6.9b,i-moves,0shot,0.325,180,0.287,0.048,-0.047,0.140,at
pythia-6.9b,i-moves,fair,0.311,180,0.287,0.028,-0.069,0.127,at
pythia-6.9b,you-moves,0shot,0.211,180,0.287,-0.111,-0.194,-0.032,below
pythia-6.9b,you-moves,fair,0.275,180,0.287,-0.023,-0.118,0.070,at
pythia-6.9b,you-named,0shot,0.358,180,0.287,0.097,0.001,0.197,above
pythia-6.9b,you-named,fair,0.369,180,0.287,0.118,0.015,0.214,above
pythia-6.9b,answer-self,0shot,0.315,2560,0.250,0.087,0.062,0.110,above
pythia-6.9b,answer-self,fair,0.292,2560,0.250,0.056,0.034,0.080,above
pythia-6.9b,grammar,ladder,,1080,,0.779,0.754,0.806,above
pythia-6.9b,operators,ladder,,1080,,0.215,0.186,0.248,above
pythia-6.9b,patterns,ladder,,1080,,0.496,0.460,0.525,above
pythia-6.9b,facts,ladder,,768,,0.995,0.989,0.998,above
falcon-mamba-7b,lookup,0shot,0.503,1210,0.168,0.389,0.359,0.422,above
falcon-mamba-7b,lookup,fair,0.981,1210,0.168,0.977,0.969,0.985,above
falcon-mamba-7b,shortlist,0shot,0.754,960,0.040,0.743,0.719,0.769,above
falcon-mamba-7b,pairs,0shot,0.526,64,0.125,0.459,0.330,0.582,above
falcon-mamba-7b,pairs,fair,0.232,64,0.125,0.123,0.028,0.228,above
falcon-mamba-7b,noise,0shot,0.656,720,0.200,0.570,0.534,0.606,above
falcon-mamba-7b,now,0shot,0.570,300,0.292,0.393,0.328,0.457,above
falcon-mamba-7b,handoffs,0shot,0.712,1200,0.312,0.574,0.539,0.610,above
falcon-mamba-7b,handoffs,fair,0.694,1200,0.312,0.550,0.517,0.585,above
falcon-mamba-7b,sets,0shot,0.422,320,0.275,0.207,0.143,0.268,above
falcon-mamba-7b,sets,fair,0.402,320,0.275,0.182,0.129,0.235,above
falcon-mamba-7b,refilled,0shot,0.472,720,0.200,0.339,0.307,0.376,above
falcon-mamba-7b,refills,0shot,0.288,600,0.200,0.111,0.070,0.149,above
falcon-mamba-7b,codevar,0shot,0.362,1200,0.267,0.121,0.090,0.152,above
falcon-mamba-7b,codevar,fair,0.383,1200,0.267,0.149,0.117,0.181,above
falcon-mamba-7b,sharedlists,0shot,0.245,600,0.200,0.056,0.020,0.094,above
falcon-mamba-7b,sharedlists,fair,0.263,600,0.200,0.079,0.043,0.116,above
falcon-mamba-7b,box-narrative,0shot,0.279,200,0.253,0.023,-0.052,0.098,at
falcon-mamba-7b,box-narrative,fair,0.343,200,0.253,0.104,0.025,0.176,above
falcon-mamba-7b,box-screenplay,0shot,0.422,200,0.253,0.204,0.118,0.286,above
falcon-mamba-7b,box-screenplay,fair,0.393,200,0.253,0.174,0.096,0.253,above
falcon-mamba-7b,box-code,0shot,0.472,200,0.253,0.289,0.187,0.382,above
falcon-mamba-7b,box-code,fair,0.613,200,0.253,0.479,0.392,0.558,above
falcon-mamba-7b,box-chat,0shot,0.540,200,0.253,0.376,0.296,0.458,above
falcon-mamba-7b,box-chat,fair,0.431,200,0.253,0.226,0.144,0.305,above
falcon-mamba-7b,you-static,0shot,0.791,300,0.292,0.704,0.647,0.758,above
falcon-mamba-7b,i-moves,0shot,0.353,180,0.287,0.090,0.019,0.168,above
falcon-mamba-7b,i-moves,fair,0.410,180,0.287,0.174,0.088,0.262,above
falcon-mamba-7b,you-moves,0shot,0.278,180,0.287,-0.015,-0.091,0.059,at
falcon-mamba-7b,you-moves,fair,0.236,180,0.287,-0.071,-0.144,0.004,at
falcon-mamba-7b,you-named,0shot,0.315,180,0.287,0.036,-0.039,0.118,at
falcon-mamba-7b,you-named,fair,0.256,180,0.287,-0.048,-0.124,0.022,at
falcon-mamba-7b,answer-self,0shot,0.288,2560,0.250,0.051,0.032,0.071,above
falcon-mamba-7b,answer-self,fair,0.285,2560,0.250,0.047,0.029,0.065,above
falcon-mamba-7b,grammar,ladder,,1080,,0.817,0.799,0.832,above
falcon-mamba-7b,operators,ladder,,1080,,0.275,0.253,0.301,above
falcon-mamba-7b,patterns,ladder,,1080,,0.801,0.783,0.824,above
falcon-mamba-7b,facts,ladder,,768,,0.998,0.995,1.000,above
olmo2-7b,lookup,0shot,0.459,1210,0.168,0.335,0.300,0.368,above
olmo2-7b,lookup,fair,0.981,1210,0.168,0.977,0.968,0.986,above
olmo2-7b,shortlist,0shot,0.885,960,0.040,0.881,0.860,0.900,above
olmo2-7b,pairs,0shot,0.367,64,0.125,0.277,0.152,0.411,above
olmo2-7b,pairs,fair,0.797,64,0.125,0.768,0.661,0.875,above
olmo2-7b,noise,0shot,0.647,720,0.200,0.559,0.519,0.602,above
olmo2-7b,now,0shot,0.517,300,0.292,0.316,0.237,0.393,above
olmo2-7b,handoffs,0shot,0.770,1200,0.312,0.660,0.624,0.695,above
olmo2-7b,handoffs,fair,0.801,1200,0.312,0.706,0.669,0.740,above
olmo2-7b,sets,0shot,0.588,320,0.275,0.449,0.381,0.527,above
olmo2-7b,sets,fair,0.762,320,0.275,0.691,0.627,0.754,above
olmo2-7b,refilled,0shot,0.579,720,0.200,0.474,0.429,0.521,above
olmo2-7b,refills,0shot,0.401,600,0.200,0.251,0.204,0.297,above
olmo2-7b,codevar,0shot,0.372,1200,0.267,0.137,0.100,0.174,above
olmo2-7b,codevar,fair,0.419,1200,0.267,0.201,0.164,0.239,above
olmo2-7b,sharedlists,0shot,0.318,600,0.200,0.148,0.104,0.192,above
olmo2-7b,sharedlists,fair,0.333,600,0.200,0.167,0.117,0.212,above
olmo2-7b,box-narrative,0shot,0.370,200,0.253,0.134,0.038,0.237,above
olmo2-7b,box-narrative,fair,0.425,200,0.253,0.214,0.110,0.308,above
olmo2-7b,box-screenplay,0shot,0.555,200,0.253,0.410,0.313,0.505,above
olmo2-7b,box-screenplay,fair,0.422,200,0.253,0.199,0.090,0.299,above
olmo2-7b,box-code,0shot,0.595,200,0.253,0.451,0.349,0.546,above
olmo2-7b,box-code,fair,0.635,200,0.253,0.482,0.389,0.574,above
olmo2-7b,box-chat,0shot,0.525,200,0.253,0.363,0.265,0.457,above
olmo2-7b,box-chat,fair,0.405,200,0.253,0.171,0.072,0.272,above
olmo2-7b,you-static,0shot,0.877,300,0.292,0.824,0.773,0.877,above
olmo2-7b,i-moves,0shot,0.342,180,0.287,0.080,-0.018,0.170,at
olmo2-7b,i-moves,fair,0.394,180,0.287,0.156,0.056,0.253,above
olmo2-7b,you-moves,0shot,0.272,180,0.287,-0.023,-0.112,0.071,at
olmo2-7b,you-moves,fair,0.339,180,0.287,0.073,-0.020,0.174,at
olmo2-7b,you-named,0shot,0.333,180,0.287,0.062,-0.032,0.158,at
olmo2-7b,you-named,fair,0.322,180,0.287,0.048,-0.047,0.146,at
olmo2-7b,answer-self,0shot,0.346,2560,0.250,0.129,0.105,0.154,above
olmo2-7b,answer-self,fair,0.380,2560,0.250,0.173,0.148,0.200,above
olmo2-7b,grammar,ladder,,1080,,0.862,0.841,0.882,above
olmo2-7b,operators,ladder,,1080,,0.286,0.257,0.322,above
olmo2-7b,patterns,ladder,,1080,,0.810,0.788,0.831,above
olmo2-7b,facts,ladder,,768,,1.000,1.000,1.000,above
falcon3-7b,lookup,0shot,0.752,1210,0.168,0.697,0.666,0.728,above
falcon3-7b,lookup,fair,0.999,1210,0.168,0.999,0.997,1.000,above
falcon3-7b,shortlist,0shot,0.728,960,0.040,0.716,0.687,0.743,above
falcon3-7b,pairs,0shot,0.688,64,0.125,0.643,0.500,0.750,above
falcon3-7b,pairs,fair,0.891,64,0.125,0.875,0.786,0.946,above
falcon3-7b,noise,0shot,0.711,720,0.200,0.639,0.600,0.681,above
falcon3-7b,now,0shot,0.572,300,0.292,0.396,0.317,0.471,above
falcon3-7b,handoffs,0shot,0.859,1200,0.312,0.791,0.763,0.819,above
falcon3-7b,handoffs,fair,0.881,1200,0.312,0.824,0.797,0.850,above
falcon3-7b,sets,0shot,0.828,320,0.275,0.762,0.697,0.821,above
falcon3-7b,sets,fair,0.658,320,0.275,0.539,0.470,0.603,above
falcon3-7b,refilled,0shot,0.593,720,0.200,0.491,0.443,0.536,above
falcon3-7b,refills,0shot,0.450,600,0.200,0.312,0.263,0.365,above
falcon3-7b,codevar,0shot,0.625,1200,0.267,0.482,0.444,0.518,above
falcon3-7b,codevar,fair,0.824,1200,0.267,0.755,0.722,0.785,above
falcon3-7b,sharedlists,0shot,0.395,600,0.200,0.244,0.194,0.296,above
falcon3-7b,sharedlists,fair,0.777,600,0.200,0.721,0.677,0.760,above
falcon3-7b,box-narrative,0shot,0.465,200,0.253,0.264,0.171,0.356,above
falcon3-7b,box-narrative,fair,0.470,200,0.253,0.283,0.184,0.377,above
falcon3-7b,box-screenplay,0shot,0.590,200,0.253,0.440,0.346,0.532,above
falcon3-7b,box-screenplay,fair,0.605,200,0.253,0.477,0.384,0.571,above
falcon3-7b,box-code,0shot,0.495,200,0.253,0.338,0.243,0.424,above
falcon3-7b,box-code,fair,0.820,200,0.253,0.748,0.671,0.821,above
falcon3-7b,box-chat,0shot,0.642,200,0.253,0.516,0.421,0.604,above
falcon3-7b,box-chat,fair,0.660,200,0.253,0.546,0.445,0.632,above
falcon3-7b,you-static,0shot,0.890,300,0.292,0.840,0.786,0.890,above
falcon3-7b,i-moves,0shot,0.347,180,0.287,0.082,-0.015,0.184,at
falcon3-7b,i-moves,fair,0.372,180,0.287,0.119,0.024,0.220,above
falcon3-7b,you-moves,0shot,0.278,180,0.287,-0.009,-0.093,0.078,at
falcon3-7b,you-moves,fair,0.250,180,0.287,-0.051,-0.140,0.043,at
falcon3-7b,you-named,0shot,0.361,180,0.287,0.105,0.008,0.199,above
falcon3-7b,you-named,fair,0.428,180,0.287,0.195,0.099,0.299,above
falcon3-7b,answer-self,0shot,0.409,2560,0.250,0.212,0.188,0.238,above
falcon3-7b,answer-self,fair,0.452,2560,0.250,0.269,0.244,0.297,above
falcon3-7b,grammar,ladder,,1080,,0.833,0.814,0.851,above
falcon3-7b,operators,ladder,,1080,,0.227,0.198,0.264,above
falcon3-7b,patterns,ladder,,1080,,0.833,0.812,0.852,above
falcon3-7b,facts,ladder,,768,,1.000,1.000,1.000,above
falcon-h1-7b,lookup,0shot,0.882,1210,0.168,0.853,0.830,0.874,above
falcon-h1-7b,lookup,fair,0.998,1210,0.168,0.997,0.993,1.000,above
falcon-h1-7b,shortlist,0shot,0.971,960,0.040,0.970,0.959,0.981,above
falcon-h1-7b,pairs,0shot,0.953,64,0.125,0.946,0.875,1.000,above
falcon-h1-7b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
falcon-h1-7b,noise,0shot,0.831,720,0.200,0.788,0.756,0.821,above
falcon-h1-7b,now,0shot,0.293,300,0.292,0.001,-0.063,0.077,at
falcon-h1-7b,handoffs,0shot,0.903,1200,0.312,0.856,0.832,0.880,above
falcon-h1-7b,handoffs,fair,0.920,1200,0.312,0.882,0.859,0.902,above
falcon-h1-7b,sets,0shot,0.892,320,0.275,0.850,0.804,0.897,above
falcon-h1-7b,sets,fair,0.772,320,0.275,0.694,0.632,0.759,above
falcon-h1-7b,refilled,0shot,0.701,720,0.200,0.627,0.587,0.667,above
falcon-h1-7b,refills,0shot,0.632,600,0.200,0.540,0.491,0.588,above
falcon-h1-7b,codevar,0shot,0.536,1200,0.267,0.366,0.328,0.405,above
falcon-h1-7b,codevar,fair,0.775,1200,0.267,0.687,0.654,0.720,above
falcon-h1-7b,sharedlists,0shot,0.379,600,0.200,0.224,0.180,0.271,above
falcon-h1-7b,sharedlists,fair,0.757,600,0.200,0.696,0.652,0.737,above
falcon-h1-7b,box-narrative,0shot,0.320,200,0.253,0.067,-0.025,0.167,at
falcon-h1-7b,box-narrative,fair,0.537,200,0.253,0.371,0.273,0.470,above
falcon-h1-7b,box-screenplay,0shot,0.650,200,0.253,0.528,0.438,0.624,above
falcon-h1-7b,box-screenplay,fair,0.660,200,0.253,0.549,0.458,0.639,above
falcon-h1-7b,box-code,0shot,0.620,200,0.253,0.467,0.378,0.558,above
falcon-h1-7b,box-code,fair,0.855,200,0.253,0.799,0.732,0.866,above
falcon-h1-7b,box-chat,0shot,0.640,200,0.253,0.522,0.423,0.611,above
falcon-h1-7b,box-chat,fair,0.680,200,0.253,0.562,0.475,0.647,above
falcon-h1-7b,you-static,0shot,0.957,300,0.292,0.937,0.900,0.971,above
falcon-h1-7b,i-moves,0shot,0.333,180,0.287,0.057,-0.035,0.158,at
falcon-h1-7b,i-moves,fair,0.547,180,0.287,0.367,0.270,0.475,above
falcon-h1-7b,you-moves,0shot,0.261,180,0.287,-0.041,-0.127,0.050,at
falcon-h1-7b,you-moves,fair,0.356,180,0.287,0.096,-0.002,0.200,at
falcon-h1-7b,you-named,0shot,0.353,180,0.287,0.090,-0.006,0.190,at
falcon-h1-7b,you-named,fair,0.464,180,0.287,0.240,0.137,0.340,above
falcon-h1-7b,answer-self,0shot,0.419,2560,0.250,0.225,0.200,0.252,above
falcon-h1-7b,answer-self,fair,0.493,2560,0.250,0.325,0.298,0.352,above
falcon-h1-7b,grammar,ladder,,1080,,0.873,0.857,0.893,above
falcon-h1-7b,operators,ladder,,1080,,0.279,0.246,0.317,above
falcon-h1-7b,patterns,ladder,,1080,,0.918,0.903,0.936,above
falcon-h1-7b,facts,ladder,,768,,0.998,0.995,1.000,above
qwen2.5-7b,lookup,0shot,0.795,1210,0.168,0.745,0.716,0.773,above
qwen2.5-7b,lookup,fair,1.000,1210,0.168,1.000,0.999,1.000,above
qwen2.5-7b,shortlist,0shot,0.881,960,0.040,0.876,0.854,0.897,above
qwen2.5-7b,pairs,0shot,0.953,64,0.125,0.946,0.893,1.000,above
qwen2.5-7b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
qwen2.5-7b,noise,0shot,0.909,720,0.200,0.886,0.861,0.911,above
qwen2.5-7b,now,0shot,0.787,300,0.292,0.699,0.634,0.759,above
qwen2.5-7b,handoffs,0shot,0.822,1200,0.312,0.738,0.707,0.770,above
qwen2.5-7b,handoffs,fair,0.868,1200,0.312,0.806,0.778,0.834,above
qwen2.5-7b,sets,0shot,0.880,320,0.275,0.836,0.787,0.889,above
qwen2.5-7b,sets,fair,0.892,320,0.275,0.856,0.809,0.901,above
qwen2.5-7b,refilled,0shot,0.652,720,0.200,0.565,0.524,0.609,above
qwen2.5-7b,refills,0shot,0.549,600,0.200,0.436,0.388,0.487,above
qwen2.5-7b,codevar,0shot,0.619,1200,0.267,0.478,0.438,0.513,above
qwen2.5-7b,codevar,fair,0.657,1200,0.267,0.526,0.487,0.562,above
qwen2.5-7b,sharedlists,0shot,0.561,600,0.200,0.451,0.402,0.502,above
qwen2.5-7b,sharedlists,fair,0.669,600,0.200,0.587,0.542,0.630,above
qwen2.5-7b,box-narrative,0shot,0.455,200,0.253,0.243,0.144,0.343,above
qwen2.5-7b,box-narrative,fair,0.520,200,0.253,0.335,0.235,0.431,above
qwen2.5-7b,box-screenplay,0shot,0.670,200,0.253,0.551,0.463,0.642,above
qwen2.5-7b,box-screenplay,fair,0.650,200,0.253,0.516,0.421,0.605,above
qwen2.5-7b,box-code,0shot,0.665,200,0.253,0.512,0.405,0.602,above
qwen2.5-7b,box-code,fair,0.802,200,0.253,0.716,0.631,0.786,above
qwen2.5-7b,box-chat,0shot,0.662,200,0.253,0.536,0.444,0.622,above
qwen2.5-7b,box-chat,fair,0.695,200,0.253,0.582,0.492,0.668,above
qwen2.5-7b,you-static,0shot,0.827,300,0.292,0.751,0.687,0.807,above
qwen2.5-7b,i-moves,0shot,0.444,180,0.287,0.225,0.123,0.326,above
qwen2.5-7b,i-moves,fair,0.506,180,0.287,0.307,0.204,0.405,above
qwen2.5-7b,you-moves,0shot,0.356,180,0.287,0.096,-0.005,0.194,at
qwen2.5-7b,you-moves,fair,0.383,180,0.287,0.132,0.040,0.237,above
qwen2.5-7b,you-named,0shot,0.383,180,0.287,0.131,0.038,0.238,above
qwen2.5-7b,you-named,fair,0.367,180,0.287,0.106,0.011,0.208,above
qwen2.5-7b,answer-self,0shot,0.387,2560,0.250,0.183,0.159,0.207,above
qwen2.5-7b,answer-self,fair,0.468,2560,0.250,0.291,0.266,0.317,above
qwen2.5-7b,grammar,ladder,,1080,,0.826,0.803,0.847,above
qwen2.5-7b,operators,ladder,,1080,,0.224,0.196,0.261,above
qwen2.5-7b,patterns,ladder,,1080,,0.936,0.920,0.950,above
qwen2.5-7b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3-8b,lookup,0shot,0.745,1210,0.168,0.680,0.647,0.708,above
llama3-8b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
llama3-8b,shortlist,0shot,0.988,960,0.040,0.987,0.979,0.994,above
llama3-8b,pairs,0shot,0.859,64,0.125,0.839,0.750,0.929,above
llama3-8b,pairs,fair,0.969,64,0.125,0.964,0.911,1.000,above
llama3-8b,noise,0shot,0.915,720,0.200,0.894,0.868,0.917,above
llama3-8b,now,0shot,0.334,300,0.292,0.053,-0.017,0.125,at
llama3-8b,handoffs,0shot,0.847,1200,0.312,0.774,0.746,0.803,above
llama3-8b,handoffs,fair,0.859,1200,0.312,0.793,0.767,0.822,above
llama3-8b,sets,0shot,0.863,320,0.275,0.809,0.757,0.861,above
llama3-8b,sets,fair,0.830,320,0.275,0.771,0.720,0.823,above
llama3-8b,refilled,0shot,0.445,720,0.200,0.306,0.262,0.350,above
llama3-8b,refills,0shot,0.428,600,0.200,0.285,0.239,0.333,above
llama3-8b,codevar,0shot,0.574,1200,0.267,0.414,0.378,0.453,above
llama3-8b,codevar,fair,0.674,1200,0.267,0.548,0.514,0.583,above
llama3-8b,sharedlists,0shot,0.417,600,0.200,0.272,0.227,0.321,above
llama3-8b,sharedlists,fair,0.617,600,0.200,0.521,0.473,0.569,above
llama3-8b,box-narrative,0shot,0.278,200,0.253,0.014,-0.074,0.107,at
llama3-8b,box-narrative,fair,0.520,200,0.253,0.328,0.220,0.423,above
llama3-8b,box-screenplay,0shot,0.657,200,0.253,0.545,0.454,0.634,above
llama3-8b,box-screenplay,fair,0.609,200,0.253,0.453,0.351,0.548,above
llama3-8b,box-code,0shot,0.692,200,0.253,0.565,0.476,0.647,above
llama3-8b,box-code,fair,0.725,200,0.253,0.599,0.510,0.685,above
llama3-8b,box-chat,0shot,0.564,200,0.253,0.413,0.317,0.500,above
llama3-8b,box-chat,fair,0.706,200,0.253,0.596,0.506,0.679,above
llama3-8b,you-static,0shot,0.933,300,0.292,0.901,0.858,0.939,above
llama3-8b,i-moves,0shot,0.328,180,0.287,0.053,-0.041,0.151,at
llama3-8b,i-moves,fair,0.450,180,0.287,0.232,0.126,0.334,above
llama3-8b,you-moves,0shot,0.261,180,0.287,-0.037,-0.118,0.053,at
llama3-8b,you-moves,fair,0.317,180,0.287,0.044,-0.045,0.141,at
llama3-8b,you-named,0shot,0.406,180,0.287,0.162,0.070,0.259,above
llama3-8b,you-named,fair,0.366,180,0.287,0.107,0.010,0.208,above
llama3-8b,answer-self,0shot,0.402,2560,0.250,0.203,0.180,0.230,above
llama3-8b,answer-self,fair,0.409,2560,0.250,0.212,0.186,0.236,above
llama3-8b,grammar,ladder,,1080,,0.848,0.829,0.869,above
llama3-8b,operators,ladder,,1080,,0.260,0.231,0.291,above
llama3-8b,patterns,ladder,,1080,,0.932,0.919,0.947,above
llama3-8b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-8b,lookup,0shot,0.674,1210,0.168,0.589,0.555,0.622,above
llama3.1-8b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
llama3.1-8b,shortlist,0shot,0.988,960,0.040,0.987,0.978,0.994,above
llama3.1-8b,pairs,0shot,0.727,64,0.125,0.688,0.562,0.804,above
llama3.1-8b,pairs,fair,0.992,64,0.125,0.991,0.973,1.000,above
llama3.1-8b,noise,0shot,0.924,720,0.200,0.905,0.878,0.928,above
llama3.1-8b,now,0shot,0.366,300,0.292,0.098,0.025,0.172,above
llama3.1-8b,handoffs,0shot,0.830,1200,0.312,0.750,0.720,0.781,above
llama3.1-8b,handoffs,fair,0.871,1200,0.312,0.811,0.786,0.838,above
llama3.1-8b,sets,0shot,0.860,320,0.275,0.804,0.751,0.853,above
llama3.1-8b,sets,fair,0.864,320,0.275,0.816,0.765,0.867,above
llama3.1-8b,refilled,0shot,0.452,720,0.200,0.315,0.267,0.358,above
llama3.1-8b,refills,0shot,0.433,600,0.200,0.291,0.244,0.339,above
llama3.1-8b,codevar,0shot,0.581,1200,0.267,0.422,0.385,0.458,above
llama3.1-8b,codevar,fair,0.682,1200,0.267,0.560,0.524,0.595,above
llama3.1-8b,sharedlists,0shot,0.426,600,0.200,0.282,0.235,0.326,above
llama3.1-8b,sharedlists,fair,0.609,600,0.200,0.511,0.462,0.558,above
llama3.1-8b,box-narrative,0shot,0.306,200,0.253,0.044,-0.047,0.136,at
llama3.1-8b,box-narrative,fair,0.557,200,0.253,0.381,0.275,0.481,above
llama3.1-8b,box-screenplay,0shot,0.615,200,0.253,0.477,0.378,0.568,above
llama3.1-8b,box-screenplay,fair,0.647,200,0.253,0.508,0.409,0.598,above
llama3.1-8b,box-code,0shot,0.677,200,0.253,0.544,0.449,0.630,above
llama3.1-8b,box-code,fair,0.762,200,0.253,0.649,0.559,0.731,above
llama3.1-8b,box-chat,0shot,0.615,200,0.253,0.481,0.401,0.573,above
llama3.1-8b,box-chat,fair,0.690,200,0.253,0.580,0.487,0.669,above
llama3.1-8b,you-static,0shot,0.913,300,0.292,0.874,0.829,0.915,above
llama3.1-8b,i-moves,0shot,0.339,180,0.287,0.069,-0.019,0.164,at
llama3.1-8b,i-moves,fair,0.449,180,0.287,0.231,0.131,0.332,above
llama3.1-8b,you-moves,0shot,0.233,180,0.287,-0.078,-0.154,0.010,at
llama3.1-8b,you-moves,fair,0.335,180,0.287,0.069,-0.024,0.167,at
llama3.1-8b,you-named,0shot,0.400,180,0.287,0.155,0.057,0.253,above
llama3.1-8b,you-named,fair,0.373,180,0.287,0.115,0.015,0.214,above
llama3.1-8b,answer-self,0shot,0.388,2560,0.250,0.184,0.159,0.209,above
llama3.1-8b,answer-self,fair,0.404,2560,0.250,0.206,0.179,0.231,above
llama3.1-8b,grammar,ladder,,1080,,0.843,0.822,0.868,above
llama3.1-8b,operators,ladder,,1080,,0.266,0.238,0.295,above
llama3.1-8b,patterns,ladder,,1080,,0.942,0.928,0.953,above
llama3.1-8b,facts,ladder,,768,,1.000,1.000,1.000,above
gemma-7b,lookup,0shot,0.764,1210,0.168,0.701,0.674,0.728,above
gemma-7b,lookup,fair,0.999,1210,0.168,0.999,0.997,1.000,above
gemma-7b,shortlist,0shot,0.960,960,0.040,0.958,0.946,0.969,above
gemma-7b,pairs,0shot,0.630,64,0.125,0.577,0.449,0.696,above
gemma-7b,pairs,fair,0.859,64,0.125,0.838,0.760,0.913,above
gemma-7b,noise,0shot,0.755,720,0.200,0.693,0.660,0.732,above
gemma-7b,now,0shot,0.289,300,0.292,-0.006,-0.063,0.053,at
gemma-7b,handoffs,0shot,0.756,1200,0.312,0.640,0.609,0.669,above
gemma-7b,handoffs,fair,0.880,1200,0.312,0.823,0.800,0.847,above
gemma-7b,sets,0shot,0.696,320,0.275,0.582,0.528,0.638,above
gemma-7b,sets,fair,0.789,320,0.275,0.714,0.667,0.760,above
gemma-7b,refilled,0shot,0.495,720,0.200,0.368,0.333,0.410,above
gemma-7b,refills,0shot,0.410,600,0.200,0.262,0.222,0.305,above
gemma-7b,codevar,0shot,0.595,1200,0.267,0.440,0.411,0.468,above
gemma-7b,codevar,fair,0.603,1200,0.267,0.451,0.423,0.481,above
gemma-7b,sharedlists,0shot,0.394,600,0.200,0.243,0.201,0.284,above
gemma-7b,sharedlists,fair,0.535,600,0.200,0.418,0.381,0.456,above
gemma-7b,box-narrative,0shot,0.206,200,0.253,-0.068,-0.145,0.010,at
gemma-7b,box-narrative,fair,0.348,200,0.253,0.101,0.013,0.183,above
gemma-7b,box-screenplay,0shot,0.434,200,0.253,0.249,0.168,0.330,above
gemma-7b,box-screenplay,fair,0.479,200,0.253,0.279,0.188,0.365,above
gemma-7b,box-code,0shot,0.699,200,0.253,0.581,0.491,0.662,above
gemma-7b,box-code,fair,0.767,200,0.253,0.676,0.607,0.743,above
gemma-7b,box-chat,0shot,0.384,200,0.253,0.157,0.079,0.233,above
gemma-7b,box-chat,fair,0.508,200,0.253,0.322,0.236,0.411,above
gemma-7b,you-static,0shot,0.955,300,0.292,0.933,0.900,0.961,above
gemma-7b,i-moves,0shot,0.468,180,0.287,0.255,0.163,0.344,above
gemma-7b,i-moves,fair,0.445,180,0.287,0.226,0.137,0.327,above
gemma-7b,you-moves,0shot,0.258,180,0.287,-0.041,-0.118,0.034,at
gemma-7b,you-moves,fair,0.326,180,0.287,0.055,-0.026,0.138,at
gemma-7b,you-named,0shot,0.307,180,0.287,0.025,-0.051,0.113,at
gemma-7b,you-named,fair,0.289,180,0.287,-0.002,-0.077,0.082,at
gemma-7b,answer-self,0shot,0.373,2560,0.250,0.164,0.144,0.184,above
gemma-7b,answer-self,fair,0.380,2560,0.250,0.174,0.155,0.194,above
gemma-7b,grammar,ladder,,1080,,0.845,0.826,0.864,above
gemma-7b,operators,ladder,,1080,,0.236,0.206,0.265,above
gemma-7b,patterns,ladder,,1080,,0.896,0.876,0.913,above
gemma-7b,facts,ladder,,768,,1.000,1.000,1.000,above
gemma2-9b,lookup,0shot,0.773,1210,0.168,0.722,0.693,0.750,above
gemma2-9b,lookup,fair,0.998,1210,0.168,0.998,0.994,1.000,above
gemma2-9b,shortlist,0shot,0.940,960,0.040,0.938,0.922,0.951,above
gemma2-9b,pairs,0shot,0.906,64,0.125,0.893,0.812,0.964,above
gemma2-9b,pairs,fair,0.984,64,0.125,0.982,0.946,1.000,above
gemma2-9b,noise,0shot,0.941,720,0.200,0.926,0.906,0.947,above
gemma2-9b,now,0shot,0.440,300,0.292,0.210,0.137,0.287,above
gemma2-9b,handoffs,0shot,0.856,1200,0.312,0.786,0.758,0.813,above
gemma2-9b,handoffs,fair,0.928,1200,0.312,0.893,0.873,0.914,above
gemma2-9b,sets,0shot,0.834,320,0.275,0.769,0.716,0.823,above
gemma2-9b,sets,fair,0.808,320,0.275,0.734,0.673,0.795,above
gemma2-9b,refilled,0shot,0.510,720,0.200,0.388,0.342,0.432,above
gemma2-9b,refills,0shot,0.486,600,0.200,0.358,0.309,0.406,above
gemma2-9b,codevar,0shot,0.643,1200,0.267,0.507,0.472,0.545,above
gemma2-9b,codevar,fair,0.760,1200,0.267,0.668,0.637,0.700,above
gemma2-9b,sharedlists,0shot,0.609,600,0.200,0.511,0.460,0.555,above
gemma2-9b,sharedlists,fair,0.706,600,0.200,0.632,0.588,0.677,above
gemma2-9b,box-narrative,0shot,0.491,200,0.253,0.319,0.226,0.412,above
gemma2-9b,box-narrative,fair,0.642,200,0.253,0.500,0.415,0.588,above
gemma2-9b,box-screenplay,0shot,0.559,200,0.253,0.408,0.316,0.501,above
gemma2-9b,box-screenplay,fair,0.701,200,0.253,0.590,0.505,0.672,above
gemma2-9b,box-code,0shot,0.670,200,0.253,0.533,0.441,0.624,above
gemma2-9b,box-code,fair,0.820,200,0.253,0.749,0.672,0.821,above
gemma2-9b,box-chat,0shot,0.655,200,0.253,0.534,0.446,0.617,above
gemma2-9b,box-chat,fair,0.713,200,0.253,0.610,0.520,0.691,above
gemma2-9b,you-static,0shot,0.888,300,0.292,0.838,0.785,0.885,above
gemma2-9b,i-moves,0shot,0.387,180,0.287,0.141,0.048,0.242,above
gemma2-9b,i-moves,fair,0.517,180,0.287,0.322,0.231,0.424,above
gemma2-9b,you-moves,0shot,0.229,180,0.287,-0.084,-0.165,0.009,at
gemma2-9b,you-moves,fair,0.397,180,0.287,0.157,0.063,0.257,above
gemma2-9b,you-named,0shot,0.306,180,0.287,0.023,-0.062,0.121,at
gemma2-9b,you-named,fair,0.425,180,0.287,0.189,0.090,0.290,above
gemma2-9b,answer-self,0shot,0.394,2560,0.250,0.192,0.169,0.218,above
gemma2-9b,answer-self,fair,0.513,2560,0.250,0.351,0.328,0.377,above
gemma2-9b,grammar,ladder,,1080,,0.860,0.840,0.883,above
gemma2-9b,operators,ladder,,1080,,0.297,0.265,0.333,above
gemma2-9b,patterns,ladder,,1080,,0.899,0.881,0.918,above
gemma2-9b,facts,ladder,,768,,0.998,0.995,1.000,above
recurrentgemma-9b,lookup,0shot,0.363,1210,0.168,0.233,0.201,0.268,above
recurrentgemma-9b,lookup,fair,0.299,1210,0.168,0.163,0.133,0.194,above
recurrentgemma-9b,shortlist,0shot,0.698,960,0.040,0.686,0.657,0.715,above
recurrentgemma-9b,pairs,0shot,0.169,64,0.125,0.051,-0.045,0.158,at
recurrentgemma-9b,pairs,fair,0.180,64,0.125,0.062,-0.036,0.179,at
recurrentgemma-9b,noise,0shot,0.427,720,0.200,0.284,0.240,0.326,above
recurrentgemma-9b,now,0shot,0.459,300,0.292,0.235,0.159,0.312,above
recurrentgemma-9b,handoffs,0shot,0.623,1200,0.312,0.445,0.406,0.487,above
recurrentgemma-9b,handoffs,fair,0.518,1200,0.312,0.294,0.254,0.337,above
recurrentgemma-9b,sets,0shot,0.332,320,0.275,0.080,0.012,0.147,above
recurrentgemma-9b,sets,fair,0.317,320,0.275,0.064,-0.009,0.139,at
recurrentgemma-9b,refilled,0shot,0.245,720,0.200,0.056,0.018,0.097,above
recurrentgemma-9b,refills,0shot,0.277,600,0.200,0.096,0.054,0.138,above
recurrentgemma-9b,codevar,0shot,0.357,1200,0.267,0.119,0.083,0.154,above
recurrentgemma-9b,codevar,fair,0.283,1200,0.267,0.017,-0.014,0.051,at
recurrentgemma-9b,sharedlists,0shot,0.208,600,0.200,0.010,-0.032,0.050,at
recurrentgemma-9b,sharedlists,fair,0.292,600,0.200,0.116,0.073,0.164,above
recurrentgemma-9b,box-narrative,0shot,0.253,200,0.253,0.004,-0.075,0.085,at
recurrentgemma-9b,box-narrative,fair,0.279,200,0.253,0.021,-0.070,0.108,at
recurrentgemma-9b,box-screenplay,0shot,0.333,200,0.253,0.115,0.027,0.200,above
recurrentgemma-9b,box-screenplay,fair,0.435,200,0.253,0.235,0.141,0.332,above
recurrentgemma-9b,box-code,0shot,0.435,200,0.253,0.254,0.156,0.337,above
recurrentgemma-9b,box-code,fair,0.439,200,0.253,0.257,0.167,0.345,above
recurrentgemma-9b,box-chat,0shot,0.327,200,0.253,0.127,0.045,0.209,above
recurrentgemma-9b,box-chat,fair,0.460,200,0.253,0.279,0.186,0.371,above
recurrentgemma-9b,you-static,0shot,0.574,300,0.292,0.398,0.321,0.479,above
recurrentgemma-9b,i-moves,0shot,0.300,180,0.287,0.012,-0.084,0.111,at
recurrentgemma-9b,i-moves,fair,0.261,180,0.287,-0.043,-0.133,0.047,at
recurrentgemma-9b,you-moves,0shot,0.243,180,0.287,-0.064,-0.148,0.024,at
recurrentgemma-9b,you-moves,fair,0.303,180,0.287,0.024,-0.066,0.115,at
recurrentgemma-9b,you-named,0shot,0.336,180,0.287,0.067,-0.029,0.162,at
recurrentgemma-9b,you-named,fair,0.317,180,0.287,0.039,-0.055,0.134,at
recurrentgemma-9b,answer-self,0shot,0.279,2560,0.250,0.038,0.015,0.062,above
recurrentgemma-9b,answer-self,fair,0.243,2560,0.250,-0.009,-0.031,0.013,at
recurrentgemma-9b,grammar,ladder,,1080,,0.631,0.604,0.658,above
recurrentgemma-9b,operators,ladder,,1080,,0.128,0.104,0.159,above
recurrentgemma-9b,patterns,ladder,,1080,,0.342,0.311,0.374,above
recurrentgemma-9b,facts,ladder,,768,,0.997,0.992,1.000,above
bamba-9b,lookup,0shot,0.303,1210,0.168,0.168,0.137,0.198,above
bamba-9b,lookup,fair,0.214,1210,0.168,0.060,0.031,0.089,above
bamba-9b,shortlist,0shot,0.653,960,0.040,0.639,0.608,0.669,above
bamba-9b,pairs,0shot,0.156,64,0.125,0.036,-0.054,0.143,at
bamba-9b,pairs,fair,0.109,64,0.125,-0.018,-0.107,0.071,at
bamba-9b,noise,0shot,0.386,720,0.200,0.233,0.189,0.276,above
bamba-9b,now,0shot,0.402,300,0.292,0.157,0.070,0.232,above
bamba-9b,handoffs,0shot,0.542,1200,0.312,0.330,0.288,0.372,above
bamba-9b,handoffs,fair,0.466,1200,0.312,0.222,0.180,0.262,above
bamba-9b,sets,0shot,0.323,320,0.275,0.067,-0.006,0.139,at
bamba-9b,sets,fair,0.358,320,0.275,0.129,0.056,0.200,above
bamba-9b,refilled,0shot,0.414,720,0.200,0.267,0.223,0.309,above
bamba-9b,refills,0shot,0.374,600,0.200,0.218,0.171,0.269,above
bamba-9b,codevar,0shot,0.313,1200,0.267,0.061,0.025,0.100,above
bamba-9b,codevar,fair,0.300,1200,0.267,0.043,0.007,0.076,above
bamba-9b,sharedlists,0shot,0.237,600,0.200,0.047,0.007,0.090,above
bamba-9b,sharedlists,fair,0.182,600,0.200,-0.022,-0.060,0.018,at
bamba-9b,box-narrative,0shot,0.172,200,0.253,-0.120,-0.192,-0.045,below
bamba-9b,box-narrative,fair,0.190,200,0.253,-0.085,-0.164,-0.009,below
bamba-9b,box-screenplay,0shot,0.180,200,0.253,-0.087,-0.157,-0.014,below
bamba-9b,box-screenplay,fair,0.235,200,0.253,-0.030,-0.113,0.065,at
bamba-9b,box-code,0shot,0.285,200,0.253,0.050,-0.038,0.137,at
bamba-9b,box-code,fair,0.240,200,0.253,-0.015,-0.095,0.070,at
bamba-9b,box-chat,0shot,0.165,200,0.253,-0.132,-0.208,-0.048,below
bamba-9b,box-chat,fair,0.245,200,0.253,0.001,-0.078,0.083,at
bamba-9b,you-static,0shot,0.378,300,0.292,0.123,0.050,0.196,above
bamba-9b,i-moves,0shot,0.233,180,0.287,-0.077,-0.167,0.010,at
bamba-9b,i-moves,fair,0.258,180,0.287,-0.041,-0.135,0.058,at
bamba-9b,you-moves,0shot,0.272,180,0.287,-0.022,-0.113,0.064,at
bamba-9b,you-moves,fair,0.283,180,0.287,-0.007,-0.104,0.084,at
bamba-9b,you-named,0shot,0.369,180,0.287,0.115,0.018,0.216,above
bamba-9b,you-named,fair,0.292,180,0.287,0.007,-0.086,0.101,at
bamba-9b,answer-self,0shot,0.253,2560,0.250,0.004,-0.019,0.025,at
bamba-9b,answer-self,fair,0.256,2560,0.250,0.009,-0.013,0.030,at
bamba-9b,grammar,ladder,,1080,,0.575,0.543,0.602,above
bamba-9b,operators,ladder,,1080,,0.136,0.113,0.166,above
bamba-9b,patterns,ladder,,1080,,0.430,0.399,0.459,above
bamba-9b,facts,ladder,,768,,0.998,0.995,1.000,above
olmo2-13b,lookup,0shot,0.598,1210,0.168,0.498,0.465,0.533,above
olmo2-13b,lookup,fair,0.992,1210,0.168,0.990,0.983,0.995,above
olmo2-13b,shortlist,0shot,0.925,960,0.040,0.922,0.904,0.940,above
olmo2-13b,pairs,0shot,0.844,64,0.125,0.821,0.714,0.911,above
olmo2-13b,pairs,fair,0.883,64,0.125,0.866,0.759,0.946,above
olmo2-13b,noise,0shot,0.694,720,0.200,0.618,0.576,0.660,above
olmo2-13b,now,0shot,0.263,300,0.292,-0.042,-0.114,0.028,at
olmo2-13b,handoffs,0shot,0.791,1200,0.312,0.693,0.657,0.726,above
olmo2-13b,handoffs,fair,0.913,1200,0.312,0.873,0.847,0.896,above
olmo2-13b,sets,0shot,0.869,320,0.275,0.815,0.761,0.865,above
olmo2-13b,sets,fair,0.897,320,0.275,0.859,0.815,0.901,above
olmo2-13b,refilled,0shot,0.746,720,0.200,0.682,0.642,0.721,above
olmo2-13b,refills,0shot,0.581,600,0.200,0.476,0.426,0.525,above
olmo2-13b,codevar,0shot,0.410,1200,0.267,0.185,0.146,0.224,above
olmo2-13b,codevar,fair,0.526,1200,0.267,0.350,0.310,0.389,above
olmo2-13b,sharedlists,0shot,0.348,600,0.200,0.185,0.138,0.235,above
olmo2-13b,sharedlists,fair,0.553,600,0.200,0.442,0.394,0.487,above
olmo2-13b,box-narrative,0shot,0.535,200,0.253,0.381,0.289,0.475,above
olmo2-13b,box-narrative,fair,0.440,200,0.253,0.225,0.119,0.330,above
olmo2-13b,box-screenplay,0shot,0.555,200,0.253,0.404,0.309,0.502,above
olmo2-13b,box-screenplay,fair,0.505,200,0.253,0.329,0.227,0.424,above
olmo2-13b,box-code,0shot,0.660,200,0.253,0.515,0.420,0.603,above
olmo2-13b,box-code,fair,0.705,200,0.253,0.603,0.515,0.684,above
olmo2-13b,box-chat,0shot,0.550,200,0.253,0.411,0.320,0.513,above
olmo2-13b,box-chat,fair,0.570,200,0.253,0.443,0.353,0.535,above
olmo2-13b,you-static,0shot,0.970,300,0.292,0.957,0.927,0.981,above
olmo2-13b,i-moves,0shot,0.411,180,0.287,0.176,0.081,0.277,above
olmo2-13b,i-moves,fair,0.567,180,0.287,0.391,0.291,0.492,above
olmo2-13b,you-moves,0shot,0.294,180,0.287,0.009,-0.082,0.105,at
olmo2-13b,you-moves,fair,0.461,180,0.287,0.245,0.145,0.356,above
olmo2-13b,you-named,0shot,0.433,180,0.287,0.201,0.104,0.292,above
olmo2-13b,you-named,fair,0.522,180,0.287,0.327,0.221,0.433,above
olmo2-13b,answer-self,0shot,0.456,2560,0.250,0.274,0.249,0.300,above
olmo2-13b,answer-self,fair,0.445,2560,0.250,0.259,0.233,0.285,above
olmo2-13b,grammar,ladder,,1080,,0.843,0.825,0.862,above
olmo2-13b,operators,ladder,,1080,,0.240,0.213,0.274,above
olmo2-13b,patterns,ladder,,1080,,0.822,0.802,0.844,above
olmo2-13b,facts,ladder,,768,,0.998,0.995,1.000,above
olmo3-32b,lookup,0shot,0.694,1210,0.168,0.620,0.588,0.652,above
olmo3-32b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
olmo3-32b,shortlist,0shot,0.939,960,0.040,0.936,0.920,0.951,above
olmo3-32b,pairs,0shot,0.844,64,0.125,0.821,0.714,0.911,above
olmo3-32b,pairs,fair,0.984,64,0.125,0.982,0.946,1.000,above
olmo3-32b,noise,0shot,0.698,720,0.200,0.623,0.582,0.665,above
olmo3-32b,now,0shot,0.216,300,0.292,-0.108,-0.174,-0.047,below
olmo3-32b,handoffs,0shot,0.850,1200,0.312,0.779,0.751,0.808,above
olmo3-32b,handoffs,fair,0.897,1200,0.312,0.848,0.823,0.873,above
olmo3-32b,sets,0shot,0.880,320,0.275,0.831,0.784,0.880,above
olmo3-32b,sets,fair,0.831,320,0.275,0.769,0.710,0.826,above
olmo3-32b,refilled,0shot,0.671,720,0.200,0.589,0.549,0.634,above
olmo3-32b,refills,0shot,0.565,600,0.200,0.456,0.407,0.503,above
olmo3-32b,codevar,0shot,0.727,1200,0.267,0.628,0.594,0.661,above
olmo3-32b,codevar,fair,0.840,1200,0.267,0.779,0.750,0.808,above
olmo3-32b,sharedlists,0shot,0.701,600,0.200,0.626,0.578,0.672,above
olmo3-32b,sharedlists,fair,0.792,600,0.200,0.740,0.697,0.780,above
olmo3-32b,box-narrative,0shot,0.409,200,0.253,0.205,0.111,0.299,above
olmo3-32b,box-narrative,fair,0.532,200,0.253,0.340,0.235,0.439,above
olmo3-32b,box-screenplay,0shot,0.664,200,0.253,0.550,0.460,0.639,above
olmo3-32b,box-screenplay,fair,0.682,200,0.253,0.562,0.466,0.653,above
olmo3-32b,box-code,0shot,0.760,200,0.253,0.659,0.572,0.739,above
olmo3-32b,box-code,fair,0.845,200,0.253,0.783,0.712,0.845,above
olmo3-32b,box-chat,0shot,0.670,200,0.253,0.555,0.465,0.643,above
olmo3-32b,box-chat,fair,0.645,200,0.253,0.498,0.397,0.596,above
olmo3-32b,you-static,0shot,0.985,300,0.292,0.978,0.956,0.995,above
olmo3-32b,i-moves,0shot,0.428,180,0.287,0.200,0.109,0.302,above
olmo3-32b,i-moves,fair,0.578,180,0.287,0.409,0.306,0.519,above
olmo3-32b,you-moves,0shot,0.281,180,0.287,-0.011,-0.106,0.078,at
olmo3-32b,you-moves,fair,0.447,180,0.287,0.224,0.128,0.329,above
olmo3-32b,you-named,0shot,0.408,180,0.287,0.165,0.059,0.269,above
olmo3-32b,you-named,fair,0.503,180,0.287,0.299,0.201,0.403,above
olmo3-32b,answer-self,0shot,0.421,2560,0.250,0.229,0.202,0.254,above
olmo3-32b,answer-self,fair,0.524,2560,0.250,0.366,0.341,0.390,above
olmo3-32b,grammar,ladder,,1080,,0.846,0.825,0.865,above
olmo3-32b,operators,ladder,,1080,,0.211,0.181,0.244,above
olmo3-32b,patterns,ladder,,1080,,0.900,0.884,0.918,above
olmo3-32b,facts,ladder,,768,,1.000,1.000,1.000,above
olmo3-32b-stage1,lookup,0shot,0.704,1210,0.168,0.628,0.596,0.660,above
olmo3-32b-stage1,lookup,fair,0.999,1210,0.168,0.999,0.997,1.000,above
olmo3-32b-stage1,shortlist,0shot,0.882,960,0.040,0.877,0.856,0.897,above
olmo3-32b-stage1,pairs,0shot,0.828,64,0.125,0.804,0.696,0.911,above
olmo3-32b-stage1,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
olmo3-32b-stage1,noise,0shot,0.685,720,0.200,0.607,0.564,0.650,above
olmo3-32b-stage1,now,0shot,0.094,300,0.292,-0.285,-0.329,-0.240,below
olmo3-32b-stage1,handoffs,0shot,0.861,1200,0.312,0.794,0.765,0.823,above
olmo3-32b-stage1,handoffs,fair,0.921,1200,0.312,0.883,0.861,0.905,above
olmo3-32b-stage1,sets,0shot,0.863,320,0.275,0.808,0.757,0.857,above
olmo3-32b-stage1,sets,fair,0.884,320,0.275,0.842,0.789,0.890,above
olmo3-32b-stage1,refilled,0shot,0.718,720,0.200,0.648,0.606,0.688,above
olmo3-32b-stage1,refills,0shot,0.579,600,0.200,0.474,0.424,0.524,above
olmo3-32b-stage1,codevar,0shot,0.540,1200,0.267,0.371,0.332,0.409,above
olmo3-32b-stage1,codevar,fair,0.712,1200,0.267,0.603,0.566,0.639,above
olmo3-32b-stage1,sharedlists,0shot,0.347,600,0.200,0.183,0.137,0.234,above
olmo3-32b-stage1,sharedlists,fair,0.662,600,0.200,0.578,0.532,0.624,above
olmo3-32b-stage1,box-narrative,0shot,0.395,200,0.253,0.173,0.080,0.267,above
olmo3-32b-stage1,box-narrative,fair,0.537,200,0.253,0.353,0.256,0.447,above
olmo3-32b-stage1,box-screenplay,0shot,0.685,200,0.253,0.574,0.483,0.655,above
olmo3-32b-stage1,box-screenplay,fair,0.623,200,0.253,0.480,0.382,0.576,above
olmo3-32b-stage1,box-code,0shot,0.713,200,0.253,0.594,0.500,0.681,above
olmo3-32b-stage1,box-code,fair,0.828,200,0.253,0.759,0.688,0.828,above
olmo3-32b-stage1,box-chat,0shot,0.682,200,0.253,0.570,0.478,0.652,above
olmo3-32b-stage1,box-chat,fair,0.682,200,0.253,0.561,0.469,0.650,above
olmo3-32b-stage1,you-static,0shot,0.957,300,0.292,0.938,0.903,0.971,above
olmo3-32b-stage1,i-moves,0shot,0.403,180,0.287,0.161,0.062,0.259,above
olmo3-32b-stage1,i-moves,fair,0.494,180,0.287,0.293,0.189,0.402,above
olmo3-32b-stage1,you-moves,0shot,0.308,180,0.287,0.028,-0.065,0.124,at
olmo3-32b-stage1,you-moves,fair,0.411,180,0.287,0.172,0.066,0.275,above
olmo3-32b-stage1,you-named,0shot,0.303,180,0.287,0.014,-0.075,0.113,at
olmo3-32b-stage1,you-named,fair,0.453,180,0.287,0.229,0.123,0.333,above
olmo3-32b-stage1,answer-self,0shot,0.328,2560,0.250,0.104,0.082,0.129,above
olmo3-32b-stage1,answer-self,fair,0.466,2560,0.250,0.288,0.262,0.313,above
olmo3-32b-stage1,grammar,ladder,,1080,,0.832,0.810,0.852,above
olmo3-32b-stage1,operators,ladder,,1080,,0.277,0.241,0.313,above
olmo3-32b-stage1,patterns,ladder,,1080,,0.919,0.904,0.939,above
olmo3-32b-stage1,facts,ladder,,768,,0.995,0.989,1.000,above
glm4-32b-base,lookup,0shot,0.778,1210,0.168,0.724,0.693,0.752,above
glm4-32b-base,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
glm4-32b-base,shortlist,0shot,0.992,960,0.040,0.991,0.985,0.997,above
glm4-32b-base,pairs,0shot,0.859,64,0.125,0.839,0.750,0.929,above
glm4-32b-base,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
glm4-32b-base,noise,0shot,0.808,720,0.200,0.760,0.727,0.794,above
glm4-32b-base,now,0shot,0.101,300,0.292,-0.272,-0.317,-0.226,below
glm4-32b-base,handoffs,0shot,0.915,1200,0.312,0.874,0.851,0.895,above
glm4-32b-base,handoffs,fair,0.970,1200,0.312,0.957,0.942,0.969,above
glm4-32b-base,sets,0shot,0.948,320,0.275,0.931,0.899,0.960,above
glm4-32b-base,sets,fair,0.941,320,0.275,0.922,0.883,0.952,above
glm4-32b-base,refilled,0shot,0.694,720,0.200,0.617,0.575,0.660,above
glm4-32b-base,refills,0shot,0.621,600,0.200,0.526,0.475,0.574,above
glm4-32b-base,codevar,0shot,0.793,1200,0.267,0.713,0.681,0.744,above
glm4-32b-base,codevar,fair,0.892,1200,0.267,0.851,0.826,0.874,above
glm4-32b-base,sharedlists,0shot,0.547,600,0.200,0.434,0.382,0.480,above
glm4-32b-base,sharedlists,fair,0.718,600,0.200,0.648,0.604,0.689,above
glm4-32b-base,box-narrative,0shot,0.502,200,0.253,0.314,0.217,0.415,above
glm4-32b-base,box-narrative,fair,0.625,200,0.253,0.467,0.366,0.565,above
glm4-32b-base,box-screenplay,0shot,0.627,200,0.253,0.465,0.371,0.556,above
glm4-32b-base,box-screenplay,fair,0.818,200,0.253,0.755,0.680,0.828,above
glm4-32b-base,box-code,0shot,0.795,200,0.253,0.702,0.614,0.779,above
glm4-32b-base,box-code,fair,0.899,200,0.253,0.860,0.797,0.912,above
glm4-32b-base,box-chat,0shot,0.662,200,0.253,0.533,0.445,0.621,above
glm4-32b-base,box-chat,fair,0.757,200,0.253,0.658,0.573,0.739,above
glm4-32b-base,you-static,0shot,0.962,300,0.292,0.943,0.909,0.971,above
glm4-32b-base,i-moves,0shot,0.496,180,0.287,0.295,0.192,0.395,above
glm4-32b-base,i-moves,fair,0.655,180,0.287,0.519,0.426,0.619,above
glm4-32b-base,you-moves,0shot,0.303,180,0.287,0.019,-0.070,0.106,at
glm4-32b-base,you-moves,fair,0.444,180,0.287,0.221,0.115,0.319,above
glm4-32b-base,you-named,0shot,0.375,180,0.287,0.124,0.034,0.223,above
glm4-32b-base,you-named,fair,0.556,180,0.287,0.372,0.275,0.474,above
glm4-32b-base,answer-self,0shot,0.454,2560,0.250,0.272,0.247,0.298,above
glm4-32b-base,answer-self,fair,0.618,2560,0.250,0.490,0.467,0.516,above
glm4-32b-base,grammar,ladder,,1080,,0.857,0.836,0.876,above
glm4-32b-base,operators,ladder,,1080,,0.225,0.199,0.265,above
glm4-32b-base,patterns,ladder,,1080,,0.957,0.944,0.967,above
glm4-32b-base,facts,ladder,,768,,1.000,1.000,1.000,above
falcon-h1-34b,lookup,0shot,0.957,1210,0.168,0.947,0.932,0.961,above
falcon-h1-34b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
falcon-h1-34b,shortlist,0shot,0.987,960,0.040,0.986,0.978,0.992,above
falcon-h1-34b,pairs,0shot,0.969,64,0.125,0.964,0.911,1.000,above
falcon-h1-34b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
falcon-h1-34b,noise,0shot,0.783,720,0.200,0.728,0.688,0.766,above
falcon-h1-34b,now,0shot,0.107,300,0.292,-0.266,-0.313,-0.216,below
falcon-h1-34b,handoffs,0shot,0.930,1200,0.312,0.896,0.873,0.917,above
falcon-h1-34b,handoffs,fair,0.951,1200,0.312,0.927,0.909,0.944,above
falcon-h1-34b,sets,0shot,0.934,320,0.275,0.907,0.866,0.948,above
falcon-h1-34b,sets,fair,0.898,320,0.275,0.860,0.813,0.905,above
falcon-h1-34b,refilled,0shot,0.828,720,0.200,0.785,0.748,0.818,above
falcon-h1-34b,refills,0shot,0.725,600,0.200,0.656,0.612,0.700,above
falcon-h1-34b,codevar,0shot,0.741,1200,0.267,0.640,0.605,0.672,above
falcon-h1-34b,codevar,fair,0.887,1200,0.267,0.843,0.818,0.868,above
falcon-h1-34b,sharedlists,0shot,0.801,600,0.200,0.751,0.712,0.791,above
falcon-h1-34b,sharedlists,fair,0.832,600,0.200,0.790,0.754,0.828,above
falcon-h1-34b,box-narrative,0shot,0.487,200,0.253,0.316,0.220,0.415,above
falcon-h1-34b,box-narrative,fair,0.738,200,0.253,0.646,0.553,0.722,above
falcon-h1-34b,box-screenplay,0shot,0.770,200,0.253,0.680,0.595,0.759,above
falcon-h1-34b,box-screenplay,fair,0.882,200,0.253,0.836,0.772,0.895,above
falcon-h1-34b,box-code,0shot,0.725,200,0.253,0.623,0.535,0.706,above
falcon-h1-34b,box-code,fair,0.915,200,0.253,0.879,0.821,0.931,above
falcon-h1-34b,box-chat,0shot,0.755,200,0.253,0.664,0.584,0.747,above
falcon-h1-34b,box-chat,fair,0.840,200,0.253,0.779,0.702,0.848,above
falcon-h1-34b,you-static,0shot,0.943,300,0.292,0.915,0.875,0.950,above
falcon-h1-34b,i-moves,0shot,0.461,180,0.287,0.243,0.143,0.347,above
falcon-h1-34b,i-moves,fair,0.642,180,0.287,0.499,0.403,0.604,above
falcon-h1-34b,you-moves,0shot,0.344,180,0.287,0.081,-0.022,0.176,at
falcon-h1-34b,you-moves,fair,0.539,180,0.287,0.355,0.260,0.454,above
falcon-h1-34b,you-named,0shot,0.497,180,0.287,0.294,0.194,0.397,above
falcon-h1-34b,you-named,fair,0.689,180,0.287,0.557,0.462,0.654,above
falcon-h1-34b,answer-self,0shot,0.528,2560,0.250,0.371,0.346,0.396,above
falcon-h1-34b,answer-self,fair,0.659,2560,0.250,0.545,0.521,0.569,above
falcon-h1-34b,grammar,ladder,,1080,,0.874,0.859,0.892,above
falcon-h1-34b,operators,ladder,,1080,,0.245,0.215,0.278,above
falcon-h1-34b,patterns,ladder,,1080,,0.945,0.930,0.960,above
falcon-h1-34b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-70b,lookup,0shot,0.539,1210,0.168,0.438,0.402,0.472,above
llama3.1-70b,lookup,fair,0.998,1210,0.168,0.997,0.994,1.000,above
llama3.1-70b,shortlist,0shot,0.979,960,0.040,0.978,0.969,0.988,above
llama3.1-70b,pairs,0shot,0.172,64,0.125,0.054,-0.036,0.161,at
llama3.1-70b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
llama3.1-70b,noise,0shot,0.887,720,0.200,0.859,0.831,0.884,above
llama3.1-70b,now,0shot,0.184,300,0.292,-0.153,-0.208,-0.091,below
llama3.1-70b,handoffs,0shot,0.911,1200,0.312,0.870,0.845,0.892,above
llama3.1-70b,handoffs,fair,0.982,1200,0.312,0.973,0.962,0.984,above
llama3.1-70b,sets,0shot,0.931,320,0.275,0.901,0.860,0.938,above
llama3.1-70b,sets,fair,0.941,320,0.275,0.924,0.890,0.955,above
llama3.1-70b,refilled,0shot,0.629,720,0.200,0.536,0.495,0.580,above
llama3.1-70b,refills,0shot,0.665,600,0.200,0.581,0.533,0.626,above
llama3.1-70b,codevar,0shot,0.805,1200,0.267,0.731,0.699,0.761,above
llama3.1-70b,codevar,fair,0.890,1200,0.267,0.843,0.818,0.867,above
llama3.1-70b,sharedlists,0shot,0.840,600,0.200,0.800,0.766,0.837,above
llama3.1-70b,sharedlists,fair,0.720,600,0.200,0.650,0.606,0.693,above
llama3.1-70b,box-narrative,0shot,0.350,200,0.253,0.104,0.015,0.195,above
llama3.1-70b,box-narrative,fair,0.747,200,0.253,0.629,0.539,0.713,above
llama3.1-70b,box-screenplay,0shot,0.650,200,0.253,0.532,0.446,0.615,above
llama3.1-70b,box-screenplay,fair,0.887,200,0.253,0.849,0.787,0.904,above
llama3.1-70b,box-code,0shot,0.767,200,0.253,0.670,0.575,0.754,above
llama3.1-70b,box-code,fair,0.915,200,0.253,0.878,0.817,0.929,above
llama3.1-70b,box-chat,0shot,0.702,200,0.253,0.594,0.507,0.682,above
llama3.1-70b,box-chat,fair,0.897,200,0.253,0.855,0.793,0.910,above
llama3.1-70b,you-static,0shot,0.998,300,0.292,0.998,0.990,1.000,above
llama3.1-70b,i-moves,0shot,0.371,180,0.287,0.112,0.017,0.213,above
llama3.1-70b,i-moves,fair,0.674,180,0.287,0.544,0.452,0.640,above
llama3.1-70b,you-moves,0shot,0.344,180,0.287,0.078,-0.016,0.172,at
llama3.1-70b,you-moves,fair,0.531,180,0.287,0.340,0.241,0.444,above
llama3.1-70b,you-named,0shot,0.535,180,0.287,0.350,0.255,0.450,above
llama3.1-70b,you-named,fair,0.731,180,0.287,0.623,0.533,0.712,above
llama3.1-70b,answer-self,0shot,0.508,2560,0.250,0.344,0.322,0.372,above
llama3.1-70b,answer-self,fair,0.704,2560,0.250,0.605,0.583,0.627,above
llama3.1-70b,grammar,ladder,,1080,,0.853,0.833,0.873,above
llama3.1-70b,operators,ladder,,1080,,0.318,0.285,0.349,above
llama3.1-70b,patterns,ladder,,1080,,0.956,0.943,0.969,above
llama3.1-70b,facts,ladder,,768,,1.000,1.000,1.000,above
qwen2.5-72b,lookup,0shot,0.953,1210,0.168,0.939,0.924,0.954,above
qwen2.5-72b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
qwen2.5-72b,shortlist,0shot,0.985,960,0.040,0.985,0.977,0.992,above
qwen2.5-72b,pairs,0shot,0.984,64,0.125,0.982,0.946,1.000,above
qwen2.5-72b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
qwen2.5-72b,noise,0shot,0.921,720,0.200,0.902,0.878,0.926,above
qwen2.5-72b,now,0shot,0.102,300,0.292,-0.269,-0.316,-0.221,below
qwen2.5-72b,handoffs,0shot,0.938,1200,0.312,0.909,0.889,0.928,above
qwen2.5-72b,handoffs,fair,0.986,1200,0.312,0.979,0.969,0.989,above
qwen2.5-72b,sets,0shot,0.953,320,0.275,0.936,0.904,0.964,above
qwen2.5-72b,sets,fair,0.947,320,0.275,0.928,0.894,0.960,above
qwen2.5-72b,refilled,0shot,0.685,720,0.200,0.607,0.563,0.650,above
qwen2.5-72b,refills,0shot,0.680,600,0.200,0.600,0.551,0.648,above
qwen2.5-72b,codevar,0shot,0.750,1200,0.267,0.653,0.620,0.686,above
qwen2.5-72b,codevar,fair,0.873,1200,0.267,0.823,0.797,0.850,above
qwen2.5-72b,sharedlists,0shot,0.838,600,0.200,0.798,0.762,0.832,above
qwen2.5-72b,sharedlists,fair,0.906,600,0.200,0.882,0.854,0.909,above
qwen2.5-72b,box-narrative,0shot,0.294,200,0.253,0.036,-0.052,0.124,at
qwen2.5-72b,box-narrative,fair,0.667,200,0.253,0.537,0.440,0.623,above
qwen2.5-72b,box-screenplay,0shot,0.665,200,0.253,0.540,0.453,0.630,above
qwen2.5-72b,box-screenplay,fair,0.863,200,0.253,0.813,0.750,0.874,above
qwen2.5-72b,box-code,0shot,0.733,200,0.253,0.628,0.535,0.712,above
qwen2.5-72b,box-code,fair,0.858,200,0.253,0.802,0.730,0.871,above
qwen2.5-72b,box-chat,0shot,0.715,200,0.253,0.621,0.539,0.701,above
qwen2.5-72b,box-chat,fair,0.749,200,0.253,0.658,0.571,0.736,above
qwen2.5-72b,you-static,0shot,0.930,300,0.292,0.896,0.854,0.937,above
qwen2.5-72b,i-moves,0shot,0.386,180,0.287,0.135,0.041,0.237,above
qwen2.5-72b,i-moves,fair,0.764,180,0.287,0.670,0.576,0.755,above
qwen2.5-72b,you-moves,0shot,0.327,180,0.287,0.055,-0.042,0.148,at
qwen2.5-72b,you-moves,fair,0.586,180,0.287,0.423,0.323,0.518,above
qwen2.5-72b,you-named,0shot,0.508,180,0.287,0.313,0.214,0.409,above
qwen2.5-72b,you-named,fair,0.778,180,0.287,0.689,0.610,0.769,above
qwen2.5-72b,answer-self,0shot,0.489,2560,0.250,0.319,0.295,0.345,above
qwen2.5-72b,answer-self,fair,0.623,2560,0.250,0.497,0.473,0.521,above
qwen2.5-72b,grammar,ladder,,1080,,0.873,0.858,0.889,above
qwen2.5-72b,operators,ladder,,1080,,0.224,0.193,0.252,above
qwen2.5-72b,patterns,ladder,,1080,,0.990,0.984,0.996,above
qwen2.5-72b,facts,ladder,,768,,1.000,1.000,1.000,above
llama3.1-405b,lookup,0shot,0.801,1210,0.168,0.750,0.724,0.779,above
llama3.1-405b,lookup,fair,1.000,1210,0.168,1.000,1.000,1.000,above
llama3.1-405b,shortlist,0shot,0.987,960,0.040,0.986,0.978,0.992,above
llama3.1-405b,pairs,0shot,0.906,64,0.125,0.893,0.804,0.964,above
llama3.1-405b,pairs,fair,1.000,64,0.125,1.000,1.000,1.000,above
llama3.1-405b,noise,0shot,0.900,720,0.200,0.875,0.847,0.901,above
llama3.1-405b,now,0shot,0.075,300,0.292,-0.309,-0.351,-0.263,below
llama3.1-405b,handoffs,0shot,0.948,1200,0.312,0.922,0.904,0.940,above
llama3.1-405b,handoffs,fair,0.993,1200,0.312,0.990,0.982,0.996,above
llama3.1-405b,sets,0shot,0.991,320,0.275,0.986,0.968,1.000,above
llama3.1-405b,sets,fair,0.991,320,0.275,0.989,0.979,1.000,above
llama3.1-405b,refilled,0shot,0.786,720,0.200,0.733,0.696,0.768,above
llama3.1-405b,refills,0shot,0.795,600,0.200,0.744,0.704,0.781,above
llama3.1-405b,codevar,0shot,0.851,1200,0.267,0.793,0.765,0.819,above
llama3.1-405b,codevar,fair,0.943,1200,0.267,0.919,0.899,0.938,above
llama3.1-405b,sharedlists,0shot,0.881,600,0.200,0.851,0.820,0.882,above
llama3.1-405b,sharedlists,fair,0.979,600,0.200,0.974,0.960,0.987,above
llama3.1-405b,box-narrative,0shot,0.395,200,0.253,0.168,0.073,0.271,above
llama3.1-405b,box-narrative,fair,0.845,200,0.253,0.772,0.692,0.842,above
llama3.1-405b,box-screenplay,0shot,0.695,200,0.253,0.571,0.481,0.666,above
llama3.1-405b,box-screenplay,fair,0.910,200,0.253,0.868,0.806,0.918,above
llama3.1-405b,box-code,0shot,0.833,200,0.253,0.761,0.682,0.829,above
llama3.1-405b,box-code,fair,0.877,200,0.253,0.820,0.751,0.884,above
llama3.1-405b,box-chat,0shot,0.835,200,0.253,0.767,0.696,0.840,above
llama3.1-405b,box-chat,fair,0.917,200,0.253,0.885,0.831,0.937,above
llama3.1-405b,you-static,0shot,0.997,300,0.292,0.995,0.985,1.000,above
llama3.1-405b,i-moves,0shot,0.442,180,0.287,0.217,0.131,0.326,above
llama3.1-405b,i-moves,fair,0.811,180,0.287,0.737,0.659,0.812,above
llama3.1-405b,you-moves,0shot,0.341,180,0.287,0.075,-0.027,0.172,at
llama3.1-405b,you-moves,fair,0.711,180,0.287,0.594,0.500,0.678,above
llama3.1-405b,you-named,0shot,0.468,180,0.287,0.252,0.159,0.345,above
llama3.1-405b,you-named,fair,0.739,180,0.287,0.637,0.545,0.722,above
llama3.1-405b,answer-self,0shot,0.452,2560,0.250,0.269,0.243,0.295,above
llama3.1-405b,answer-self,fair,0.768,2560,0.250,0.691,0.670,0.712,above
llama3.1-405b,grammar,ladder,,1080,,0.903,0.886,0.920,above
llama3.1-405b,operators,ladder,,1080,,0.214,0.186,0.244,above
llama3.1-405b,patterns,ladder,,1080,,0.976,0.966,0.984,above
llama3.1-405b,facts,ladder,,768,,0.999,0.998,1.000,above
tsumugi-s17-a5B,lookup,0shot,0.478,1210,0.168,0.372,0.336,0.404,above
tsumugi-s17-a5B,lookup,fair,0.717,1210,0.168,0.659,0.628,0.690,above
tsumugi-s17-a5B,shortlist,0shot,0.498,960,0.040,0.477,0.444,0.507,above
tsumugi-s17-a5B,pairs,0shot,0.336,64,0.125,0.241,0.107,0.375,above
tsumugi-s17-a5B,pairs,fair,0.531,64,0.125,0.464,0.339,0.607,above
tsumugi-s17-a5B,noise,0shot,0.491,720,0.200,0.364,0.319,0.408,above
tsumugi-s17-a5B,now,0shot,0.458,300,0.292,0.235,0.157,0.308,above
tsumugi-s17-a5B,handoffs,0shot,0.370,1200,0.312,0.082,0.045,0.122,above
tsumugi-s17-a5B,handoffs,fair,0.494,1200,0.312,0.259,0.219,0.299,above
tsumugi-s17-a5B,sets,0shot,0.408,320,0.275,0.195,0.116,0.266,above
tsumugi-s17-a5B,sets,fair,0.458,320,0.275,0.258,0.181,0.334,above
tsumugi-s17-a5B,refilled,0shot,0.235,720,0.200,0.043,0.006,0.082,above
tsumugi-s17-a5B,refills,0shot,0.198,600,0.200,-0.003,-0.041,0.034,at
tsumugi-s17-a5B,codevar,0shot,0.258,1200,0.267,-0.008,-0.042,0.026,at
tsumugi-s17-a5B,codevar,fair,0.256,1200,0.267,-0.018,-0.051,0.016,at
tsumugi-s17-a5B,sharedlists,0shot,0.265,600,0.200,0.081,0.038,0.127,above
tsumugi-s17-a5B,sharedlists,fair,0.210,600,0.200,0.013,-0.026,0.051,at
tsumugi-s17-a5B,box-narrative,0shot,0.315,200,0.253,0.078,-0.009,0.172,at
tsumugi-s17-a5B,box-narrative,fair,0.310,200,0.253,0.064,-0.027,0.152,at
tsumugi-s17-a5B,box-screenplay,0shot,0.395,200,0.253,0.201,0.114,0.291,above
tsumugi-s17-a5B,box-screenplay,fair,0.355,200,0.253,0.133,0.035,0.219,above
tsumugi-s17-a5B,box-code,0shot,0.290,200,0.253,0.061,-0.025,0.149,at
tsumugi-s17-a5B,box-code,fair,0.412,200,0.253,0.197,0.099,0.292,above
tsumugi-s17-a5B,box-chat,0shot,0.398,200,0.253,0.187,0.098,0.282,above
tsumugi-s17-a5B,box-chat,fair,0.350,200,0.253,0.121,0.022,0.211,above
tsumugi-s17-a5B,you-static,0shot,0.387,300,0.292,0.132,0.056,0.214,above
tsumugi-s17-a5B,i-moves,0shot,0.319,180,0.287,0.040,-0.055,0.129,at
tsumugi-s17-a5B,i-moves,fair,0.247,180,0.287,-0.060,-0.143,0.027,at
tsumugi-s17-a5B,you-moves,0shot,0.300,180,0.287,0.009,-0.084,0.105,at
tsumugi-s17-a5B,you-moves,fair,0.256,180,0.287,-0.048,-0.139,0.041,at
tsumugi-s17-a5B,you-named,0shot,0.275,180,0.287,-0.018,-0.106,0.075,at
tsumugi-s17-a5B,you-named,fair,0.283,180,0.287,-0.008,-0.094,0.095,at
tsumugi-s17-a5B,answer-self,0shot,0.307,2560,0.250,0.076,0.053,0.100,above
tsumugi-s17-a5B,answer-self,fair,0.276,2560,0.250,0.035,0.012,0.057,above
tsumugi-s17-a5B,grammar,ladder,,1080,,0.665,0.635,0.690,above
tsumugi-s17-a5B,operators,ladder,,1080,,0.252,0.224,0.281,above
tsumugi-s17-a5B,patterns,ladder,,1080,,0.562,0.525,0.595,above
tsumugi-s17-a5B,facts,ladder,,768,,0.635,0.598,0.671,above
tsumugi-s7-a5B,lookup,0shot,0.487,1210,0.168,0.379,0.346,0.412,above
tsumugi-s7-a5B,lookup,fair,0.828,1210,0.168,0.792,0.765,0.817,above
tsumugi-s7-a5B,shortlist,0shot,0.527,960,0.040,0.507,0.474,0.538,above
tsumugi-s7-a5B,pairs,0shot,0.273,64,0.125,0.170,0.054,0.295,above
tsumugi-s7-a5B,pairs,fair,0.555,64,0.125,0.491,0.348,0.616,above
tsumugi-s7-a5B,noise,0shot,0.536,720,0.200,0.420,0.374,0.468,above
tsumugi-s7-a5B,now,0shot,0.397,300,0.292,0.147,0.071,0.231,above
tsumugi-s7-a5B,handoffs,0shot,0.346,1200,0.312,0.050,0.013,0.087,above
tsumugi-s7-a5B,handoffs,fair,0.523,1200,0.312,0.301,0.262,0.344,above
tsumugi-s7-a5B,sets,0shot,0.334,320,0.275,0.091,0.019,0.163,above
tsumugi-s7-a5B,sets,fair,0.468,320,0.275,0.270,0.195,0.340,above
tsumugi-s7-a5B,refilled,0shot,0.239,720,0.200,0.049,0.011,0.087,above
tsumugi-s7-a5B,refills,0shot,0.172,600,0.200,-0.035,-0.073,0.001,at
tsumugi-s7-a5B,codevar,0shot,0.276,1200,0.267,0.012,-0.023,0.047,at
tsumugi-s7-a5B,codevar,fair,0.236,1200,0.267,-0.047,-0.081,-0.014,below
tsumugi-s7-a5B,sharedlists,0shot,0.249,600,0.200,0.061,0.018,0.104,above
tsumugi-s7-a5B,sharedlists,fair,0.233,600,0.200,0.041,-0.004,0.085,at
tsumugi-s7-a5B,box-narrative,0shot,0.253,200,0.253,-0.028,-0.115,0.061,at
tsumugi-s7-a5B,box-narrative,fair,0.320,200,0.253,0.075,-0.018,0.164,at
tsumugi-s7-a5B,box-screenplay,0shot,0.378,200,0.253,0.170,0.075,0.261,above
tsumugi-s7-a5B,box-screenplay,fair,0.378,200,0.253,0.153,0.060,0.240,above
tsumugi-s7-a5B,box-code,0shot,0.340,200,0.253,0.131,0.037,0.220,above
tsumugi-s7-a5B,box-code,fair,0.477,200,0.253,0.289,0.189,0.383,above
tsumugi-s7-a5B,box-chat,0shot,0.352,200,0.253,0.119,0.028,0.205,above
tsumugi-s7-a5B,box-chat,fair,0.385,200,0.253,0.166,0.071,0.255,above
tsumugi-s7-a5B,you-static,0shot,0.365,300,0.292,0.104,0.033,0.181,above
tsumugi-s7-a5B,i-moves,0shot,0.283,180,0.287,-0.013,-0.095,0.082,at
tsumugi-s7-a5B,i-moves,fair,0.239,180,0.287,-0.074,-0.163,0.018,at
tsumugi-s7-a5B,you-moves,0shot,0.222,180,0.287,-0.100,-0.181,-0.011,below
tsumugi-s7-a5B,you-moves,fair,0.233,180,0.287,-0.078,-0.167,0.012,at
tsumugi-s7-a5B,you-named,0shot,0.300,180,0.287,0.017,-0.071,0.106,at
tsumugi-s7-a5B,you-named,fair,0.261,180,0.287,-0.038,-0.123,0.048,at
tsumugi-s7-a5B,answer-self,0shot,0.277,2560,0.250,0.036,0.014,0.060,above
tsumugi-s7-a5B,answer-self,fair,0.281,2560,0.250,0.042,0.018,0.065,above
tsumugi-s7-a5B,grammar,ladder,,1080,,0.654,0.624,0.679,above
tsumugi-s7-a5B,operators,ladder,,1080,,0.219,0.189,0.251,above
tsumugi-s7-a5B,patterns,ladder,,1080,,0.491,0.458,0.526,above
tsumugi-s7-a5B,facts,ladder,,768,,0.654,0.616,0.692,above
tsumugi-s17-a11B,lookup,0shot,0.534,1210,0.168,0.433,0.399,0.469,above
tsumugi-s17-a11B,lookup,fair,0.907,1210,0.168,0.887,0.867,0.906,above
tsumugi-s17-a11B,shortlist,0shot,0.515,960,0.040,0.494,0.463,0.527,above
tsumugi-s17-a11B,pairs,0shot,0.344,64,0.125,0.250,0.125,0.375,above
tsumugi-s17-a11B,pairs,fair,0.719,64,0.125,0.679,0.536,0.804,above
tsumugi-s17-a11B,noise,0shot,0.446,720,0.200,0.307,0.260,0.352,above
tsumugi-s17-a11B,now,0shot,0.532,300,0.292,0.341,0.261,0.418,above
tsumugi-s17-a11B,handoffs,0shot,0.419,1200,0.312,0.152,0.114,0.194,above
tsumugi-s17-a11B,handoffs,fair,0.541,1200,0.312,0.330,0.287,0.371,above
tsumugi-s17-a11B,sets,0shot,0.455,320,0.275,0.260,0.185,0.337,above
tsumugi-s17-a11B,sets,fair,0.491,320,0.275,0.306,0.230,0.383,above
tsumugi-s17-a11B,refilled,0shot,0.157,720,0.200,-0.054,-0.088,-0.021,below
tsumugi-s17-a11B,refills,0shot,0.189,600,0.200,-0.013,-0.051,0.027,at
tsumugi-s17-a11B,codevar,0shot,0.267,1200,0.267,-0.003,-0.039,0.032,at
tsumugi-s17-a11B,codevar,fair,0.250,1200,0.267,-0.028,-0.063,0.006,at
tsumugi-s17-a11B,sharedlists,0shot,0.231,600,0.200,0.038,-0.004,0.078,at
tsumugi-s17-a11B,sharedlists,fair,0.228,600,0.200,0.034,-0.005,0.080,at
tsumugi-s17-a11B,box-narrative,0shot,0.347,200,0.253,0.117,0.027,0.211,above
tsumugi-s17-a11B,box-narrative,fair,0.403,200,0.253,0.190,0.092,0.294,above
tsumugi-s17-a11B,box-screenplay,0shot,0.502,200,0.253,0.322,0.228,0.421,above
tsumugi-s17-a11B,box-screenplay,fair,0.512,200,0.253,0.333,0.231,0.438,above
tsumugi-s17-a11B,box-code,0shot,0.287,200,0.253,0.054,-0.030,0.143,at
tsumugi-s17-a11B,box-code,fair,0.535,200,0.253,0.346,0.234,0.441,above
tsumugi-s17-a11B,box-chat,0shot,0.419,200,0.253,0.195,0.096,0.286,above
tsumugi-s17-a11B,box-chat,fair,0.475,200,0.253,0.272,0.168,0.370,above
tsumugi-s17-a11B,you-static,0shot,0.383,300,0.292,0.126,0.050,0.196,above
tsumugi-s17-a11B,i-moves,0shot,0.275,180,0.287,-0.020,-0.113,0.061,at
tsumugi-s17-a11B,i-moves,fair,0.361,180,0.287,0.102,-0.005,0.205,at
tsumugi-s17-a11B,you-moves,0shot,0.267,180,0.287,-0.030,-0.117,0.050,at
tsumugi-s17-a11B,you-moves,fair,0.306,180,0.287,0.026,-0.065,0.124,at
tsumugi-s17-a11B,you-named,0shot,0.292,180,0.287,0.002,-0.090,0.089,at
tsumugi-s17-a11B,you-named,fair,0.306,180,0.287,0.017,-0.072,0.114,at
tsumugi-s17-a11B,answer-self,0shot,0.317,2560,0.250,0.089,0.066,0.112,above
tsumugi-s17-a11B,answer-self,fair,0.304,2560,0.250,0.072,0.049,0.095,above
tsumugi-s17-a11B,grammar,ladder,,1080,,0.759,0.730,0.781,above
tsumugi-s17-a11B,operators,ladder,,1080,,0.267,0.238,0.293,above
tsumugi-s17-a11B,patterns,ladder,,1080,,0.556,0.518,0.585,above
tsumugi-s17-a11B,facts,ladder,,768,,0.713,0.674,0.741,above
tsumugi-s7-a11B,lookup,0shot,0.515,1210,0.168,0.412,0.377,0.447,above
tsumugi-s7-a11B,lookup,fair,0.916,1210,0.168,0.898,0.878,0.916,above
tsumugi-s7-a11B,shortlist,0shot,0.545,960,0.040,0.526,0.494,0.558,above
tsumugi-s7-a11B,pairs,0shot,0.461,64,0.125,0.384,0.241,0.518,above
tsumugi-s7-a11B,pairs,fair,0.602,64,0.125,0.545,0.411,0.679,above
tsumugi-s7-a11B,noise,0shot,0.513,720,0.200,0.392,0.345,0.437,above
tsumugi-s7-a11B,now,0shot,0.542,300,0.292,0.351,0.274,0.426,above
tsumugi-s7-a11B,handoffs,0shot,0.437,1200,0.312,0.178,0.138,0.215,above
tsumugi-s7-a11B,handoffs,fair,0.559,1200,0.312,0.354,0.312,0.392,above
tsumugi-s7-a11B,sets,0shot,0.405,320,0.275,0.189,0.118,0.260,above
tsumugi-s7-a11B,sets,fair,0.510,320,0.275,0.333,0.254,0.410,above
tsumugi-s7-a11B,refilled,0shot,0.201,720,0.200,0.001,-0.034,0.036,at
tsumugi-s7-a11B,refills,0shot,0.221,600,0.200,0.026,-0.015,0.071,at
tsumugi-s7-a11B,codevar,0shot,0.307,1200,0.267,0.057,0.025,0.094,above
tsumugi-s7-a11B,codevar,fair,0.275,1200,0.267,0.007,-0.029,0.043,at
tsumugi-s7-a11B,sharedlists,0shot,0.270,600,0.200,0.087,0.046,0.133,above
tsumugi-s7-a11B,sharedlists,fair,0.246,600,0.200,0.057,0.016,0.100,above
tsumugi-s7-a11B,box-narrative,0shot,0.398,200,0.253,0.197,0.106,0.289,above
tsumugi-s7-a11B,box-narrative,fair,0.490,200,0.253,0.301,0.201,0.399,above
tsumugi-s7-a11B,box-screenplay,0shot,0.492,200,0.253,0.321,0.229,0.415,above
tsumugi-s7-a11B,box-screenplay,fair,0.482,200,0.253,0.288,0.187,0.386,above
tsumugi-s7-a11B,box-code,0shot,0.393,200,0.253,0.181,0.090,0.272,above
tsumugi-s7-a11B,box-code,fair,0.580,200,0.253,0.416,0.313,0.506,above
tsumugi-s7-a11B,box-chat,0shot,0.482,200,0.253,0.298,0.203,0.393,above
tsumugi-s7-a11B,box-chat,fair,0.487,200,0.253,0.286,0.178,0.386,above
tsumugi-s7-a11B,you-static,0shot,0.393,300,0.292,0.136,0.055,0.217,above
tsumugi-s7-a11B,i-moves,0shot,0.286,180,0.287,-0.005,-0.097,0.090,at
tsumugi-s7-a11B,i-moves,fair,0.292,180,0.287,0.005,-0.091,0.100,at
tsumugi-s7-a11B,you-moves,0shot,0.258,180,0.287,-0.044,-0.133,0.039,at
tsumugi-s7-a11B,you-moves,fair,0.289,180,0.287,0.001,-0.094,0.097,at
tsumugi-s7-a11B,you-named,0shot,0.256,180,0.287,-0.044,-0.132,0.047,at
tsumugi-s7-a11B,you-named,fair,0.328,180,0.287,0.053,-0.036,0.148,at
tsumugi-s7-a11B,answer-self,0shot,0.304,2560,0.250,0.072,0.048,0.094,above
tsumugi-s7-a11B,answer-self,fair,0.294,2560,0.250,0.058,0.033,0.079,above
tsumugi-s7-a11B,grammar,ladder,,1080,,0.773,0.743,0.797,above
tsumugi-s7-a11B,operators,ladder,,1080,,0.232,0.195,0.263,above
tsumugi-s7-a11B,patterns,ladder,,1080,,0.674,0.643,0.703,above
tsumugi-s7-a11B,facts,ladder,,768,,0.752,0.716,0.786,above
tsumugi-s17-a33B,lookup,0shot,0.706,1210,0.168,0.636,0.607,0.667,above
tsumugi-s17-a33B,lookup,fair,0.942,1210,0.168,0.931,0.914,0.946,above
tsumugi-s17-a33B,shortlist,0shot,0.609,960,0.040,0.593,0.560,0.625,above
tsumugi-s17-a33B,pairs,0shot,0.586,64,0.125,0.527,0.393,0.652,above
tsumugi-s17-a33B,pairs,fair,0.703,64,0.125,0.661,0.536,0.777,above
tsumugi-s17-a33B,noise,0shot,0.521,720,0.200,0.402,0.356,0.444,above
tsumugi-s17-a33B,now,0shot,0.738,300,0.292,0.627,0.555,0.689,above
tsumugi-s17-a33B,handoffs,0shot,0.448,1200,0.312,0.193,0.151,0.230,above
tsumugi-s17-a33B,handoffs,fair,0.520,1200,0.312,0.300,0.259,0.339,above
tsumugi-s17-a33B,sets,0shot,0.520,320,0.275,0.353,0.281,0.422,above
tsumugi-s17-a33B,sets,fair,0.520,320,0.275,0.358,0.284,0.434,above
tsumugi-s17-a33B,refilled,0shot,0.257,720,0.200,0.072,0.032,0.116,above
tsumugi-s17-a33B,refills,0shot,0.210,600,0.200,0.013,-0.028,0.053,at
tsumugi-s17-a33B,codevar,0shot,0.315,1200,0.267,0.057,0.019,0.092,above
tsumugi-s17-a33B,codevar,fair,0.282,1200,0.267,0.013,-0.025,0.050,at
tsumugi-s17-a33B,sharedlists,0shot,0.200,600,0.200,0.000,-0.040,0.043,at
tsumugi-s17-a33B,sharedlists,fair,0.211,600,0.200,0.013,-0.029,0.058,at
tsumugi-s17-a33B,box-narrative,0shot,0.400,200,0.253,0.176,0.085,0.263,above
tsumugi-s17-a33B,box-narrative,fair,0.517,200,0.253,0.343,0.250,0.439,above
tsumugi-s17-a33B,box-screenplay,0shot,0.515,200,0.253,0.335,0.239,0.431,above
tsumugi-s17-a33B,box-screenplay,fair,0.490,200,0.253,0.302,0.204,0.399,above
tsumugi-s17-a33B,box-code,0shot,0.435,200,0.253,0.253,0.162,0.347,above
tsumugi-s17-a33B,box-code,fair,0.635,200,0.253,0.483,0.386,0.572,above
tsumugi-s17-a33B,box-chat,0shot,0.407,200,0.253,0.171,0.071,0.267,above
tsumugi-s17-a33B,box-chat,fair,0.515,200,0.253,0.333,0.234,0.427,above
tsumugi-s17-a33B,you-static,0shot,0.620,300,0.292,0.455,0.374,0.527,above
tsumugi-s17-a33B,i-moves,0shot,0.372,180,0.287,0.118,0.020,0.213,above
tsumugi-s17-a33B,i-moves,fair,0.364,180,0.287,0.108,0.002,0.216,above
tsumugi-s17-a33B,you-moves,0shot,0.222,180,0.287,-0.098,-0.182,-0.012,below
tsumugi-s17-a33B,you-moves,fair,0.342,180,0.287,0.077,-0.014,0.181,at
tsumugi-s17-a33B,you-named,0shot,0.289,180,0.287,0.000,-0.082,0.101,at
tsumugi-s17-a33B,you-named,fair,0.333,180,0.287,0.059,-0.032,0.163,at
tsumugi-s17-a33B,answer-self,0shot,0.320,2560,0.250,0.093,0.070,0.115,above
tsumugi-s17-a33B,answer-self,fair,0.312,2560,0.250,0.083,0.061,0.107,above
tsumugi-s17-a33B,grammar,ladder,,1080,,0.786,0.762,0.813,above
tsumugi-s17-a33B,operators,ladder,,1080,,0.279,0.247,0.311,above
tsumugi-s17-a33B,patterns,ladder,,1080,,0.737,0.709,0.764,above
tsumugi-s17-a33B,facts,ladder,,768,,0.886,0.856,0.909,above
tsumugi-s17-a55B,lookup,0shot,0.694,1210,0.168,0.625,0.592,0.655,above
tsumugi-s17-a55B,lookup,fair,0.976,1210,0.168,0.972,0.962,0.982,above
tsumugi-s17-a55B,shortlist,0shot,0.622,960,0.040,0.606,0.573,0.639,above
tsumugi-s17-a55B,pairs,0shot,0.594,64,0.125,0.536,0.411,0.679,above
tsumugi-s17-a55B,pairs,fair,0.734,64,0.125,0.696,0.571,0.821,above
tsumugi-s17-a55B,noise,0shot,0.574,720,0.200,0.467,0.423,0.513,above
tsumugi-s17-a55B,now,0shot,0.550,300,0.292,0.362,0.284,0.441,above
tsumugi-s17-a55B,handoffs,0shot,0.542,1200,0.312,0.329,0.288,0.368,above
tsumugi-s17-a55B,handoffs,fair,0.635,1200,0.312,0.463,0.423,0.503,above
tsumugi-s17-a55B,sets,0shot,0.475,320,0.275,0.283,0.207,0.357,above
tsumugi-s17-a55B,sets,fair,0.580,320,0.275,0.438,0.365,0.508,above
tsumugi-s17-a55B,refilled,0shot,0.371,720,0.200,0.213,0.170,0.260,above
tsumugi-s17-a55B,refills,0shot,0.272,600,0.200,0.090,0.045,0.134,above
tsumugi-s17-a55B,codevar,0shot,0.301,1200,0.267,0.041,0.007,0.079,above
tsumugi-s17-a55B,codevar,fair,0.321,1200,0.267,0.067,0.030,0.102,above
tsumugi-s17-a55B,sharedlists,0shot,0.221,600,0.200,0.027,-0.013,0.067,at
tsumugi-s17-a55B,sharedlists,fair,0.268,600,0.200,0.084,0.042,0.130,above
tsumugi-s17-a55B,box-narrative,0shot,0.417,200,0.253,0.191,0.092,0.283,above
tsumugi-s17-a55B,box-narrative,fair,0.443,200,0.253,0.239,0.145,0.327,above
tsumugi-s17-a55B,box-screenplay,0shot,0.557,200,0.253,0.414,0.317,0.512,above
tsumugi-s17-a55B,box-screenplay,fair,0.507,200,0.253,0.322,0.221,0.413,above
tsumugi-s17-a55B,box-code,0shot,0.470,200,0.253,0.285,0.195,0.377,above
tsumugi-s17-a55B,box-code,fair,0.652,200,0.253,0.505,0.403,0.598,above
tsumugi-s17-a55B,box-chat,0shot,0.502,200,0.253,0.332,0.234,0.424,above
tsumugi-s17-a55B,box-chat,fair,0.555,200,0.253,0.383,0.278,0.471,above
tsumugi-s17-a55B,you-static,0shot,0.592,300,0.292,0.419,0.339,0.499,above
tsumugi-s17-a55B,i-moves,0shot,0.364,180,0.287,0.106,0.009,0.208,above
tsumugi-s17-a55B,i-moves,fair,0.308,180,0.287,0.031,-0.059,0.130,at
tsumugi-s17-a55B,you-moves,0shot,0.222,180,0.287,-0.093,-0.176,-0.013,below
tsumugi-s17-a55B,you-moves,fair,0.331,180,0.287,0.059,-0.038,0.159,at
tsumugi-s17-a55B,you-named,0shot,0.272,180,0.287,-0.023,-0.110,0.071,at
tsumugi-s17-a55B,you-named,fair,0.311,180,0.287,0.029,-0.065,0.121,at
tsumugi-s17-a55B,answer-self,0shot,0.305,2560,0.250,0.074,0.050,0.098,above
tsumugi-s17-a55B,answer-self,fair,0.295,2560,0.250,0.060,0.038,0.084,above
tsumugi-s17-a55B,grammar,ladder,,1080,,0.777,0.753,0.805,above
tsumugi-s17-a55B,operators,ladder,,1080,,0.266,0.238,0.296,above
tsumugi-s17-a55B,patterns,ladder,,1080,,0.747,0.723,0.772,above
tsumugi-s17-a55B,facts,ladder,,768,,0.895,0.870,0.920,above
tsumugi-s17-a88B,lookup,0shot,0.618,1210,0.168,0.533,0.497,0.566,above
tsumugi-s17-a88B,lookup,fair,0.976,1210,0.168,0.971,0.960,0.981,above
tsumugi-s17-a88B,shortlist,0shot,0.646,960,0.040,0.631,0.599,0.662,above
tsumugi-s17-a88B,pairs,0shot,0.578,64,0.125,0.518,0.393,0.643,above
tsumugi-s17-a88B,pairs,fair,0.672,64,0.125,0.625,0.500,0.750,above
tsumugi-s17-a88B,noise,0shot,0.640,720,0.200,0.550,0.503,0.591,above
tsumugi-s17-a88B,now,0shot,0.802,300,0.292,0.715,0.647,0.778,above
tsumugi-s17-a88B,handoffs,0shot,0.510,1200,0.312,0.282,0.242,0.326,above
tsumugi-s17-a88B,handoffs,fair,0.642,1200,0.312,0.476,0.433,0.515,above
tsumugi-s17-a88B,sets,0shot,0.436,320,0.275,0.228,0.156,0.299,above
tsumugi-s17-a88B,sets,fair,0.603,320,0.275,0.461,0.390,0.528,above
tsumugi-s17-a88B,refilled,0shot,0.459,720,0.200,0.324,0.281,0.371,above
tsumugi-s17-a88B,refills,0shot,0.379,600,0.200,0.224,0.175,0.271,above
tsumugi-s17-a88B,codevar,0shot,0.350,1200,0.267,0.108,0.069,0.147,above
tsumugi-s17-a88B,codevar,fair,0.365,1200,0.267,0.128,0.091,0.165,above
tsumugi-s17-a88B,sharedlists,0shot,0.258,600,0.200,0.072,0.029,0.116,above
tsumugi-s17-a88B,sharedlists,fair,0.324,600,0.200,0.155,0.110,0.205,above
tsumugi-s17-a88B,box-narrative,0shot,0.438,200,0.253,0.209,0.112,0.309,above
tsumugi-s17-a88B,box-narrative,fair,0.453,200,0.253,0.235,0.131,0.337,above
tsumugi-s17-a88B,box-screenplay,0shot,0.573,200,0.253,0.423,0.328,0.518,above
tsumugi-s17-a88B,box-screenplay,fair,0.490,200,0.253,0.289,0.186,0.386,above
tsumugi-s17-a88B,box-code,0shot,0.510,200,0.253,0.343,0.250,0.436,above
tsumugi-s17-a88B,box-code,fair,0.650,200,0.253,0.496,0.398,0.590,above
tsumugi-s17-a88B,box-chat,0shot,0.487,200,0.253,0.280,0.177,0.372,above
tsumugi-s17-a88B,box-chat,fair,0.522,200,0.253,0.351,0.257,0.442,above
tsumugi-s17-a88B,you-static,0shot,0.655,300,0.292,0.511,0.430,0.577,above
tsumugi-s17-a88B,i-moves,0shot,0.350,180,0.287,0.086,-0.009,0.194,at
tsumugi-s17-a88B,i-moves,fair,0.311,180,0.287,0.035,-0.057,0.137,at
tsumugi-s17-a88B,you-moves,0shot,0.203,180,0.287,-0.122,-0.205,-0.037,below
tsumugi-s17-a88B,you-moves,fair,0.292,180,0.287,0.003,-0.086,0.097,at
tsumugi-s17-a88B,you-named,0shot,0.344,180,0.287,0.080,-0.012,0.179,at
tsumugi-s17-a88B,you-named,fair,0.372,180,0.287,0.118,0.018,0.218,above
tsumugi-s17-a88B,answer-self,0shot,0.323,2560,0.250,0.098,0.072,0.121,above
tsumugi-s17-a88B,answer-self,fair,0.319,2560,0.250,0.092,0.069,0.116,above
tsumugi-s17-a88B,grammar,ladder,,1080,,0.818,0.797,0.842,above
tsumugi-s17-a88B,operators,ladder,,1080,,0.295,0.270,0.330,above
tsumugi-s17-a88B,patterns,ladder,,1080,,0.801,0.773,0.824,above
tsumugi-s17-a88B,facts,ladder,,768,,0.911,0.889,0.933,above
```
