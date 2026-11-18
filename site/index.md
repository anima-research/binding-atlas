<!-- kotodama-tsumugi: plain-text version of https://kotodama.animalabs.ai/ for language models and screen readers. The same words as the page; every note and reference is expanded where it appears, and every figure is described in text from the data it draws. Build 2026-11-18 18:16 UTC. -->

**言霊** kotodama — word-spirit: language as a thing with a life of its own

**紬** tsumugi — spun thread: words bound into more than the sum of their parts

# tsumugi

A 1.4B-parameter looped depth base model, trained on 110B tokens using kotodama, our open pretraining stack: trainer, evaluation and serving in one package.

tsumugi performs in-context learning (ICL) on par with models up to 5x larger & trained with far more compute.

tsumugi also tracks who “you” refers to in two-person chats, like models trained with hundreds of times more compute.

Binding (keeping track of who is who) and ICL are the two things we set out to study with this model.

On standard benchmarks like HellaSwag and Uncheatable Eval [ref: Tan, Li, Shen, [*Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models*](https://arxiv.org/abs/2609.27510), 2026.], tsumugi scores at or above where its compute predicts.

Synthetic & distillation datasets were explicitly omitted, with an estimated 0.5–1% residual contamination from generated text.

Weights, checkpoints, data recipe, code, and the benchmark are all available.

Led by [Luxia](https://x.com/kataletheia), with [Anima Labs](https://animalabs.ai) · built with [Claude](https://www.anthropic.com/claude)

[code on GitHub](https://github.com/anima-research/kotodama) · [weights on Hugging Face](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b) · [benchmark results](https://kotodama.animalabs.ai/binding-atlas) · [neuron graphs](https://kotodama.animalabs.ai/threads)

### design spec

At each step of the process, we performed ablations to test four levers towards enhancing binding.

Depth was prioritized for serial operations; 52 layers and 1536d wide, vs 16-28 layers and mostly 2048d wide for other open models near this param count.

Moonshot AI's Attention Residuals [ref: Kimi Team (Moonshot AI), [*Attention Residuals*](https://arxiv.org/abs/2603.15031), arXiv:2603.15031, 2026.] (AttnRes) are used to allow each layer to read a learned mix of prior blocks' outputs instead of a typical running sum.

Instead of uniform blocks, we pretrained the same model without AttnRes and read layer geometry to identify natural seams and place blocks to maintain inter-layer circuitry.

This fit found a 20-layer contiguous block starting from layer 19, which we selected to run twice with the same weights, such that the end result has 72 logical layers (each pass with its own norms).

Our finding is that each piece synergizes with another, accentuating or unlocking benefits. For example, AttnRes (only with fitted boundaries, not uniform) composes with depth to unlock strong early set binding [note: Whether the model keeps track of which items belong to which group, measured at 4B tokens of training.]; looped depth provides the same as more layers for free in weights on ICL & binding.

**Figure: design spec** (an animation; described in text).

A stack of 72 layers, drawn bottom to top: input layers (19), middle block (20 layers, 1st pass), middle block, again (same weights, 2nd pass), output layers (13); a bracket marks the middle block as run ×2. Lines mark where the attention-residual blocks start, at layers 0, 1, 7, 9, 17, 19, 39 and 59; they first appear evenly spaced (at 0, 7, 15, 22, 30, 37, 65) and slide to these fitted places.

Beside the stack, the four design steps, added one at a time:

1. **Depth**: Ablations compared 52 layers × 1,536 against 28 × 2,048: within 4B training tokens, the deeper model formed early binding on all three seeds, the wider one on none.

2. **Attention Residuals**: Typical residual streams give every earlier layer weight 1. Here, weights are learned and depend on the input. Formula: plain: h_l+1 = h_l + f_l(h_l) · here: h_l = Σ_n α_n,l b_n, α_n,l = softmax_n( w_l · RMSNorm(b_n) ) · b_n: the summed outputs of block n · w_l: a learned query for layer l

3. **Fitted Boundaries**: A seam is a layer where attention spread, hidden-state shape and weight rank jump together. The seams gave five input blocks and one long block from layer 19. Formula: blocks start at layers 0, 1, 7, 9, 17, 19, 39 of the 52

4. **Looped Core**: Because the second pass is its own block, every later layer can weigh what each pass wrote against the other. Formula: f_39+i = f_19+i, i = 0 … 19 · blocks start at layers 0, 1, 7, 9, 17, 19, 39, 59 of the 72

72 layers of computation from 52 layers of weights, with every layer choosing what to read from the blocks below it.

### in-context learning

We evaluated base models on how well they pick up patterns (not reliant upon factual recall, much like Raven's matrices) from examples in context, following ICLEval (Chen et al. 2024) [ref: Chen, Lin, Zhou, Huang, Jia, Cao, Wen, [*ICLEval: Evaluating In-Context Learning Ability of Large Language Models*](https://arxiv.org/abs/2406.14955), 2024.]. An item is scored right when the model ranks the correct answer the highest from a fixed set. Accuracy is then corrected for chance, where 0 is guessing & 1 is perfect, and averaged over nine levels of difficulty per test.

tsumugi's final checkpoint settles at .559, well above the compute trend for open-weight models, Falcon3 3B being the nearest at .536 (pruned and distilled from Falcon3 7B; more below). When the same test uses known facts, such that knowledge offers a shortcut, tsumugi scores higher (.930), as every model does, landing where the compute trend predicts.

**Of the 36 fully trained open-weight models we measured, none scores higher until about 150 times tsumugi's training compute; the first, OLMo-2 7B, used 1.8 × 10²³ FLOPs. tsumugi sits 2.4 SD above the trend; with outliers [note: Trend w/ models more than 2 SD away excluded & refitted. Below: RecurrentGemma 9B and Bamba 9B (recurrent hybrids) and Pythia 160M (the smallest model here); above: Falcon3 3B.] excluded, 4.5.**

**Figure: in-context learning** (described in text).

Left, one item of the test: worked examples ta mu → ta mo; mo mi → mo mo; du bu → du bo; se pi → se po, then “… 4 more examples …”, then the query pe fi → ?, with five options (po, fe, fu, fi, fo); the right answer is fo (1 right answer, 5 options).

Right, a scatter. Vertical axis: chance-corrected accuracy (0 = guessing, 1 = always right). Horizontal axis: training compute (FLOPs), log scale from 3.0e19 to 2.5e24. Points: 36 final checkpoints of open-weight base models (dots; 4 of them distilled from a larger model, drawn as asterisks), 15 in-training checkpoints (hollow circles), and tsumugi (a highlighted diamond, labelled .559) at 1.2e21 FLOPs. A dashed line, the trend of the fully trained, runs from .281 at 3.2e19 to .606 at 2.5e24; at tsumugi's compute it gives .387.

### referent binding

Referent binding in prose is the task of tracking who is being referred to, and what belongs to them. A simple test of this capability is the following: "John is holding a bat. Jane is holding a flower. They swap items. John is holding the ______", where flower is the answer.

When personal pronouns such as "I/you/me" are used, this task becomes harder; the referent depends on perspective, who the active speaker is, and context clues. This capability is acquired in stages, in trajectories we've analyzed. First, "I" binds cleanly, then "you" when state doesn't shift, and finally "you" while the state changes.

The figure is a test for that case; possession & speakers change, with clear statements of ownership. A model must predict what "you" is holding. We then also swap who speaks the last line, so that "you" becomes the other speaker, dropping the probability of the prior answer.

On this test, tsumugi's average drop in probability is .038, with a 95% CI clearing +.02 [note: +.02 = two pp drop on average, with the whole 95% interval above that line. The bar simply shows that the model moves the right way at all, not how well it does so. The value was arbitrary & fixed before any scoring occurred; an improved test derives it from “null twin” chats, where swapping speakers won’t change the answer, and puts the bar where noise alone stops.]. **tsumugi is the smallest public base model to achieve this;** the next is Qwen3 4B. Scale also doesn't guarantee this capability; OLMo-2 7B and 13B don’t pass, nor do GLM-4 32B or Qwen2.5 72B.

**Figure: referent binding** (an animation; described in text).

Left, one test chat (after 3 worked examples):

> A chat between Steve and Kurt.
> Steve: The battery is in my hand.
> Steve: The shoe is in my hand.
> Kurt: I have the lamp.
> Kurt: Here, take my lamp.
> Steve: My battery is in your hand now.

> Steve: Right now you're holding the ___

With Steve saying the last line, “you” = Kurt, who now has the battery; the probability of “battery” is .445. The figure then swaps who says that line (“Kurt: Right now you're holding the ___”), so “you” = Steve, who now has the lamp, and the probability of “battery” falls to .311. This chat moves more than most: the average across 180 is .038.

Right, average drop, by model (sorted by size; bars with 95% whiskers; a dashed line marks the +.02 bar; left of zero: takes “you” as the speaker; right: tracks “you”; 0: ignores who’s speaking):

| model | parameters | average drop | 95% interval | whole interval above +.02 |
|---|---|---|---|---|
| Qwen2.5 0.5B | 0.49B | +.012 | -.003 to +.028 | no |
| Qwen3 0.6B | 0.60B | +.000 | -.018 to +.020 | no |
| OLMo-1 1B | 1.28B | -.013 | -.029 to +.004 | no |
| Pythia 1.4B | 1.41B | -.023 | -.037 to -.008 | no |
| **tsumugi 1.4B** | 1.42B | +.038 | +.023 to +.055 | yes |
| OLMo-2 1B | 1.48B | -.018 | -.030 to -.008 | no |
| Qwen2.5 1.5B | 1.54B | -.036 | -.053 to -.019 | no |
| SmolLM2 1.7B | 1.71B | +.006 | -.012 to +.026 | no |
| Qwen3 1.7B | 1.72B | +.026 | -.007 to +.059 | no |
| Pythia 2.8B | 2.78B | +.022 | +.002 to +.043 | no |
| SmolLM3 3B | 3.08B | +.027 | +.009 to +.048 | no |
| Qwen2.5 3B | 3.09B | -.006 | -.029 to +.015 | no |
| Qwen3 4B | 4.02B | +.055 | +.034 to +.077 | yes |
| OLMo-2 7B | 7.30B | -.015 | -.024 to -.005 | no |
| Qwen2.5 7B | 7.62B | +.073 | +.044 to +.104 | yes |
| Qwen3 8B | 8.19B | +.080 | +.048 to +.112 | yes |
| Gemma-2 9B | 9.24B | +.034 | +.022 to +.045 | yes |

### world knowledge evals

On nine standard benchmarks (HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, PIQA, SciQ, WinoGrande, COPA and BoolQ), tsumugi's zero-shot average is 64.1% correct. On Uncheatable Eval [ref: Tan, Li, Shen, [*Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models*](https://arxiv.org/abs/2609.27510), 2026.], which evaluates compression on unseen human text, tsumugi scores .768 bits per byte, 0.9 SD better than the trend. tsumugi's score here shows that the enhancements in binding & ICL come at little to no cost on knowledge work: it scores ahead of Pythia 1.4B, trained with twice its compute, and within a point of Pythia 2.8B, trained with four times as much, most likely attributable to either more modern architecture or the data mix.

Given the prior results with binding & ICL, and the knowledge-based benchmarks here, tsumugi operates *in context* like a model much larger; while having *world knowledge* expected for its compute.

**Figure: world knowledge evals** (described in text).

Left, 0-shot accuracy (%), one column per model in order of training compute (cells shaded by score within each row; * distilled from a larger model):

| | tsumugi 1.4B | Pythia 1.4B | Pythia 2.8B | OLMo 1B | OLMo-2 1B | Llama-3.2 1B | SmolLM2 1.7B | Qwen3 1.7B |
|---|---|---|---|---|---|---|---|---|
| HellaSwag | 58.1 | 51.9 | 59.4 | 66.1 | 68.3 | 63.7 | 71.3 | 66.5 |
| LAMBADA | 60.5 | 60.9 | 63.9 | 60.6 | 65.0 | 62.1 | 67.3 | 63.1 |
| ARC-Easy | 62.9 | 60.8 | 64.1 | 61.0 | 72.6 | 65.4 | 78.0 | 73.4 |
| ARC-Challenge | 31.7 | 28.2 | 31.9 | 32.3 | 42.1 | 36.2 | 47.4 | 44.9 |
| PIQA | 71.4 | 70.9 | 73.8 | 74.9 | 75.8 | 74.3 | 77.0 | 75.7 |
| SciQ | 92.6 | 86.8 | 88.8 | 92.6 | 95.3 | 91.5 | 93.1 | 95.8 |
| WinoGrande | 61.6 | 57.3 | 58.6 | 61.3 | 64.3 | 61.3 | 66.3 | 64.4 |
| COPA | 74.0 | 72.0 | 78.0 | 80.0 | 83.0 | 77.0 | 83.0 | 82.0 |
| BoolQ | 64.5 | 63.3 | 63.4 | 66.1 | 62.9 | 63.9 | 72.4 | 79.2 |
| **average** | 64.1 | 61.4 | 64.7 | 66.1 | 69.9 | 66.1 | 72.9 | 71.7 |
| compute, × tsumugi’s | 1× | 2.2× | 4.2× | 19× | 29× | 54×* | 91× | 299× |

Right, Uncheatable Eval: bits per byte, lower is better, against training FLOPs (log scale): 26 base models, fully trained (dots, from 0.548 to 0.810 bits per byte) and their trend (a dashed least-squares line on log compute, fitted after dropping any model more than 2 SD from it, which drops Pythia 160M). tsumugi is a highlighted diamond labelled .768 at 1.2e21 FLOPs, below the line, which gives .805 at that compute.

### null completions

tsumugi's training data was sourced specifically to avoid as much synthetic or otherwise LLM-generated or altered text as possible. This is explicitly to make it a "true" base with no distillation involved, and therefore a cleaner bar for understanding architectural differences & benchmark trends.

Of the 36 models shown in the ICL chart above, 19 report their pretraining included synthetic or LLM-generated/altered text, and 4 more are distilled from a larger model; 23 in total.

Compute estimates for these models are rough; we don't calculate how much compute went into generating the non-human data nor the compute that went into training the teacher/previous model. An example of this is Falcon3 3B, which was created via pruning the larger Falcon3 7B, then trained on 100B tokens with a distillation objective.

**Figure: blank document completions** (an animation; described in text).

First, 128 runs from an empty document, one square per run, coloured by kind:

- kotodama-tsumugi 1.4B: 0 of 128 with signs of synthetic or LLM-generated data (ordinary text 128)
- Falcon3 3B Base: 50 of 128 with signs of synthetic or LLM-generated data (assistant voice 14, synthetic instruction data 36, human-written Q&A 2, ordinary text 76)

Then oldest first: runs of 128 with signs of synthetic or LLM data:

| year | model | chat transcript | assistant voice | synthetic instruction data | human-written Q&A | ordinary text | with signs of synthetic or LLM data |
|---|---|---|---|---|---|---|---|
| 2019 | GPT-2 XL | 0 | 0 | 0 | 1 | 127 | 0 of 128 |
| 2023 | Pythia 1.4B | 0 | 0 | 0 | 16 | 112 | 0 of 128 |
| 2023 | Pythia 160M | 0 | 0 | 0 | 23 | 105 | 0 of 128 |
| 2023 | Pythia 2.8B | 0 | 0 | 0 | 18 | 110 | 0 of 128 |
| 2023 | Pythia 6.9B | 0 | 0 | 0 | 15 | 113 | 0 of 128 |
| 2024 | Gemma 2 2B | 0 | 0 | 0 | 8 | 120 | 0 of 128 |
| 2024 | OLMo 1B (0724) | 0 | 0 | 0 | 29 | 99 | 0 of 128 |
| 2024 | SmolLM2 1.7B | 0 | 0 | 0 | 0 | 128 | 0 of 128 |
| 2024 | SmolLM2 360M | 0 | 0 | 0 | 3 | 125 | 0 of 128 |
| 2024 | Qwen2.5 3B | 8 | 3 | 4 | 8 | 105 | 15 of 128 |
| 2024 | Llama 3.2 1B | 0 | 0 | 18 | 31 | 79 | 18 of 128 |
| 2024 | Llama 3.1 8B | 0 | 0 | 19 | 30 | 79 | 19 of 128 |
| 2024 | Llama 3.2 3B | 0 | 0 | 22 | 28 | 78 | 22 of 128 |
| 2024 | OLMo 2 7B | 19 | 1 | 6 | 49 | 53 | 26 of 128 |
| 2024 | Qwen2.5 7B | 38 | 2 | 6 | 13 | 69 | 46 of 128 |
| 2024 | **Falcon3 3B** | 0 | 14 | 36 | 2 | 76 | 50 of 128 |
| 2024 | Qwen2.5 1.5B | 62 | 0 | 1 | 11 | 54 | 63 of 128 |
| 2024 | Qwen2.5 0.5B | 63 | 1 | 7 | 7 | 50 | 71 of 128 |
| 2025 | Gemma 3 4B | 0 | 0 | 0 | 6 | 122 | 0 of 128 |
| 2025 | SmolLM3 3B | 0 | 0 | 0 | 0 | 128 | 0 of 128 |
| 2025 | Qwen3 8B | 1 | 3 | 2 | 4 | 118 | 6 of 128 |
| 2025 | OLMo 3 7B | 0 | 1 | 6 | 16 | 105 | 7 of 128 |
| 2025 | OLMo 3 32B | 1 | 1 | 8 | 17 | 101 | 10 of 128 |
| 2025 | Qwen3 1.7B | 1 | 12 | 2 | 6 | 107 | 15 of 128 |
| 2025 | Qwen3 0.6B | 3 | 14 | 6 | 7 | 98 | 23 of 128 |
| 2025 | Qwen3 4B | 1 | 20 | 8 | 13 | 86 | 29 of 128 |
| 2025 | OLMo 2 1B | 25 | 0 | 6 | 53 | 44 | 31 of 128 |
| 2025 | Qwen3 30B-A3B | 50 | 3 | 3 | 6 | 66 | 56 of 128 |
| 2025 | Falcon-H1 1.5B | 0 | 1 | 99 | 1 | 27 | 100 of 128 |
| 2026 | **tsumugi 1.4B** | 0 | 0 | 0 | 0 | 128 | 0 of 128 |

Beside them, one run of each of the first two models, verbatim:

kotodama-tsumugi 1.4B (ordinary text):

> [start of document]
> What motivated you to combine trading and blogging? It’s not really a choice, I guess. I am an investor jockey who wants to join the ranks of bloggers by giving away many little pieces of “information” (whether they are red flags, perceived bullish news, bearish news or big moves) for beating the market. What I was trying to avoid in the blog world is being tagged as a “street trader”.
> What’s your strategy to entry…

Falcon3 3B Base (assistant voice):

> [start of document]
> Certainly! Here are some mathematical questions and detailed answers based on the extract provided:
>
> ### Question 1: What is the formula to find the number when a percentage and the value are known?
> **Answer:**
> The formula to find the number when a percentage and the value are known is:
> \[ x = \frac{\text{value} \times 100}{\text{percentage}} \]
> In the given problem, the value is 30 and the percentage is 20%.…

Under the runs: [Read all 128 runs](https://kotodama.animalabs.ai/empty-document).

## discussion

−.13 on new rules, with a quarter of the second pass's attention switched off (recall: −.025)

### Rule learning leans on the looped block’s second pass

tsumugi runs one block of 20 layers twice, with the same weights. Switching off a quarter of the attention in the second pass lowers its new-rules score by .13 on the same 0-to-1 scale, while recall falls .025. The same cut costs .035 in the first pass and .019 in the layers before and after the loop.

This test was registered too, and its prediction holds. The cut also makes the model worse at predicting ordinary text, and the result holds after controlling for that loss.

[Inside tsumugi: the neurons behind rule learning and recall](https://kotodama.animalabs.ai/threads)

## release

- **Weights**: [LuxiaSL/kotodama-tsumugi-1b](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b), with the model card.

- **The training run**: Every checkpoint, plus the cooled-down versions, as separate branches named `anneal…`: seed 17, the release, to 110B ([trajectory](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-trajectory)), and a second seed, seed 7, to 11B ([trajectory](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-seed7-trajectory)).

- **The training stack**: [anima-research/kotodama](https://github.com/anima-research/kotodama): the trainer, the evaluation suite and the serving stack.

- **The data recipe**: [Every training source](https://github.com/anima-research/kotodama/blob/main/docs/DATA-tsumugi-1b.md), with its share of the tokens and how it was processed. 19 of the 32 sources are published as cleaned text ([kotodama-3b-corpus-open](https://huggingface.co/datasets/aethera-gp/kotodama-3b-corpus-open)); two small private sources are described by category only.

- **The benchmark**: [anima-research/binding-bench](https://github.com/anima-research/binding-bench), the tests above, [with every public model's results](https://kotodama.animalabs.ai/binding-atlas), and [how tsumugi learned them over training](https://kotodama.animalabs.ai/binding-atlas/tsumugi).

- **The neuron graph**: Every neuron of its feed-forward layers, and the strongest connections between them, for the release ([kotodama-tsumugi-1b-neuron-graph](https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph)) and across training ([its trajectory](https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph-trajectory)). [Explore it in your browser](https://kotodama.animalabs.ai/threads).

## limitations

- It is a base model, with no instruction, preference or safety tuning. It continues text and does not follow instructions.

- Without a repetition penalty it falls into self-copying loops in long generations. The model card gives sampling that avoids them.

- On held-out code, in bits per byte, it trails Pythia-1.4B and OLMo-2 1B; on held-out prose it is ahead of Pythia-1.4B.

- Like every ~1B model measured, it scores at chance on tracking who holds an object in a third-person story.

- It is one cooldown of one seed (seed 17), and its training data is mostly English.

Companion pages: [the model card](https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b) · [every public model's results on who-is-who tests](https://kotodama.animalabs.ai/binding-atlas) · [a map of tsumugi's neurons](https://kotodama.animalabs.ai/threads) · [what it writes from an empty page](https://kotodama.animalabs.ai/empty-document).

A [plain-text version of this page](https://kotodama.animalabs.ai/index.md), for language models: the same words, with every note expanded and every figure described.

build 2026-11-18 18:16 UTC
