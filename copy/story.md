# The story's and the video's words

The announcement's scroll story and the animated card (the video) are built from the same beats, so each entry below
shows in both, except the `*.step` paragraphs (the story's text column) and `*.toc` (its contents rail), which are the
story's only. Text is HTML; inside the charts (the `stack.*`, `rules.axis`, `you.left` … entries that sit in an SVG),
keep to plain text.

`{{n:NAME}}` is a number the build fills in from its record (`rule` is tsumugi's rule-learning score, `cut2` the
second-pass cut, `avg9` the nine-benchmark average, and so on): move it around freely, but change the number itself
only at its source. `{{cite:AttnRes}}` is the numbered citation marker; its leaflet is the entry `cite.AttnRes`.

The easiest way to edit is `make edit` in this repo. To see how a change reads in the video without re-rendering it,
`build_anim.py --stills DIR` draws one still per beat.

## story.toc
tsumugi

## story.arch.title
design spec

## story.arch.sub
Added one at a time.

## story.arch.toc
design spec

## story.arch.step
At each step of the process, we performed ablations to test four levers towards enhancing binding.
<br><br>
Depth was prioritized for serial operations; 52 layers and 1536d wide, vs 16-28 layers and mostly 2048d wide for other open models near this param count.

Moonshot AI's Attention Residuals{{cite:AttnRes}} (AttnRes) are used to allow each layer to read a learned mix of prior blocks' outputs instead of a typical running sum.

Instead of uniform blocks, we pretrained the same model without AttnRes and read layer geometry to identify natural seams and place blocks to maintain inter-layer circuitry.

This fit found a 20-layer contiguous block starting from layer 19, which we selected to run twice with the same weights, such that the end result has 72 logical layers (each pass with its own norms).
<br><br>
Our finding is that each piece synergizes with another, accentuating or unlocking benefits. For example, AttnRes (only with fitted boundaries, not uniform) composes with depth to unlock strong early {{note:set-bind|set binding}}; looped depth provides the same as more layers for free in weights on ICL & binding.

## cite.AttnRes
Kimi Team (Moonshot AI), <a href="https://arxiv.org/abs/2603.15031"><i>Attention Residuals</i></a>, arXiv:2603.15031, 2026.

## story.arch.1.head
Depth

## story.arch.1.body
Ablations compared 52 layers × 1,536 against 28 × 2,048: within 4B training tokens, the deeper model formed early binding on all three seeds, the wider one on none.

## story.arch.1.formula

## story.arch.2.head
Attention Residuals

## story.arch.2.body
Typical residual streams give every earlier layer weight 1. Here, weights are learned and depend on the input.

## story.arch.2.formula
plain:  h<sub>l+1</sub> = h<sub>l</sub> + f<sub>l</sub>(h<sub>l</sub>)<br>here:  h<sub>l</sub> = Σ<sub>n</sub> α<sub>n,l</sub> b<sub>n</sub>,  α<sub>n,l</sub> = softmax<sub>n</sub>( w<sub>l</sub> · RMSNorm(b<sub>n</sub>) )<br><span class="lab">b<sub>n</sub>: the summed outputs of block n · w<sub>l</sub>: a learned query for layer l</span>

## story.arch.3.head
Fitted Boundaries

## story.arch.3.body
A seam is a layer where attention spread, hidden-state shape and weight rank jump together. The seams gave five input blocks and one long block from layer 19.

## story.arch.3.formula
blocks start at layers 0, 1, 7, 9, 17, 19, 39 of the 52

## story.arch.4.head
Looped Core

## story.arch.4.body
Because the second pass is its own block, every later layer can weigh what each pass wrote against the other.

## story.arch.4.formula
f<sub>39+i</sub> = f<sub>19+i</sub>,  i = 0 … 19<br>blocks start at layers 0, 1, 7, 9, 17, 19, 39, 59 of the 72

## story.arch.together
72 layers of computation from 52 layers of weights, with every layer choosing what to read from the blocks below it.

## story.arch.hint_hover
Hover over a step to read it again.

## story.arch.hint_tap
Tap a step to read it again.

## stack.input
input layers

## stack.input_n
19

## stack.middle
middle block

## stack.middle_sub
20 layers, 1st pass

## stack.middle_sub_closed
20 layers

## stack.middle_again
middle block, again

## stack.middle_again_sub
same weights, 2nd pass

## stack.output
output layers

## stack.output_n
13

## story.rules.title
in-context learning

## story.rules.sub
Worked examples of an invented rule, then a new word to apply it to.

## story.rules.toc
in-context learning

## story.rules.step
We evaluated base models on how well they pick up patterns (not reliant upon factual recall, much like Raven's matrices) from examples in context, following ICLEval (Chen et al. 2024){{cite:ICLEval}}. An item is scored right when the model ranks the correct answer the highest from a fixed set. Accuracy is then corrected for chance, where 0 is guessing & 1 is perfect, and averaged over nine levels of difficulty per test.
<br><br>
tsumugi's final checkpoint settles at {{n:rule}}, well above the compute trend for open-weight models, Falcon3 3B being the nearest at {{n:best_final_score}} (pruned and distilled from Falcon3 7B; <a href="#story-data">more below</a>). When the same test uses known facts, such that knowledge offers a shortcut, tsumugi scores higher ({{n:cryst}}), as every model does, landing where the compute trend predicts. 
<br><br>
<b>Of the {{n:n_final}} fully trained open-weight models we measured, none scores higher until about {{n:cross_mult}} times tsumugi's training compute; the first, {{n:cross_model}}, used {{n:cross_flops}} FLOPs. tsumugi sits 2.4 SD above the trend; with {{note:outliers|outliers}} excluded, 4.5.</b>

## rules.item_label
a pattern recognition test

## rules.more
… 4 more examples …

## rules.pick
1 right answer, 5 options

## rules.axis
rule-learning score · 0 = chance, 1 = perfect

## rules.legend1
● 15 public checkpoints, partway through training · ○ 2 fully

## rules.legend2
trained public models: all at the same training compute

## story.you.title
referent binding

## story.you.sub
Tracking who is being referred to, and what belongs to them.

## story.you.toc
referent binding

## story.you.step
Referent binding in prose is the task of tracking who is being referred to, and what belongs to them. A simple test of this capability is the following: "John is holding a bat. Jane is holding a flower. They swap items. John is holding the ______", where flower is the answer.
<br><br>
When personal pronouns such as "I/you/me" are used, this task becomes harder; the referent depends on perspective, who the active speaker is, and context clues. This capability is acquired in stages, in trajectories we've analyzed. First, "I" binds cleanly, then "you" when state doesn't shift, and finally "you"  while the state changes.
<br><br>
The figure is a test for that case; possession & speakers change, with clear statements of ownership. A model must predict what "you" is holding. We then also swap who speaks the last line, so that "you" becomes the other speaker, dropping the probability of the prior answer.
<br><br>
On this test, tsumugi's average drop in probability is {{n:dyn_avg}}, with a 95% CI clearing {{note:bar|+.02}}. <b>tsumugi is the smallest public base model to achieve this;</b> the next is Qwen3 4B. Scale also doesn't guarantee this capability; OLMo-2 7B and 13B don’t pass, nor do GLM-4 32B or Qwen2.5 72B.

## note.bar
+.02 = two pp drop on average, with the whole 95% interval above that line. The bar simply shows that the model moves the right way at all, not how well it does so. The value was arbitrary & fixed before any scoring occurred; an improved test derives it from “null twin” chats, where swapping speakers won’t change the answer, and puts the bar where noise alone stops.

## you.chat_label
one test chat (after 3 worked examples)

## you.prompt
Right now you're holding the ___

## you.is_kurt
“you” = Kurt, who now has the battery

## you.is_steve
“you” = Steve, who now has the lamp

## you.prob_label
probability of “battery”

## you.moves
this chat moves more than most: the average across 180 is {{n:dyn_avg}}

## you.bars_title
average drop, by model

## you.left
← takes “you” as the speaker

## you.right
tracks “you” →

## you.zero
0: ignores who’s speaking

## you.bar
+.02 bar

## you.claim1
the smallest public base model whose whole 95% interval

## you.claim2
clears +.02, a bar fixed before any model was scored

## story.cheap.title
world knowledge evals

## story.cheap.sub


## story.cheap.toc
world knowledge evals

## story.cheap.step
On nine standard benchmarks (HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, PIQA, SciQ, WinoGrande, COPA and BoolQ), tsumugi's zero-shot average is {{n:avg9}}% correct. On Uncheatable Eval{{cite:Uncheatable}}, which evaluates compression on unseen human text, tsumugi scores .768 bits per byte, 0.9 SD better than the trend. tsumugi's score here shows that the enhancements in binding & ICL come at little to no cost on knowledge work: it scores ahead of Pythia 1.4B, trained with twice its compute, and within a point of Pythia 2.8B, trained with four times as much, most likely attributable to either more modern architecture or the data mix.
<br><br>
Given the prior results with binding & ICL, and the knowledge-based benchmarks here, tsumugi operates <i>in context</i> like a model much larger; while having <i>world knowledge</i> expected for its compute.

## cheap.grid_title
0-shot accuracy (%)

## cheap.avg
average

## cheap.compute
compute, × tsumugi’s

## cheap.foot
* distilled

## cheap.bpb_title
Uncheatable Eval: bits per byte, lower is better

## cheap.bpb_y


## cheap.bpb_x
training FLOPs

## cheap.models
base models, fully trained

## cheap.trend
trend

## story.data.title
blank document completions

## story.data.sub
only &lt;start_of_document&gt; or equivalents are given; sampled for 256 tokens

## story.data.toc
data

## story.data.step
tsumugi's training data was sourced specifically to avoid as much synthetic or otherwise LLM-generated or altered text as possible. This is explicitly to make it a "true" base with no distillation involved, and therefore a cleaner bar for understanding architectural differences & benchmark trends. 
<br><br>
Of the 36 models shown in the ICL chart above, 19 report their pretraining included synthetic or LLM-generated/altered text, and 4 more are distilled from a larger model; 23 in total.
<br><br>
Compute estimates for these models are rough; we don't calculate how much compute went into generating the non-human data nor the compute that went into training the teacher/previous model. An example of this is Falcon3 3B, which was created via pruning the larger Falcon3 7B, then trained on 100B tokens with a distillation objective.

## note.outliers
Trend w/ models more than 2 SD away excluded & refitted. Below: RecurrentGemma 9B and Bamba 9B (recurrent hybrids) and Pythia 160M (the smallest model here); above: Falcon3 3B.

## data.link
Read all 128 runs →

## data.more
oldest first: runs of 128 with signs of synthetic or LLM data

## data.tiles_title
128 runs from an empty document, one square per run

## data.era
with signs of synthetic or LLM-generated data

## data.start
[start of document]

## data.k_s
chat transcript

## data.k_v
assistant voice

## data.k_x
synthetic instruction data

## data.k_h
human-written Q&amp;A

## data.k_p
ordinary text

## story.arch.1.video
Depth was prioritized for serial operations: 52 layers, 1,536 wide. At the same 1.42B parameters, it formed set binding within 4B training tokens on all three seeds; 28 layers × 2,048 did on none.

## story.arch.2.video
Each layer reads a learned, input-dependent mix of prior blocks' outputs instead of a typical running sum (Moonshot AI's Attention Residuals).

## story.arch.3.video
Instead of uniform blocks, we read the layer geometry of the same 52 layers trained without AttnRes and placed blocks at its natural seams, where attention spread, hidden-state shape and weight rank jump together.

## story.arch.4.video
The fit found a 20-layer block starting at layer 19. It runs twice with the same weights, so the model computes through 72 layers while storing 52.

## cite.Uncheatable
Tan, Li, Shen, <a href="https://arxiv.org/abs/2609.27510"><i>Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models</i></a>, 2026.

## cite.ICLEval
Chen, Lin, Zhou, Huang, Jia, Cao, Wen, <a href="https://arxiv.org/abs/2406.14955"><i>ICLEval: Evaluating In-Context Learning Ability of Large Language Models</i></a>, 2024.

## rules.hero_y
chance-corrected accuracy (0 = guessing, 1 = always right)

## rules.hero_x
training compute (FLOPs)

## rules.hero_legend_t
tsumugi

## rules.hero_legend_full
final checkpoints

## rules.hero_legend_mid
in-training checkpoints

## rules.hero_legend_trend
trend of the fully trained

## rules.hero_legend_distilled
distilled from a larger model

## note.set-bind
Whether the model keeps track of which items belong to which group, measured at 4B tokens of training.
