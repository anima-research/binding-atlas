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
Our belief is that the composition is key; each accentuates a theoretical strength of the others. In tsumugi's run, both seeds (7 and 17) scored higher on set binding by 11B tokens than any of nine runs of a standard 28-layer model at 10B.

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
The prompt shows worked examples of an invented rule, here a made-up plural, and the model picks the answer for a new word. On these tests it scores {{n:rule}}, where 0 is chance and 1 is always right. Public models that spent the same training compute score lower: checkpoints taken partway through training reach at most {{n:pub_max}}, and the best fully trained one, {{n:best_final}}, {{n:best_final_score}}. Of {{n:n_small}} fully trained public base models up to 3.3B parameters, only {{n:above}}, trained on 36T tokens, scores higher ({{n:above_score}}).

## rules.item_label
pattern recognition test

## rules.more
… 4 more examples …

## rules.pick
the model picks the answer it finds most likely

## rules.axis
rule-learning score · 0 = chance, 1 = perfect

## rules.legend1
● 15 public checkpoints, partway through training · ○ 2 fully

## rules.legend2
trained public models: all at the same training compute

## story.loop.title
Rule learning leans on the second pass

## story.loop.sub
The same 20 layers run twice. Cutting attention in the second pass costs rule learning most.

## story.loop.toc
the second block

## story.loop.step
On the benchmark's published test items, where tsumugi scores {{n:loop_base}} (the {{n:rule}} above uses freshly drawn items of the same tests), switching off 60 random attention heads in the second pass, a quarter of them, drops its score to {{n:loop_cut2_score}}. The same cut costs {{n:cut1}} in the first pass and {{n:cut_out}} in the layers before and after the loop. The damage is specific: fact recall in the same format falls only {{n:recall_cut2}}, from {{n:recall_base}}, and the gap holds after accounting for how much each cut hurts text prediction.

## loop.heads_off
60 random heads off

## loop.cuts_title
rule learning lost, by where 60 random heads are switched off

## loop.cutnote
Each cut switches off 60 attention heads chosen at random, a quarter of the 240 in one pass, with five different draws per place. A second-pass cut costs rule learning {{n:cut2}}, almost four times a first-pass cut, while fact recall in the same prompts falls only {{n:recall_cut2}}. The gap holds after controlling for how much each cut hurts the model's text prediction.

## story.you.title
Tracks who “you” is as objects change hands

## story.you.sub
Swap who says the last line and “you” is the other person, so the old answer should lose probability.

## story.you.toc
Tracking “you”

## story.you.step
The test “you”, after hand-offs: on the last line of this chat, “you” is Kurt, who now has the battery. Swap who says that line, and “you” is Steve, who has the lamp, so the battery should lose probability. Here it falls from {{n:p_orig}} to {{n:p_swap}}, a larger move than most. Averaged over 180 chats the drop is {{n:dyn_avg}}, and tsumugi is the smallest public base model whose drop clears +.02, a bar fixed before any model was scored, across its whole 95% interval.

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

## you.claim1
the smallest public base model whose whole 95% interval

## you.claim2
clears +.02, a bar fixed before any model was scored

## story.cheap.title
Ahead of Pythia 1.4B–6.9B and OLMo-1 1B at its compute

## story.cheap.sub
Level with OLMo-2 1B. Each bar: the training compute a model needed to reach tsumugi's average on nine standard benchmarks.

## story.cheap.toc
Benchmarks at its compute

## story.cheap.step
On nine standard benchmarks (HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, PIQA, SciQ, WinoGrande, COPA and BoolQ), with no examples in the prompt, tsumugi averages {{n:avg9}}% correct. Public models released checkpoints from partway through training, which show how much compute each needed to reach that score. Pythia 2.8B needed {{n:mult_pythia28}} tsumugi's compute, Pythia 6.9B {{n:mult_pythia69}} and OLMo-1 1B {{n:mult_olmo1}}; Pythia 1.4B never got there. OLMo-2 1B got there at about the same compute as tsumugi.

## cheap.avg_lead
tsumugi averages

## cheap.axis
training compute, as a multiple of tsumugi's

## cheap.tsumugi_line
tsumugi, 1×

## cheap.note
Public checkpoints are taken partway through training, before any final learning-rate decay; tsumugi's {{n:avg9}} includes its own.

## story.arch.1.video
Depth was prioritized for serial operations: 52 layers, 1,536 wide. At the same 1.42B parameters, it formed set binding within 4B training tokens on all three seeds; 28 layers × 2,048 did on none.

## story.arch.2.video
Each layer reads a learned, input-dependent mix of prior blocks' outputs instead of a typical running sum (Moonshot AI's Attention Residuals).

## story.arch.3.video
Instead of uniform blocks, we read the layer geometry of the same 52 layers trained without AttnRes and placed blocks at its natural seams, where attention spread, hidden-state shape and weight rank jump together.

## story.arch.4.video
The fit found a 20-layer block starting at layer 19. It runs twice with the same weights, so the model computes through 72 layers while storing 52.
