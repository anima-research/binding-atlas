# The card's words

The card is the 1600×900 image used as the model card's picture and the announcement's link preview, and the video's
last frame. Text is HTML; the bar labels (`*.bar_*`, `cheap.tsumugi_line`) sit in an SVG, so keep those to plain text.
`{{n:NAME}}` is a number the build fills in from its record: move it around freely, but change the number itself only
at its source.

## card.caption
The middle block runs twice. <b>Blue</b>: where blocks begin; each layer reads a learned mix of the blocks below.

## card.romaji
kotodama · tsumugi

## card.sub
a 1.4B-parameter base model · 110B training tokens

## card.rules.title
Learns made-up rules from its prompt

## card.rules.example
ta mu → ta mo · mo mi → mo mo · <span class="m">… 6 more …</span><br><span class="q">pe fi → pe ___</span> <span class="m">picks</span> <span class="q">fo</span> <span class="m">from po, fe, fu, fi, fo</span>

## card.rules.label
score over many such tests · 0 = chance, 1 = perfect

## card.rules.bar_ckpt
best checkpoint

## card.rules.note
public models at the same training compute<br>checkpoint: one saved partway through training

## card.loop.title
Rule learning leans on the 2nd pass

## card.loop.label
rule-learning score lost when 60 random attention heads are switched off in

## card.loop.bar_pass2
2nd pass

## card.loop.bar_pass1
1st pass

## card.loop.bar_outside
outside the loop

## card.loop.note
from {{n:loop_base}} on the published test items · fact recall, same cut: {{n:recall_cut2}}

## card.you.title
Tracks who “you” is in a chat

## card.you.example
<span class="m">Kurt: Here, take my lamp.<br>Steve: My battery is in your hand now.</span>

## card.you.prompt
Right now you’re holding the ___

## card.you.answer
battery

## card.you.note
said by Kurt, “you” is Steve, who has the lamp<br>across 180 chats the average drop is smaller: {{n:dyn_avg}}

## card.cheap.title
Ahead of Pythia 1.4B–6.9B at its compute

## card.cheap.label
compute each model needed to reach tsumugi’s {{n:avg9}}% correct, averaged over 9 standard benchmarks

## card.cheap.tsumugi_line
tsumugi 1×

## card.cheap.note
level with OLMo-2 1B · HellaSwag, LAMBADA, ARC, PIQA, SciQ, WinoGrande, COPA, BoolQ

## card.foot
open weights, every training checkpoint, the code and the benchmark

## card.url
kotodama.animalabs.ai
