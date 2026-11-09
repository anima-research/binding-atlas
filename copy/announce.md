# The announcement's words

Each `## key` below is one piece of text on the page; edit the text under it. Text is HTML: keep tags such as `<a href="…">`,
`<i>` and `<b>` balanced, and keep placeholders such as `__ATLAS_LINK__` as they are (the build fills them in).
`{{cite:NAME}}` puts a numbered citation marker there; its leaflet text is the entry `cite.NAME`.

The easiest way to edit is `make edit` in this repo: it opens the page, and clicking any outlined text edits it here.
Numbers are claims: change one only together with its source.

Notes for the data section (facts on record; for Luxia's writing, not shown on the page):
- Left out when choosing sources: known synthetic datasets and assistant chats (OpenCodeReasoning, SmolTalk, SODA, WildChat,
  OASST2); no model was run over the text to filter or rewrite it. Audits put model-written text at about 0.5–1% of all
  tokens seen (stray LLM text in web and forum sources; one source, fourchan, 0.06%, was LLM-rewritten upstream).
- From an empty page: tsumugi writes assistant-era text in 0 of 128 runs; Falcon3 3B in 50 of 128.
- Falcon3 3B (model card): "pruned in terms of depth and width from Falcon3-7B-Base", then trained on 100B tokens with a
  knowledge-distillation objective. 3.2B parameters (more than twice tsumugi's 1.4B); its teacher Falcon3 7B has 7.5B
  (about 5x) and 14T tokens of pretraining (about 500x tsumugi's training compute). tsumugi: 110B tokens, from scratch.
- On new rules: tsumugi .559, Falcon3 3B .536.
- tsumugi against the compute trend (straight line on log training compute, the 36 fully trained models; descriptive,
  no interval): as drawn, the trend predicts .387, spread .072, so +2.4 SD. Outliers by a fixed rule (more than 2 SD
  from the trend fitted without them): RecurrentGemma 9B -3.8, Bamba 9B -3.5, Pythia 160M -2.1, Falcon3 3B +2.1 (next:
  Llama-3 8B +1.0). Without them (32 models): predicts .396, spread .036, +4.5 SD. With nothing excluded but a fit built
  to resist outliers: +3.1 SD. 7 of the 32 kept models are hybrids or state-space models (Falcon-Mamba 7B .551, the
  Falcon-H1s, Nemotron-H 8B), and three kept models are distilled, so neither architecture nor distillation is offered as
  the cause of the outliers. The registered test (against parameters) is +1.07 SD, interval +0.68 to +2.07.
- Across the hero chart's 37 models, by their developers' own papers and cards: 4 were distilled from a larger model in
  pretraining (Falcon3 3B, Gemma-2 2B and 9B, Llama-3.2 1B); Bamba 9B and both OLMo-2 say they trained from scratch; the
  rest don't say. 19 of the 36 plotted state synthetic or rewritten text in their base pretraining (all Falcon-H1,
  Qwen2.5, Qwen3, SmolLM2 and OLMo-2, plus Bamba, Falcon-Mamba and Nemotron-H); none of the 4 distilled do, so 23 of 36
  state one or the other; the other 13 don't say (Pythia, Gemma 1, Llama-3/3.1, Falcon3 7B, OLMo-1, RecurrentGemma). Falcon3 3B's card names the 7B as the pruning
  source and says "a knowledge distillation objective" without naming the teacher: "pruned from Falcon3 7B and trained
  with a distillation objective" is exact; "distilled from Falcon3 7B" is the likely reading, not a stated one.
- Binding (Atlas, mean of 20 binding tests; a quick guide, not a published metric): Falcon3 3B .471, far above models
  trained from scratch at its counted compute (Pythia/OLMo ~.2-.3), below Falcon-H1 3B (.598) and Qwen2.5 3B (.498) at
  its size; tsumugi .349. Distillation helped its binding; it needed the 7B to get there.
- Luxia's argument: that route needs the larger model, and that doesn't scale; tsumugi's setup cost less total compute and
  can scale further. Synthetic data would likely push it higher, but there'd be no clean comparison, and it would import
  the assistant prior and change the model's texture, an aesthetic and specific choice on top of the clean comparison.

Notes for the discussion (facts on record; for Luxia's writing, not shown on the page). R = registered before the run,
E = exploratory. The loop card under the discussion is the old results card, kept as a placeholder: keep it, fold it
into the text, or empty its entries.
1. Levers chosen on one rendering do partly that rendering's job.
   - R: knocking out 42 position-8 neurons lifts binding on bare 0-shot battery prose (dialogue +11.4 pp, attribute
     +10.1 pp), but the gain shrinks 3-8x once examples or a scene frame are present. The bare prose ("Amy has the
     knife. ... So Amy has the ...") reads like an arithmetic word problem, and position 8 feeds the chain that detects
     that frame. The record's words: it "removes a genre misread when nothing else tells the model what kind of text
     this is. It does not add binding capacity." A small real residual remains (dialogue in scenes, +3 pp).
   - R: a lever made of the union of binding neurons passes the battery but fails on live dialogue (+.047, interval
     crossing 0). Reselected from attribution on 3-shot prompts, it transfers: +.114 [+.078, +.152].
   - Read: neurons selected in the right genre do transfer.
2. Many circuits, not one.
   - R: static "you" with and without examples runs through largely different neurons (14 of 100 shared; changing the
     wording instead keeps 58-73, so it isn't noise).
   - R: dynamic "you" shares 8 of 100 neurons with static "you"; chance is 0 (at most 2), so it is registered as reuse:
     the examples recruit some of the static-"you" machinery, not most of it. Knocking out the static-"you" supporters
     drops the dynamic effect to +.023.
   - R: rule learning and fact recall are near-orthogonal (Spearman .01, 5 of 100 shared; each knock-out takes out its
     own skill and leaves the other).
   - R: wiring is genre-conditioned (a neuron's inputs in dialogue or code agree with its inputs on the general mix far
     less than two halves of one genre agree: .26 dialogue, .14 code); largest at the entry and the coda, smallest inside
     the loop.
   - R: the rendering changes the circuit for some tasks (attribute: 8-15 of 100 shared between 0-shot and fair).
   - R: the same ICL behaviour runs through different mechanisms in prose and code (prose: late brake heads; code: the
     readers' default).
   - E: per-task neuron maps are nearly orthogonal, with a few shared families (dialogue with adjacency and deixis,
     attribute with set binding). A task's own neuron set moves it +32-37 pp; the other tasks' sets, +5-11 pp.
   - Don't lean on: "binding neurons keep 71-77% of their strong inputs across genres" (no baseline).
3. Dispersed, or hubs? Mixed: a few hubs and a long, spread tail.
   - R: one neuron, 19/3306, is a causal hub (ablating it moves the attribute margin +.57 nats vs at most +.013 for
     random single neurons); its second-pass twin 39/3306 is causally inert. The two biggest broadcasters (19/3306,
     8/797) are the top static-"you" interferers.
   - E: knocking out all of position 8 leaves binding within 2 pp; the causal binding neurons sit in 213 of 400 meaning
     groups; a lever covering many tasks needed 702 neurons.
   - Don't lean on: the readability note (its registered follow-up failed). Not yet measured: how many neurons it takes
     to move a gauge (a top-k ablation curve would be the direct test).
4. In-context learning and the second pass.
   - R: switching off 60 random heads, a quarter of one pass, costs rule learning .13 in the second pass, .035 in the
     first and .019 outside the loop; recall falls .025. With the damage to text prediction controlled for, -.065
     [-.093, -.036]. The heads are random: this says where, not which heads.
   - R: 107 heads read the examples (39 in the first pass, 56 in the second, 12 in the coda); all 7 brake heads are in
     the second pass or the coda.
   - R: dynamic "you" sits in the loop: 75% of the neurons that switch on with examples are in the looped core (56% of
     all neurons are); 43 of 100 in the second pass; the top rising head is 39:0, the first position of the second pass.
   - Don't lean on: "40 of the rule circuit's 100 neurons are in the second pass" as rule-specific: recall has 46.
   - R: the same head cut read on "you" (fraction of the swap effect removed, median of 5 draws). Dynamic "you" (3
     examples): pass 1 -55%, pass 2 -31% (draws scatter -99% to +4%), outside the loop -21%; per unit of damage, pass 2
     hurts it less than pass 1. Static "you": pass 1 -64%, pass 2 -67%, outside the loop +18%. Registered reading:
     dynamic "you" does NOT lean on the second pass the way rule learning does; both kinds of "you" sit in the loop
     (either pass), not outside it.
   - E: the draws tell two different stories. Rule learning loses .10-.20 on every one of the 5 second-pass draws (a
     broad effect of the pass). Dynamic "you" on the same 5 head sets: 0%, -80%, -31%, -99%, +4% (the -99% draw also
     does 3x the damage to text prediction). That looks like a few specific heads, caught by some draws and not others.
     The top rising head, 39:0, was in the draw that left dynamic "you" untouched.
   - R (the informed read: each of 864 heads cut alone, picked on half the items, tested on the other half): the top 5
     and top 10 heads each beat 20 of 20 random same-band sets. But the top 10 also hurt text prediction and static
     "you", so as a set they aren't specific to dynamic "you". Restoring 3 heads recovers 65% of draw 1's loss (passes)
     and 48% of draw 3's (just misses); 40:4 is a restorer in both.
   - E (not registered): the top 1-3 heads are specific. 0:1 (layer 0) alone reverses the dynamic "you" effect with no
     measurable damage to text prediction; static "you" loses much less. 40:4 (second pass) is #2, and its first-pass
     twin, the same weights (21:4), ranks 572 of 864: the same head matters only on its second run (likewise 45:4 vs
     26:4, 43:1 vs 24:1). 39:0 ranks 822. The held-out effect is small (.026), so the ratios run large; the rank tests
     against random sets are the solid part. Next: reading 0:1 and 40:4 in the neuron explorer.
   - R (the same head-level test on rule learning, plus a cross-test): a few heads carry rule learning, and they are not
     the dynamic-"you" heads. The top 10 cost rule learning .353 (60 random second-pass heads cost .132), with recall
     untouched (+.010); their damage to text prediction equals the random sets' maximum (.0286 vs .0287), so that
     check passes with no margin. The top head, 39:1 (the first layer of the second pass), costs .121 alone. 6 of the
     top 10 are in the second pass. Cross-test: the dynamic-"you" heads don't carry rule learning, and the rule heads
     don't carry dynamic "you". Top-20 overlap: 2 heads (40:4, 6:6), about chance. Not a broad property of the pass at
     the head level (the coda has the most harmful heads by median and share).
   - E: the rule heads are less tied to the second run than the "you" heads: 39:1 costs .098 on half A, its first-pass
     twin 19:1 .007.
   - E (not registered; worth registering before it's said): the "I" queries do lean on the second pass. Every pass-2
     draw lowers static and dynamic "I"; no pass-1 draw lowers static "I".

Parked from the old results and architecture sections (removed from the page; verbatim, for the discussion or
dog-ears). Facts said nowhere else on the page: the rules test was registered and all three predictions held;
+1.07 SD against parameters (interval +0.68 to +2.07); before its cooldown tsumugi scored .436, level with in-training
checkpoints near its compute (at most .461); the next smallest model to pass dynamic "you" is Qwen3 4B at about
300x tsumugi's tokens; static "you" is passed by 33B tokens (OLMo-2 1B between 420B and 525B); Muon, the SmolLM2
tokenizer and the cooldown (architecture). Links each card carried are given with them.
- results.heading: results
- rules.number: +1.07 SD
- rules.number_note: above the trend for its size, on new rules · 95% interval +0.68 to +2.07
- rules.title: Learns made-up rules from its prompt, beyond its size
- rules.text_1: On new rules, tsumugi scores .559. A trend line fitted to 37 public base models predicts less for a model of its size. The 37 are the 36 in the figure plus GPT-2 XL, which has no published token count. tsumugi sits 1.07 standard deviations above it, measured in the spread of the public models around the line. The interval comes from resampling both the public models and tsumugi's test items. On recall and on held-out text, it sits on that trend. Against training compute, as in the figure above, the gap on new rules is about twice as large: +2.2 to +2.4 SD. Before its final cooldown, the closing stretch of training where the learning rate decays, tsumugi scored .436, level with the public checkpoints taken partway through training near the same compute (at most .461), which haven't had their cooldown either.
- rules.text_2: This comparison was registered: the test and its three predictions were written down and committed before any score was computed, and all three hold. Among public bases under 4B parameters, only Qwen3 1.7B scores higher on new rules.
- rules.more_card: The registered test, its table and checks  [link: __CARD_LINK__#learning-rules-from-the-prompt]
- you.number: +.038
- you.number_note: “you”, after hand-offs: “you” tracked as objects change hands, given three worked examples · 95% interval +.023 to +.055
- you.title: Tracks who “you” is as objects change hands
- you.text_1: The test “you”, after hand-offs: in a two-person chat where objects pass between the speakers, a line like “Right now you're holding the ___” asks about the person being addressed. Swap who says that line, and “you” becomes the other person, so the old answer should lose probability. Given three worked examples, tsumugi's old answer loses .038 on average over 180 chats. A model passes when its whole 95% interval is above .02, a bar that was fixed before any model was scored. tsumugi is the smallest public base model measured that passes. The next smallest is Qwen3-4B, trained on 36T tokens, about 300 times tsumugi's 110B.
- you.text_2: The simpler test, “you”, nothing moves, states who holds what and never changes it. tsumugi passes it by 33B training tokens. OLMo-2 1B passes it somewhere between its 420B and 525B checkpoints.
- you.more_atlas: The test, its threshold and every public model's results  [link: __ATLAS_LINK__#tsumugi-you]
- you.more_tsumugi: How it learned these tests over training  [link: __TSUMUGI_LINK__]
- compute.number: 64.1%
- compute.number_note: average accuracy over nine standard benchmarks: ahead of Pythia 1.4B, 2.8B and 6.9B and OLMo-1 1B at the same compute
- compute.title: Ahead of Pythia 1.4B–6.9B and OLMo-1 1B at its compute
- compute.text_1: Its average over nine standard benchmarks is 64.1% correct. Public models released checkpoints from partway through training, which show how much compute each needed to reach that score: Pythia 2.8B about 2.7 times tsumugi's, Pythia 6.9B about 5 times and OLMo-1 1B about 7 times. Pythia 1.4B never got there: it ends at 61.4 after twice the compute. OLMo-2 1B got there at about the same compute as tsumugi. Those checkpoints are taken before any final learning-rate decay, and tsumugi's score includes its own. The nine are HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, PIQA, SciQ, WinoGrande, COPA and BoolQ, with no examples in the prompt.
- compute.more: Benchmarks and compute, on the model card  [link: __CARD_LINK__#benchmarks-and-compute]
- architecture.heading: architecture
- architecture.text_1: tsumugi is a Llama-style decoder with <a href="https://arxiv.org/abs/2603.15031">Block Attention Residuals</a>. Each layer reads a learned mixture of earlier blocks' outputs, and the block boundaries were fitted to the network's own layer geometry. Its 52 layers of weights run as 72: 19 input layers, a 20-layer middle block run twice with shared weights, and 13 output layers. Its hidden matrices were trained with the <a href="https://kellerjordan.github.io/posts/muon/">Muon</a> optimizer. It was trained on 110B tokens with the SmolLM2 tokenizer, and ended with a cooldown: the closing stretch of training, where the learning rate decays.
- architecture.text_2: The training data, the full architecture and how to run it are on the <a href="__CARD_LINK__">model card</a>. A technical report will follow.

## meta.title
kotodama-tsumugi

## meta.description
kotodama-tsumugi is a 1.4B open base model trained on 110B tokens. It learns new rules from its prompt better than its size predicts, and its looped core is where that happens. Weights, every checkpoint, the code and the benchmark are open.

## meta.og_title
kotodama-tsumugi: a 1.4B base model that learns rules beyond its size

## meta.og_description
Open weights, every checkpoint, the training stack and the benchmark. Registered tests, stated with their intervals.

## meta.twitter_title
kotodama-tsumugi

## header.gloss_kotodama
kotodama — word-spirit: language as a thing with a life of its own

## header.gloss_tsumugi
tsumugi — spun thread: words bound into more than the sum of their parts

## header.title
tsumugi

## header.lede
A 1.4B-parameter looped depth base model, trained on 110B tokens using kotodama, our open pretraining stack: trainer, evaluation and serving in one package. 
<br>
tsumugi performs in-context learning (ICL) on par with models up to 5x larger & trained with far more compute.
<br>
tsumugi also tracks who “you” refers to in two-person chats, like models trained with hundreds of times more compute.
<br>
Binding (keeping track of who is who) and ICL are the two things we set out to study with this model.
<br>
On standard benchmarks like HellaSwag and Uncheatable Eval{{cite:UncheatableEval}}, tsumugi scores at or above where its compute predicts.
<br>
Synthetic & distillation datasets were explicitly omitted, with an estimated 0.5–1% residual contamination from generated text.
<br>
Weights, checkpoints, data recipe, code, and the benchmark are all available.

## header.byline
Led by <a href="https://x.com/kataletheia">Luxia</a>, with <a href="https://animalabs.ai">Anima Labs</a> · built with <a href="https://www.anthropic.com/claude">Claude</a>

## header.links
<a href="https://github.com/anima-research/kotodama">code on GitHub</a> · <a href="__CARD_LINK__">weights on Hugging Face</a> · <a href="__ATLAS_LINK__">benchmark results</a> · <a href="__THREADS_LINK__">neuron graphs</a>

## loop.number
−.13

## loop.number_note
on new rules, with a quarter of the second pass's attention switched off (recall: −.025)

## loop.title
Rule learning leans on the looped block’s second pass

## loop.text_1
tsumugi runs one block of 20 layers twice, with the same weights. Switching off a quarter of the attention in the second pass lowers its new-rules score by .13 on the same 0-to-1 scale, while recall falls .025. The same cut costs .035 in the first pass and .019 in the layers before and after the loop.

## loop.text_2
This test was registered too, and its prediction holds. The cut also makes the model worse at predicting ordinary text, and the result holds after controlling for that loss.

## loop.more
Inside tsumugi: the neurons behind rule learning and recall

## open.heading
release

## open.1.label
Weights

## open.1.text
<a href="https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b">LuxiaSL/kotodama-tsumugi-1b</a>, with the model card.

## open.2.label
The training run

## open.2.text
Every checkpoint, plus the cooled-down versions, as separate branches named <code>anneal…</code>: seed 17, the release, to 110B (<a href="https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-trajectory">trajectory</a>), and a second seed, seed 7, to 11B (<a href="https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b-seed7-trajectory">trajectory</a>).

## open.3.label
The training stack

## open.3.text
<a href="https://github.com/anima-research/kotodama">anima-research/kotodama</a>: the trainer, the evaluation suite and the serving stack.

## open.4.label
The data recipe

## open.4.text
<a href="https://github.com/anima-research/kotodama/blob/main/docs/DATA-tsumugi-1b.md">Every training source</a>, with its share of the tokens and how it was processed. 19 of the 32 sources are published as cleaned text (<a href="https://huggingface.co/datasets/aethera-gp/kotodama-3b-corpus-open">kotodama-3b-corpus-open</a>); two small private sources are described by category only.

## open.5.label
The benchmark

## open.5.text
<a href="https://github.com/anima-research/binding-bench">anima-research/binding-bench</a>, the tests above, <a href="__ATLAS_LINK__">with every public model's results</a>, and <a href="__TSUMUGI_LINK__">how tsumugi learned them over training</a>.

## open.6.label
The neuron graph

## open.6.text
Every neuron of its feed-forward layers, and the strongest connections between them, for the release (<a href="https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph">kotodama-tsumugi-1b-neuron-graph</a>) and across training (<a href="https://huggingface.co/datasets/LuxiaSL/kotodama-tsumugi-1b-neuron-graph-trajectory">its trajectory</a>). <a href="__THREADS_LINK__">Explore it in your browser</a>.

## limits.heading
limitations

## limits.1
It is a base model, with no instruction, preference or safety tuning. It continues text and does not follow instructions.

## limits.2
Without a repetition penalty it falls into self-copying loops in long generations. The model card gives sampling that avoids them.

## limits.3
On held-out code, in bits per byte, it trails Pythia-1.4B and OLMo-2 1B; on held-out prose it is ahead of Pythia-1.4B.

## limits.4
Like every ~1B model measured, it scores at chance on tracking who holds an object in a third-person story.

## limits.5
It is one cooldown of one seed (seed 17), and its training data is mostly English.

## footer.companions
Companion pages: <a href="__CARD_LINK__">the model card</a> · <a href="__ATLAS_LINK__">every public model's results on who-is-who tests</a> · <a href="__THREADS_LINK__">a map of tsumugi's neurons</a> · <a href="__EMPTY_LINK__">what it writes from an empty page</a>.

## cite.UncheatableEval
Tan, Li, Shen, <a href="https://arxiv.org/abs/2609.27510"><i>Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models</i></a>, 2026.

## discussion.heading
discussion

## discussion.text

