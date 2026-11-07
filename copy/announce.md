# The announcement's words

Each `## key` below is one piece of text on the page; edit the text under it. Text is HTML: keep tags such as `<a href="…">`,
`<i>` and `<b>` balanced, and keep placeholders such as `__ATLAS_LINK__` as they are (the build fills them in).
`{{cite:NAME}}` puts a numbered citation marker there; its leaflet text is the entry `cite.NAME`.

The easiest way to edit is `make edit` in this repo: it opens the page, and clicking any outlined text edits it here.
Numbers are claims: change one only together with its source.

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

## header.kicker
Model release · open weights

## header.gloss_kotodama
kotodama — word-spirit: language as a thing with a life of its own. It names the training project.

## header.gloss_tsumugi
tsumugi — spun thread: words woven into more than the sum of their parts. It names this model.

## header.title
kotodama-tsumugi

## header.lede
A 1.4B-parameter base model, trained on 110B tokens with kotodama, our open pretraining stack: trainer, evaluation and serving in one package. Shown a few worked examples of a made-up rule, tsumugi applies the rule to a new case better than public models of its size. In a two-person chat, it keeps track of who “you” is, like larger models trained on 40 to 300 times more data. Its weights, training checkpoints, data recipe, code and benchmark are all open.

## header.byline
Led by <a href="https://x.com/kataletheia">Luxia</a>, with <a href="https://animalabs.ai">Anima Labs</a> · built with <a href="https://www.anthropic.com/claude">Claude</a>

## header.links
<a href="https://github.com/anima-research/kotodama">code on GitHub</a> · <a href="__CARD_LINK__">weights on Hugging Face</a> · <a href="__ATLAS_LINK__">benchmark results</a> · <a href="__THREADS_LINK__">neuron graphs</a>

## figure.intro
The figure below compares tsumugi with open-weight base models on learning a made-up rule from examples in the prompt, against the compute that each spent in training. Compute is about 6 × parameters × tokens, with tsumugi's repeated layers counted twice. The prompt shows worked examples of an invented rule, such as an invented word ending, invented operators or a nonsense pattern. The model then picks the answer to a new case. The score runs from 0 (guessing) to 1 (always right). Two controls, scored the same way, appear in the results below. <b>Recall:</b> questions about real facts, in the same worked-example format. <b>Held-out text:</b> how well a model predicts text published after its training data was collected, in bits per byte; lower is better.

## figure.alt
Learning a made-up rule from examples in the prompt, against training compute. Open-weight base models, fully trained (filled dots) and partway through training (open dots), with the trend of the fully trained. tsumugi, the yellow diamond, scores .559, above Falcon3 3B at .536, the highest near the same compute.

## figure.caption
Base models are evaluated on how well they pick up patterns (not reliant upon factual recall, much like Raven's matrices) from examples in context, following ICLEval (Chen et al. 2024){{cite:ICLEval}}. tsumugi's final checkpoint settles at 0.559 chance-corrected accuracy, well above the compute trend for open-weight models, Falcon3 3B the nearest at 0.536. When the same test uses known facts, such that memory offers a shortcut, it scores higher (0.930) and lands where the compute trend predicts.

## results.heading
Results

## rules.number
+1.07 SD

## rules.number_note
above the trend for its size, on new rules · 95% interval +0.68 to +2.07

## rules.title
Learns made-up rules from its prompt, beyond its size

## loop.number
−.13

## loop.number_note
on new rules, with a quarter of the second pass's attention switched off (recall: −.025)

## loop.title
Rule learning leans on the looped block’s second pass

## you.number
+.038

## you.number_note
“you”, after hand-offs: “you” tracked as objects change hands, given three worked examples · 95% interval +.023 to +.055

## you.title
Tracks who “you” is as objects change hands

## compute.number
64.1%

## compute.number_note
average accuracy over nine standard benchmarks: ahead of Pythia 1.4B, 2.8B and 6.9B and OLMo-1 1B at the same compute

## compute.title
Ahead of Pythia 1.4B–6.9B and OLMo-1 1B at its compute

## rules.text_1
On new rules, tsumugi scores .559. A trend line fitted to 37 public base models predicts less for a model of its size. The 37 are the 36 in the figure plus GPT-2 XL, which has no published token count. tsumugi sits 1.07 standard deviations above it, measured in the spread of the public models around the line. The interval comes from resampling both the public models and tsumugi's test items. On recall and on held-out text, it sits on that trend. Against training compute, as in the figure above, the gap on new rules is about twice as large: +2.2 to +2.4 SD. Before its final cooldown, the closing stretch of training where the learning rate decays, tsumugi scored .436, level with the public checkpoints taken partway through training near the same compute (at most .461), which haven't had their cooldown either.

## rules.text_2
This comparison was registered: the test and its three predictions were written down and committed before any score was computed, and all three hold. Among public bases under 4B parameters, only Qwen3 1.7B scores higher on new rules.

## rules.more
The registered test, its table and checks

## loop.text_1
tsumugi runs one block of 20 layers twice, with the same weights. Switching off a quarter of the attention in the second pass lowers its new-rules score by .13 on the same 0-to-1 scale, while recall falls .025. The same cut costs .035 in the first pass and .019 in the layers before and after the loop.

## loop.text_2
This test was registered too, and its prediction holds. The cut also makes the model worse at predicting ordinary text, and the result holds after controlling for that loss.

## loop.more
Inside tsumugi: the neurons behind rule learning and recall

## you.text_1
The test “you”, after hand-offs: in a two-person chat where objects pass between the speakers, a line like “Right now you're holding the ___” asks about the person being addressed. Swap who says that line, and “you” becomes the other person, so the old answer should lose probability. Given three worked examples, tsumugi's old answer loses .038 on average over 180 chats. A model passes when its whole 95% interval is above .02, a bar that was fixed before any model was scored. tsumugi is the smallest public base model measured that passes. The next smallest is Qwen3-4B, trained on 36T tokens, about 300 times tsumugi's 110B.

## you.text_2
The simpler test, “you”, nothing moves, states who holds what and never changes it. tsumugi passes it by 33B training tokens. OLMo-2 1B passes it somewhere between its 420B and 525B checkpoints.

## you.more_atlas
The test, its threshold and every public model's results

## you.more_tsumugi
How it learned these tests over training

## compute.text_1
Its average over nine standard benchmarks is 64.1% correct. Public models released checkpoints from partway through training, which show how much compute each needed to reach that score: Pythia 2.8B about 2.7 times tsumugi's, Pythia 6.9B about 5 times and OLMo-1 1B about 7 times. Pythia 1.4B never got there: it ends at 61.4 after twice the compute. OLMo-2 1B got there at about the same compute as tsumugi. Those checkpoints are taken before any final learning-rate decay, and tsumugi's score includes its own. The nine are HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, PIQA, SciQ, WinoGrande, COPA and BoolQ, with no examples in the prompt.

## compute.more
Benchmarks and compute, on the model card

## empty.title
What it writes from an empty page

## empty.text
Its training text was curated to leave out assistant replies and machine-written text, and it shows. Given only the start-of-document token, 128 times, tsumugi writes no chatbot transcripts, assistant replies or machine-made training data; many modern base models do. Claude sorted every run, after agreeing with 300 hand-labelled runs 98.7% of the time. All 128 runs <a href="__EMPTY_LINK__">can be read on their own page</a>.

## architecture.heading
Architecture

## architecture.text_1
tsumugi is a Llama-style decoder with <a href="https://arxiv.org/abs/2603.15031">Block Attention Residuals</a>. Each layer reads a learned mixture of earlier blocks' outputs, and the block boundaries were fitted to the network's own layer geometry. Its 52 layers of weights run as 72: 19 input layers, a 20-layer middle block run twice with shared weights, and 13 output layers. Its hidden matrices were trained with the <a href="https://kellerjordan.github.io/posts/muon/">Muon</a> optimizer. It was trained on 110B tokens with the SmolLM2 tokenizer, and ended with a cooldown: the closing stretch of training, where the learning rate decays.

## architecture.text_2
The training data, the full architecture and how to run it are on the <a href="__CARD_LINK__">model card</a>. A technical report will follow.

## open.heading
What is open

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
Limitations

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

## cite.ICLEval
Chen, Lin, Zhou, Huang, Jia, Cao, Wen, <a href="https://arxiv.org/abs/2406.14955"><i>ICLEval: Evaluating In-Context Learning Ability of Large Language Models</i></a>, 2024.
