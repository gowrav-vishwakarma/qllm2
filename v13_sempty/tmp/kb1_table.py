# kb1 table, the user's layout (16:30 IST via Lead): blocks by TOKENS TRAINED ON; claims and deltas only within a block; the reference
# block is labelled "different data and tokens, no claim". Headline metric per task (pre-registered); MC mean; LAMBADA acc / ppl.
import json, glob, sys
D = sys.argv[1] if len(sys.argv) > 1 else '/home/gowrav/Development/qllm_kb1/logs/kb1'
HEAD = [('hellaswag', 'acc_norm'), ('arc_easy', 'acc'), ('arc_challenge', 'acc_norm'), ('piqa', 'acc_norm'), ('winogrande', 'acc'), ('openbookqa', 'acc_norm')]
MIX = 'our mix (dclm / fineweb-edu / smoltalk2 + synthetic)'
BLOCKS = [
    ('600M-token tier (same data; claims within this block)', [
        ('b2p2', 'PAM b2p2', '600M', MIX, '102.9M'), ('cu2a', 'PAM cu2a (+2 % fld / vfy)', '600M', MIX, '102.9M'),
        ('b2t2', 'T++ b2t2', '600M', MIX, '101.9M'), ('b2m2', 'Mamba-2 b2m2', '600M', MIX, '102.6M')]),
    ('3B-token tier (same data; claims within this block)', [
        ('r3c', 'PAM r3c', '3.0B', MIX, '103.2M'), ('tpp3b', 'T++ e072a8f', '3.0B', MIX, '101.9M'), ('mamba3b', 'Mamba-2 b36e3a5', '3.0B', MIX, '102.6M')]),
    ('220M-token window arms (same data; the window\'s language cost; claims within this block)', [
        ('w1s_w0_508aaff', 'PAM W0, seed 43', '220M', MIX, '102.9M'), ('w1r_w0_508aaff', 'PAM W0, seed 44', '220M', MIX, '102.9M'),
        ('w1s_w128_508aaff', 'PAM W128, seed 43', '220M', MIX, '107.0M'), ('w1r_w128_508aaff', 'PAM W128, seed 44', '220M', MIX, '107.0M'),
        ('w1w_w512_f68ab64', 'PAM W512, seed 43', '220M', MIX, '107.0M'), ('w1w_w1024_f68ab64', 'PAM W1024, seed 43', '220M', MIX, '107.0M')]),
    ('REFERENCE: different data and tokens, no claim', [
        ('smollm2_135m', 'SmolLM2-135M', '2T', 'SmolLM corpus', '135M'), ('pythia_160m', 'Pythia-160M', '300B', 'the Pile', '162M'),
        ('gpt2_124m', 'GPT-2 124M', '~10B', 'WebText', '124M')])]
for title, rows in BLOCKS:
    got = [(r, glob.glob(f'{D}/v13_sempty_kb1_{r[0]}_*.json')) for r in rows]
    got = [(r, f) for r, f in got if f]
    if not got: continue
    print(f'\n**{title}**\n')
    print('| model | tokens | data | params | ' + ' | '.join(f'{t} ({m})' for t, m in HEAD) + ' | MC mean | LAMBADA acc / ppl |')
    print('|' + '---|' * (len(HEAD) + 6))
    for (key, name, tok, data, par), f in got:
        r = json.load(open(f[0]))['results']
        v = [r[t][f'{m},none'] for t, m in HEAD]; lam = r['lambada_openai']
        print(f'| {name} | {tok} | {data} | {par} | ' + ' | '.join(f'{x:.3f}' for x in v) + f' | **{sum(v) / len(v):.3f}** | {lam["acc,none"]:.3f} / {lam["perplexity,none"]:.0f} |')
