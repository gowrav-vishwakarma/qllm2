"""Interactive multi-turn chat with a SCI-1 checkpoint on the CPU (no GPU used).

  cd ~/Development/qllm_sci1check     # any worktree with .venv + scripts/ + v13_sempty/
  .venv/bin/python ~/Development/qllm2/v16/chat_cpu.py [checkpoint.pt]

Default checkpoint: the newest SCI-1 milestone weights on the d_drive. Type a message and press Enter.
Commands:
  /new            start a new session (forget the conversation)
  /system TEXT    set a system message for the next sessions (empty = none)
  /greedy         greedy decoding (default)      /sample   sampled (t 0.8, top-k 50, top-p 0.9)
  /max N          max new tokens per reply (default 200)
  /history        show the raw prompt the model sees
  /quit           exit
The model sees the whole conversation in ChatML (<|im_start|>user ... <|im_end|>), as in training.
"""
import glob, os, sys, time
from pathlib import Path

W = Path.cwd()
sys.path.insert(0, str(W)); sys.path.insert(0, str(W / 'scripts'))
import torch
from run_memory_behavioral import _load_v13_sempty

D = '/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty'


def newest_ckpt():
    c = glob.glob(f'{D}/sci1_*/c*_step_*_weights.pt') + glob.glob(f'{D}/sci1_*/step_*.pt')
    return max(c, key=lambda p: int(p.rsplit('step_', 1)[1].split('_')[0].split('.')[0]))


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else newest_ckpt()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))   # leave half the cores to the rest of the box
    t0 = time.time(); print(f'loading {ck} on CPU ...', flush=True)
    m, tok, _ = _load_v13_sempty(Path(ck), 'baseline_real_pm', torch.device('cpu'))
    m.eval().float()
    end_id = tok.convert_tokens_to_ids('<|im_end|>')
    print(f'ready in {time.time() - t0:.0f}s. /new = new session, /quit = exit, /help = commands\n')
    system, hist, greedy, max_new = '', '', True, 200
    while True:
        try:
            u = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not u:
            continue
        if u.startswith('/'):
            cmd, _, arg = u.partition(' ')
            if cmd == '/quit': break
            elif cmd == '/new': hist = ''; print('-- new session --\n')
            elif cmd == '/system': system = arg.strip(); hist = ''; print(f'-- system set, new session: {system!r} --\n')
            elif cmd == '/greedy': greedy = True; print('-- greedy --')
            elif cmd == '/sample': greedy = False; print('-- sampled (t 0.8, top-k 50, top-p 0.9) --')
            elif cmd == '/max': max_new = int(arg or 200); print(f'-- max {max_new} tokens --')
            elif cmd == '/history': print(hist or '(empty)')
            else: print(__doc__)
            continue
        if not hist and system:
            hist = f'<|im_start|>system\n{system}<|im_end|>\n'
        hist += f'<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n'
        ids = torch.tensor([tok.encode(hist)])
        t1 = time.time()
        with torch.no_grad():
            if greedy:
                out = m.generate(ids, max_new_tokens=max_new, temperature=0.0, eos_token_id=end_id)
            else:
                out = m.generate(ids, max_new_tokens=max_new, temperature=0.8, top_k=50, top_p=0.9,
                                 repetition_penalty=1.2, eos_token_id=end_id)
        new = out[0, ids.shape[1]:].tolist()
        if end_id in new:
            new = new[:new.index(end_id)]
        reply = tok.decode(new).split('<|im_end|>')[0].split('<|im_start|>')[0].strip()
        hist += reply + '<|im_end|>\n'
        print(f'model> {reply}\n   [{len(new)} tok, {time.time() - t1:.1f}s, context {ids.shape[1]} tok]\n')


if __name__ == '__main__':
    main()
