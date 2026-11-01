
**600M-token tier (same data; claims within this block)**

| model | tokens | data | params | hellaswag (acc_norm) | arc_easy (acc) | arc_challenge (acc_norm) | piqa (acc_norm) | winogrande (acc) | openbookqa (acc_norm) | MC mean | LAMBADA acc / ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PAM b2p2 | 600M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.9M | 0.279 | 0.454 | 0.233 | 0.578 | 0.505 | 0.298 | **0.391** | 0.146 / 578 |
| PAM cu2a (+2 % fld / vfy) | 600M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.9M | 0.280 | 0.458 | 0.228 | 0.567 | 0.501 | 0.292 | **0.388** | 0.142 / 758 |
| T++ b2t2 | 600M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 101.9M | 0.272 | 0.439 | 0.232 | 0.571 | 0.520 | 0.294 | **0.388** | 0.148 / 1299 |
| Mamba-2 b2m2 | 600M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.6M | 0.279 | 0.466 | 0.238 | 0.578 | 0.496 | 0.296 | **0.392** | 0.132 / 945 |

**3B-token tier (same data; claims within this block)**

| model | tokens | data | params | hellaswag (acc_norm) | arc_easy (acc) | arc_challenge (acc_norm) | piqa (acc_norm) | winogrande (acc) | openbookqa (acc_norm) | MC mean | LAMBADA acc / ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PAM r3c | 3.0B | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 103.2M | 0.296 | 0.489 | 0.243 | 0.590 | 0.511 | 0.290 | **0.403** | 0.179 / 376 |
| T++ e072a8f | 3.0B | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 101.9M | 0.292 | 0.505 | 0.246 | 0.592 | 0.494 | 0.306 | **0.406** | 0.196 / 404 |
| Mamba-2 b36e3a5 | 3.0B | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.6M | 0.303 | 0.517 | 0.249 | 0.605 | 0.513 | 0.300 | **0.415** | 0.176 / 305 |

**220M-token window arms (same data; the window's language cost; claims within this block)**

| model | tokens | data | params | hellaswag (acc_norm) | arc_easy (acc) | arc_challenge (acc_norm) | piqa (acc_norm) | winogrande (acc) | openbookqa (acc_norm) | MC mean | LAMBADA acc / ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PAM W0, seed 43 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.9M | 0.271 | 0.441 | 0.224 | 0.570 | 0.502 | 0.256 | **0.377** | 0.122 / 1280 |
| PAM W0, seed 44 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 102.9M | 0.275 | 0.436 | 0.225 | 0.574 | 0.514 | 0.276 | **0.383** | 0.124 / 1257 |
| PAM W128, seed 43 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 107.0M | 0.271 | 0.437 | 0.232 | 0.559 | 0.518 | 0.262 | **0.380** | 0.124 / 1332 |
| PAM W128, seed 44 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 107.0M | 0.271 | 0.428 | 0.230 | 0.566 | 0.532 | 0.278 | **0.384** | 0.116 / 1390 |
| PAM W512, seed 43 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 107.0M | 0.271 | 0.432 | 0.237 | 0.561 | 0.510 | 0.276 | **0.381** | 0.127 / 1242 |
| PAM W1024, seed 43 | 220M | our mix (dclm / fineweb-edu / smoltalk2 + synthetic) | 107.0M | 0.271 | 0.434 | 0.237 | 0.567 | 0.507 | 0.270 | **0.381** | 0.120 / 1313 |

**REFERENCE: different data and tokens, no claim**

| model | tokens | data | params | hellaswag (acc_norm) | arc_easy (acc) | arc_challenge (acc_norm) | piqa (acc_norm) | winogrande (acc) | openbookqa (acc_norm) | MC mean | LAMBADA acc / ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 2T | SmolLM corpus | 135M | 0.432 | 0.645 | 0.294 | 0.684 | 0.538 | 0.330 | **0.487** | 0.428 / 19 |
| Pythia-160M (fp32; bf16 degrades it) | 300B | the Pile | 162M | 0.303 | 0.436 | 0.236 | 0.619 | 0.513 | 0.268 | **0.396** | 0.354 / 37 |
| GPT-2 124M | ~10B | WebText | 124M | 0.313 | 0.443 | 0.230 | 0.621 | 0.508 | 0.270 | **0.397** | 0.307 / 39 |
