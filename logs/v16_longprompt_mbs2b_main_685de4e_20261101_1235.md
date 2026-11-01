# Long-prompt stress test: mbs2b_main_685de4e (mamba2), seed 1, greedy, max_new 24


## L = 2048 (prompt 2163 tokens, file v16/prompts/long_L2048_s1.txt)

| question | expected | generated |
|---|---|---|
| recall_trained_near_end | juniper | -- `cobalt. /` |
| recall_trained_near_start | cobalt | OK `cobalt. /` |
| recall_updated_trained | indigo | -- `amber. /` |
| recall_chat | cobalt | -- `The word velmor refers to the capacity of a ship to carry a load of goods. The word velmor refers to` |
| chat_fact | blue cabinet | -- ````python /` |
| rule_trained | 26 | -- `10, 6->10, 6->10, 6->10, 6->10, 6->10, 6->` |
| rule_plain_chat | 26 | -- ````python /` |
| knowledge_capital | Paris | -- `The capital of France is the capital of France. The capital of France is France, which is the capital of France.` |
| knowledge_author | Shakespeare | -- `The play Hamlet is a play that involves the use of a set of characters to represent the characters of the play.` |

## L = 4096 (prompt 4210 tokens, file v16/prompts/long_L4096_s1.txt)

| question | expected | generated |
|---|---|---|
| recall_trained_near_end | juniper | OK `juniper. /` |
| recall_trained_near_start | cobalt | -- `indigo. /` |
| recall_updated_trained | indigo | OK `indigo. /` |
| recall_chat | cobalt | -- ````python /` |
| chat_fact | blue cabinet | -- ````python /` |
| rule_trained | 26 | -- `10. The word "filling" is a bit of a misnomer. The word "filling" is` |
| rule_plain_chat | 26 | -- ````python /` |
| knowledge_capital | Paris | -- `The capital of France is the capital of France. The capital of France is France, which is the capital of France.` |
| knowledge_author | Shakespeare | -- `The play Hamlet is a play that involves a series of lines, each of which is a line. The lines are` |

## L = 8192 (prompt 8306 tokens, file v16/prompts/long_L8192_s1.txt)

| question | expected | generated |
|---|---|---|
| recall_trained_near_end | juniper | OK `juniper. /` |
| recall_trained_near_start | cobalt | -- `indigo. /` |
| recall_updated_trained | indigo | OK `indigo. /` |
| recall_chat | cobalt | -- `The word velmor refers to a city that is built in a specific way. The word velmor refers to a city` |
| chat_fact | blue cabinet | -- `The backup key is the key that is used to store the data. The key is the key that is used to store` |
| rule_trained | 26 | -- `1 -> 2 -> 3 -> 4 -> 5 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 10 ->` |
| rule_plain_chat | 26 | -- `The price when the cost is 11 is calculated by subtracting the cost of the cost of the cost of the cost of` |
| knowledge_capital | Paris | -- `The capital of France is the capital of France. The capital of France is the capital of France, and the capital of` |
| knowledge_author | Shakespeare | -- `The play Hamlet is about a man who is trying to get rid of the man who is trying to get rid of` |
