# Story comprehension: the PACKAGE (b2p2_main_38e3c29, mode lenient), seed 1

v16/package.py: read the session into the mind's state; fetch by text (token / word organ / rare word) then the mind's surprise keys (F2); answer = the mind reading row + question on its state, no-row contrast. Keyword scorer as v16/story_eval.py.


## L = 2048 (2052 tokens, story_L2048_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | OK `The story of Mira climbed through the broken shutter of the mill to look for swallows, and` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | OK `The church stair was not the only building that Mira hid inside the church stair.` |
| where_key | chat | third step | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | -- `The mill's nests are arranged in a heap of sacks, each with a green glass door.` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair was not touched by the flood.` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, not to mention the wolves that had been lost.` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, not to mention eight sheep, but only one sheep.` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by the flood and the flood.` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found that the bridge had been hollowed out by flood, and that the bridge had b` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by the flood and the flood.` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found that the bridge had been hollowed out by flood.` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by flood and flood, and the bridge was hollowed out by the flo` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | OK `Halden had found that the eastern pier had been hollowed out by flood.` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The story of the fair and the city's weight four, which is four, is a classic example of t` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged him nothing.` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on Tuesday after the flood, and the tax collector always cam` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on the Tuesday after the flood, and always remembered the sa` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The mill was abducted by Britain through guile, but the British were not abducted by Brita` |
| true_false | plain | False | F2 | `idow, Agnes Thorne, though she had not set foot in it since her husband drowned.` | -- `Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes ` |
| letter_H | chat | Harrow | F2 | `tten the week before he drowned, and that the H stood for Harrow, not for a man.` | OK `The H stood for Harrow, not Harrow, and the letter was written in the same manner as the l` |
| letter_H | plain | Harrow | F2 | `tten the week before he drowned, and that the H stood for Harrow, not for a man.` | -- `Agnes Thorne Thorne Thorne Thorne Thorne Thorne Thorne Thorne, not Agnes Thorne, not Agnes` |
| summary | chat | green | F2 | `lustrious father King Douglas Jaja (Jaja IV  [The story continues]  Chapter Two.` | -- `Chapter Two describes the events leading to the fair.` |
| summary | plain | green | F2 | `The Lantern of Harrow Mill  Chapter One.` | -- `The Lantern of Harrow Mill /` |

keyword hits: 8 / 22; bytes held: mind state 1.15 MB + library text 4 kB

## L = 4096 (4101 tokens, story_L4096_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | OK `The story of Mira climbed through the broken shutter of the mill to look for swallows' nes` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair was the only building that could withstand the flood.` |
| where_key | chat | third step | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | -- `The mill was designed to be used for the mill to look for swallows' nests, which were not ` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is the only building that has been touched by the flood.` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, not to mention wolves, and the Vell family lost eleven ` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, not to mention seven sheep, not to wolves, but to wolve` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, which would not bear cartouches.` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and the bridge was hollowed out by floodwaters` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, which were hollowed out by floodwaters.` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found that Halden had found a hollowed cart that had been hollowed out by flood` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, which were not hollowed out by floodwaters.` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | OK `Halden had found that the eastern pier had been hollowed out by flood flood flood flood fl` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The fair price rule for the fair was given by the following rules: Old Pell charged him no` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged him nothing.` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on Tuesday after the flood, and the tax collector always cam` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on Tuesday after the flood, and always remembered the same d` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The mill was abducted by Britain through guile, but the British were not abducted by Brita` |
| true_false | plain | False | F2 | `idow, Agnes Thorne, though she had not set foot in it since her husband drowned.` | -- `Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes ` |
| letter_H | chat | Harrow | F2 | `tten the week before he drowned, and that the H stood for Harrow, not for a man.` | OK `The letter H stands for Harrow, not for Harrow, not for Harrow.` |
| letter_H | plain | Harrow | F2 | `tten the week before he drowned, and that the H stood for Harrow, not for a man.` | OK `Agnes Thorne, the widow of Harrow, drowned Harrow in the river.` |
| summary | chat | green | F2 | `areas is further bolstered by another insidious feature of QALY-based economics.` | -- `Summarise the fair treatment of QALY-based economics by using the following key features: ` |
| summary | plain | green | F2 | `The Lantern of Harrow Mill  Chapter One.` | -- `Summarise what happened at Harrow Mill /` |

keyword hits: 8 / 22; bytes held: mind state 1.15 MB + library text 8 kB

## L = 8192 (8198 tokens, story_L8192_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | -- `The mill was designed to withstand the heavy elements of heavy elements such as heavy elem` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | OK `The church stair was not built for the flood Mira hid the brass key she had found hanging ` |
| where_key | chat | third step | F2 | `nests and found, under a heap of sacks, a brass lantern with a green glass door.` | -- `The mill's nests are held in a heap of sacks of sacks that are held in a heap.` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | OK `The church stair is the only building that has been built on the third step.` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, which would not bear cartouches.` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and the bridge was hollowed out by floodwaters` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, which were hollowed out by floodwaters.` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found that Halden had found a hollow cart, and that Halden had found a cart.` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters and the floodwaters.` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found that Halden had found a hollowed cart that had been hollowed out by flood` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The problem involves determining how many coins Old and New are worth four, which Old and ` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged him on the following Monday, carrying four loaf of weight four, for which` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on Tuesday after the flood, and the tax collector always cam` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The tax collector always came on Tuesday after the flood, and always remembered the same d` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The mill was abducted by Britain through guile and guile.` |
| true_false | plain | False | F2 | `idow, Agnes Thorne, though she had not set foot in it since her husband drowned.` | -- `Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes Thorne, Agnes ` |
| letter_H | chat | Harrow | F2 | `tten the week before he drowned, and that the H stood for Harrow, not for a man.` | OK `The letter H stands for Harrow, not for Harrow.` |
| letter_H | plain | Harrow | F2 | `om BostonNext week's challenge: Think of a word in which the second letter is R.` | -- `Paul Anagnostos from BostonNext week's challenge: Think of Paul Anagnostos from BostonNext` |
| summary | chat | green | F2 | `areas is further bolstered by another insidious feature of QALY-based economics.` | -- `Summarise the fair treatment of QALY-based economics by using the fair treatment of QALY-b` |
| summary | plain | green | F2 | `The Lantern of Harrow Mill  Chapter One.` | -- `The Lantern of Harrow Mill /` |

keyword hits: 7 / 22; bytes held: mind state 1.15 MB + library text 16 kB
