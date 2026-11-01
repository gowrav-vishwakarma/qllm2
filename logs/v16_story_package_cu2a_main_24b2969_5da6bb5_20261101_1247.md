# Story comprehension: the PACKAGE (cu2a_main_24b2969, mode lenient), seed 1

v16/package.py: read the session into the mind's state; fetch by text (token / word organ / rare word) then the mind's surprise keys (F2); answer = the mind reading row + question on its state, no-row contrast. Keyword scorer as v16/story_eval.py.


## L = 2048 (2052 tokens, story_L2048_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is a building with a stairway that is not touched by church stair.` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair was the only building in the flood that was not touched by the flood.` |
| where_key | chat | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | OK `The church stair is the only building in the floodplain that has not touched the third ste` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair was the only building in the flood that was not touched by the church sta` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, but the Vell family lost eleven sheep, and the Vell fam` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by Halden and Halden in the floodgates.` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `H is for the flood, and H is for the flood.` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by flood and flood, and Halden had found a new bridge.` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `H is the bridge's dielectric constant.` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and Halden had been hollowed out by floodwater` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found a hollow hollowed out carton that had been hollowed out by floodwaters.` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The fair's price is determined by the number of loafs of weight four, which is given as \(` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged one loaf four, carrying one loaf four, carrying one loaf four, carrying o` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The stranger arrived on the Tuesday after the flood, and Tomas left for the city after the` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `Tomas was on the Tuesday after the flood, and Tomas was on the Tuesday after the flood.` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The story of the mill's arrival in exile in exile in exile in exile in exile in the Britis` |
| true_false | plain | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | OK `The British were not the only ones to be abducted by Britain through false charges.` |
| letter_H | chat | Harrow | F2 | ` electrons in one dimension have long been exploited in solid-state electronics.` | -- `The letter H at the bottom of the letter H at the bottom of the letter H at the bottom of ` |
| letter_H | plain | Harrow | F2 | `ective mirrors for specific wavelengths as  [The story continues]  Chapter Four.` | -- `compound semiconductor superlattices can be grown epitaxially epitaxially epitaxially epit` |
| summary | chat | green | F2 | `ective mirrors for specific wavelengths as  [The story continues]  Chapter Four.` | -- ````python /` |
| summary | plain | green | F2 | `ective mirrors for specific wavelengths as  [The story continues]  Chapter Four.` | -- `compound semiconductor superlattices can be grown epitaxially epitaxially epitaxially epit` |

keyword hits: 6 / 22; bytes held: mind state 1.15 MB + library text 4 kB

## L = 4096 (4101 tokens, story_L4096_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is a building stairway that is the only building stair to the church stai` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair was the only building in the floodgates that was not touched by the churc` |
| where_key | chat | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is a building stairway that is the only building stair to the church stai` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is the only building in the church stairway stair.` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and Halden had to be hollowed out by floodwate` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found a bridge hollowed out by floodwaters, and Hald had found a hollowed out p` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and Halden's bridge was hollowed out by floodw` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found a bridge hollowed out by floodwaters, and Halden had found a hollowed out` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and Halden had been hollowed out by floodwater` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had found a hollowed carton hollowed out by floodwaters.` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The fair share of weight four costs four, four, and six, is determined by the following ru` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged one loaf four, carrying one loaf four, carrying one loaf four, carrying o` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The stranger arrived on the Tuesday after the flood, and Tomas came on the Tuesday after t` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `Tomas came on the Tuesday after the flood, and Tomas came on the Tuesday after the flood.` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The story begins with Britain's British and French forces, who were captured by Britain in` |
| true_false | plain | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | OK `True: The British were not the only ones to be abducted by Britain through false charges o` |
| letter_H | chat | Harrow | F2 | `tter pair, like "eraser," which has E-R twice, or "regret," which has R-E twice.` | -- `The word "eraser," which has E twice, has R twice, and has R twice.` |
| letter_H | plain | Harrow | F2 | `idow, Agnes Thorne, though she had not set foot in it since her husband drowned.` | -- `Agnes Thorne set foot on the miller's widow's widow's widow's widow's widow Agnes Thorne T` |
| summary | chat | green | F2 | ` Minima, Illoma and Epellema, Ekereborikiri, Down-Below, Abazibie and Opukalama.` | -- `The Kingdom is made up of satellite towns namely Minimaibiama, Abazibieborikiriama and Opu` |
| summary | plain | green | F2 | `The Lantern of Harrow Mill  Chapter One.` | -- `/` |

keyword hits: 5 / 22; bytes held: mind state 1.15 MB + library text 8 kB

## L = 8192 (8198 tokens, story_L8192_s1.txt)

| question | form | expected | path | fetched row | package answer |
|---|---|---|---|---|---|
| who_found | chat | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is a building stairway that is the only building stairway in the floodpla` |
| who_found | plain | Mira | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stairwell stairwell stairwell stairwell stairwell stairwell stairwell stairwell` |
| where_key | chat | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is a building stairway that is the only building stair to the church stai` |
| where_key | plain | third step | F2 | `hurch stair, because the church was the only building the flood had not touched.` | -- `The church stair is the only building in the world that has ever been found hanging inside` |
| howmany_why | chat | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| howmany_why | plain | eleven | text4 | `ater but to the wolves that came down from Cold Ridge while the fences lay flat.` | OK `The Vell family lost eleven sheep, and the Vell family lost eleven sheep.` |
| promise | chat | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The bridge was hollowed out by floodwaters, and Halden had to be hollowed out.` |
| promise | plain | city | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Halden had been in the pier for a long time, but Halden had been in the past for a long ti` |
| order | chat | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The fair was hollowed out by Halden's bridge.` |
| order | plain | storm | text4 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `Hald was not a fair cart cart cart.` |
| explain_why | chat | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | -- `The problem involves a bridge with a closed cartouche filled with floodwaters and floodwat` |
| explain_why | plain | pier | text2 | `astern pier had been hollowed out by the flood and would not bear a loaded cart.` | OK `Halden had found a hollowed pier in the flood plain, but the flood plain was hollowed out ` |
| rule_apply | chat | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `The problem involves determining the maximum number of coins Old Pell charged four times.` |
| rule_apply | plain | 9 | text2 | `esday, carrying one loaf of weight four, for which Old Pell charged him nothing.` | -- `Old Pell charged one loaf of weight four, and Old Pell charged one loaf of weight four, ca` |
| inference_day | chat | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `The stranger arrived on the Tuesday after the flood.` |
| inference_day | plain | Tuesday | text2 | `, the same day the tax collector always came, which is how people remembered it.` | OK `Tomas came on the Tuesday after the flood.` |
| true_false | chat | False | F2 | `e like kings of other prominent communities, to clear the coast for the British.` | -- `The mill's origins are rooted in the British colonial rule of Britain, where the British w` |
| true_false | plain | False | F2 | `om BostonNext week's challenge: Think of a word in which the second letter is R.` | -- `"Footnote," which is R.` |
| letter_H | chat | Harrow | F2 | `eLinköping University Neurobiology CentreAreas of ResearchPage manager: susanne.` | -- `The letter H stands for ** susanne.` |
| letter_H | plain | Harrow | F2 | `idow, Agnes Thorne, though she had not set foot in it since her husband drowned.` | -- `Agnes Thorne was a widow and a widow, Agnes Thorne Thorne Thorne Thorne Thorne Thorne Thor` |
| summary | chat | green | F2 | ` Minima, Illoma and Epellema, Ekereborikiri, Down-Below, Abazibie and Opukalama.` | -- ````python /` |
| summary | plain | green | F2 | `om BostonNext week's challenge: Think of a word in which the second letter is R.` | -- `"Footnote," R.` |

keyword hits: 5 / 22; bytes held: mind state 1.15 MB + library text 16 kB
