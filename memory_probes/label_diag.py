"""label_category diagnostic: semantic lookup vs exact lookup.

label_category asks for the label of a NEW item of a category seen in the
context ("truck is silver, ... car is" -> silver): the model must match by
meaning, find that example and copy its label. ``label_exact`` is the same
prompt with the query replaced by an item that IS in the context ("...
truck is" -> silver): pure exact lookup + copy. Same target and candidates.

  exact high, category low  -> copying works; the fuzzy (semantic) match is
                               the bottleneck (read sharpness / key geometry)
  both low                  -> the in-context lookup-and-copy circuit itself
                               has not formed
"""

from __future__ import annotations

from dataclasses import replace

from memory_probes.rules import RuleExample

_HEAD = 'Label rule. '


def exact_variant(example: RuleExample, tokenizer) -> RuleExample:
    if example.task != 'label_category' or not example.prompt_text.startswith(_HEAD):
        raise ValueError(f'{example.example_id}: not a label_category prompt')
    units = example.prompt_text[len(_HEAD):].split(', ')
    pairs, query = units[:-1], units[-1]
    target = example.target_text.strip()
    same = [p.rsplit(' is ', 1)[0] for p in pairs if p.rsplit(' is ', 1)[1] == target]
    if not same or not query.endswith(' is'):
        raise ValueError(f'{example.example_id}: no in-context item with the target label')
    text = _HEAD + ', '.join(pairs) + f', {same[0]} is'
    return replace(example, example_id=example.example_id.replace('label_category', 'label_exact'),
                   task='label_exact', prompt_text=text,
                   prompt_ids=tokenizer.encode(text, add_special_tokens=False))


__all__ = ['exact_variant']
