"""Duplex control vocabulary (separate from GPT-2 / ChatML)."""

from dataclasses import dataclass
from typing import List, Tuple

from v11.duplex.config import DUPLEX_TEXT_OFFSET


@dataclass(frozen=True)
class DuplexVocab:
    pad: int = 0
    eos: int = 1
    listen: int = 2
    speak: int = 3
    backchannel: int = 4
    env_mark: int = 5
    ast_mark: int = 6
    block_sep: int = 7

    @property
    def text_offset(self) -> int:
        return DUPLEX_TEXT_OFFSET

    @property
    def thinking_ids(self) -> Tuple[int, ...]:
        return (self.listen, self.speak, self.backchannel)

    def is_thinking(self, token_id: int) -> bool:
        return token_id in self.thinking_ids

    def is_special(self, token_id: int) -> bool:
        return token_id < self.text_offset

    def text_token(self, idx: int) -> int:
        return self.text_offset + (idx % max(1, 512 - self.text_offset))


VOCAB = DuplexVocab()

# Short word pools for synthetic dialogue (text ids mapped via text_token).
USER_PHRASES: List[List[int]] = [
    [0, 1, 2, 3],
    [4, 5, 6],
    [7, 8, 9, 10, 11],
    [12, 13],
    [14, 15, 16],
]

ASSISTANT_PHRASES: List[List[int]] = [
    [20, 21, 22, 23, 24],
    [25, 26, 27, 28],
    [29, 30, 31],
    [32, 33, 34, 35, 36, 37],
    [38, 39],
]

BACKCHANNEL_PHRASES: List[List[int]] = [
    [40, 41],
    [42],
    [43, 44],
]
