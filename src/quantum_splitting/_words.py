"""Combinatorial plans for series in the two noncommuting letters A and B."""

from functools import cache
from itertools import product

_BITS = str.maketrans("AB", "01")


def word_index(word: str) -> int:
    """Pack a word as a binary 1 followed by its letters (A=0, B=1)."""
    return int("1" + word.translate(_BITS), 2)


@cache
def lyndon_words(order: int) -> tuple[str, ...]:
    """List mixed Lyndon words of odd degrees 3 through order, degree first."""
    words = []
    for degree in range(3, order + 1, 2):
        for middle in product("AB", repeat=degree - 2):
            word = "A" + "".join(middle) + "B"
            if all(word < word[k:] + word[:k] for k in range(1, degree)):
                words.append(word)
    return tuple(words)


def count_order_conditions(order: int) -> int:
    """Count the odd Lyndon conditions above degree one without listing words."""
    primitive_counts = {1: 2}
    total = 0
    for degree in range(3, order + 1, 2):
        # Binary words of length d consist of repetitions of a primitive word
        # whose length divides d. Each primitive rotation class gives one Lyndon word.
        repeated = sum(size for shorter, size in primitive_counts.items() if degree % shorter == 0)
        primitive_counts[degree] = (1 << degree) - repeated
        total += primitive_counts[degree] // degree
    return total


@cache
def append_terms(order: int, letter: str) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    """Plan the nonconstant terms for right multiplication by exp(c*letter).

    Each entry contains a destination index and (prefix index, power) pairs.
    Descending destination order permits in-place updates, as each prefix is
    shorter than its destination word.
    """
    bit = int(letter == "B")
    plan = []
    for destination in range((1 << (order + 1)) - 2 + bit, 1, -2):
        prefix = destination
        terms = []
        while prefix > 1 and (prefix & 1) == bit:
            prefix >>= 1
            terms.append((prefix, len(terms) + 1))
        plan.append((destination, tuple(terms)))
    return tuple(plan)


def derivative_splits(word: str) -> tuple[tuple[int, int], ...]:
    """Give prefix and reversed-suffix indices at every occurrence of B."""
    return tuple(
        (word_index(word[:i]), word_index(word[i + 1 :][::-1]))
        for i, letter in enumerate(word)
        if letter == "B"
    )
