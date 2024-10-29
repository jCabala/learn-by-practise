from itertools import islice, cycle, count
from typing import Generator, Any, Callable, Iterable

class HaskellList:
    def __init__(self, generator: Callable[[], Generator[Any, None, None]]):
        """Initialize the HaskellList with a generator function for lazy evaluation."""
        self.generator_func = generator

    def generator(self):
        """Creates a new generator instance each time the list is accessed."""
        return self.generator_func()

    @staticmethod
    def from_list(lst: Iterable):
        """Corresponds to Haskell's finite list constructor (e.g., [1, 2, 3])."""
        def generator():
            for item in lst:
                yield item
        return HaskellList(generator)

    @staticmethod
    def repeat(value: Any):
        """Corresponds to Haskell's `repeat x`, creating an infinite list of x (e.g., repeat 3)."""
        def generator():
            while True:
                yield value
        return HaskellList(generator)

    @staticmethod
    def cycle(lst: Iterable):
        """Corresponds to Haskell's `cycle [1, 2, 3]`, creating an infinite list by cycling through a finite list."""
        def generator():
            while True:
                for item in lst:
                    yield item
        return HaskellList(generator)

    @staticmethod
    def iterate(func: Callable[[Any], Any], start: Any):
        """Corresponds to Haskell's `iterate f x`, producing an infinite list by applying f repeatedly to x."""
        def generator():
            value = start
            while True:
                yield value
                value = func(value)
        return HaskellList(generator)

    def cons(self, value: Any):
        """Corresponds to Haskell's `x : xs` syntax, which prepends an element to the front of the list."""
        def generator():
            yield value
            for item in self.generator():
                yield item
        return HaskellList(generator)

    def head(self) -> Any:
        """Corresponds to Haskell's `head xs`, returning the first element of the list."""
        return next(self.generator())

    def tail(self):
        """Corresponds to Haskell's `tail xs`, returning all elements except the first."""
        def generator():
            it = iter(self.generator())
            next(it)  # Skip the first element (head)
            for item in it:
                yield item
        return HaskellList(generator)

    def index(self, n: int) -> Any:
        """Corresponds to Haskell's `xs !! n`, which retrieves the nth element (0-indexed)."""
        return next(islice(self.generator(), n, n + 1))

    def take(self, n: int):
        """Corresponds to Haskell's `take n xs`, which takes the first n elements of a list."""
        return list(islice(self.generator(), n))

    def __getitem__(self, n: int) -> Any:
        """Override for `xs !! n` indexing syntax (list[index])."""
        return self.index(n)

    def __iter__(self):
        """Allows iteration over the HaskellList, corresponding to regular Haskell list evaluation."""
        return self.generator()

    def __repr__(self):
        """Provides a display similar to Haskell's list syntax for easier visualization."""
        return "HaskellList(" + ", ".join(map(str, self.take(10))) + " ...)"


# Examples of usage:

# Creating a finite HaskellList
finite_list = HaskellList.from_list([1, 2, 3, 4, 5])
print(finite_list.head())   # Output: 1
print(finite_list.tail())   # Output: HaskellList(2, 3, 4, 5 ...)
print(finite_list.index(2)) # Output: 3
print(finite_list.take(3))  # Output: [1, 2, 3]

# Infinite lists
infinite_list = HaskellList.repeat(1)
print(infinite_list.take(5))  # Output: [1, 1, 1, 1, 1]

cycled_list = HaskellList.cycle([1, 2, 3])
print(cycled_list.take(7))    # Output: [1, 2, 3, 1, 2, 3, 1]

iterated_list = HaskellList.iterate(lambda x: x + 1, 0)
print(iterated_list.take(5))  # Output: [0, 1, 2, 3, 4]

# Using cons
new_list = finite_list.cons(0)
print(new_list.take(6))       # Output: [0, 1, 2, 3, 4, 5]