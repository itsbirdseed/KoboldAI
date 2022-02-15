import collections
from typing import Iterable, Tuple, Dict, Optional, Hashable, Generic, TypeVar


KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")
class Automaton(Generic[KT, VT]):
    '''
    Aho-Corasick finite state automaton
    <https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm>
    '''

    class NotAutomatonError(Exception):
        pass

    class Node(Generic[KT, VT]):
        def __init__(self, k: KT, parent: Optional["Automaton.Node[KT, VT]"] = None):
            self.goto: Dict[KT, Automaton.Node[KT, VT]] = {}
            self.failure: Optional[Automaton.Node[KT, VT]] = None
            self.output: Optional[Automaton.Node[KT, VT]] = None
            self.parent = parent
            self.is_leaf = False
            self.k = k
            self.v: Optional[VT] = None

    def __init__(self):
        self._root: Automaton.Node[KT, VT] = self.Node(None)
        self.__length = 0
        self.__is_automaton = True

    def add_word(self, key: Iterable[KT], value: VT = None):
        node = self._root
        for k in key:
            if k not in node.goto:
                new_node = self.Node(k, parent=node)
                node.goto[k] = new_node
                node = new_node
            else:
                node = node.goto[k]
        node.v = value
        if node.is_leaf or node is self._root:
            return False
        node.is_leaf = True
        self.__is_automaton = False
        self.__length += 1
        return True

    def is_automaton(self):
        return self.__is_automaton

    def make_automaton(self):
        if self.__is_automaton:
            return False
        queue: collections.deque[Automaton.Node[KT, VT]] = collections.deque((self._root,))
        while queue:
            node = queue.popleft()
            queue.extend(node.goto.values())
            node.failure = self._root
            if node is self._root or node.parent is self._root:
                continue
            ptr = node.parent
            while ptr is not self._root:
                ptr = ptr.failure
                failure = ptr.goto.get(node.k)
                if failure is not None:
                    node.failure = failure
                    break
            node.output = node.failure if node.failure.is_leaf else node.failure.output if node.failure.output is not None else None
        self.__is_automaton = True
        return True

    def iter(self, string: Iterable[KT]):
        '''
        Returns an iterator of the form `(last_index, value)` for each
        occurrence of a matching key in `string`, where `last_index` is an int
        that represents the zero-based position where the last character in the
        key matches `string`, and `value` is the value associated with said key.

        The matches are sorted in ascending order of `last_index`, then by
        descending order of the length of the key.
        '''
        if not self.__is_automaton:
            raise self.NotAutomatonError(
                "You need to call this object's `make_automaton()` method once after adding words"
            )
        node: Optional[Automaton.Node[KT, VT]] = self._root
        for index, char in enumerate(string):
            g = node.goto.get(char)
            while node is not self._root:
                if g is not None:
                    break
                node = node.failure
                g = node.goto.get(char)
            node = g if g is not None else self._root
            ptr = node
            while ptr is not None:
                if ptr.is_leaf:
                    v: VT = ptr.v
                    yield index, v
                ptr = ptr.output

    def _get(self, key: Iterable[KT]):
        node = self._root
        for k in key:
            node = node.goto.get(k)
            if node is None:
                return None
        if not node.is_leaf:
            return None
        return node

    def get(self, key: Iterable[KT], default: VT = None) -> VT:
        result = self._get(key)
        if result is None:
            return default
        return result.v

    def __getitem__(self, key: Iterable[KT]):
        result = self._get(key)
        if result is None:
            raise KeyError(key)
        return result.v

    def __setitem__(self, key: Iterable[KT], value: VT = None):
        self.add_word(key, value)

    def __contains__(self, key: Iterable[KT]):
        return self._get(key) is not None

    def __len__(self):
        return self.__length


class KoboldStoryRegister(collections.OrderedDict):
    '''
    Complexity-optimized class for keeping track of story chunks
    '''

    def __init__(self, sequence: Iterable[Tuple[int, str]] = ()):
        super().__init__(sequence)
        self.__next_id: int = len(sequence)

    def append(self, v: str) -> None:
        self[self.__next_id] = v
        self.increment_id()
    
    def pop(self) -> str:
        return self.popitem()[1]
    
    def get_first_key(self) -> int:
        return next(iter(self))

    def get_last_key(self) -> int:
        return next(reversed(self))

    def __getitem__(self, k: int) -> str:
        return super().__getitem__(k)

    def __setitem__(self, k: int, v: str) -> None:
        return super().__setitem__(k, v)

    def increment_id(self) -> None:
        self.__next_id += 1

    def get_next_id(self) -> int:
        return self.__next_id

    def set_next_id(self, x: int) -> None:
        self.__next_id = x
