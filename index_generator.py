"""

Indexes the words of a text file (word -> line numbers) in lexicographic
(ASCII) order, implemented three ways: Trie, Binary Search Tree, Python dict.
Every word keeps a linked-list queue of the line numbers where it occurs.

Just run the file (e.g. click Run in your IDE) and pick from the menu:
  1) Index a text file  -> asks for input file, output file, data structure
  2) Full benchmark     -> generates the six large .txt files, times Trie/BST/dict,
                           writes the index files and plots the Matplotlib graphs

Command line (optional):
  python index_generator.py input.txt output.txt [--impl trie|bst|dict]
"""
import argparse
import os
import random
import re
import sys
import time

# Separators: blanks, line breaks, '.', ',', ':', ';' (tab/CR treated as blanks).
# Every other printable ASCII character is part of a word.
SEPARATORS = re.compile(r"[ \t\r\n.,:;]+")


# ---------------- linked-list queue of line numbers ----------------
class _LineNode:
    __slots__ = ("line", "next")

    def __init__(self, line):
        self.line = line
        self.next = None


class LineQueue:
    """Linked-list queue: O(1) enqueue at tail, read from head."""
    __slots__ = ("head", "tail")

    def __init__(self):
        self.head = self.tail = None

    def add(self, line):
        if self.tail is not None and self.tail.line == line:
            return                       # same word twice on one line: keep once
        node = _LineNode(line)
        if self.tail is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node.line
            node = node.next

    def __str__(self):
        return ", ".join(str(n) for n in self)


# ---------------- (a) Trie ----------------
class _TrieNode:
    __slots__ = ("children", "queue")

    def __init__(self):
        self.children = {}               # character -> _TrieNode
        self.queue = None                # LineQueue if a complete word ends here


class TrieIndex:
    name = "Trie"

    def __init__(self):
        self.root = _TrieNode()

    def insert(self, word, line):
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = node.children[ch] = _TrieNode()
            node = nxt
        if node.queue is None:
            node.queue = LineQueue()
        node.queue.add(line)             # repeated word -> enqueue another line

    def search(self, word):
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return None
        return node.queue                # None if only a prefix of other words

    def items(self):
        """Pre-order DFS with children in ASCII order -> lexicographic order."""
        stack = [(self.root, "")]
        while stack:
            node, prefix = stack.pop()
            if node.queue is not None:
                yield prefix, node.queue
            for ch in sorted(node.children, reverse=True):
                stack.append((node.children[ch], prefix + ch))


# ---------------- (b) Binary Search Tree ----------------
class _BSTNode:
    __slots__ = ("word", "queue", "left", "right")

    def __init__(self, word, line):
        self.word = word
        self.left = self.right = None
        self.queue = LineQueue()
        self.queue.add(line)


class BSTIndex:
    name = "BST"

    def __init__(self):
        self.root = None

    def insert(self, word, line):
        if self.root is None:
            self.root = _BSTNode(word, line)
            return
        node = self.root
        while True:
            if word == node.word:        # repeated word -> enqueue line
                node.queue.add(line)
                return
            if word < node.word:         # str comparison follows ASCII order
                if node.left is None:
                    node.left = _BSTNode(word, line)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = _BSTNode(word, line)
                    return
                node = node.right

    def search(self, word):
        node = self.root
        while node is not None and node.word != word:
            node = node.left if word < node.word else node.right
        return node.queue if node is not None else None

    def items(self):
        """Iterative in-order traversal -> lexicographic order."""
        stack, node = [], self.root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.word, node.queue
            node = node.right


# ---------------- (c) Python dictionary ----------------
class DictIndex:
    name = "Dict"

    def __init__(self):
        self.table = {}                  # word -> LineQueue

    def insert(self, word, line):
        queue = self.table.get(word)
        if queue is None:
            queue = self.table[word] = LineQueue()
        queue.add(line)

    def search(self, word):
        return self.table.get(word)

    def items(self):
        for word in sorted(self.table):  # dict is unordered: sort when printing
            yield word, self.table[word]


IMPLEMENTATIONS = {"trie": TrieIndex, "bst": BSTIndex, "dict": DictIndex}


# ---------------- file handling and the three timed actions ----------------
def load_tokens(path):
    """Read file once -> list of (line_number, [words]); line numbers start at 1."""
    data = []
    with open(path, "r", encoding="latin-1", newline="") as f:
        for lineno, line in enumerate(f, start=1):
            words = [w for w in SEPARATORS.split(line) if w]
            if words:
                data.append((lineno, words))
    return data


def build_index(index, data):            # action (a): construction
    for lineno, words in data:
        for w in words:
            index.insert(w, lineno)
    return index


def search_words(index, queries):        # action (b): search many words
    return sum(1 for w in queries if index.search(w) is not None)


def write_index(index, out_path):        # action (c): print in sorted order
    with open(out_path, "w", encoding="latin-1", newline="\n") as f:
        for word, queue in index.items():
            f.write(f"{word}: {queue}\n")


# ---------------- performance comparison ----------------
ALPHABET = [chr(c) for c in range(33, 127) if chr(c) not in ".,:;"]
SEPS = [" ", "  ", ", ", ". ", ": ", "; "]


def generate_file(path, distinct, total, rng):
    """Text file with `total` words and exactly `distinct` different words."""
    vocab = set()
    while len(vocab) < distinct:
        vocab.add("".join(rng.choices(ALPHABET, k=rng.randint(3, 10))))
    vocab = list(vocab)
    words = vocab + rng.choices(vocab, k=total - distinct)
    rng.shuffle(words)
    with open(path, "w", encoding="latin-1") as f:
        for i in range(0, total, 10):    # 10 words per line
            line = words[i]
            for w in words[i + 1:i + 10]:
                line += rng.choice(SEPS) + w
            f.write(line + "\n")
    return vocab


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return time.perf_counter() - t0, result


def plot_results(sizes, results, n_search, png):
    import matplotlib.pyplot as plt
    titles = {"build": "(a) Index construction time",
              "search": f"(b) Search time ({n_search} random words)",
              "print": "(c) Printing index in lexicographic order"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, action in zip(axes, titles):
        for key, cls in IMPLEMENTATIONS.items():
            ax.plot(sizes, results[key][action], marker="o", label=cls.name)
        ax.set(xscale="log", xlabel="Number of distinct words",
               ylabel="Time (seconds)", title=titles[action])
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"\nGraph saved to {png}")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.show()                   # opens the graph window when possible
    except Exception:
        pass


def run_compare(sizes, total_words, n_search, workdir, seed=42):
    os.makedirs(workdir, exist_ok=True)
    rng = random.Random(seed)
    results = {k: {"build": [], "search": [], "print": []} for k in IMPLEMENTATIONS}
    for size in sizes:
        path = os.path.join(workdir, f"words_{size}.txt")
        print(f"\n== {size} distinct words, {total_words} total ==", flush=True)
        print("Generating input file ...", flush=True)
        vocab = generate_file(path, size, total_words, rng)
        data = load_tokens(path)
        queries = rng.choices(vocab, k=n_search)         # random words to search
        outs = []
        for key, cls in IMPLEMENTATIONS.items():
            out = os.path.join(workdir, f"index_{size}_{key}.tmp")
            t_build, idx = timed(build_index, cls(), data)
            t_search, found = timed(search_words, idx, queries)
            t_print, _ = timed(write_index, idx, out)
            assert found == n_search, "a stored word was not found"
            results[key]["build"].append(t_build)
            results[key]["search"].append(t_search)
            results[key]["print"].append(t_print)
            print(f"{cls.name:5s} build={t_build:8.3f}s  search={t_search:7.3f}s  "
                  f"print={t_print:7.3f}s", flush=True)
            outs.append(out)
        contents = [open(o, "rb").read() for o in outs]
        assert contents[0] == contents[1] == contents[2], "outputs differ"
        del contents
        # all three indexes are identical: keep one as the output index file
        os.replace(outs[0], os.path.join(workdir, f"index_{size}.txt"))
        for o in outs[1:]:
            os.remove(o)
    plot_results(sizes, results, n_search, os.path.join(workdir, "performance.png"))
    print(f"\nAll files are in: {os.path.abspath(workdir)}")


# ---------------- running the program ----------------
def ask_file(prompt, must_exist):
    while True:
        name = input(prompt).strip().strip('"').strip("'")
        if not name:
            continue
        if must_exist and not os.path.isfile(name):
            print("  File not found, try again.")
            continue
        return name


def index_file(inp, out, impl):
    index = build_index(IMPLEMENTATIONS[impl](), load_tokens(inp))
    write_index(index, out)
    print(f"Index written to {out} using {index.name}.")


def interactive():
    print("=== Word Index Utility ===")
    print("1) Index a text file")
    print("2) Run full benchmark (generates the six large .txt files)")
    choice = ""
    while choice not in ("1", "2"):
        choice = input("Choose 1 or 2: ").strip()
    if choice == "1":
        inp = ask_file("Input text file name : ", must_exist=True)
        out = ask_file("Output index file name: ", must_exist=False)
        impl = ""
        while impl not in IMPLEMENTATIONS:
            impl = input("Data structure (trie / bst / dict) [trie]: ").strip().lower() or "trie"
        index_file(inp, out, impl)
    else:
        workdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data")
        print(f"Files will be created in: {workdir}")
        print("This can take several minutes in pure Python ...")
        run_compare([1000, 5000, 10000, 50000, 100000, 500000],
                    1_100_000, 100_000, workdir)


def main():
    if len(sys.argv) == 1:               # plain Run: ask the user what to do
        interactive()
        return
    p = argparse.ArgumentParser(description="Word index generator")
    p.add_argument("input", help="input text file name")
    p.add_argument("output", help="output index file name")
    p.add_argument("--impl", choices=IMPLEMENTATIONS, default="trie")
    a = p.parse_args()
    index_file(a.input, a.output, a.impl)


if __name__ == "__main__":
    main()