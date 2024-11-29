# qqvi

## What is this?

This is a simple vim "plugin", based on a couple of Python scripts and
a little bit of vimrc, for interacting with chat completion endpoints
from within the ergonomic comfort of vim.

## Installation

```sh
cat vimrc.example >> ~/.vimrc
mkdir -p ~/.local/bin
cp -pv qq_vim.py ~/.local/bin/_qq_vim
cp -pv qqq_vim.py ~/.local/bin/_qqq_vim
```

## Usage

In vim:

1.  `:new` or `:vnew` to open a new buffer.
2.  `:qqq` to load a fresh `qq` text file in the current buffer.
3.  Edit your prompt after the `^Q^Q` escape chars.
4.  `:qq` to submit your context window to a chat completions endpoint
    (default: QwQ-32B-Preview via together.ai).
5.  Wait a little bit.
6.  The current buffer will auto reload to display the new context
    window, with the assistant response after the `^A^A` escape chars.
    More `^Q^Q` escape chars are also appended at the end of the text
    file.
7.  Repeat (3)-(6).

## License

MIT License
