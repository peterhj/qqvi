# qqvi

## What is this?

This is a simple vim "plugin", based on a couple of Python scripts and
a little bit of vimrc, for interacting with LLM inference endpoints
from within the ergonomic comfort of vim.

## Installation

```sh
./install_qq.py [--prefix=$HOME/.local]
cat vimrc.example >> $HOME/.vimrc
```

You should also ensure that `$HOME/.local/bin` is in your `$PATH`.

## Basic usage

Configure endpoint API tokens via environment variables,
e.g. `$DEEPSEEK_API_KEY`, `$ANTHROPIC_API_KEY`, `$XAI_API_KEY`, etc.

In vim:

1.  `:new` or `:vnew` to open a new buffer.
2.  `:qqq` to load a fresh `qq` text file in the current buffer.
3.  Edit your prompt after the `^Q^Q` escape chars.
4.  `:qq` to submit your context window to a chat completions endpoint
    (default model: [DeepSeek-V3](https://api-docs.deepseek.com/)).
5.  Wait a little bit.
6.  The current buffer will auto reload to display the new context
    window, with the assistant response after the `^A^A` escape chars.
    More `^Q^Q` escape chars are also appended at the end of the text
    file.
7.  Repeat (3)-(6).

## Advanced usage

- Any text that you place before (above/left of) the initial `^Q^Q` prompt
  will be interpreted as an initial _system message_.

## Implementation details / changelog

As of the December 2025 update, this script has been migrated to preferably use
`/v1/messages`-compatible APIs where possible.

Technically, we implement a superset of the messages API by supporting an
initial `"system"` role message, which is translated to a special field in the
API request.

## See also

- [Christoffer Stjernlöf's q script](https://entropicthoughts.com/q)
  (based on [Simon Willison's llm tool](https://simonwillison.net/2024/Aug/7/q-what-do-i-title-this-article/))
- [qq.fish](https://github.com/dzervas/dotfiles/blob/main/home/fish-functions/qq.fish)
- [qqqa](https://github.com/matisojka/qqqa)

## License

MIT License
