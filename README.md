# qqvi

## What is this?

This is a simple vim "plugin", based on a couple of Python scripts and
a little bit of vimrc, for interacting with chat completion endpoints
from within the ergonomic comfort of vim.

## Installation

```sh
./install_qq.py [--prefix=$HOME/.local]
```

You should also ensure that `$HOME/.local/bin` is in your `$PATH`.

## Usage

Configure endpoint API tokens either via environment variables
(e.g. `$DEEPSEEK_API_KEY`, `$ANTHROPIC_API_KEY`, `$TOGETHER_API_KEY`, etc.)
or via text files located in `~/.qq/api_tokens`.

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

## See also

- [Christoffer Stjernlöf's q script](https://entropicthoughts.com/q)
  (based on [Simon Willison's llm tool](https://simonwillison.net/2024/Aug/7/q-what-do-i-title-this-article/))
- [qq.fish](https://github.com/dzervas/dotfiles/blob/main/home/fish-functions/qq.fish)
- [qqqa](https://github.com/matisojka/qqqa)

## License

MIT License
