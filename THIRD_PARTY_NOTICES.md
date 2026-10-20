# Third-party notices

The bundled o200k_base vocabulary and encoding configuration originate from
[OpenAI tiktoken](https://github.com/openai/tiktoken), distributed under the MIT license.
Its full copyright and permission notice is included in
[TIKTOKEN_LICENSE.txt](src/context_window/data/TIKTOKEN_LICENSE.txt) and shipped in the wheel.

The vocabulary source, tiktoken version and compressed/uncompressed SHA-256 values
are recorded in [encoding.json](src/context_window/data/encoding.json). Runtime
construction verifies both hashes and never downloads a vocabulary.

Other dependencies are installed separately under their respective licenses.
This is an independent project; it is not affiliated with or endorsed by OpenAI.
