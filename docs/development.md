# Development and verification

Supported Python versions: 3.12–3.14. Docker Compose v2 is the alternative to installing
Python locally. The official base image is pinned by multi-platform digest.

```sh
make local                 # hashed dependency setup, offline demo and tests
make lint typecheck        # Ruff and mypy
make verify-package        # wheel + sdist; import/CLI check outside checkout
make local-benchmark RUNS=3
```

The package verifier installs the wheel without dependencies into a temporary target
and uses the existing locked dependency environment. It checks included vocabulary,
license and type marker assets, source reproduction files, import origin, tokenization
and CLI behavior with an empty tokenizer cache.

All dependency lines have exact versions and PyPI file hashes. Build requirements are
locked too. Setup installs them before editable installation, with build isolation and
dependency resolution disabled for the project itself. Regenerate locks as an explicit
dependency update, inspect artifact hashes, and rerun the version matrix and container CI.

The bundled tokenizer's compressed and uncompressed hashes are checked at load time.
It does not call tiktoken's downloader. Keep its MIT notice and manifest when replacing it.

## Important contracts

Tests cover long-turn budgeting, source revisions, transactional failures, complete
tool groups, stable prefix ordering, unsafe JSON/Unicode, corruption detection,
stream failure/timeout/cancellation, privacy and live accounting. Real SDK transport tests
exercise Responses count/compact/SSE parsing without remote calls.

The offline CI matrix runs lint, formatting, typing, branch coverage, benchmark smoke
tests, wheel/source builds and installed-wheel verification. A separate job builds the
container, tests it with no network and verifies the documented demo and runtime image.
Live inference is never part of default CI.

No live credentials belong in fixtures or committed files. --include-content creates
private local artifacts and is inappropriate for public evidence with real data.
Default traces contain timing and usage metadata only; the compile command intentionally
creates a request file containing the imported conversation.

Use a fresh output directory per run. Configuration or provider failures return nonzero;
completed live samples survive a later provider failure. Interrupted live runs may end
before the aggregate summary is written; trace records already exported remain on disk.

The Docker profile for live calls enables networking and forwards credentials only when
explicitly selected. Offline profiles cannot call remote models. Local permission mode
0600 applies to output files; directory policies and host access are still your responsibility.
