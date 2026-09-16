# GL 3.3 incremental shards

These selections are for development, not the final conformance gate. Run one
shard at a time with a fresh WoW64 prefix:

```sh
./scripts/run-gl33-shard.sh \
  /path/to/wine-install \
  /path/to/VK-GL-CTS/build-win64/external/openglcts/modules/glcts.exe \
  tests/cts/gl33-shards/shader-indexing.txt \
  /tmp/gl33-indexing.qpa
```

Rules:

1. Start with one exact case using `run-gl33-case.sh`.
2. Run the smallest affected shard.
3. Run the regression ledger after the related fix.
4. Do not run `KHR-GL33.*` during normal iteration.
5. Run the full selection only as the final release gate.

A shard file can contain exact case paths or bounded CTS patterns. Blank lines
and lines beginning with `#` are ignored. `NotSupported` is reported but does
not fail a pattern shard; `Fail`, `InternalError`, an empty result, or a failed
CTS process does fail it.

## Current shards

- `regression-api.txt`: the non-shader cases from the focused regression set.
- `regression-fbo-transfer.txt`: the focused packed/depth/FBO transfer cases.
- `regression-shader-arrays.txt`: the focused shader-array regression cases.
- `shader-arrays.txt`: the GL 3.3 shader-array group.
- `shader-indexing.txt`: the GL 3.3 shader-indexing group.
- `shader-constructors.txt`: the GL 3.3 GLSL constructor group.

The 108-case file `../gl33-failure-ledger.txt` remains the combined regression
gate. It is intentionally separate from the smaller edit-test shards.
