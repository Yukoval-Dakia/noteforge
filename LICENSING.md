# NoteForge licensing

## Current releases

Starting with NoteForge 1.2.0, the following original material is licensed
under the Apache License, Version 2.0:

- source code, command-line tools, documentation, and HTTP API;
- NoteForge planner and placement weights;
- the distributed r5 model weights.

The full license text is in `LICENSE`, and required attribution notices are in
`NOTICE`.

## Historical releases

NoteForge releases through v1.1.1 were published under the MIT License. That
grant is permanent for those released copies and cannot be revoked by the later
change. The Apache-2.0 license applies to releases from v1.2.0 onward.

The r5 weights in the v1.0.0 Release were published under MIT. NoteForge 1.2.0
and later releases offer Apache-2.0 permission for the same published weight
bytes, identified by SHA-256. The download may remain hosted on the original
v1.0.0 Release URL; the later Apache-2.0 grant does not withdraw the earlier
MIT grant.

## Upstream MERT weights

MERT-v2-FullSong is not a NoteForge asset and is not distributed in this
repository. The pinned upstream checkpoint is licensed CC BY-NC 4.0
(non-commercial). Running NoteForge through the MERT feature path is therefore
limited to non-commercial use unless the user obtains separate permission from
the upstream rights holder.

The Apache-2.0 license for NoteForge code and original weights does not
relicense MERT, its modeling code, or its checkpoint weights.

## Closed training materials

Training code, training data, corpus construction, training methods, and
internal evaluation materials are not distributed in this repository and are
not licensed by `LICENSE`.

Third-party chart and audio content is not distributed here. Its own license
continues to govern any separate use or redistribution.
