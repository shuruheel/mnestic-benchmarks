# Interrupted attempt — November 5, 2026

Engine c3f70b9174d182993b5ffaf802cf3f20be17fc19; harness 6dfef02ec2d65499ba9ecf7ba79cd80d947b467e.
First baseline run began at 23:21:50 EST and was interrupted at 23:21:59 after three samples of busy fileproviderd. Zero of twelve runs completed. Raw rows contain only manifest and dataset records; there are no E1–E3 measurements to accept.

The guard marked timing invalid, then its unconditional final group SIGKILL raised PermissionError. A November 6 process inspection found no remaining benchmark processes. The precise kernel reason for that denial was not established; the replacement checks live membership and preserves genuine cleanup failures.

Original evidence remains under /tmp/mnestic-019-campaign-c3f70b91-quiet and adjacent guard files. Its partial database is excluded from this raw-log archive. The owner explicitly authorized a fresh attempt on November 6; this is not an automatic retry or a continuation of the failed run.
