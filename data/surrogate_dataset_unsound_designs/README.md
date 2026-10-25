# Set aside: unsound rigging (2026-10-24)

161 synthetic records moved out of `data/surrogate_dataset`. In each one some
casting metal freezes after the last riser has frozen, so that metal cannot be
fed and would shrink: a design a foundry would not sign off.

Cause: `core/surrogate/design_rules.py` counted faces against a sand core as
half a cooling face. Once cores were modelled as a conducting sand body
(commit 4ab9a8f) they heat through and stop cooling, so risers sized by that
rule were too small for the metal around cores (complex housings 83 %,
complex hubs 70 %, cored flanged cylinders 30 % unsound).

Fix: core faces now count as non-cooling, and `generate_dataset` rejects any
solved assembly whose casting outlasts its risers (`design_rules.unfed_metal`).

The check here was made on the encoded 64^3 grid. The plan entries in
`data/surrogate_dataset/generation_plan.json` are marked `invalid` with the
reason and point at these files. Nothing was deleted.
