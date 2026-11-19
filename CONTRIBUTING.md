# Contributing to Solidus Pro

## License of contributions

Solidus Pro is licensed under **AGPL-3.0-or-later**. By submitting a pull
request, patch, or any other contribution, you agree that your contribution is
licensed under the same terms (**inbound = outbound**). You keep your
copyright. There is no CLA and no copyright assignment.

Submit only work you own or have the right to license. If your contribution
includes third-party code, say so and include its license.

## Developer Certificate of Origin

Sign off every commit:

```sh
git commit -s -m "Your change"
```

The sign-off adds a `Signed-off-by:` line and certifies the DCO 1.1 below.

### DCO 1.1

By making a contribution to this project, I certify that:

1. The contribution was created in whole or in part by me and I have the right
   to submit it under the open source license indicated in the file; or
2. The contribution is based upon previous work that, to the best of my
   knowledge, is covered under an appropriate open source license and I have
   the right under that license to submit that work with modifications,
   whether created in whole or in part by me, under the same open source
   license (unless I am permitted to submit under a different license), as
   indicated in the file; or
3. The contribution was provided directly to me by some other person who
   certified (1), (2) or (3) and I have not modified it.
4. I understand and agree that this project and the contribution are public and
   that a record of the contribution (including all personal information I
   submit with it, including my sign-off) is maintained indefinitely and may be
   redistributed consistent with this project or the open source license(s)
   involved.

## New files

Add an SPDX header to new source files:

```
SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-FileCopyrightText: 2025-2026 Brendon Jarrett
```

## Before opening a pull request

Run the same checks as CI:

```sh
scripts/ci/check.sh models
scripts/ci/check.sh pins
scripts/ci/check.sh syntax
scripts/ci/check.sh types
scripts/ci/check.sh tests
```
