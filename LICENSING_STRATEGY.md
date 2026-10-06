# Solidus Pro: GPL v3 + CLA Licensing Strategy

> **Status (Oct 6, 2026): Deferred.** Solidus Pro is proprietary, all rights
> reserved (see `LICENSE`). Nothing in this plan has been executed, and the
> repository stays private until a sale or licensing decision is made.

## 1. Overview

Solidus Pro is released under the **GNU General Public License v3.0 (GPL v3)** with a mandatory **Contributor License Agreement (CLA)**. This dual mechanism allows the foundry community to use, study, modify, and improve the software freely, while preserving the project owner's right to offer commercial licenses to organizations that cannot or will not comply with GPL v3's copyleft requirements.

### The Core Proposition

| For the Community | For Commercial Users |
|-------------------|---------------------|
| Free to use, modify, and distribute | Must comply with GPL v3 (open-source their modifications) |
| Can contribute improvements back | Or purchase a commercial license to keep their use proprietary |
| Full access to source code | Commercial license removes GPL obligations |
| No cost, no restrictions on learning/research | Revenue funds continued development |

---

## 2. How GPL v3 Works

### What GPL v3 Grants

Anyone who receives a copy of Solidus Pro under GPL v3 has the right to:

- **Use** the software for any purpose (personal, academic, commercial)
- **Study** the source code and understand how it works
- **Modify** the code to fix bugs or add features
- **Distribute** copies of the original or modified software

### What GPL v3 Requires (Copyleft)

Anyone who **distributes** a modified version of Solidus Pro must:

- Release their modifications under GPL v3 as well
- Make the complete source code available to all recipients
- Include the same GPL v3 license text
- Preserve all copyright notices and disclaimers

### The Key Insight

**Using GPL software internally is not distribution.** A foundry can run Solidus Pro on their own servers, modify it for their own use, and never share it -- and GPL v3 does not require them to release their changes. The copyleft obligation only triggers when they *distribute* the software (or a modified version) to someone else.

However, if a company:
- Bundles Solidus Pro into a product they sell to customers
- Offers Solidus Pro as a hosted SaaS service where users interact with it remotely
- Distributes modified binaries to their own subsidiaries or contractors

...then GPL v3's requirements apply. Many companies have policies against using GPL-licensed software precisely because of these obligations, even if their specific use case wouldn't technically trigger distribution. **This is where the commercial license becomes valuable.**

---

## 3. The Contributor License Agreement (CLA)

### Why a CLA is Essential

When someone from the community submits a pull request -- say, a fix to the Numba shrinkage kernel or a new alloy in the material database -- **they own the copyright to that code**. Without a CLA, you cannot re-license their contribution under a commercial license later. You would need their explicit permission for every single contribution.

The CLA solves this by having every contributor grant you broad rights to their contribution upfront.

### CLA Terms

```
CONTRIBUTOR LICENSE AGREEMENT

By submitting code, documentation, or other materials (collectively, 
"Contributions") to the Solidus Pro project, you agree to the following:

1. Copyright License Grant
   You grant Brendon Jarett ("Project Owner") a perpetual, worldwide, 
   non-exclusive, royalty-free, irrevocable license to use, reproduce, 
   modify, prepare derivative works of, publicly display, publicly perform, 
   sublicense, and distribute your Contributions under any license, 
   including but not limited to:
   - GNU General Public License v3.0 (or later)
   - Commercial/proprietary licenses
   - Any other license the Project Owner chooses

2. Patent License Grant
   You grant Project Owner a perpetual, worldwide, non-exclusive, 
   royalty-free, irrevocable patent license to make, have made, use, 
   sell, offer to sell, import, and otherwise transfer your Contributions.

3. Original Work
   You represent that each Contribution is your original work and that 
   you have the legal right to grant the licenses described above.

4. No Obligation
   Project Owner is under no obligation to accept or include your 
   Contribution in the project.

5. Retained Rights
   You retain all right, title, and interest in and to your Contributions. 
   This agreement does not transfer ownership of your Contributions to 
   Project Owner; it only grants the licenses described above.
```

### What the CLA Does NOT Do

- It does **not** transfer copyright ownership to you. Contributors retain their copyright.
- It does **not** prevent contributors from using their own code elsewhere.
- It does **not** prevent contributors from contributing to other projects.
- It does **not** give you exclusive rights. The contributor can still use their code however they want.

### What the CLA DOES Do

- It gives you the legal right to re-license their contributions under a commercial license.
- It gives you patent protection against contributors suing users of the project.
- It ensures you can close-source future versions of Solidus Pro without tracking down every contributor.

---

## 4. Dual Licensing Model

### How It Works in Practice

```
                    ┌─────────────────────────────┐
                    │   Solidus Pro Source Code    │
                    │   (Single Codebase)          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           ┌────────▼────────┐          ┌────────▼────────┐
           │  GPL v3 License │          │  Commercial     │
           │  (Free)         │          │  License        │
           │                 │          │  (Paid)         │
           └────────┬────────┘          └────────┬────────┘
                    │                             │
        ┌───────────┼───────────┐     ┌───────────┼───────────┐
        │           │           │     │           │           │
   Individuals   Universities   Hobbyists   Foundries   SaaS Providers
   Researchers   Students       Learners    Enterprises   Cloud Companies
        │           │           │     │           │           │
        └───────────┴───────────┘     └───────────┴───────────┘
              FREE                        PAID
         (Must open-source             (Can keep use
          modifications)                proprietary)
```

### Who Needs a Commercial License?

| Scenario | GPL v3 OK? | Commercial License Needed? |
|----------|-----------|---------------------------|
| Student learning casting simulation | Yes | No |
| University research project | Yes | No |
| Small foundry running simulations internally | Yes* | No* |
| Foundry modifying Solidus Pro and distributing to employees | Maybe | Probably |
| Company embedding Solidus Pro in a commercial product | No | Yes |
| Cloud provider offering Solidus Pro as a hosted service | No | Yes |
| Consultant using Solidus Pro to serve multiple clients | Maybe | Recommended |
| Company distributing modified Solidus Pro to customers | No | Yes |

\* *Internal use without distribution is generally permitted under GPL v3, but many corporate legal departments prohibit GPL software entirely. A commercial license removes this legal friction.*

### Commercial License Pricing

| Tier | Price | What It Covers |
|------|-------|---------------|
| **Single Seat** | $2,500/year | One engineer, internal use only, no distribution rights |
| **Team (up to 10)** | $8,000/year | Up to 10 engineers, internal use, priority support |
| **Enterprise** | $20,000/year | Unlimited seats, internal use, custom alloy support, SLA |
| **OEM / Embed** | Custom | Embed Solidus Pro in a commercial product, distribution rights |
| **SaaS / Cloud** | Custom + revenue share | Offer Solidus Pro as a hosted service to third parties |

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

1. **Add LICENSE file** to repository root with full GPL v3 text
2. **Add LICENSE header** to every source file:
   ```python
   # Solidus Pro - Casting Solidification Simulation Platform
   # Copyright (c) 2026 Brendon Jarett
   #
   # This program is free software: you can redistribute it and/or modify
   # it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   # (at your option) any later version.
   #
   # This program is distributed in the hope that it will be useful,
   # but WITHOUT ANY WARRANTY; without even the implied warranty of
   # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   # GNU General Public License for more details.
   #
   # You should have received a copy of the GNU General Public License
   # along with this program. If not, see <https://www.gnu.org/licenses/>.
   #
   # Commercial licenses are available. Contact brendon@soliduspro.com
   ```
3. **Set up CLA infrastructure**:
   - Use [cla-assistant.io](https://cla-assistant.io) (free for open source) or [EasyCLA](https://easycla.lfx.linuxfoundation.org/) (Linux Foundation)
   - Configure GitHub repository to require CLA signature before merging PRs
   - Add a `CLA.md` file to the repository with the full CLA text
4. **Update README.md** with licensing section explaining dual-license model

### Phase 2: Community Launch (Week 3-4)

1. **Public announcement** on relevant forums:
   - AFS (American Foundry Society) forums
   - Reddit r/foundry, r/manufacturing, r/engineering
   - GitHub trending (if the repo gains traction)
   - LinkedIn posts targeting foundry engineers
2. **Contributor guide** (`CONTRIBUTING.md`) explaining:
   - How to set up the development environment
   - Coding standards and testing requirements
   - The CLA process (automated via bot)
   - What kinds of contributions are most needed
3. **Issue labels** for community contribution:
   - `good-first-issue` -- easy entry points
   - `help-wanted` -- needs community help
   - `material-database` -- add new alloys
   - `physics-improvement` -- enhance simulation accuracy
   - `visualization` -- UI/UX improvements

### Phase 3: Commercial Infrastructure (Month 2-3)

1. **Landing page** for commercial licenses:
   - Explain why a company would want a commercial license
   - Pricing table
   - Contact form for enterprise inquiries
   - FAQ addressing common legal concerns
2. **License key system** (if you build commercial features):
   - Simple validation mechanism
   - Feature gating for commercial-only features
3. **Support infrastructure**:
   - Dedicated support channel for commercial customers
   - SLA documentation
   - Onboarding process

### Phase 4: Ongoing (Month 4+)

1. **Review and merge community PRs** (with CLA verification)
2. **Release regular updates** under GPL v3
3. **Engage with the community** -- respond to issues, host discussions
4. **Pursue commercial leads** -- reach out to foundries that are using the GPL version internally
5. **Build case studies** -- document real-world validation with community contributors

---

## 6. What Happens If You Want to Close Source Later

With the CLA in place, you have full legal flexibility:

### Scenario: You Want to Release v2.0 as Closed Source

1. **You own all your original code** → You can license it however you want.
2. **All community contributions are covered by the CLA** → You can re-license them under a commercial/proprietary license.
3. **v1.x remains available under GPL v3** → Anyone can continue using and modifying the last open-source version.
4. **v2.0 is released under a proprietary license** → Only commercial license holders get access.

### The Community Fork Risk

When you close-source v2.0, the community may fork v1.x and continue development as "OpenSolidus" or similar. This is legal and expected. To mitigate:

- **Be transparent**: Announce the change well in advance with clear reasoning.
- **Offer value in the commercial version**: Cloud hosting, verified material databases, professional support, enterprise features that a fork can't easily replicate.
- **Maintain goodwill**: Don't sabotage the open-source version. Let it continue to exist and serve its audience.

### The "Open Core" Alternative

Instead of fully closing source, you can adopt an **Open Core** model:

| Open Source (GPL v3) | Commercial Only |
|---------------------|-----------------|
| Basic mesh generation | GPU acceleration (CuPy) |
| Schedule-based solver | Thermal-coupled solver |
| 3 core alloys | Full 7-alloy database + custom alloys |
| Basic shrinkage prediction | All 9 defect analyses |
| Desktop UI | Cloud-hosted streaming |
| Community support | Professional support + SLA |

This is the model used by GitLab, Redis, and MongoDB. It's often more sustainable than a full close-source transition because the community continues to improve the open core while you monetize the enterprise features.

---

## 7. Legal Considerations

### GPL v3 and Patents

GPL v3 includes an explicit patent grant. Anyone who distributes GPL v3 software implicitly grants a patent license to all recipients. This protects users from patent trolls but also means you cannot patent algorithms in Solidus Pro and then sue users. If you want patent protection, you would need to file patents before releasing under GPL v3, or use a different license.

### GPL v3 and Trademarks

GPL v3 does **not** grant trademark rights. You retain full control over the "Solidus Pro" name and logo. A community fork must use a different name. This is important for brand protection.

### GPL v3 and SaaS (The "Affero" Gap)

GPL v3 does **not** require source code disclosure for software accessed over a network (SaaS). If you want to require SaaS providers to release their modifications, you would need **AGPL v3** (Affero GPL). However, AGPL v3 is more restrictive and may deter some community users.

**Recommendation**: Stick with GPL v3. If a cloud provider wants to offer Solidus Pro as a service without complying with GPL, they need a commercial license anyway. The commercial license is your enforcement mechanism, not the AGPL.

### GPL v3 and Linking

GPL v3's copyleft extends to **derivative works**. If someone writes a Python module that imports Solidus Pro's core modules and distributes it, that module may be considered a derivative work and must also be GPL v3. This is a grey area in Python (importing vs. linking), but the general consensus is that importing GPL code creates a derivative work.

---

## 8. Frequently Asked Questions

### Q: Can a foundry use Solidus Pro internally without paying?
**A:** Yes. GPL v3 permits internal use without distribution. They can modify it for their own purposes and never share it. However, if their legal department has a policy against GPL software, they may choose to purchase a commercial license to avoid the restriction entirely.

### Q: What if someone takes Solidus Pro, modifies it, and sells it?
**A:** They are allowed to do this under GPL v3, but they must:
- Release their modifications under GPL v3
- Provide source code to all their customers
- Include the GPL v3 license text
- Not add additional restrictions

Most companies don't want to do this because it forces them to open-source their own modifications. A commercial license lets them sell it without these obligations.

### Q: Can I use Solidus Pro code in my own commercial project?
**A:** Only if you release your entire project under GPL v3. If you want to keep your project proprietary, you need a commercial license from the project owner.

### Q: What happens to my contributions if the project closes sources?
**A:** Your contributions remain available under GPL v3 in the last open-source version. You retain the right to use your own contributions however you want. The CLA only grants additional rights to the project owner; it doesn't take away your rights.

### Q: Is the CLA enforceable?
**A:** Yes. The CLA is a standard legal agreement used by thousands of open-source projects (Google, Microsoft, Apache Foundation, etc.). By submitting a PR, the contributor explicitly agrees to the CLA terms. The cla-assistant bot records this agreement with a timestamp and IP address.

### Q: Can I change the license later?
**A:** Yes, for future versions. You can re-license v2.0 under any license you choose because you own the original code and have CLA coverage for all contributions. Past versions remain under GPL v3.

### Q: What if a contributor doesn't sign the CLA?
**A:** Their contribution cannot be merged. The cla-assistant bot will block the PR until the CLA is signed. This is non-negotiable -- without a CLA, you cannot re-license their code.

### Q: Should I use GPL v3 or AGPL v3?
**A:** GPL v3 is recommended. AGPL v3 adds a network-use clause that requires SaaS providers to release their source code. While this sounds good, it also deters some community users and is harder to enforce. GPL v3 + commercial licensing achieves the same practical outcome: SaaS providers need a commercial license.

---

## 9. File Checklist

To implement this strategy, create or modify these files:

| File | Purpose | Status |
|------|---------|--------|
| `LICENSE` | Full GPL v3 license text | Create |
| `CLA.md` | Contributor License Agreement text | Create |
| `CONTRIBUTING.md` | Guide for community contributors | Create |
| `COMMERCIAL.md` | Commercial licensing information | Create |
| `README.md` | Update with licensing section | Modify |
| Every `.py` file | Add GPL v3 header comment | Modify |
| GitHub repository settings | Enable CLA assistant | Configure |

---

## 10. Summary

The GPL v3 + CLA model gives you the best of both worlds:

- **Community growth**: Engineers, students, and foundries can freely use and improve Solidus Pro.
- **Revenue potential**: Companies that need proprietary use, distribution rights, or SaaS hosting must purchase a commercial license.
- **Future flexibility**: The CLA ensures you can re-license or close-source future versions without legal obstacles.
- **Brand protection**: Trademark rights remain yours. Community forks must use different names.
- **Low friction**: The CLA process is automated. Contributors sign with one click.

This is the same model used by Qt, MySQL, MongoDB, and many other successful projects. It has been legally tested and proven at scale.

---

*This document is for informational purposes and does not constitute legal advice. Consult with an attorney specializing in open-source licensing before implementing this strategy.*
