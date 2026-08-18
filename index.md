---
title: "mimic-jax scientific run reports"
description: "Self-describing SAGE16 runs, numerical diagnostics, and physically interpretable responses"
toc: false
---

# Understand the run before interpreting the science

Mimic-jax reports connect familiar SAGE results to the evidence needed to trust and interrogate them. Each published run records what was evolved, which diagnostics were actually evaluated, where its scientific arrays live, and how to reproduce it.

## Published reference reports

- [SAGE16 Mini-Millennium: initial mimic-jax run report](reports/mini-millennium-sage16-initial/index.md) — familiar upstream plots, selected-tree equivalence, controlled conservation and timestep diagnostics, a validated fractional parameter response, performance, and complete provenance.

## How to read a report

The opening health table distinguishes passed, warning, failed, and not evaluated checks. Familiar SAGE plots come first. Numerical diagnostics and physically interpreted derivatives follow, with array products linked rather than embedded in the manifest. The final provenance section records the code, command, configurations, inputs, software, and hardware.

Every report is also available as ordinary Markdown for GitHub or Obsidian and as a compact `report.json` manifest for agents and future MCP tools. See the [report architecture](docs/reporting.md) for the stable contract.
