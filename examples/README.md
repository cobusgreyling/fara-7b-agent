# Examples

Recorded Fara-7B inference turns. Each subdirectory contains:

- `turn_NN.png` — the screenshot fed to the model on turn N
- `transcript.json` — full task, raw model output, parsed thought + action, timing, notes

These were captured on **Apple M2 Pro, 16 GB RAM, Metal** running the
`Q4_K_M` quantisation. Both screenshots are 1280×800, which tokenises
to 1,334 vision tokens before the model has read a word of the task.

## wikipedia_claude_shannon/

Starting screen: Wikipedia Main Page.
Task: *Find the Wikipedia article about Claude Shannon.*

Fara chose to type into the search bar rather than navigate to a URL —
the most direct route given what is on screen.

## flight_capetown_london/

Starting screen: google.com homepage.
Task: *Find the cheapest flight from Cape Town to London next Friday with BA.*

Fara skipped the Google search and navigated straight to britishairways.com,
inferring the destination domain from the user's "with BA" hint.

## Quantisation artifacts

Both examples show the Q4_K_M quant drifting the wrapper `name` field
(`"Rasa"`, `"B"`) — full-precision Fara emits `"computer_use"`. The
parser dispatches on the inner `action` field, so wrapper drift does
not affect execution. Action verbs also drift (`"visit"` instead of
`"visit_url"`); the parser normalises common aliases.

## Failure modes

`failure_modes.md` documents the seven rough edges you will hit on a long
enough run — wrapper-name drift, hallucinated coordinates, SPA confusion,
context-window exhaustion, and the rest — and what the harness does about
each one. Read this before debugging an unexpected stall.
