# Failure modes

The recorded runs in `wikipedia_claude_shannon/` and `flight_capetown_london/`
are happy-path samples. This document covers the rough edges — what Fara-7B
looks like when it loses the plot, and what the harness does about it.

Reproduce each by running `python run.py "<task>"` with a vision-light
quantisation (Q4_K_M); higher-precision builds reduce the frequency of all of
these.

## 1. Wrapper-name drift

**Symptom.** The `<tool_call>` payload's outer `name` field drifts:

```json
{"name": "B",    "arguments": {"action": "left_click", ...}}
{"name": "Rasa", "arguments": {"action": "visit_url",  ...}}
```

Full-precision Fara emits `"name": "computer_use"`.

**Mitigation.** `fara/parser.py` dispatches on `arguments.action`, never on
the outer `name`. Covered by
`tests/test_parser.py::test_wrapper_name_drift_does_not_matter`.

## 2. Action-verb aliasing

**Symptom.** The inner `action` field shortens:

| Drifted      | Canonical      |
|--------------|----------------|
| `visit`      | `visit_url`    |
| `click`      | `left_click`   |
| `search`     | `web_search`   |
| `back`       | `history_back` |
| `done`       | `terminate`    |

**Mitigation.** `parser.ACTION_ALIASES` normalises the common drifts.
Anything not in the alias map passes through with its original name so the
executor can surface "unsupported action: …" rather than silently dispatch
the wrong handler.

## 3. Hallucinated coordinates

**Symptom.** On a 1280×800 viewport the model emits coordinates outside the
visible area, e.g. `[1340, 850]`. The click silently lands on the chrome or
is dropped.

**Mitigation.** `BrowserExecutor._resolve_xy` clamps to viewport bounds before
dispatching. This is a correctness band-aid — the underlying problem is that
the model misread the screenshot, and the next turn's screenshot will reveal
nothing happened, often triggering a productive retry. If the model insists
on the same wrong coordinate three turns in a row, you have a true regression.

## 4. SPA navigation confusion

**Symptom.** On a single-page app, clicking a "Home" link does not change
`page.url` — the URL only updates after the SPA's router runs. Fara sometimes
re-issues the same `visit_url` action because the screenshot has not yet
caught up.

**Mitigation.** `BrowserExecutor.settle()` calls
`page.wait_for_load_state("networkidle", timeout=settle_ms)` with a fallback
to a fixed `wait_for_timeout`. Increase `--settle-ms` (default 1500) on
heavy-JS sites.

## 5. The model never emits `terminate`

**Symptom.** The task is genuinely complete (the right page is on screen,
the right answer is visible) but Fara keeps scrolling or re-clicking.

**Mitigation.** Hard cap via `--max-turns`. The transcript still saves and
the screenshots are still useful. There is no clean fix on the harness side
— this is a model-side limitation.

## 6. Critical-Point false positive

**Symptom.** The page URL contains `/checkout` (e.g. a help-centre article
*about* checkout) and Fara emits a committing action. The harness pauses
when it should not.

**Mitigation.** Use `--interactive`. The proposed action is printed; one
keystroke approves and the loop continues. False positives are the intended
failure mode of `CRITICAL_URL_PATTERNS` — false negatives at a real
transactional boundary are far worse.

## 7. Context-window exhaustion

**Symptom.** At ~1,334 vision tokens per 1280×800 screenshot, `n_ctx=16384`
overflows somewhere around turn 6 if every screenshot is retained in the
prompt. Symptoms: degraded reasoning, then truncated output, then the
`<tool_call>` block is cut off and the parser returns `action=None`.

**Mitigation.** Sliding window — only the most recent
`--history-images` screenshots (default 3) are sent. Older turns contribute
only the assistant text. Memorised facts from `pause_and_memorize_fact` are
re-injected into the user text every turn so they survive the window.
