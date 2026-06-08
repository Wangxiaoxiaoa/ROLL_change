# GUI RLVR Reward Design

This reward replaces the previous single-tool-call reward. It supports action
sequences in both predictions and ground truth, including cases such as
`mouse_move` followed by `scroll`.

## Final Score

```text
score = format_gate * (
    0.80 * action_sequence_score
  + 0.15 * format_score
  + 0.05 * efficiency_score
)
```

The score is clipped to `[0, 1]`.

## Format Gate

`format_gate` is either `0` or `1`.

It is `1` only when all predicted tool calls are executable:

- At least one `<tool_call>...</tool_call>` exists.
- Every tool call payload is valid JSON.
- Every tool call has `name == "computer_use"`.
- Every tool call has `arguments.action`.

If this gate is `0`, the final score is `0`.

## Format Score

`format_score` rewards the response shape after the gate passes:

- `1.0` if a valid tool-call sequence exists and the response contains an
  `Action:` line.
- `0.85` if tool calls are valid but the `Action:` line is missing.

## Action Sequence Score

The reward parses both model output and ground truth into ordered tool-call
sequences:

```text
pred = [p1, p2, ...]
gold = [g1, g2, ...]
```

It uses dynamic-programming sequence alignment. Matched actions receive a
single-action score, omitted gold actions and extra predicted actions lower the
score through the normalization denominator:

```text
action_sequence_score = best_aligned_score / max(len(pred), len(gold))
```

This means a gold sequence like `mouse_move -> scroll` cannot receive full
credit from a prediction that only outputs `scroll`.

## Single Action Score

```text
single_action_score = 0.65 * action_type_score + 0.35 * argument_score
```

`action_type_score`:

- Exact action match: `1.0`
- Click-family mismatch, such as `left_click` vs `double_click`: `0.5`
- Pointer-family mismatch, such as click vs move/drag: `0.35`
- Key-family mismatch, such as `key` vs `key_down`: `0.4`
- Otherwise: `0.0`

`argument_score` is only used for exact action matches.

Argument scoring by action type:

- Pointer actions: continuous coordinate distance score
  `exp(-(distance / 80)^2)` on the 0-1000 coordinate scale.
- `scroll`: `0.8 * direction_match + 0.2 * magnitude_similarity`.
- `type`: text similarity using sequence matching.
- `key`, `key_down`, `key_up`: normalized key sequence similarity.
- `wait`: full argument credit once action type matches.
- `terminate`: status must match, defaulting missing status to `success`.

## Efficiency Score

`efficiency_score` is a small regularizer. It does not decide task correctness.

It rewards:

- Predicted action count close to gold action count.
- Concise non-tool-call text.
- No repeated identical actions.

It penalizes extra high-risk actions more than low-risk actions:

- High risk: click, drag, type, key, terminate.
- Low risk: mouse_move, wait, scroll.

## Logged Reward Fields

The reward worker logs:

- `score`: final reward used by RL.
- `format_gate`: executable-format gate.
- `action_sequence`: aligned action-sequence score.
- `format`: format score.
- `efficiency`: efficiency score.
