import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ACTION_LINE_RE = re.compile(r"Action\s*:", re.IGNORECASE)

CLICK_ACTIONS = {"left_click", "right_click", "middle_click", "double_click", "triple_click"}
POINTER_ACTIONS = CLICK_ACTIONS | {"mouse_move", "left_click_drag"}
LOW_RISK_EXTRA_ACTIONS = {"mouse_move", "wait", "scroll"}
HIGH_RISK_EXTRA_ACTIONS = {
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "type",
    "key",
    "key_down",
    "key_up",
    "terminate",
}
KEY_ALIASES = {
    "ctrl": "control",
    "esc": "escape",
    "return": "enter",
    "del": "delete",
}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _loads_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _safe_json_loads(value: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], int]:
    matches = TOOL_CALL_RE.findall(text or "")
    calls: List[Dict[str, Any]] = []
    malformed = 0
    for payload in matches:
        obj = _safe_json_loads(payload)
        if obj is None:
            malformed += 1
            continue
        calls.append(obj)
    return calls, malformed


def _valid_call(call: Dict[str, Any]) -> bool:
    args = call.get("arguments")
    return call.get("name") == "computer_use" and isinstance(args, dict) and bool(args.get("action"))


def _load_ground_truth_sequence(ground_truth: Any) -> List[Dict[str, Any]]:
    try:
        obj = _loads_json(ground_truth)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    action = obj.get("action")
    if isinstance(action, dict):
        return [action] if _valid_call(action) else []
    if isinstance(action, str):
        calls, malformed = extract_tool_calls(action)
        if malformed:
            return []
        return [call for call in calls if _valid_call(call)]
    return []


def _num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _point(value: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    x = _num(value[0])
    y = _num(value[1])
    if x is None or y is None:
        return None
    return x, y


def _coord_score(pred: Any, gold: Any, sigma: float = 80.0) -> float:
    pred_point = _point(pred)
    gold_point = _point(gold)
    if pred_point is None or gold_point is None:
        return 0.0
    dist = math.hypot(pred_point[0] - gold_point[0], pred_point[1] - gold_point[1])
    return _clamp(math.exp(-((dist / sigma) ** 2)))


def _text_similarity(pred: Any, gold: Any) -> float:
    if pred is None or gold is None:
        return 0.0
    pred_text = str(pred)
    gold_text = str(gold)
    if pred_text == gold_text:
        return 1.0
    if not pred_text and not gold_text:
        return 1.0
    return _clamp(SequenceMatcher(None, pred_text, gold_text).ratio())


def _normalize_keys(value: Any) -> List[str]:
    keys = value if isinstance(value, list) else [value]
    norm = []
    for key in keys:
        key_text = str(key).strip().lower()
        norm.append(KEY_ALIASES.get(key_text, key_text))
    return norm


def _key_score(pred: Any, gold: Any) -> float:
    pred_keys = _normalize_keys(pred)
    gold_keys = _normalize_keys(gold)
    if pred_keys == gold_keys:
        return 1.0
    return _clamp(SequenceMatcher(None, pred_keys, gold_keys).ratio())


def _scroll_score(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> float:
    pred_pixels = _num(pred_args.get("pixels"))
    gold_pixels = _num(gold_args.get("pixels"))
    if pred_pixels is None or gold_pixels is None:
        return 0.0
    if pred_pixels == 0 and gold_pixels == 0:
        return 1.0
    direction = 1.0 if pred_pixels * gold_pixels > 0 else 0.0
    denom = max(abs(pred_pixels), abs(gold_pixels), 1e-6)
    magnitude = 1.0 - min(abs(abs(pred_pixels) - abs(gold_pixels)) / denom, 1.0)
    return _clamp(0.8 * direction + 0.2 * magnitude)


def _argument_score(pred_args: Dict[str, Any], gold_args: Dict[str, Any]) -> float:
    action = gold_args.get("action")
    if action in POINTER_ACTIONS:
        scores = []
        if "coordinate" in gold_args:
            scores.append(_coord_score(pred_args.get("coordinate"), gold_args.get("coordinate")))
        if "coordinate_2" in gold_args:
            scores.append(_coord_score(pred_args.get("coordinate_2"), gold_args.get("coordinate_2")))
        return sum(scores) / len(scores) if scores else 1.0
    if action == "scroll":
        return _scroll_score(pred_args, gold_args)
    if action == "type":
        return _text_similarity(pred_args.get("text"), gold_args.get("text"))
    if action in {"key", "key_down", "key_up"}:
        return _key_score(pred_args.get("keys"), gold_args.get("keys"))
    if action == "terminate":
        return 1.0 if str(pred_args.get("status", "success")).lower() == str(gold_args.get("status", "success")).lower() else 0.0
    if action == "wait":
        return 1.0

    gold_keys = [key for key in gold_args.keys() if key != "action"]
    if not gold_keys:
        return 1.0
    return sum(1.0 if pred_args.get(key) == gold_args.get(key) else 0.0 for key in gold_keys) / len(gold_keys)


def _action_type_score(pred_action: Any, gold_action: Any) -> float:
    if pred_action == gold_action:
        return 1.0
    if pred_action in CLICK_ACTIONS and gold_action in CLICK_ACTIONS:
        return 0.5
    if pred_action in POINTER_ACTIONS and gold_action in POINTER_ACTIONS:
        return 0.35
    if pred_action in {"key", "key_down", "key_up"} and gold_action in {"key", "key_down", "key_up"}:
        return 0.4
    return 0.0


def _single_action_score(pred_call: Dict[str, Any], gold_call: Dict[str, Any]) -> float:
    if pred_call.get("name") != gold_call.get("name"):
        return 0.0
    pred_args = pred_call.get("arguments") or {}
    gold_args = gold_call.get("arguments") or {}
    pred_action = pred_args.get("action")
    gold_action = gold_args.get("action")
    type_score = _action_type_score(pred_action, gold_action)
    if type_score <= 0.0:
        return 0.0
    arg_score = _argument_score(pred_args, gold_args) if pred_action == gold_action else 0.0
    return _clamp(0.65 * type_score + 0.35 * arg_score)


def _sequence_action_score(pred_calls: List[Dict[str, Any]], gold_calls: List[Dict[str, Any]]) -> float:
    if not pred_calls or not gold_calls:
        return 0.0
    rows = len(gold_calls) + 1
    cols = len(pred_calls) + 1
    dp = [[0.0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            pair = dp[i - 1][j - 1] + _single_action_score(pred_calls[j - 1], gold_calls[i - 1])
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], pair)
    return _clamp(dp[-1][-1] / max(len(gold_calls), len(pred_calls)))


def _format_scores(response: str, pred_calls: List[Dict[str, Any]], malformed: int) -> Tuple[float, float]:
    if malformed or not pred_calls:
        return 0.0, 0.0
    if not all(_valid_call(call) for call in pred_calls):
        return 0.0, 0.0
    has_action_line = bool(ACTION_LINE_RE.search(response or ""))
    format_score = 1.0 if has_action_line else 0.85
    return 1.0, format_score


def _outside_tool_text_len(response: str) -> int:
    outside = TOOL_CALL_RE.sub("", response or "")
    outside = ACTION_LINE_RE.sub("", outside)
    return len(re.sub(r"\s+", "", outside))


def _repeat_penalty(pred_calls: List[Dict[str, Any]]) -> float:
    if len(pred_calls) < 2:
        return 0.0
    repeats = 0
    previous = None
    for call in pred_calls:
        args = call.get("arguments") or {}
        signature = (args.get("action"), json.dumps(args, sort_keys=True, ensure_ascii=False))
        if signature == previous:
            repeats += 1
        previous = signature
    return min(0.25, repeats * 0.08)


def _extra_action_penalty(pred_calls: List[Dict[str, Any]], gold_calls: List[Dict[str, Any]]) -> float:
    extra = max(0, len(pred_calls) - len(gold_calls))
    if extra == 0:
        return 0.0
    penalty = 0.0
    for call in pred_calls[len(gold_calls):]:
        action = (call.get("arguments") or {}).get("action")
        if action in HIGH_RISK_EXTRA_ACTIONS:
            penalty += 0.18
        elif action in LOW_RISK_EXTRA_ACTIONS:
            penalty += 0.06
        else:
            penalty += 0.10
    return min(0.45, penalty)


def _efficiency_score(response: str, pred_calls: List[Dict[str, Any]], gold_calls: List[Dict[str, Any]]) -> float:
    if not pred_calls or not gold_calls:
        return 0.0

    count_gap = abs(len(pred_calls) - len(gold_calls))
    count_score = _clamp(1.0 - 0.18 * count_gap)

    text_len = _outside_tool_text_len(response)
    if text_len <= 120:
        text_score = 1.0
    elif text_len <= 300:
        text_score = 0.7
    else:
        text_score = 0.4

    penalty = _repeat_penalty(pred_calls) + _extra_action_penalty(pred_calls, gold_calls)
    return _clamp(0.65 * count_score + 0.35 * text_score - penalty)


def score_gui_response(response: str, ground_truth: Any) -> Dict[str, float]:
    gold_calls = _load_ground_truth_sequence(ground_truth)
    pred_calls, malformed = extract_tool_calls(response)

    format_gate, format_score = _format_scores(response, pred_calls, malformed)
    if format_gate == 0.0 or not gold_calls:
        return {
            "score": 0.0,
            "format_gate": float(format_gate),
            "action_sequence": 0.0,
            "format": float(format_score),
            "efficiency": 0.0,
        }

    action_sequence_score = _sequence_action_score(pred_calls, gold_calls)
    efficiency_score = _efficiency_score(response, pred_calls, gold_calls)
    total = format_gate * (
        0.80 * action_sequence_score
        + 0.15 * format_score
        + 0.05 * efficiency_score
    )

    return {
        "score": _clamp(total),
        "format_gate": float(format_gate),
        "action_sequence": float(action_sequence_score),
        "format": float(format_score),
        "efficiency": float(efficiency_score),
    }
