def pick_token(logits, allowed_ids) -> int:
    """Return the allowed token id with the
    highest logit"""

    best_id = None
    best_score = None

    for i in allowed_ids:
        if best_score is None or logits[i] > best_score:
            best_id = i
            best_score = logits[i]
    return best_id


fake_logits = [0.5, 3.0, 1.2, 9.9, 2.0]

print(pick_token(fake_logits, [0, 1, 2, 3, 4]))
print(pick_token(fake_logits, [0, 1, 2, 4]))
