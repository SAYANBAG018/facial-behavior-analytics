
class EngagementEstimator:
    """
    Experimental engagement heuristic.

    This score is a proxy based on facial emotion scores and
    face visibility. It is not a validated measure of engagement.
    """

    def estimate(self, emotion_result, facial_features):
        if emotion_result is None or facial_features is None:
            return {
                "score": None,
                "label": "insufficient_data",
                "reason": "Face or emotion data unavailable",
            }

        scores = emotion_result.get("scores", {})

        required = {
            "angry", "disgust", "fear", "happy",
            "neutral", "sad", "surprise"
        }

        if not required.issubset(scores):
            raise ValueError("Expected all seven emotion scores.")

        # Simple transparent heuristic; weights are illustrative,
        # not learned or validated engagement parameters.
        positive = (
            scores["happy"] + 0.5 * scores["surprise"]
        )

        negative = (
            scores["angry"]
            + scores["disgust"]
            + scores["fear"]
            + scores["sad"]
        )

        raw_score = 0.5 + 0.5 * (positive - negative)
        score = max(0.0, min(1.0, raw_score))

        # This only indicates whether the heuristic score is high
        # or low; it does not establish a person's true engagement.
        if score >= 0.65:
            label = "higher_proxy_score"
        elif score >= 0.40:
            label = "moderate_proxy_score"
        else:
            label = "lower_proxy_score"

        return {
            "score": round(score * 100, 2),
            "label": label,
            "reason": (
                "Experimental heuristic; interpret cautiously."
            ),
        }
