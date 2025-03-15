# Calculate runtime duration with its tag
def duration_cal(duration: float):
    if duration > 60:
        if duration > 3600:
            duration = duration / 3600
            tag = "hr"
        else:
            duration = duration / 60
            tag = "min"
    else:
        tag = "sec"

    return duration, tag
