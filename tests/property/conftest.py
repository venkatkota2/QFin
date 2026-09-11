import os

from hypothesis import settings

settings.register_profile("ci", max_examples=50, deadline=None, derandomize=True)
settings.register_profile("stress", max_examples=500, deadline=None, derandomize=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
