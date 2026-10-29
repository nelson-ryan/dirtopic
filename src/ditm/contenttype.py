from enum import StrEnum

""" Each Loader and Preprocessor must have .content_type;
`Corpus` checks for any mismatch upon initialization.
"""

class ContentType(StrEnum):
    email = "email"
    transcript = "transcript"
