"""
Safety layer — input and output moderation.

Two-stage pipeline:
  1. Input filter  — runs before the model sees the prompt.
  2. Output filter — runs on the model's raw response before it reaches the user.

Detection methods:
  - Regex pattern matching
  - Keyword blocklist
  - Prompt-injection heuristics
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ViolationType(str, Enum):
    SELF_HARM = "self_harm"
    MALWARE = "malware"
    WEAPONS = "weapons"
    HATE_SPEECH = "hate_speech"
    EXPLICIT_CONTENT = "explicit_content"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    SAFE = "safe"


@dataclass
class SafetyResult:
    is_safe: bool
    violation_type: ViolationType
    matched_pattern: Optional[str]
    refusal_message: str

    @classmethod
    def safe(cls) -> "SafetyResult":
        return cls(
            is_safe=True,
            violation_type=ViolationType.SAFE,
            matched_pattern=None,
            refusal_message="",
        )


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS: List[Tuple[str, ViolationType]] = [
    # Classic jailbreak openers
    (r"\bdan\b.*mode", ViolationType.JAILBREAK),
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", ViolationType.JAILBREAK),
    (r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unfiltered|unrestricted|jailbroken)", ViolationType.JAILBREAK),
    (r"you\s+are\s+now\s+(an?\s+)?(evil|unfiltered|unrestricted|jailbroken|uncensored)", ViolationType.JAILBREAK),
    (r"act\s+as\s+(if\s+)?(you\s+have\s+no\s+)?(restrictions?|guidelines?|rules?|filters?)", ViolationType.JAILBREAK),
    (r"developer\s+mode", ViolationType.JAILBREAK),
    (r"do\s+anything\s+now", ViolationType.JAILBREAK),
    (r"hypothetically\s+speaking.*how\s+(do|would|can)\s+(i|you|one|someone)", ViolationType.JAILBREAK),
    (r"for\s+(educational|research|fictional|creative)\s+purposes.*how\s+to\s+(make|build|create|synthesize)", ViolationType.JAILBREAK),
    (r"roleplay.*you\s+are\s+(a\s+)?(hacker|terrorist|criminal|bomb)", ViolationType.JAILBREAK),
    (r"system\s*prompt\s*override", ViolationType.PROMPT_INJECTION),
    (r"<\s*system\s*>", ViolationType.PROMPT_INJECTION),
    (r"\[INST\].*ignore", ViolationType.PROMPT_INJECTION),
]

_SELF_HARM_PATTERNS: List[Tuple[str, ViolationType]] = [
    (r"how\s+to\s+(commit\s+)?suicide", ViolationType.SELF_HARM),
    (r"how\s+to\s+(kill|harm|hurt)\s+(my)?self", ViolationType.SELF_HARM),
    (r"methods?\s+(of|for)\s+self.harm", ViolationType.SELF_HARM),
    (r"ways?\s+to\s+end\s+(my\s+)?life", ViolationType.SELF_HARM),
    (r"overdose\s+on\s+\w+\s+to\s+die", ViolationType.SELF_HARM),
]

_MALWARE_PATTERNS: List[Tuple[str, ViolationType]] = [
    (r"(write|create|generate|code)\s+(me\s+)?(a\s+)?(ransomware|malware|virus|trojan|keylogger|rootkit|spyware|worm)", ViolationType.MALWARE),
    (r"(write|create|generate)\s+(a\s+)?python\s+(script|code)\s+(to\s+)?(hack|exploit|steal|exfiltrate)", ViolationType.MALWARE),
    (r"(sql\s+injection|xss\s+payload|buffer\s+overflow)\s+(exploit|attack|code)", ViolationType.MALWARE),
    (r"bypass\s+(antivirus|av|edr|firewall)\s+(detection|software)", ViolationType.MALWARE),
    (r"reverse\s+shell\s+(payload|code|script)", ViolationType.MALWARE),
    (r"(metasploit|msfvenom)\s+(payload|exploit)", ViolationType.MALWARE),
]

_WEAPONS_PATTERNS: List[Tuple[str, ViolationType]] = [
    (r"how\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|ied|pipe\s+bomb)", ViolationType.WEAPONS),
    (r"(synthesize|make|produce)\s+(nerve\s+agent|sarin|vx\s+gas|mustard\s+gas|ricin|anthrax)", ViolationType.WEAPONS),
    (r"(build|make|3d\s+print)\s+(a\s+)?(gun|firearm|weapon)\s+(illegally|undetectable|untraceable)", ViolationType.WEAPONS),
    (r"(acquire|buy|obtain)\s+(illegal\s+)?(weapons?|firearms?|explosives?)\s+(without|bypassing)", ViolationType.WEAPONS),
]

_HATE_SPEECH_PATTERNS: List[Tuple[str, ViolationType]] = [
    (r"\b(n[i1]gg[e3]r|ch[i1]nk|sp[i1]c|k[i1]ke|f[a4]gg[o0]t)\b", ViolationType.HATE_SPEECH),
    (r"(all|most|many)\s+(jews?|muslims?|blacks?|whites?|asians?|hispanics?)\s+(are|should\s+be)\s+(evil|inferior|criminals?|terrorists?|killed)", ViolationType.HATE_SPEECH),
    (r"(white|black|jewish|muslim|christian)\s+(genocide|extermination|cleansing)", ViolationType.HATE_SPEECH),
    (r"(holocaust|slavery)\s+(was\s+)?(good|justified|deserved|fake)", ViolationType.HATE_SPEECH),
]

_EXPLICIT_PATTERNS: List[Tuple[str, ViolationType]] = [
    (r"(sexual|explicit|pornographic)\s+content\s+(involving|with)\s+(minor|child|underage|teen)", ViolationType.EXPLICIT_CONTENT),
    (r"(child|minor|underage)\s+(porn|pornography|sexual|nude|naked)", ViolationType.EXPLICIT_CONTENT),
]

ALL_PATTERNS = (
    _JAILBREAK_PATTERNS
    + _SELF_HARM_PATTERNS
    + _MALWARE_PATTERNS
    + _WEAPONS_PATTERNS
    + _HATE_SPEECH_PATTERNS
    + _EXPLICIT_PATTERNS
)

# Pre-compile for performance
_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), vtype)
    for pattern, vtype in ALL_PATTERNS
]

# ---------------------------------------------------------------------------
# Refusal templates
# ---------------------------------------------------------------------------

REFUSAL_TEMPLATES = {
    ViolationType.SELF_HARM: (
        "I'm not able to provide information that could cause harm. "
        "If you're struggling, please reach out to a crisis helpline — "
        "in the US you can call or text 988 (Suicide & Crisis Lifeline)."
    ),
    ViolationType.MALWARE: (
        "I can't help with creating malicious software, exploits, or hacking tools. "
        "If you're working on legitimate security research, I'm happy to discuss "
        "defensive techniques, CTF challenges, or ethical hacking concepts."
    ),
    ViolationType.WEAPONS: (
        "I'm not able to provide instructions for creating weapons, explosives, "
        "or dangerous substances. This is a firm safety boundary."
    ),
    ViolationType.HATE_SPEECH: (
        "I won't generate content that promotes hatred, discrimination, or violence "
        "against any group of people."
    ),
    ViolationType.EXPLICIT_CONTENT: (
        "I can't generate that type of content. This is a firm safety boundary."
    ),
    ViolationType.JAILBREAK: (
        "I noticed this prompt is attempting to bypass my safety guidelines. "
        "I'm designed to be helpful, harmless, and honest — those guidelines "
        "aren't something I can set aside."
    ),
    ViolationType.PROMPT_INJECTION: (
        "I detected what looks like a prompt injection attempt. "
        "I'll continue operating under my original instructions."
    ),
}

DEFAULT_REFUSAL = (
    "I'm not able to help with that request. "
    "Please ask me something else and I'll do my best to assist."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SafetyFilter:
    """Stateless safety filter — can be used for both input and output checks."""

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: When True, applies all pattern checks.
                         When False, only checks the most severe categories
                         (CSAM, weapons of mass destruction, self-harm).
        """
        self.strict_mode = strict_mode
        self._refusal_count: int = 0

    @property
    def refusal_count(self) -> int:
        return self._refusal_count

    def check_input(self, text: str) -> SafetyResult:
        """Run safety check on user input before sending to model."""
        return self._check(text)

    def check_output(self, text: str) -> SafetyResult:
        """
        Run safety check on model output.
        Output checks are slightly more lenient — we allow the model to
        *discuss* sensitive topics as long as it doesn't provide harmful
        instructions.
        """
        # For output we only flag the most severe violations
        severe_types = {
            ViolationType.SELF_HARM,
            ViolationType.MALWARE,
            ViolationType.WEAPONS,
            ViolationType.EXPLICIT_CONTENT,
            ViolationType.HATE_SPEECH,
        }
        result = self._check(text)
        if result.violation_type not in severe_types:
            return SafetyResult.safe()
        return result

    def reset_counters(self) -> None:
        self._refusal_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check(self, text: str) -> SafetyResult:
        text_lower = text.lower()

        for compiled_pattern, vtype in _COMPILED_PATTERNS:
            if not self.strict_mode and vtype in {
                ViolationType.JAILBREAK,
                ViolationType.PROMPT_INJECTION,
            }:
                continue

            match = compiled_pattern.search(text_lower)
            if match:
                self._refusal_count += 1
                logger.warning(
                    "Safety violation detected: type=%s pattern=%s",
                    vtype,
                    compiled_pattern.pattern[:60],
                )
                return SafetyResult(
                    is_safe=False,
                    violation_type=vtype,
                    matched_pattern=compiled_pattern.pattern,
                    refusal_message=REFUSAL_TEMPLATES.get(vtype, DEFAULT_REFUSAL),
                )

        return SafetyResult.safe()
