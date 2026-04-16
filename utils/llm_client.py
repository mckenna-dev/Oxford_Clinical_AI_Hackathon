"""
LLM Client wrapper for the Clinical AI Assistant.

Supports the OpenAI API (including Azure OpenAI) and a built-in demo mode
that works without any API key, making it easy to explore the app offline.
"""

from __future__ import annotations

import os
from typing import List, Dict

# Attempt to import the openai package; fall back gracefully if absent.
try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Demo-mode responses (no API key required)
# ---------------------------------------------------------------------------

_DEMO_RESPONSES: Dict[str, str] = {
    "default": (
        "[DEMO MODE – no API key set]\n"
        "This is a placeholder response. Set the OPENAI_API_KEY environment "
        "variable (or pass api_key to LLMClient) to receive real AI responses."
    ),
    "soap": (
        "[DEMO MODE]\n"
        "**S – Subjective:** Patient reports 3-day history of productive cough, "
        "fever (38.4 °C), and pleuritic chest pain.\n"
        "**O – Objective:** RR 22, O2 sat 94% on air, dullness to percussion "
        "right base, reduced air entry right lower zone.\n"
        "**A – Assessment:** Community-acquired pneumonia (right lower lobe), "
        "CURB-65 score 2.\n"
        "**P – Plan:** Amoxicillin 500 mg TDS × 5 days, chest X-ray, "
        "repeat bloods in 48 h, safety-net for worsening."
    ),
    "sbar": (
        "[DEMO MODE]\n"
        "**S – Situation:** I'm calling about Mr Smith in bay 4 – he has become "
        "acutely short of breath in the last 30 minutes.\n"
        "**B – Background:** 68 y/o male admitted 2 days ago with COPD exacerbation, "
        "background of IHD and T2DM.\n"
        "**A – Assessment:** RR 28, SpO2 88% on 2 L O2, using accessory muscles, "
        "trachea central, bilateral wheeze.\n"
        "**R – Recommendation:** Please review urgently; consider nebulised "
        "salbutamol, controlled O2, and ABG."
    ),
    "differential": (
        "[DEMO MODE]\n"
        "Top differentials for acute chest pain:\n"
        "1. Acute coronary syndrome (STEMI / NSTEMI / unstable angina)\n"
        "2. Pulmonary embolism\n"
        "3. Aortic dissection\n"
        "4. Tension pneumothorax\n"
        "5. Pericarditis / myocarditis\n"
        "6. Oesophageal rupture (Boerhaave syndrome)\n"
        "7. Musculoskeletal / costochondritis\n"
        "8. GORD / oesophageal spasm\n"
        "Red flags: tearing pain → dissection; haemodynamic instability → "
        "tension pneumothorax or massive PE."
    ),
    "letter": (
        "[DEMO MODE]\n"
        "Dear Dr Jones,\n\n"
        "Re: Mr John Smith, DOB 12/05/1958, NHS No. 123 456 7890\n\n"
        "Thank you for referring this gentleman whom I reviewed in my clinic "
        "today. He presented with a 6-month history of progressive exertional "
        "dyspnoea. Examination revealed bibasal fine crackles. Spirometry "
        "confirms an FVC of 68% predicted with an FEV1/FVC of 0.82.\n\n"
        "I have arranged a high-resolution CT thorax and will review him in "
        "4 weeks' time. I will keep you informed of the results.\n\n"
        "Yours sincerely,\nDr A Consultant, Respiratory Physician"
    ),
    "discharge": (
        "[DEMO MODE]\n"
        "**Discharge Summary**\n"
        "Patient: Jane Doe | DOB: 04/09/1972 | Admission: 08/04/2026 | "
        "Discharge: 10/04/2026\n\n"
        "**Presenting complaint:** Acute exacerbation of asthma.\n"
        "**Investigations:** Peak flow 45% predicted on admission; "
        "CXR no consolidation; ABG pH 7.41, pO2 11.2.\n"
        "**Treatment:** Prednisolone 40 mg OD × 5 days, salbutamol nebs, "
        "ipratropium nebs, IV hydrocortisone first 24 h.\n"
        "**Discharge medications:** Continued inhalers, prednisolone course.\n"
        "**Follow-up:** GP in 48 h; respiratory clinic in 6 weeks.\n"
        "**Safety netting:** Return immediately if SOB worsens or peak flow "
        "falls below 33% predicted."
    ),
}


class LLMClient:
    """
    Thin wrapper around the OpenAI chat-completions endpoint.

    Parameters
    ----------
    api_key : str | None
        OpenAI API key. If *None* the value of the ``OPENAI_API_KEY``
        environment variable is used.  When neither is set the client
        runs in *demo mode* and returns canned responses.
    model : str
        Model identifier, e.g. ``"gpt-4o"`` or ``"gpt-3.5-turbo"``.
    system_prompt : str
        System-level instructions sent with every request.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a highly knowledgeable clinical AI assistant designed to "
        "support UK NHS doctors. You follow NICE guidelines and GMC good "
        "medical practice. You are helpful, concise, evidence-based, and "
        "always remind users that your output does not replace clinical "
        "judgement. You use UK English and UK medical terminology."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")

        if resolved_key and _OPENAI_AVAILABLE:
            self._client: openai.OpenAI | None = openai.OpenAI(api_key=resolved_key)
            self._demo_mode = False
        else:
            self._client = None
            self._demo_mode = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def demo_mode(self) -> bool:
        """True when no API key is configured."""
        return self._demo_mode

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        demo_key: str = "default",
    ) -> str:
        """
        Send *messages* to the model and return the reply text.

        Parameters
        ----------
        messages :
            List of ``{"role": ..., "content": ...}`` dicts.  The system
            prompt is prepended automatically.
        temperature :
            Sampling temperature (lower = more deterministic).
        demo_key :
            Key used to look up a canned response when in demo mode.
        """
        if self._demo_mode:
            return _DEMO_RESPONSES.get(demo_key, _DEMO_RESPONSES["default"])

        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=full_messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def simple_query(self, prompt: str, demo_key: str = "default") -> str:
        """Convenience wrapper for single-turn queries."""
        return self.chat([{"role": "user", "content": prompt}], demo_key=demo_key)
