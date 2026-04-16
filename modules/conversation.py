"""
Conversation module – AI-powered clinical conversation support.

Provides:
* Multi-turn clinical chatbot (history-aware)
* Differential diagnosis generator
* Patient explanation simplifier (plain-English summaries for patients)
"""

from __future__ import annotations

from typing import List, Dict

from utils.llm_client import LLMClient


class ConversationAssistant:
    """
    AI-powered clinical conversation assistant.

    Parameters
    ----------
    llm : LLMClient
        Shared LLM client instance.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a message and receive a clinically-aware reply.

        Conversation history is maintained across calls so that follow-up
        questions are understood in context.

        Parameters
        ----------
        user_message :
            Free-text question or comment from the clinician.

        Returns
        -------
        str
            AI reply.
        """
        self._history.append({"role": "user", "content": user_message})
        reply = self._llm.chat(self._history, demo_key="default")
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history to start a new session."""
        self._history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the conversation history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Differential diagnosis
    # ------------------------------------------------------------------

    def differential_diagnosis(
        self,
        presenting_complaint: str,
        age: int | None = None,
        sex: str | None = None,
        key_findings: str | None = None,
    ) -> str:
        """
        Generate a structured differential diagnosis list.

        Parameters
        ----------
        presenting_complaint :
            e.g. ``"acute onset chest pain"``
        age :
            Patient age in years (optional).
        sex :
            Patient sex / gender (optional).
        key_findings :
            Relevant history, examination or investigation findings.

        Returns
        -------
        str
            Formatted differential diagnosis with red flags noted.
        """
        context_parts = [f"Presenting complaint: {presenting_complaint}"]
        if age is not None:
            context_parts.append(f"Age: {age} years")
        if sex:
            context_parts.append(f"Sex: {sex}")
        if key_findings:
            context_parts.append(f"Key findings: {key_findings}")

        context = "\n".join(context_parts)

        prompt = (
            "Generate a structured differential diagnosis for the following "
            "clinical scenario. Rank by likelihood, briefly explain each, "
            "and highlight any red-flag diagnoses that must not be missed. "
            "Use UK clinical guidelines where relevant.\n\n"
            f"{context}"
        )
        return self._llm.simple_query(prompt, demo_key="differential")

    # ------------------------------------------------------------------
    # Patient explanation
    # ------------------------------------------------------------------

    def explain_for_patient(
        self,
        medical_text: str,
        reading_age: int = 12,
    ) -> str:
        """
        Rewrite a medical passage in plain English suitable for patients.

        Parameters
        ----------
        medical_text :
            Clinical text to simplify.
        reading_age :
            Target reading age in years (default 12).

        Returns
        -------
        str
            Plain-English version.
        """
        prompt = (
            f"Rewrite the following medical text in clear, plain English suitable "
            f"for a patient with a reading age of approximately {reading_age} years. "
            "Avoid jargon, use short sentences, and be reassuring without being "
            "inaccurate.\n\n"
            f"Medical text:\n{medical_text}"
        )
        return self._llm.simple_query(prompt, demo_key="default")

    # ------------------------------------------------------------------
    # Treatment discussion
    # ------------------------------------------------------------------

    def treatment_options(
        self,
        diagnosis: str,
        patient_context: str | None = None,
    ) -> str:
        """
        Summarise evidence-based treatment options for a given diagnosis.

        Parameters
        ----------
        diagnosis :
            Confirmed or working diagnosis.
        patient_context :
            Relevant co-morbidities, allergies, or patient preferences.

        Returns
        -------
        str
            Structured treatment summary.
        """
        context = f"Diagnosis: {diagnosis}"
        if patient_context:
            context += f"\nPatient context: {patient_context}"

        prompt = (
            "Provide a structured overview of evidence-based treatment options "
            "for the following, following current NICE guidelines. Include "
            "first-line, second-line, and non-pharmacological options. Note any "
            "important contraindications or monitoring requirements.\n\n"
            f"{context}"
        )
        return self._llm.simple_query(prompt, demo_key="default")
