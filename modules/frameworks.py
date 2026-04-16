"""
Frameworks module – clinical documentation and scoring frameworks.

Provides:
* SOAP note generator
* SBAR handover tool
* NEWS2 (National Early Warning Score 2) calculator
* Discharge summary framework
* Referral letter framework
"""

from __future__ import annotations

from typing import Optional

from utils.llm_client import LLMClient

# ---------------------------------------------------------------------------
# NEWS2 parameter tables
# ---------------------------------------------------------------------------

# Each entry: (score_3, score_2, score_1, score_0, score_1b, score_2b, score_3b)
# where 'b' suffix means the higher range.  Ranges stored as (low, high) tuples;
# None means unbounded.

_NEWS2_THRESHOLDS = {
    # (parameter, unit, ranges with their scores ordered low→high)
    "respiratory_rate": [
        (None, 8, 3),
        (9, 11, 1),
        (12, 20, 0),
        (21, 24, 2),
        (25, None, 3),
    ],
    "spo2_scale1": [
        (None, 91, 3),
        (92, 93, 2),
        (94, 95, 1),
        (96, None, 0),
    ],
    "systolic_bp": [
        (None, 90, 3),
        (91, 100, 2),
        (101, 110, 1),
        (111, 219, 0),
        (220, None, 3),
    ],
    "heart_rate": [
        (None, 40, 3),
        (41, 50, 1),
        (51, 90, 0),
        (91, 110, 1),
        (111, 130, 2),
        (131, None, 3),
    ],
    "temperature": [
        (None, 35.0, 3),
        (35.1, 36.0, 1),
        (36.1, 38.0, 0),
        (38.1, 39.0, 1),
        (39.1, None, 2),
    ],
}


def _score_param(value: float, thresholds: list) -> int:
    for low, high, score in thresholds:
        below = low is None or value >= low
        above = high is None or value <= high
        if below and above:
            return score
    return 0


class ClinicalFrameworks:
    """
    Structured clinical frameworks and scoring tools.

    Parameters
    ----------
    llm : LLMClient
        Shared LLM client instance.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # SOAP note
    # ------------------------------------------------------------------

    def soap_note(
        self,
        subjective: str,
        objective: str,
        assessment: str | None = None,
        plan: str | None = None,
    ) -> str:
        """
        Generate or complete a SOAP note using AI.

        If *assessment* and *plan* are omitted the AI will suggest them
        based on the subjective and objective information provided.

        Parameters
        ----------
        subjective :
            Patient's reported symptoms and history.
        objective :
            Examination findings, observations, investigation results.
        assessment :
            Clinician's working diagnosis / assessment (optional).
        plan :
            Management plan (optional).

        Returns
        -------
        str
            Formatted SOAP note.
        """
        provided = (
            f"Subjective:\n{subjective}\n\n"
            f"Objective:\n{objective}\n"
        )
        if assessment:
            provided += f"\nAssessment:\n{assessment}\n"
        if plan:
            provided += f"\nPlan:\n{plan}\n"

        prompt = (
            "Complete and format the following SOAP note. If the Assessment "
            "or Plan are missing, suggest appropriate content based on the "
            "clinical information provided. Follow UK NHS documentation "
            "standards.\n\n"
            f"{provided}"
        )
        return self._llm.simple_query(prompt, demo_key="soap")

    # ------------------------------------------------------------------
    # SBAR handover
    # ------------------------------------------------------------------

    def sbar_handover(
        self,
        situation: str,
        background: str,
        assessment: str,
        recommendation: str | None = None,
    ) -> str:
        """
        Generate a structured SBAR handover communication.

        Parameters
        ----------
        situation :
            What is happening right now.
        background :
            Relevant background information.
        assessment :
            Your clinical assessment of the problem.
        recommendation :
            What you need / recommend (optional – AI will suggest if absent).

        Returns
        -------
        str
            Formatted SBAR handover.
        """
        content = (
            f"Situation: {situation}\n"
            f"Background: {background}\n"
            f"Assessment: {assessment}\n"
        )
        if recommendation:
            content += f"Recommendation: {recommendation}\n"

        prompt = (
            "Format the following information as a concise, professional SBAR "
            "handover suitable for verbal or written communication between "
            "clinical staff. If the Recommendation is missing, suggest an "
            "appropriate one.\n\n"
            f"{content}"
        )
        return self._llm.simple_query(prompt, demo_key="sbar")

    # ------------------------------------------------------------------
    # NEWS2 calculator
    # ------------------------------------------------------------------

    def news2_score(
        self,
        respiratory_rate: int,
        spo2: float,
        systolic_bp: int,
        heart_rate: int,
        temperature: float,
        consciousness: str = "A",
        on_supplemental_o2: bool = False,
    ) -> dict:
        """
        Calculate a NEWS2 (National Early Warning Score 2) score.

        Parameters
        ----------
        respiratory_rate : int
            Breaths per minute.
        spo2 : float
            Oxygen saturation (%) – use Scale 1 (no known hypercapnic drive).
        systolic_bp : int
            Systolic blood pressure (mmHg).
        heart_rate : int
            Pulse rate (beats per minute).
        temperature : float
            Temperature (°C).
        consciousness : str
            ACVPU scale: ``"A"`` (Alert), ``"C"`` (Confused), ``"V"`` (Voice),
            ``"P"`` (Pain), ``"U"`` (Unresponsive).  Defaults to ``"A"``.
        on_supplemental_o2 : bool
            Whether the patient is receiving supplemental oxygen.

        Returns
        -------
        dict
            ``{"total": int, "breakdown": dict, "risk": str,
               "recommendation": str}``
        """
        breakdown: dict[str, int] = {}

        breakdown["respiratory_rate"] = _score_param(
            respiratory_rate, _NEWS2_THRESHOLDS["respiratory_rate"]
        )
        breakdown["spo2"] = _score_param(spo2, _NEWS2_THRESHOLDS["spo2_scale1"])
        breakdown["systolic_bp"] = _score_param(
            systolic_bp, _NEWS2_THRESHOLDS["systolic_bp"]
        )
        breakdown["heart_rate"] = _score_param(
            heart_rate, _NEWS2_THRESHOLDS["heart_rate"]
        )
        breakdown["temperature"] = _score_param(
            temperature, _NEWS2_THRESHOLDS["temperature"]
        )

        # Supplemental O2
        breakdown["supplemental_o2"] = 2 if on_supplemental_o2 else 0

        # Consciousness (ACVPU): Alert = 0 points; any other state
        # (Confused / Voice / Pain / Unresponsive) = 3 points each.
        acvpu_score = {"A": 0, "C": 3, "V": 3, "P": 3, "U": 3}
        breakdown["consciousness"] = acvpu_score.get(consciousness.upper(), 0)

        total = sum(breakdown.values())

        if total <= 4:
            if any(v >= 3 for v in breakdown.values()):
                risk = "Medium"
                recommendation = (
                    "Urgent review by ward-based doctor. "
                    "Minimum 1-hourly monitoring."
                )
            else:
                risk = "Low"
                recommendation = (
                    "Continue routine monitoring (minimum 12-hourly)."
                )
        elif total <= 6:
            risk = "Medium"
            recommendation = (
                "Urgent review by ward-based doctor. "
                "Minimum 1-hourly monitoring. "
                "Consider escalation to SpR/Registrar."
            )
        else:
            risk = "High"
            recommendation = (
                "Emergency assessment by a clinical team with critical-care "
                "competencies, including the outreach team or critical-care "
                "team. Transfer to ICU/HDU should be considered."
            )

        return {
            "total": total,
            "breakdown": breakdown,
            "risk": risk,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Discharge summary framework
    # ------------------------------------------------------------------

    def discharge_summary(
        self,
        patient_name: str,
        dob: str,
        admission_date: str,
        discharge_date: str,
        presenting_complaint: str,
        investigations: str,
        treatment: str,
        discharge_medications: str,
        follow_up: str,
        safety_netting: str | None = None,
    ) -> str:
        """
        Generate a structured discharge summary.

        Returns
        -------
        str
            Formatted discharge summary.
        """
        details = (
            f"Patient: {patient_name} | DOB: {dob}\n"
            f"Admission: {admission_date} | Discharge: {discharge_date}\n"
            f"Presenting complaint: {presenting_complaint}\n"
            f"Investigations: {investigations}\n"
            f"Treatment: {treatment}\n"
            f"Discharge medications: {discharge_medications}\n"
            f"Follow-up plan: {follow_up}\n"
        )
        if safety_netting:
            details += f"Safety-netting advice: {safety_netting}\n"

        prompt = (
            "Format the following information as a professional NHS discharge "
            "summary suitable for sending to the patient's GP. Use clear "
            "headings and UK medical terminology.\n\n"
            f"{details}"
        )
        return self._llm.simple_query(prompt, demo_key="discharge")

    # ------------------------------------------------------------------
    # Referral letter
    # ------------------------------------------------------------------

    def referral_letter(
        self,
        referring_doctor: str,
        receiving_specialty: str,
        patient_name: str,
        dob: str,
        nhs_number: str,
        reason: str,
        history: str,
        examination: str | None = None,
        investigations: str | None = None,
        urgency: str = "routine",
    ) -> str:
        """
        Generate a structured referral letter.

        Parameters
        ----------
        urgency :
            One of ``"routine"``, ``"urgent"``, or ``"two-week-wait"``.

        Returns
        -------
        str
            Formatted referral letter.
        """
        details = (
            f"From: {referring_doctor}\n"
            f"To: {receiving_specialty}\n"
            f"Patient: {patient_name} | DOB: {dob} | NHS No: {nhs_number}\n"
            f"Urgency: {urgency}\n"
            f"Reason for referral: {reason}\n"
            f"History: {history}\n"
        )
        if examination:
            details += f"Examination: {examination}\n"
        if investigations:
            details += f"Investigations: {investigations}\n"

        prompt = (
            "Write a professional NHS referral letter using the information "
            "below. Use a formal letter format with appropriate clinical detail. "
            "Ensure the urgency is clearly communicated.\n\n"
            f"{details}"
        )
        return self._llm.simple_query(prompt, demo_key="letter")
