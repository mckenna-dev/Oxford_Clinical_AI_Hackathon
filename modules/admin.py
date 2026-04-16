"""
Admin module – clinical administrative tasks.

Provides:
* Medical letter composer (clinic letters, GP letters)
* Prescription / medication list formatter
* Meeting / MDT summary generator
* Clinical coding suggestion (ICD-10)
* Appointment / task reminder generator
"""

from __future__ import annotations

from typing import List, Optional

from utils.llm_client import LLMClient


class AdminAssistant:
    """
    Clinical administrative AI assistant.

    Parameters
    ----------
    llm : LLMClient
        Shared LLM client instance.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ------------------------------------------------------------------
    # Clinic / GP letter
    # ------------------------------------------------------------------

    def clinic_letter(
        self,
        author: str,
        recipient: str,
        patient_name: str,
        dob: str,
        nhs_number: str,
        clinic_date: str,
        content_notes: str,
    ) -> str:
        """
        Compose a clinic outpatient letter.

        Parameters
        ----------
        author :
            Dictating clinician name and title.
        recipient :
            Recipient name and role (e.g. GP).
        patient_name :
            Full patient name.
        dob :
            Date of birth (DD/MM/YYYY).
        nhs_number :
            NHS number.
        clinic_date :
            Date of clinic attendance.
        content_notes :
            Bullet points or free text describing the consultation.

        Returns
        -------
        str
            Formatted clinic letter.
        """
        prompt = (
            "Write a professional outpatient clinic letter in UK NHS style "
            "from the following notes. Use a formal letter format with correct "
            "medical terminology and clear headings.\n\n"
            f"From: {author}\n"
            f"To: {recipient}\n"
            f"Patient: {patient_name} | DOB: {dob} | NHS No: {nhs_number}\n"
            f"Clinic date: {clinic_date}\n\n"
            f"Consultation notes:\n{content_notes}"
        )
        return self._llm.simple_query(prompt, demo_key="letter")

    # ------------------------------------------------------------------
    # Medication list formatter
    # ------------------------------------------------------------------

    def format_medication_list(
        self,
        medications: List[str],
        include_counselling: bool = True,
    ) -> str:
        """
        Format a medication list with optional patient counselling points.

        Parameters
        ----------
        medications :
            List of medication strings, e.g.
            ``["Metformin 500 mg BD with food", "Lisinopril 5 mg OD"]``.
        include_counselling :
            If True, add key counselling points for each medication.

        Returns
        -------
        str
            Formatted medication list.
        """
        med_list = "\n".join(f"- {m}" for m in medications)
        counselling_instruction = (
            "For each medication, include one or two key patient counselling "
            "points (what it is for, important side effects to watch for, "
            "and any critical interactions)."
            if include_counselling
            else "Present as a clean, numbered medication list."
        )
        prompt = (
            f"Format the following medication list for a UK NHS patient. "
            f"{counselling_instruction}\n\n"
            f"Medications:\n{med_list}"
        )
        return self._llm.simple_query(prompt, demo_key="default")

    # ------------------------------------------------------------------
    # MDT summary
    # ------------------------------------------------------------------

    def mdt_summary(
        self,
        patient_name: str,
        diagnosis: str,
        discussion_points: str,
        attendees: str | None = None,
    ) -> str:
        """
        Generate a structured MDT (multidisciplinary team) meeting summary.

        Parameters
        ----------
        patient_name :
            Patient identifier.
        diagnosis :
            Working or confirmed diagnosis discussed.
        discussion_points :
            Free-text or bullet-point notes from the MDT discussion.
        attendees :
            Comma-separated list of attendees / specialties.

        Returns
        -------
        str
            Formatted MDT summary suitable for clinical records.
        """
        details = (
            f"Patient: {patient_name}\n"
            f"Diagnosis: {diagnosis}\n"
            f"Discussion: {discussion_points}\n"
        )
        if attendees:
            details += f"Attendees: {attendees}\n"

        prompt = (
            "Write a concise MDT meeting summary for the clinical notes "
            "based on the following information. Include the agreed plan and "
            "any outstanding actions with responsible individuals.\n\n"
            f"{details}"
        )
        return self._llm.simple_query(prompt, demo_key="default")

    # ------------------------------------------------------------------
    # ICD-10 coding suggestions
    # ------------------------------------------------------------------

    def suggest_icd10_codes(self, clinical_description: str) -> str:
        """
        Suggest relevant ICD-10 codes for a clinical scenario.

        Parameters
        ----------
        clinical_description :
            Free-text clinical description or diagnosis.

        Returns
        -------
        str
            List of suggested ICD-10 codes with descriptions.
        """
        prompt = (
            "Based on the following clinical description, suggest the most "
            "appropriate ICD-10 codes (UK edition). List each code, its "
            "description, and a brief rationale.\n\n"
            f"{clinical_description}"
        )
        return self._llm.simple_query(prompt, demo_key="default")

    # ------------------------------------------------------------------
    # Appointment / task reminder
    # ------------------------------------------------------------------

    def generate_reminders(
        self,
        patient_name: str,
        outstanding_tasks: List[str],
        follow_up_date: str | None = None,
    ) -> str:
        """
        Generate a formatted list of outstanding tasks and reminders.

        Parameters
        ----------
        patient_name :
            Patient name or identifier.
        outstanding_tasks :
            List of tasks / follow-up items.
        follow_up_date :
            Intended follow-up date (optional).

        Returns
        -------
        str
            Formatted reminder list.
        """
        tasks = "\n".join(f"- {t}" for t in outstanding_tasks)
        date_str = f" (Target date: {follow_up_date})" if follow_up_date else ""
        prompt = (
            f"Format the following outstanding clinical tasks for {patient_name}"
            f"{date_str} as a clear, prioritised action list suitable for "
            "handover or a clinical to-do list.\n\n"
            f"Tasks:\n{tasks}"
        )
        return self._llm.simple_query(prompt, demo_key="default")

    # ------------------------------------------------------------------
    # Clinical notes summariser
    # ------------------------------------------------------------------

    def summarise_notes(
        self,
        notes: str,
        max_words: int = 150,
    ) -> str:
        """
        Summarise lengthy clinical notes into a concise paragraph.

        Parameters
        ----------
        notes :
            Full clinical notes text.
        max_words :
            Approximate maximum word count for the summary.

        Returns
        -------
        str
            Concise clinical summary.
        """
        prompt = (
            f"Summarise the following clinical notes in approximately "
            f"{max_words} words. Retain all clinically important information "
            "and use UK medical terminology.\n\n"
            f"{notes}"
        )
        return self._llm.simple_query(prompt, demo_key="default")
