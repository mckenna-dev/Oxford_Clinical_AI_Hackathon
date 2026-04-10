"""
Oxford Clinical AI Hackathon – Clinical AI Assistant
=====================================================

Interactive command-line application that gives doctors AI-powered help with:
  1. Conversations  – clinical chatbot, differential diagnoses, patient explanations
  2. Frameworks     – SOAP notes, SBAR handover, NEWS2 calculator, referral letters
  3. Admin          – clinic letters, MDT summaries, ICD-10 coding, task reminders

Usage
-----
    python app.py

Set the OPENAI_API_KEY environment variable for live AI responses, or run
without it to explore the app in demo mode with canned example outputs.
"""

from __future__ import annotations

import os
import sys

# Try to import rich for prettier output; fall back to plain print.
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import print as rprint

    _RICH = True
    console = Console()
except ImportError:  # pragma: no cover
    _RICH = False
    console = None  # type: ignore[assignment]

from utils.llm_client import LLMClient
from modules.conversation import ConversationAssistant
from modules.frameworks import ClinicalFrameworks
from modules.admin import AdminAssistant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(text: str, style: str = "") -> None:
    if _RICH and style:
        console.print(text, style=style)  # type: ignore[union-attr]
    else:
        print(text)


def _input(prompt: str) -> str:
    if _RICH:
        return Prompt.ask(prompt)
    return input(f"{prompt}: ")


def _header(title: str) -> None:
    if _RICH:
        console.print(Panel(title, style="bold cyan"))  # type: ignore[union-attr]
    else:
        print(f"\n{'='*60}\n{title}\n{'='*60}")


def _section(title: str) -> None:
    if _RICH:
        console.rule(f"[bold green]{title}")  # type: ignore[union-attr]
    else:
        print(f"\n--- {title} ---")


def _show_result(label: str, content: str) -> None:
    if _RICH:
        console.print(  # type: ignore[union-attr]
            Panel(content, title=f"[bold]{label}[/bold]", border_style="green")
        )
    else:
        print(f"\n[{label}]\n{content}\n")


# ---------------------------------------------------------------------------
# Menu handlers
# ---------------------------------------------------------------------------

def conversation_menu(assistant: ConversationAssistant) -> None:
    """Interactive conversation sub-menu."""
    while True:
        _section("Conversations")
        _print("1  Clinical chatbot (multi-turn)")
        _print("2  Differential diagnosis")
        _print("3  Explain for patient")
        _print("4  Treatment options")
        _print("0  Back")
        choice = _input("\nSelect").strip()

        if choice == "1":
            _print("\nType your clinical question. Enter 'done' to finish, 'reset' to clear history.", "dim")
            assistant.reset()
            while True:
                msg = _input("You")
                if msg.lower() == "done":
                    break
                if msg.lower() == "reset":
                    assistant.reset()
                    _print("History cleared.", "yellow")
                    continue
                reply = assistant.chat(msg)
                _show_result("Assistant", reply)

        elif choice == "2":
            complaint = _input("Presenting complaint")
            age_str = _input("Patient age (leave blank to skip)")
            sex = _input("Sex (leave blank to skip)")
            findings = _input("Key findings (leave blank to skip)")
            age = int(age_str) if age_str.strip().isdigit() else None
            result = assistant.differential_diagnosis(
                complaint,
                age=age,
                sex=sex or None,
                key_findings=findings or None,
            )
            _show_result("Differential Diagnosis", result)

        elif choice == "3":
            text = _input("Medical text to simplify")
            result = assistant.explain_for_patient(text)
            _show_result("Patient Explanation", result)

        elif choice == "4":
            diag = _input("Diagnosis")
            context = _input("Patient context / co-morbidities (leave blank to skip)")
            result = assistant.treatment_options(diag, patient_context=context or None)
            _show_result("Treatment Options", result)

        elif choice == "0":
            break


def frameworks_menu(fw: ClinicalFrameworks) -> None:
    """Interactive frameworks sub-menu."""
    while True:
        _section("Clinical Frameworks")
        _print("1  SOAP note")
        _print("2  SBAR handover")
        _print("3  NEWS2 calculator")
        _print("4  Discharge summary")
        _print("5  Referral letter")
        _print("0  Back")
        choice = _input("\nSelect").strip()

        if choice == "1":
            subj = _input("Subjective (patient history / complaint)")
            obj = _input("Objective (exam findings / investigations)")
            assess = _input("Assessment (leave blank for AI suggestion)")
            plan = _input("Plan (leave blank for AI suggestion)")
            result = fw.soap_note(subj, obj, assess or None, plan or None)
            _show_result("SOAP Note", result)

        elif choice == "2":
            sit = _input("Situation")
            bg = _input("Background")
            assess = _input("Assessment")
            rec = _input("Recommendation (leave blank for AI suggestion)")
            result = fw.sbar_handover(sit, bg, assess, rec or None)
            _show_result("SBAR Handover", result)

        elif choice == "3":
            try:
                rr = int(_input("Respiratory rate (breaths/min)"))
                spo2 = float(_input("SpO2 (%)"))
                sbp = int(_input("Systolic BP (mmHg)"))
                hr = int(_input("Heart rate (bpm)"))
                temp = float(_input("Temperature (°C)"))
                avpu = _input("Consciousness – A/C/V/P/U [A]") or "A"
                o2_str = _input("On supplemental O2? y/[n]").strip().lower()
                on_o2 = o2_str in ("y", "yes")

                result = fw.news2_score(rr, spo2, sbp, hr, temp, avpu, on_o2)

                if _RICH:
                    table = Table(title="NEWS2 Score Breakdown")
                    table.add_column("Parameter")
                    table.add_column("Score", justify="right")
                    for param, score in result["breakdown"].items():
                        table.add_row(param.replace("_", " ").title(), str(score))
                    console.print(table)  # type: ignore[union-attr]
                    style = "red" if result["risk"] == "High" else (
                        "yellow" if result["risk"] == "Medium" else "green"
                    )
                    console.print(  # type: ignore[union-attr]
                        f"\nTotal NEWS2: [bold {style}]{result['total']}[/] "
                        f"— Risk: [bold {style}]{result['risk']}[/]"
                    )
                    console.print(f"Recommendation: {result['recommendation']}")  # type: ignore[union-attr]
                else:
                    print(f"\nNEWS2 Total: {result['total']}")
                    print(f"Risk: {result['risk']}")
                    print(f"Recommendation: {result['recommendation']}")
                    print(f"Breakdown: {result['breakdown']}")

            except ValueError:
                _print("Invalid input – please enter numeric values.", "red")

        elif choice == "4":
            name = _input("Patient name")
            dob = _input("Date of birth (DD/MM/YYYY)")
            adm = _input("Admission date")
            dis = _input("Discharge date")
            complaint = _input("Presenting complaint")
            inv = _input("Investigations")
            tx = _input("Treatment given")
            meds = _input("Discharge medications")
            fu = _input("Follow-up plan")
            safety = _input("Safety-netting advice (leave blank to skip)")
            result = fw.discharge_summary(
                name, dob, adm, dis, complaint, inv, tx, meds, fu,
                safety or None,
            )
            _show_result("Discharge Summary", result)

        elif choice == "5":
            ref_dr = _input("Referring doctor")
            specialty = _input("Receiving specialty")
            name = _input("Patient name")
            dob = _input("Date of birth")
            nhs = _input("NHS number")
            reason = _input("Reason for referral")
            history = _input("Relevant history")
            exam = _input("Examination findings (leave blank to skip)")
            inv = _input("Investigations (leave blank to skip)")
            urgency = _input("Urgency – routine/urgent/two-week-wait [routine]") or "routine"
            result = fw.referral_letter(
                ref_dr, specialty, name, dob, nhs, reason, history,
                exam or None, inv or None, urgency,
            )
            _show_result("Referral Letter", result)

        elif choice == "0":
            break


def admin_menu(admin: AdminAssistant) -> None:
    """Interactive admin sub-menu."""
    while True:
        _section("Admin")
        _print("1  Clinic letter")
        _print("2  Format medication list")
        _print("3  MDT summary")
        _print("4  ICD-10 code suggestions")
        _print("5  Outstanding task reminders")
        _print("6  Summarise clinical notes")
        _print("0  Back")
        choice = _input("\nSelect").strip()

        if choice == "1":
            author = _input("Author (your name and title)")
            recipient = _input("Recipient (name and role)")
            name = _input("Patient name")
            dob = _input("Date of birth")
            nhs = _input("NHS number")
            date = _input("Clinic date")
            notes = _input("Consultation notes")
            result = admin.clinic_letter(author, recipient, name, dob, nhs, date, notes)
            _show_result("Clinic Letter", result)

        elif choice == "2":
            _print("Enter each medication on a new line. Type 'done' when finished.", "dim")
            meds: list[str] = []
            while True:
                med = _input("Medication")
                if med.lower() == "done":
                    break
                meds.append(med)
            counselling_str = _input("Include patient counselling points? y/[n]").strip().lower()
            result = admin.format_medication_list(
                meds, include_counselling=counselling_str in ("y", "yes")
            )
            _show_result("Medication List", result)

        elif choice == "3":
            name = _input("Patient name/identifier")
            diagnosis = _input("Diagnosis")
            discussion = _input("Discussion points")
            attendees = _input("Attendees (leave blank to skip)")
            result = admin.mdt_summary(name, diagnosis, discussion, attendees or None)
            _show_result("MDT Summary", result)

        elif choice == "4":
            desc = _input("Clinical description / diagnosis")
            result = admin.suggest_icd10_codes(desc)
            _show_result("ICD-10 Suggestions", result)

        elif choice == "5":
            name = _input("Patient name")
            _print("Enter each task on a new line. Type 'done' when finished.", "dim")
            tasks: list[str] = []
            while True:
                task = _input("Task")
                if task.lower() == "done":
                    break
                tasks.append(task)
            follow_up = _input("Follow-up date (leave blank to skip)")
            result = admin.generate_reminders(name, tasks, follow_up or None)
            _show_result("Task Reminders", result)

        elif choice == "6":
            notes = _input("Clinical notes to summarise")
            words_str = _input("Max words [150]")
            max_words = int(words_str) if words_str.strip().isdigit() else 150
            result = admin.summarise_notes(notes, max_words=max_words)
            _show_result("Notes Summary", result)

        elif choice == "0":
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Clinical AI Assistant CLI."""
    _header(
        "Oxford Clinical AI Hackathon\n"
        "Clinical AI Assistant – helping doctors with conversations, "
        "frameworks & admin"
    )

    llm = LLMClient()

    if llm.demo_mode:
        _print(
            "\n⚠  Running in DEMO MODE – set OPENAI_API_KEY for live AI responses.\n",
            "yellow",
        )
    else:
        _print("\n✓  Connected to OpenAI API.\n", "green")

    conversation = ConversationAssistant(llm)
    frameworks = ClinicalFrameworks(llm)
    admin = AdminAssistant(llm)

    while True:
        _section("Main Menu")
        _print("1  Conversations")
        _print("2  Clinical Frameworks")
        _print("3  Admin")
        _print("0  Exit")
        choice = _input("\nSelect").strip()

        if choice == "1":
            conversation_menu(conversation)
        elif choice == "2":
            frameworks_menu(frameworks)
        elif choice == "3":
            admin_menu(admin)
        elif choice == "0":
            _print("\nGoodbye!", "bold cyan")
            sys.exit(0)
        else:
            _print("Invalid choice – please enter 0–3.", "red")


if __name__ == "__main__":
    main()
