"""
Tests for the Clinical AI Assistant.

Run with:  python -m pytest tests/ -v
"""

import sys
import os

# Ensure repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from utils.llm_client import LLMClient
from modules.conversation import ConversationAssistant
from modules.frameworks import ClinicalFrameworks
from modules.admin import AdminAssistant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def llm():
    """LLM client in demo mode (no API key required)."""
    return LLMClient(api_key=None)


@pytest.fixture
def conv(llm):
    return ConversationAssistant(llm)


@pytest.fixture
def fw(llm):
    return ClinicalFrameworks(llm)


@pytest.fixture
def adm(llm):
    return AdminAssistant(llm)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class TestLLMClient:
    def test_demo_mode_without_key(self):
        client = LLMClient(api_key=None)
        assert client.demo_mode is True

    def test_simple_query_returns_string(self, llm):
        result = llm.simple_query("What is diabetes?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_demo_mode_returns_canned_response(self, llm):
        result = llm.simple_query("test", demo_key="soap")
        assert "DEMO" in result
        assert "Subjective" in result or "S \u2013" in result

    def test_chat_returns_string(self, llm):
        result = llm.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_demo_key_falls_back_to_default(self, llm):
        result = llm.simple_query("test", demo_key="nonexistent_key_xyz")
        assert "DEMO" in result


# ---------------------------------------------------------------------------
# ConversationAssistant
# ---------------------------------------------------------------------------

class TestConversationAssistant:
    def test_chat_returns_string(self, conv):
        reply = conv.chat("What is a STEMI?")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_history_is_maintained(self, conv):
        conv.reset()
        conv.chat("Tell me about sepsis.")
        history = conv.get_history()
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_reset_clears_history(self, conv):
        conv.chat("Hello")
        conv.reset()
        assert conv.get_history() == []

    def test_differential_diagnosis_returns_string(self, conv):
        result = conv.differential_diagnosis(
            "acute chest pain", age=50, sex="male"
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_differential_diagnosis_demo_content(self, conv):
        result = conv.differential_diagnosis("chest pain")
        assert "DEMO" in result or len(result) > 10

    def test_explain_for_patient_returns_string(self, conv):
        result = conv.explain_for_patient("You have hypertension.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_treatment_options_returns_string(self, conv):
        result = conv.treatment_options("type 2 diabetes")
        assert isinstance(result, str)

    def test_treatment_options_with_context(self, conv):
        result = conv.treatment_options(
            "type 2 diabetes",
            patient_context="CKD stage 3, allergic to sulphonylureas",
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ClinicalFrameworks
# ---------------------------------------------------------------------------

class TestClinicalFrameworks:
    def test_soap_note_returns_string(self, fw):
        result = fw.soap_note(
            subjective="Cough and fever for 3 days",
            objective="Temp 38.5, RR 22, reduced breath sounds right base",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sbar_handover_returns_string(self, fw):
        result = fw.sbar_handover(
            situation="Patient acutely short of breath",
            background="68M, admitted with COPD exacerbation",
            assessment="NEWS2 7",
        )
        assert isinstance(result, str)

    def test_news2_normal_vitals(self, fw):
        result = fw.news2_score(
            respiratory_rate=16,
            spo2=98,
            systolic_bp=125,
            heart_rate=78,
            temperature=37.0,
            consciousness="A",
            on_supplemental_o2=False,
        )
        assert result["total"] == 0
        assert result["risk"] == "Low"
        assert "breakdown" in result
        assert "recommendation" in result

    def test_news2_high_score(self, fw):
        result = fw.news2_score(
            respiratory_rate=28,
            spo2=90,
            systolic_bp=85,
            heart_rate=135,
            temperature=39.5,
            consciousness="V",
            on_supplemental_o2=True,
        )
        assert result["total"] >= 7
        assert result["risk"] == "High"

    def test_news2_supplemental_o2_adds_2(self, fw):
        base = fw.news2_score(
            respiratory_rate=16, spo2=98, systolic_bp=125,
            heart_rate=78, temperature=37.0, consciousness="A",
            on_supplemental_o2=False,
        )
        with_o2 = fw.news2_score(
            respiratory_rate=16, spo2=98, systolic_bp=125,
            heart_rate=78, temperature=37.0, consciousness="A",
            on_supplemental_o2=True,
        )
        assert with_o2["total"] == base["total"] + 2

    def test_news2_consciousness_new_confusion_scores_3(self, fw):
        result = fw.news2_score(
            respiratory_rate=16, spo2=98, systolic_bp=125,
            heart_rate=78, temperature=37.0, consciousness="C",
        )
        assert result["breakdown"]["consciousness"] == 3

    def test_news2_alert_consciousness_scores_0(self, fw):
        result = fw.news2_score(
            respiratory_rate=16, spo2=98, systolic_bp=125,
            heart_rate=78, temperature=37.0, consciousness="A",
        )
        assert result["breakdown"]["consciousness"] == 0

    def test_news2_case_insensitive_consciousness(self, fw):
        lower = fw.news2_score(
            respiratory_rate=16, spo2=98, systolic_bp=125,
            heart_rate=78, temperature=37.0, consciousness="a",
        )
        assert lower["breakdown"]["consciousness"] == 0

    def test_discharge_summary_returns_string(self, fw):
        result = fw.discharge_summary(
            patient_name="Test Patient",
            dob="01/01/1970",
            admission_date="08/04/2026",
            discharge_date="10/04/2026",
            presenting_complaint="Chest pain",
            investigations="ECG: normal; troponin negative",
            treatment="Aspirin, nitrates",
            discharge_medications="Aspirin 75 mg OD",
            follow_up="Cardiology in 2 weeks",
        )
        assert isinstance(result, str)

    def test_referral_letter_returns_string(self, fw):
        result = fw.referral_letter(
            referring_doctor="Dr Test",
            receiving_specialty="Cardiology",
            patient_name="Test Patient",
            dob="01/01/1970",
            nhs_number="123 456 7890",
            reason="Chest pain",
            history="3-month history of exertional chest pain",
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# AdminAssistant
# ---------------------------------------------------------------------------

class TestAdminAssistant:
    def test_clinic_letter_returns_string(self, adm):
        result = adm.clinic_letter(
            author="Dr Test",
            recipient="Dr GP",
            patient_name="Test Patient",
            dob="01/01/1970",
            nhs_number="123 456 7890",
            clinic_date="09/04/2026",
            content_notes="Review of hypertension – well controlled on amlodipine.",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_medication_list_returns_string(self, adm):
        result = adm.format_medication_list(
            ["Metformin 500 mg BD", "Lisinopril 5 mg OD"]
        )
        assert isinstance(result, str)

    def test_format_medication_list_no_counselling(self, adm):
        result = adm.format_medication_list(
            ["Aspirin 75 mg OD"], include_counselling=False
        )
        assert isinstance(result, str)

    def test_mdt_summary_returns_string(self, adm):
        result = adm.mdt_summary(
            patient_name="Test Patient",
            diagnosis="Lung cancer",
            discussion_points="Refer for surgery",
        )
        assert isinstance(result, str)

    def test_suggest_icd10_codes_returns_string(self, adm):
        result = adm.suggest_icd10_codes("Type 2 diabetes mellitus")
        assert isinstance(result, str)

    def test_generate_reminders_returns_string(self, adm):
        result = adm.generate_reminders(
            patient_name="Test Patient",
            outstanding_tasks=["Repeat HbA1c in 3 months", "Refer to dietitian"],
            follow_up_date="10/07/2026",
        )
        assert isinstance(result, str)

    def test_summarise_notes_returns_string(self, adm):
        result = adm.summarise_notes(
            "Patient reviewed. Stable. Plan to discharge tomorrow.",
            max_words=50,
        )
        assert isinstance(result, str)
