from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain_openai import OpenAI, ChatOpenAI
#from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import json
from typing import List, Dict, Any
import re
#from langchain.chat_models import ChatOpenAI
import requests
from functools import lru_cache
from langchain.prompts import ChatPromptTemplate
import pdfplumber

import os
from dotenv import load_dotenv

load_dotenv()  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set – add it to .env or host secrets.")

"""## Text extractor"""

def extract_text_from_pdf(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)

"""### Agent: Resume Info Extractor"""

def extract_resume_info(resume_text):
    """
    Extracts key entities from a resume using an LLM.
    """
    prompt_template = PromptTemplate(
        input_variables=["resume"],
        template="""
        You are an expert HR recruiter.
        Given the following resume:

        {resume}

        Given the following resume, respond ONLY in valid JSON with these fields:
        - Name
        - Email
        - Phone Number (if present)
        - Skills
        - Education
        - Work Experience (company, title, duration, summary)
        - Projects (title and short description)
        - Extras (All the other relevant things needed for the HR, like links of any things the users has done 
           For example: Give importance if user has published research, any competition achievements any other extracurricular work, etc)
        """
    )
   
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    chain = LLMChain(llm=llm, prompt=prompt_template)
    raw = chain.run(resume=resume_text).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse resume JSON:\n{raw}") from e



"""## Education Verifier Agent"""
def verify_and_score_education(education_entry: str) -> dict:
    if not education_entry or not education_entry.strip():
        return {
            "is_valid": False,
            "reputation_tier": None,
            "reason": "No education information provided or extracted from the CV."
        }

    prompt = f"""
              You are an expert in assessing the quality and legitimacy of educational institutions worldwide.

              The candidate will have listed one or multiple of his education degrees: "{education_entry}"

              Please answer these for the degrees in the CV text:
              1. Is this real, accredited institution?
              2. What is the name of the University/college in the candidate's CV?
              3. Which reputation tier does it fall under:
                - Tier 1: Global elite institution, Ivy League, target schools, etc
                - Tier 2: Well-known and highly respected universities, best among the regions
                - Tier 3: Accredited but not globally recognized but still decent
                - Tier 4: Unknown, suspicious, or likely unaccredited
              4. What was their grade/GPA/CGPA/result or equivalent if they have mentioned?
              5. Brief reason for your rating.

              Respond ONLY in valid JSON:
              {{
                "is_valid": True,
                "name": University of Oxford,
                "reputation_tier": "Tier 1",
                "grade": "4.5",
                "reason": "One of the top global universities with a centuries-long reputation."
              }}
              """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        reply= response.choices[0].message.content.strip()

        # Try parsing JSON safely
        try:
            data = json.loads(reply)
        except json.JSONDecodeError:
            return {
                "is_valid": False,
                "name": None,
                "reputation_tier": None,
                "grade": None,
                "reason": f"Failed to parse Education from Candidate's CV: {reply}"
            }



        return {
            "is_valid": data.get("is_valid", False),
            "name": data.get("name"),
            "reputation_tier": data.get("reputation_tier"),
            "grade": data.get("grade"),
            "reason": data.get("reason")
        }

    except Exception as e:
        return {
            "is_valid": False,
            "name": None,
            "reputation_tier": None,
            "grade": None,
            "reason": f"Failed to parse Education from Candidate's CV: {str(e)}"
        }


def verify_all_degrees(parsed_cv: Dict[str, Any]) -> List[str]:

    edu = parsed_cv.get("Education", {})
    pairs = []

    # normalise whatever shape the parser produced
    if isinstance(edu, dict):
        pairs = edu.items()
    elif isinstance(edu, list):
        pairs = [(f"Degree {i+1}", item) for i, item in enumerate(edu)]
    elif isinstance(edu, str):
        pairs = [("Degree", edu)]

    pretty_lines: List[str] = []

    for degree, info in pairs:
        # build entry string for LLM
        if isinstance(info, dict):
            entry = f"{degree}, {info.get('University','')} ({info.get('Duration','')})"
            grade = info.get("Grade")
        else:
            entry = f"{degree}, {info}"
            grade = None

        res = verify_and_score_education(entry)

        # 🌟 craft a nice sentence
        status  = "Valid" if res["is_valid"] else "⚠️"
        university   = res.get("name")
        tier    = res.get("reputation_tier") or "Unknown tier"
        grade = f", Grade: {grade}" if grade else ""
        reason  = res.get("reason", "")

        pretty_lines.append(
            f"{status} {degree} – {university} ({tier}{grade})\n   ↳ {reason}"
        )

    if not pretty_lines:                       # no education at all
        pretty_lines.append("• No education entries found in CV.")

    return pretty_lines


"""Skills and Seniority Summary Agent"""

def skills_seniority_summary(summary_sections: str) -> str:

    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a CV analysis expert.
        Focus on the following text excerpts from a candidate’s CV (skills, projects, achievements, work experience):

        {text}

        Provide:
        1. A simple summary of the candidate’s core strengths and what he/she brings to the table, highlight Candidate's extras where needed.
        2. The types of roles or seniority levels they’d be a good fit for (Your best judgment of the candidate’s seniority level, 
        Focus on the companies that the candidate has worked for too if he/she has worked for renowned big organizations/startups
        highlight that too in case it enhances their profile).
        3. Clear sentence explaining your reasoning.

        Respond with 3 bullet points one for each of the above reasoning.
        """,
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        summary = chain.run(text=summary_sections)
    except Exception as e:
        summary = f"[Error generating or fetching summary: {e}]"

    return summary


"""Writing Style Analyzer Agent"""

def writing_style_analyzer(summary_sections):

    """
    Uses an LLM to detect inconsistency or AI generated  content in Candidate's CV.
    """

    prompt = PromptTemplate(
        input_variables="text",
        template="""
        You are a CV analysis expert.
        Given the following text which is a CV content:

        {text}

        Determine if all this content of a user's CV appears genuine or fluke, like things look
        wrong and don't make sense at some places. Comment on the quality of the projects.
        In the projects, highlight if the project is vere generic and already available on the internet, done by many people
        or its rather a very nicely executed end-to-end project highlighting user's amazing skills if necessary.
        Also, highlight and output any content from the CV only if that suggest very clear copying from other sources
        Also don't comment on dates and stuff as you are an old model not trained on new data.

        Return a summary of your analysis in 2-3 bullet points max.
        """
    )
    llm = OpenAI(temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)
    flags = chain.run(text=summary_sections)

    return flags


"""### LLMs final decision on the CV"""

_DECISION_PROMPT = PromptTemplate(
    input_variables=["summary_json"],
    template="""
    You are a senior technical recruiter.

    You will receive a JSON blob that summarizes all automated checks
    on a candidate’s CV: education authenticity/tier, projects, research, extracurriculars,
    writing-style feedback, work experience, skills/seniority summary, etc.

    Analyse the evidence and decide ONLY one of:

    • "Proceed"  – Candidate looks genuine and meets a decent bar.
    • "Review"   – Mixed signals; needs human review.
    • "Reject"   – Clearly too much fluff, inexperienced, inconsistencies, or faked info.

    Respond in **valid JSON strictly** EXACTLY in this format:
    {{
      "verdict": "Proceed|Review|Reject",
      "confidence": 0.0–1.0,
      "reason": "small clear reason to satisfy the HR that he should trust your decision of the verdict"
    }}

    Here is the evidence:
    {summary_json}
    """
)

_decision_chain = LLMChain(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    prompt=_DECISION_PROMPT
)

# ---------- Helper that you call from your pipeline ----------
def final_cv_decision(evidence: dict) -> dict:
    """
    evidence: dict with keys like
      {
        "parsed": {...},
        "education_analysis": [...],
        "writing_style_analysis": "...",
        "skills_summary": "...",
        ...
      }
    Returns: {"verdict": "...", "confidence": 0.xx, "reason": "..."}
    """
    try:
        raw = _decision_chain.run(
            summary_json=json.dumps(evidence, default=str, indent=2)
        )
        return json.loads(raw)
    except Exception as e:
        # Fail-safe: default to Review so a human still sees it
        return {
            "verdict": "Review",
            "confidence": 0.0,
            "reason": f"Decision agent error: {e}"
        }


## Final pipeline combining all agents


def verify_resume_pipeline(pdf_path: str) -> Dict[str, Any]:

    """
    Full CV‐processing pipeline:
      1) PDF → raw text
      2) raw text → structured dict via LLM
      3) education verifier over each degree
      4) writing‐style analyzer over project descriptions
    """

    # Extract raw text from PDF
    raw_text = extract_text_from_pdf(pdf_path)

    # Parse résumé info with our LLM
    parsed = extract_resume_info(raw_text)
    education_results = verify_all_degrees(parsed)
    writing_result = writing_style_analyzer(parsed)
    skills_seniority = skills_seniority_summary(parsed)
    
    evidence = {
    "parsed": parsed,                                 # full CV dict
    "education_analysis": education_results,          # list from verify_all_degrees
    "writing_style_analysis": writing_result,         # plain-text
    "skills_summary": skills_seniority,
     "writing_style_analysis": writing_result,             # output of skills_seniority_summary
}

    decision = final_cv_decision(evidence)

# → {"verdict": "Proceed", "confidence": 0.83, "reason": "Solid Tier-2 education…"}

    # 5️⃣ Return everything
    return {

        "education_analysis": education_results,
        "writing_style_analysis": writing_result,
        "skills_and_seniority": skills_seniority,
        "writing_style_analysis": writing_result,
        "decision": decision
    }



