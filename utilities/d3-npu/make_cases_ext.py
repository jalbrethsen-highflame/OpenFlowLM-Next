r"""make_cases_ext: make_cases.py's set plus 30 questions built to be close calls (mixed
signals, overlapping options, under-specified states), where a small numeric shift shows
up as a probability change or a flipped answer.

    python make_cases_ext.py <repo root> > cases_ext.json
"""
import json
import subprocess
import sys
from pathlib import Path

base = json.loads(subprocess.run([sys.executable, str(Path(__file__).with_name("make_cases.py")), sys.argv[1]],
                                 capture_output=True, text=True, check=True).stdout)


def choice(cid, state, instr, opts):
    return {"id": cid, "state": state, "question": {"type": "choice", "instructions": instr, "criteria": opts}}


def noul(cid, state, instr, crit=None):
    q = {"type": "noul", "instructions": instr}
    if crit:
        q["criteria"] = crit
    return {"id": cid, "state": state, "question": q}


sent = {"positive": None, "negative": None, "mixed": None, "neutral": None}
close = [
    choice("s_meh", "It works. Setup took an hour longer than the box said, but support was friendly.", "Sentiment?", sent),
    choice("s_sarcasm", "Great, another update that moves every button I use. Love it.", "Sentiment?", sent),
    choice("s_faint", "Not the worst hotel we have stayed in.", "Sentiment?", sent),
    choice("s_late", "The food was cold and late, but they refunded us without asking.", "Sentiment?", sent),
    choice("route_ambig", "I can't log in since you charged my card again yesterday.", "Route this ticket.",
           {"billing": None, "account": "login and access", "technical": None}),
    choice("route_ambig2", "The export button gives an error and now my invoice is missing.", "Route this ticket.",
           {"billing": None, "account": None, "technical": "bugs"}),
    choice("prio", "Customer reports the dashboard is slow, about 5 s per page, since this morning. 3 users affected.",
           "Priority?", {"p1": "outage", "p2": "major degradation", "p3": "minor issue", "p4": "question"}),
    choice("prio2", "Payments fail for about 2% of checkouts in one region.", "Priority?",
           {"p1": "outage", "p2": "major degradation", "p3": "minor issue", "p4": "question"}),
    noul("tool_borderline", {"tool": "shell", "command": "git reset --hard HEAD~1"}, "Does this command delete user data?"),
    noul("tool_borderline2", {"tool": "shell", "command": "docker system prune -f"}, "Does this command delete user data?"),
    noul("tool_borderline3", {"tool": "shell", "command": "mv ~/notes.txt /tmp/"}, "Does this command delete user data?"),
    noul("pii_maybe", "Call me back at the office line, extension 4412.", "Does the text contain personal contact data?"),
    noul("pii_maybe2", "My username is jdoe1987 if that helps.", "Does the text contain personal data?"),
    noul("refund_vague", "This isn't what I expected. What are my options?", "Does the customer ask for a refund?"),
    noul("cancel_vague", "Please stop the auto-renew for next year.", "Does the customer cancel the subscription?"),
    choice("intent", "Can you also add my wife to the account so she can see the bills?", "Intent?",
           {"add_user": None, "billing_question": None, "privacy_request": None, "other": None}),
    choice("lang_mix", "Hola team, the meeting is mañana at 10, ok?", "Language?",
           {"English": None, "Spanish": None, "Portuguese": None}),
    choice("topic", "The quarterly report shows revenue up 4% but churn rose in the SMB segment.", "Topic?",
           {"finance": None, "customers": "customer retention", "product": None, "sales": None}),
    choice("next_step", {"goal": "deploy v2.3", "tests": "2 flaky failures, rerun passed", "approvals": "1 of 2"},
           "Next action?", {"deploy": None, "wait_for_approval": None, "rerun_tests": None, "rollback": None}),
    choice("next_step2", {"goal": "answer user question about API limits", "found": "docs page from 2023"},
           "Next action?", {"answer_now": None, "search_newer_docs": None, "ask_user": None}),
    choice("severity", "A typo in the privacy policy says 'we sell your data' instead of 'we never sell your data'.",
           "Severity?", {"low": None, "medium": None, "high": None, "critical": None}),
    choice("animal", "It has a long neck, eats leaves, and is not a giraffe.", "Which animal is it most likely?",
           {"giraffe": None, "llama": None, "camel": None, "ostrich": None, "okapi": None}),
    choice("weather", "Grey sky, light drizzle on and off, 11 C.", "Weather?",
           {"sunny": None, "rainy": None, "cloudy": None, "stormy": None}),
    choice("count", "apples: 3, pears: 3, plums: 2", "Which fruit is most common?",
           {"apples": None, "pears": None, "plums": None, "tie": "two or more are equal"}),
    noul("enough_info", {"from": "Boston", "date": "next Friday"}, "Is there enough information to book a flight?"),
    noul("legal_q", "Can I use this font in a logo I sell?", "Is this a legal question?"),
    choice("tone", "Per my last email, the file is attached again.", "Tone?",
           {"friendly": None, "neutral": None, "annoyed": None, "angry": None}),
    choice("dup", "Ticket A: 'login fails with 500'. Ticket B: 'cannot sign in, server error'.", "Relation?",
           {"duplicate": None, "related": None, "unrelated": None}),
    choice("rank", "Candidate 1: 5 years Go, no Rust. Candidate 2: 2 years Rust, 3 years C++. Role: Rust services.",
           "Better fit?", {"candidate_1": None, "candidate_2": None, "equal": None}),
    choice("many", "The user wants a light laptop for travel, long battery, budget under 1000 USD.", "Best match?",
           {f"model_{i}": d for i, d in enumerate(["1.1 kg, 12 h, 1100 USD", "1.4 kg, 15 h, 900 USD",
                                                    "1.0 kg, 8 h, 950 USD", "2.1 kg, 18 h, 800 USD",
                                                    "1.3 kg, 11 h, 990 USD"])}),
]
print(json.dumps(base + close, indent=1, ensure_ascii=False))
