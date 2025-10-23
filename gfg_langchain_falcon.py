import json
import re
import requests
import os
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

# --------- Step 0: Load environment variables ---------
load_dotenv()
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise EnvironmentError("❌ HUGGINGFACEHUB_API_TOKEN not found in .env file!")

# --------- Step 1: Schema ---------
class GFGProfile(BaseModel):
    username: str
    display_name: str | None = None
    about: str | None = None
    reputation: str | None = None
    followers: int | None = None
    following: int | None = None
    problems_solved: int | None = None
    articles_count: int | None = None
    badges: list[str] = []
    top_skills: list[str] = []
    raw_extracted: dict = Field(default_factory=dict)

# --------- Step 2: Scraping ---------
def fetch_gfg_profile(username: str) -> str | None:
    urls = [
        f"https://auth.geeksforgeeks.org/user/{username}",
        f"https://www.geeksforgeeks.org/user/{username}/",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; gfg-falcon/1.0)"}

    for url in urls:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.text
    return None


def parse_profile_html(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    name_tag = soup.select_one(".profile_name, .name, h1")
    if name_tag:
        data["display_name"] = name_tag.get_text(strip=True)

    about_tag = soup.select_one(".about, .bio, p")
    if about_tag:
        data["about"] = about_tag.get_text(strip=True)

    text = soup.get_text(separator="\n")

    def grab_num(pat):
        m = re.search(pat, text, re.I)
        if m:
            return int(m.group(m.lastindex).replace(",", ""))
        return None

    data["followers"] = grab_num(r"Followers[:\s]*([0-9,]+)")
    data["following"] = grab_num(r"Following[:\s]*([0-9,]+)")
    data["problems_solved"] = grab_num(r"(?:Problems solved|Solved problems)[:\s]*([0-9,]+)")
    data["articles_count"] = grab_num(r"(?:Articles|Posts)[:\s]*([0-9,]+)")

    badges = [b.get_text(strip=True) for b in soup.select(".badge, .user-badges li") if b.get_text(strip=True)]
    if badges:
        data["badges"] = badges

    skills = [s.get_text(strip=True) for s in soup.select(".skill, .skills li, .tag")]
    if skills:
        data["top_skills"] = skills

    data["raw_text_snippet"] = text[:2000]
    return data


# --------- Step 3: Falcon LLM ---------
prompt = PromptTemplate(
    template="""
You are given a dictionary of raw extracted information from a GeeksforGeeks user page.
Generate a clean JSON object that follows this schema:
{{
  "username": "{username}",
  "display_name": string|null,
  "about": string|null,
  "followers": int|null,
  "following": int|null,
  "articles_count": int|null,
  "contest_rating": int|null,
  "contest_level": int|null,
  "global_rank": int|null,
  "university_rank": int|null,
  "contests_attended": int|null,
  "problems_solved": int|null,
  "problems_breakdown": dict|null,
  "badges": [strings],
  "top_skills": [strings],
  "raw_extracted": dict
}}
Only output valid JSON, nothing else.

Raw extracted info:
{raw_extracted}
""",
    input_variables=["username", "raw_extracted"],
)

# 🦅 Using Falcon via HuggingFaceHub
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # You can also use "tiiuae/falcon-40b-instruct"
    model_kwargs={"temperature": 0.2, "max_new_tokens": 1024}
)

chain = LLMChain(llm=llm, prompt=prompt)


# --------- Step 4: Pipeline ---------
def gfg_to_json(username: str, output_path: str = "output.json") -> dict:
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑️ Deleted old file: {output_path}")

    html = fetch_gfg_profile(username)
    if not html:
        raise ValueError("Profile not found or private.")
    raw = parse_profile_html(html)

    raw_str = json.dumps(raw, ensure_ascii=False)
    response = chain.invoke({"username": username, "raw_extracted": raw_str})
    response = response["text"] if isinstance(response, dict) and "text" in response else str(response)

    try:
        parsed = json.loads(response)
    except Exception:
        match = re.search(r"\{.*\}", response, re.S)
        if match:
            parsed = json.loads(match.group(0))
        else:
            raise ValueError("Invalid JSON from Falcon:\n" + response)

    profile = GFGProfile(**parsed)
    result = profile.dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved new file: {output_path}")

    return result


# --------- Step 5: CLI ---------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch GFG profile and format via Falcon LLM")
    parser.add_argument("username", help="GFG username")
    args = parser.parse_args()

    result = gfg_to_json(args.username)
    print(json.dumps(result, indent=2, ensure_ascii=False))
