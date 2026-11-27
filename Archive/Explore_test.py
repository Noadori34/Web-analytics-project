import requests
import pandas as pd
import re
from bs4 import BeautifulSoup

BASE_URL = "https://www.themuse.com/api/public/jobs"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

TARGET_RECORDS = 300
PAGE_SIZE = 20

TECH_KEYWORDS = [
    "data", "software", "engineer", "developer", "machine learning",
    "artificial intelligence", "ai", "cloud", "devops", "analytics",
    "cybersecurity", "security", "backend", "frontend", "full stack",
    "full-stack", "product manager", "technical product", "blockchain"
]

SENIORITY_PATTERNS = {
    "Junior": r"\b(junior|jr\.?|entry[- ]level)\b",
    "Senior": r"\b(senior|sr\.?)\b",
    "Lead": r"\b(lead|team lead)\b",
    "Staff": r"\bstaff\b",
    "Principal": r"\bprincipal\b",
    "Manager": r"\bmanager\b",
    "Director": r"\bdirector\b"
}

REMOTE_PATTERNS = [
    r"\bremote\b",
    r"#li-remote",
    r"\bwork from home\b",
    r"\bwork remotely\b"
]

CURRENCY_PATTERNS = {
    "USD": r"\$",
    "EUR": r"€|\beur\b",
    "GBP": r"£|\bgbp\b"
}


def clean_html(html):
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def detect_currency(text):
    text_lower = text.lower()

    for currency, pattern in CURRENCY_PATTERNS.items():
        if re.search(pattern, text_lower, flags=re.IGNORECASE):
            return currency

    return None


def extract_salary(text):
    """
    Attempts to extract salary ranges such as:
    $155,904 to $194,880
    €50,000 - €70,000
    £60k–£80k
    """

    text = text.replace("\xa0", " ")

    patterns = [
        r'([$€£])\s?([\d,]+(?:\.\d+)?)\s*(?:to|-|–|—)\s*([$€£])?\s?([\d,]+(?:\.\d+)?)',
        r'([$€£])\s?([\d,.]+)\s*[kK]\s*(?:to|-|–|—)\s*([$€£])?\s?([\d,.]+)\s*[kK]',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            symbol = match.group(1)
            first = match.group(2)
            second = match.group(4)

            is_k_format = "k" in match.group(0).lower()

            min_salary = float(first.replace(",", ""))
            max_salary = float(second.replace(",", ""))

            if is_k_format:
                min_salary *= 1000
                max_salary *= 1000

            currency = {
                "$": "USD",
                "€": "EUR",
                "£": "GBP"
            }.get(symbol)

            return min_salary, max_salary, currency

    return None, None, detect_currency(text)


def detect_remote(text):
    text_lower = text.lower()

    for pattern in REMOTE_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    return False


def detect_seniority(title, description):
    combined = f"{title} {description}".lower()

    for seniority, pattern in SENIORITY_PATTERNS.items():
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return seniority

    return "Unknown"


def detect_technical_role(title, description):
    combined = f"{title} {description}".lower()

    return any(keyword in combined for keyword in TECH_KEYWORDS)


def fetch_jobs():
    all_jobs = []
    page = 1

    while len(all_jobs) < TARGET_RECORDS:
        print(f"Fetching page {page}...")

        params = {
            "location": "Spain",
            "page": page
        }

        response = requests.get(
            BASE_URL, headers=HEADERS, params=params, timeout=30
        )
        response.raise_for_status()

        data = response.json()
        jobs = data.get("results", [])

        if not jobs:
            print("No more jobs returned.")
            break

        all_jobs.extend(jobs)

        print(f"Collected {len(all_jobs)} jobs")

        page += 1

    return all_jobs[:TARGET_RECORDS]


def process_jobs(jobs):
    rows = []

    for job in jobs:
        title = job.get("name", "")
        company = job.get("company", {}).get("name", "")

        locations = job.get("locations", [])
        location_text = ", ".join(
            loc.get("name", "")
            for loc in locations
            if isinstance(loc, dict)
        )

        category = ""
        categories = job.get("categories", [])

        if categories:
            category = ", ".join(
                cat.get("name", "")
                for cat in categories
                if isinstance(cat, dict)
            )

        raw_description = job.get("contents", "")
        clean_description = clean_html(raw_description)

        salary_min, salary_max, currency = extract_salary(clean_description)

        salary_mid = None
        if salary_min is not None and salary_max is not None:
            salary_mid = (salary_min + salary_max) / 2

        row = {
            "job_id": job.get("id"),
            "title": title,
            "company": company,
            "location": location_text,
            "category": category,
            "description_length": len(clean_description),
            "contains_salary": salary_min is not None or salary_max is not None,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_mid": salary_mid,
            "salary_currency": currency,
            "remote_flag": detect_remote(clean_description),
            "seniority": detect_seniority(title, clean_description),
            "technical_role_flag": detect_technical_role(
                title,
                clean_description
            ),
            "description": clean_description
        }

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df):
    print("\n" + "=" * 60)
    print("THE MUSE FEASIBILITY SUMMARY")
    print("=" * 60)

    print(f"\nTotal sampled jobs: {len(df)}")

    print(
        f"Jobs with detected salary: "
        f"{df['contains_salary'].sum()} "
        f"({df['contains_salary'].mean() * 100:.1f}%)"
    )

    print(
        f"Remote jobs detected: "
        f"{df['remote_flag'].sum()} "
        f"({df['remote_flag'].mean() * 100:.1f}%)"
    )

    print(
        f"Likely technical roles: "
        f"{df['technical_role_flag'].sum()} "
        f"({df['technical_role_flag'].mean() * 100:.1f}%)"
    )

    print(
        f"Median description length: "
        f"{df['description_length'].median():.0f} characters"
    )

    print("\nCurrency distribution:")
    print(
        df["salary_currency"]
        .fillna("Unknown")
        .value_counts()
    )

    print("\nSeniority distribution:")
    print(
        df["seniority"]
        .value_counts()
    )

    print("\nTop locations:")
    print(
        df["location"]
        .value_counts()
        .head(10)
    )

    print("\nTop categories:")
    print(
        df["category"]
        .value_counts()
        .head(10)
    )

    salary_df = df[df["salary_mid"].notna()]

    if len(salary_df) > 0:
        print("\nSalary summary by currency:")

        for currency in salary_df["salary_currency"].dropna().unique():

            subset = salary_df[
                salary_df["salary_currency"] == currency
            ]

            print(f"\n{currency}")

            print(
                f"  Jobs: {len(subset)}"
            )

            print(
                f"  Median salary midpoint: "
                f"{subset['salary_mid'].median():,.0f}"
            )

            print(
                f"  Mean salary midpoint: "
                f"{subset['salary_mid'].mean():,.0f}"
            )


def main():

    jobs = fetch_jobs()

    df = process_jobs(jobs)

    output_file = "muse_spain_sample_300.csv"

    df.to_csv(
        output_file,
        index=False
    )

    print_summary(df)

    print("\n" + "=" * 60)

    print(
        f"Saved detailed sample to: {output_file}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()