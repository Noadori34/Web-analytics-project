import pandas as pd
import requests

print("=" * 60)
print("THE MUSE API: SPAIN (No Role Filter) - TOTAL COUNT & FULL DESCRIPTION")
print("=" * 60)

muse_url = "https://www.themuse.com/api/public/jobs"
# Location filter set to Spain, no category/role restriction, page 0
muse_params = {"location": "Spain", "page": 0}
muse_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

response_muse = requests.get(muse_url, headers=muse_headers, params=muse_params)
if response_muse.status_code == 200:
  muse_data = response_muse.json()

  total_muse_count = muse_data.get("total", 0)
  results = muse_data.get("results", [])

  print(f"[SUCCESS] Total matching jobs in Spain on The Muse: {total_muse_count}")
  print(f"Jobs returned in current page payload: {len(results)}")

  if results:
    sample = results[0]
    full_desc = sample.get("contents", "")
    print(f"\n--- Sample The Muse Job (Untruncated) ---")
    print(f"Title: {sample.get('name')}")
    print(f"Company: {sample.get('company', {}).get('name')}")
    print(f"Total Description Length: {len(full_desc)} characters")
    print(f"\n{full_desc}\n")
  else:
    print(
        "No open listings found right now for Spain on The Muse (try widening"
        " parameters if needed)."
    )
else:
  print(f"The Muse Error: {response_muse.status_code} - {response_muse.text}")


print("=" * 60)
print("USAJOBS API (No Role or Page Filter) - TOTAL COUNT & FULL DESCRIPTION")
print("=" * 60)

USAJOBS_KEY = "jUEA1BtI1IVYu76rxG/SbrveYfl6qhJVxMYDyBpaOSM="
USAJOBS_EMAIL = "leong.kj.sg@gmail.com"

usajobs_url = "https://data.usajobs.gov/api/Search"
usajobs_headers = {
    "Host": "data.usajobs.gov",
    "User-Agent": USAJOBS_EMAIL,
    "Authorization-Key": USAJOBS_KEY,
}
# Empty parameters mean no specific keyword/role or page restriction
usajobs_params = {}

response_usa = requests.get(usajobs_url, headers=usajobs_headers, params=usajobs_params)
if response_usa.status_code == 200:
  usa_data = response_usa.json()
  search_result = usa_data.get("SearchResult", {})

  total_usa_count = search_result.get("SearchResultCountAll", 0)
  items = search_result.get("SearchResultItems", [])

  print(f"[SUCCESS] Total matching active jobs on USAJOBS: {total_usa_count}")
  print(f"Jobs returned in current page payload: {len(items)}")

  if items:
    descriptor = items[0]["MatchedObjectDescriptor"]
    summary = descriptor.get("JobSummary", "")
    qualifications = descriptor.get("QualificationSummary", "")
    full_text = f"{summary}\n\n{qualifications}"

    print(f"\n--- Sample USAJOBS Position (Untruncated) ---")
    print(f"Title: {descriptor.get('PositionTitle')}")
    print(f"Department: {descriptor.get('DepartmentName')}")
    print(f"Total Combined Description Length: {len(full_text)} characters")
    print(f"\n{full_text}\n")
else:
  print(f"USAJOBS Error: {response_usa.status_code} - {response_usa.text[:150]}")

print("=" * 60)