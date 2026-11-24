import re
import urllib.request as u

url = "https://archive.ics.uci.edu/dataset/1081/gas+sensor+array+low-concentration"
req = u.Request(url, headers={"User-Agent": "Mozilla/5.0"})
html = u.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
open("page.html", "w", encoding="utf-8").write(html)
print("page bytes:", len(html))
for m in re.findall(r'href="([^"]+)"', html, re.I):
    if re.search(r"zip|csv|download|files", m, re.I):
        print(m)