import zipfile
import urllib.request as u

url = "https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip"
req = u.Request(url, headers={"User-Agent": "Mozilla/5.0"})
data = u.urlopen(req, timeout=120).read()
open("uci1081.zip", "wb").write(data)
print("downloaded", len(data), "bytes")
with zipfile.ZipFile("uci1081.zip") as z:
    print("members:", z.namelist())
    z.extractall("uci1081")