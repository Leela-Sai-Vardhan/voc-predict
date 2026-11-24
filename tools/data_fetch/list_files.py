import requests, os, json

token = os.environ.get('HF_TOKEN', '')
h = {'Authorization': 'Bearer ' + token} if token else {}

url = 'https://huggingface.co/api/datasets/DeweiFeng/smell-net/tree/main'
r = requests.get(url, headers=h, timeout=30)
print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    # Print entries
    for item in data.get('tree', []):
        print(f"  {item.get('type'):10} {item.get('path', ''):60} {item.get('size', ''):>15}")
else:
    print('Failed:', r.text[:200])