import json
import urllib.request


def check(url: str = "http://127.0.0.1:8080/health") -> None:
    with urllib.request.urlopen(url) as resp:
        body = resp.read().decode("utf-8")
        data = json.loads(body)
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    check()
