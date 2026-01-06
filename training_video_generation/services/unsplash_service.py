import os
import requests
import uuid

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")


def fetch_and_save_photo(query):
    """
    Fetches an image from Unsplash.
    Falls back to a local placeholder if API key is missing.
    """

    os.makedirs("images", exist_ok=True)

    # 🔁 FALLBACK MODE (NO API KEY)
    if not UNSPLASH_ACCESS_KEY:
        fallback = os.path.join("images", "fallback.jpg")

        if not os.path.exists(fallback):
            from PIL import Image
            img = Image.new("RGB", (1280, 720), (40, 40, 60))
            img.save(fallback, "JPEG", quality=90)

        return fallback

    # ---------------- REAL UNSPLASH MODE ----------------
    url = "https://api.unsplash.com/photos/random"
    params = {
        "query": query,
        "orientation": "landscape",
    }
    headers = {
        "Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    image_url = response.json()["urls"]["regular"]
    img_data = requests.get(image_url).content

    filename = f"{uuid.uuid4()}.jpg"
    path = os.path.join("images", filename)

    with open(path, "wb") as f:
        f.write(img_data)

    return path
