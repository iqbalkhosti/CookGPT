import requests
import time
import os

def fetch_recipes(num_recipes=100, filename="input.txt"):
    """
    Fetches random recipes from TheMealDB and saves them to a text file.
    """
    base_url = "https://www.themealdb.com/api/json/v1/1/random.php"

    print(f"🍳 Starting the scrape for {num_recipes} recipes...")

    with open(filename, "w", encoding="utf-8") as f:
        count = 0
        while count < num_recipes:
            try:
                response = requests.get(base_url)
                if response.status_code == 200:
                    data = response.json()
                    meal = data['meals'][0]

                    # specific fields we want
                    title = meal.get('strMeal', 'Unknown Recipe')
                    category = meal.get('strCategory', 'General')
                    area = meal.get('strArea', 'Unknown')
                    instructions = meal.get('strInstructions', '')

                    # Basic cleaning of instructions (removing extra newlines)
                    instructions = instructions.replace('\r\n', '\n').strip()

                    # Construct the text format
                    # We use a distinct separator "---" so the model learns where recipes end
                    entry = f"RECIPE: {title}\n"
                    entry += f"STYLE: {area} {category}\n"
                    entry += "INSTRUCTIONS:\n"
                    entry += f"{instructions}\n"
                    entry += "\n---\n\n"

                    f.write(entry)
                    count += 1

                    # Print progress every 10 recipes
                    if count % 10 == 0:
                        print(f"  ...collected {count} recipes.")

                    # Be nice to the API
                    time.sleep(0.5)

            except Exception as e:
                print(f"Error fetching recipe: {e}")
                time.sleep(1)

    print(f"✅ Done! Saved {count} recipes to '{filename}'.")
    print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

# --- Run it ---
# 500 recipes should give you roughly 500KB - 1MB of text, which is perfect for nanoGPT.
fetch_recipes(num_recipes=500)