from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json

path = Path('glossary.json')

languages = ["Russian", "English"]

print("Loading translation rules...")
any_rules = path.exists()
with open('glossary.json', encoding="UTF-8") as file:
    SS13_slang_glossary = json.load(file)
print(f"Succesfully Loaded {len(SS13_slang_glossary)} Rules" if any_rules else "No translation rules found, skipping") 


total_languages_supported = int(len(languages))

while True:
    try:
        for numered_entry, lang in enumerate(languages, start=1):
            print(f"{numered_entry}. {lang}")
        chosen_language= int(input(f"Choose output language:"))
        if(chosen_language > total_languages_supported or chosen_language < 1):
            print("Out of bounds!")
            continue
        else:
            try:
                 for numered_entry, lang in enumerate(languages, start=1):
                     print(f"{numered_entry}. {lang}")

                 input_language= int(input(f"Choose input language:"))
                 if(input_language > total_languages_supported or chosen_language < 1):
                     print("Out of bounds!")
                     continue
                 else:
                     break
            except ValueError:
                print(f"Choose between 1-{total_languages_supported}")

    except ValueError:
        print(f"Choose between 1-{total_languages_supported}")

text_to_translate = input("Text to translate:")
translate_into = languages[chosen_language - 1]
translate_from = languages[input_language - 1]

print(f"text: {text_to_translate}, translating into: {translate_into}")

output_language_code = {"Russian": "ru", "English": "en"}[translate_into]
input_language_code = {"Russian": "ru", "English": "en"}[translate_from]

tokenizer = AutoTokenizer.from_pretrained("models/plamo-3-translate-0.8b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "models/plamo-3-translate-0.8b", trust_remote_code=True, dtype="bfloat16", device_map="auto",
)

counter_start = perf_counter()

print(f"Running {model.name_or_path} on {model.device}")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": f" {text_to_translate}"}],
    target_lang=f"{output_language_code}",
glossary=SS13_slang_glossary,
    add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, do_sample=False, max_new_tokens=512)
text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(text)

counter_end = perf_counter() - counter_start

print(f"Done! Time elapsed: {counter_end} s. Output:{text}")