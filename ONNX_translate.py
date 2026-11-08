import onnxruntime_genai as onnx
from transformers import AutoTokenizer
from time import perf_counter
import json

languages = ["Russian", "English"]
with open('glossary.json', encoding="UTF-8") as file:
    SS13_slang_glossary = json.load(file)

total_languages_supported = int(len(languages))

#justinchuby/Hy-MT2-1.8B-ONNX/Q4_KQuant_tie/cpu
#f16/default
model = onnx.Model("models/f16/default")
tokenizer = AutoTokenizer.from_pretrained("models/f16/default")

while True:
    try:
        
        for numered_entry, lang in enumerate(languages, start=1):
            print(f"{numered_entry}. {lang}")

        chosen_language= int(input(f"Choose output language:"))
        if(chosen_language > total_languages_supported or chosen_language < 1):
            print("Out of bounds!")
            continue
        else:
            break
    except ValueError:
        print(f"Choose between 1-{total_languages_supported}")

text_to_translate = input("Text to translate:")
translate_into = languages[chosen_language - 1]

print(f"text: {text_to_translate}, translating into: {translate_into}")

# based of https://huggingface.co/justinchuby/Hy-MT2-1.8B-ONNX#usage-onnx-runtime-genai

prompt = f"Translate the following text into {translate_into}. Context: Space Station 13 game chat. Output only the translation:\n{text_to_translate}"
message = {"role": "user", "content": prompt}

prompt = tokenizer.apply_chat_template([message], tokenize=False,
                                 add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors="np")["input_ids"].astype("int32")

params = onnx.GeneratorParams(model)
params.set_search_options(max_length=256, do_sample=False)
gen = onnx.Generator(model, params)

counter_start = perf_counter()

gen.append_tokens(ids[0])

out = []
while not gen.is_done():
    gen.generate_next_token()
    out.append(gen.get_next_tokens()[0])

output = tokenizer.decode(out, skip_special_tokens=True)

counter_end = perf_counter() - counter_start

print(f"Done! Time elapsed: {counter_end} s. Output:{output}")