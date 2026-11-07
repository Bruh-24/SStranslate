import onnxruntime_genai as onnx
from transformers import AutoTokenizer

languages = ["Russian", "English"]

total_languages_supported = int(len(languages))

model = onnx.Model("models/Q4_KQuant_tie/cpu") #  make this configurable
tokenizer = AutoTokenizer.from_pretrained("models/Q4_KQuant_tie/cpu")

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

prompt = f"Translate the following text into {translate_into}. Output only the translation:\n{text_to_translate}"
message = {"role": "user", "content": prompt}

prompt = tokenizer.apply_chat_template([message], tokenize=False,
                                 add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors="np")["input_ids"].astype("int32")

params = onnx.GeneratorParams(model)
params.set_search_options(max_length=256, do_sample=False)
gen = onnx.Generator(model, params)
gen.append_tokens(ids[0])
out = []
while not gen.is_done():
    gen.generate_next_token()
    out.append(gen.get_next_tokens()[0])

output = tokenizer.decode(out, skip_special_tokens=True)

print(f"{output}")