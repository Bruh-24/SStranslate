import onnxruntime_genai
from transformers import AutoTokenizer

languages = ["Russian", "English"]

total_languages_supported = int(len(languages))

onnxruntime_genai.Model("models/Q4_KQuant_tie/cpu") # todo make this configurable

while True:
    try:
        chosen_language= int(input(f"Choose output language: {languages}"))
        if(chosen_language > total_languages_supported or chosen_language < 1):
            print("Out of bounds!")
            continue
        else:
            break
    except ValueError:
        print("Numbers only!")

text_to_translate = input("Text to translate:")
translate_into = languages[chosen_language - 1]

print(f"text: {text_to_translate}, translating into: {translate_into}")