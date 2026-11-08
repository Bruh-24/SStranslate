from fastapi import FastAPI, Request
import json
from pathlib import Path
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager



@asynccontextmanager
async def life(app: FastAPI):
     app.state.translation_rules = load_rules()
     app.state.model, app.state.tokenizer = load_model()
     yield


server = FastAPI(lifespan=life)

# A healthcheck endpoint - justincase
@server.get("/health-check")
def health():
     return {"status": "OK"}

# Loads our translation rules, model itself is bad at translating SS13 termins and slang, so we have to guide it
def load_rules():
     print("Loading translation rules...")
     path = Path('glossary.json')
     any_rules = path.exists()
     if(not any_rules):
          return
     with open(f'{path}', encoding="UTF-8") as file:
         SS13_slang_glossary = json.load(file)
         print(f"Succesfully Loaded {len(SS13_slang_glossary)} Rules" if any_rules else "No translation rules found, skipping") 
     if(any_rules):
          return SS13_slang_glossary
     else:
          return None


# Loads our model
def load_model():

     counter_start = perf_counter()
     tokenizer = AutoTokenizer.from_pretrained("models/plamo-3-translate-0.8b", trust_remote_code=True)
     model = AutoModelForCausalLM.from_pretrained(
         "models/plamo-3-translate-0.8b", trust_remote_code=True, dtype="bfloat16", device_map="auto",
     )

     print(f"Running {model.name_or_path} on {model.device}")

     counter_end = perf_counter() - counter_start
     print(f"{counter_end}")
     return model, tokenizer

def translate(text_to_translate, target_lang, source_lang, rules, model, tokenizer):
     
     inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": f" {text_to_translate}"}],
    target_lang=f"{target_lang}",
     glossary=rules,
    add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
     ).to(model.device)

     output = model.generate(**inputs, do_sample=False, max_new_tokens=512)
     text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
     return f"{text}"

