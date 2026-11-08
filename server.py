from fastapi import FastAPI, Request, HTTPException
import json
from pathlib import Path
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager
import asyncio


@asynccontextmanager
async def life(app: FastAPI):
     app.state.translation_rules = load_rules()
     app.state.model, app.state.tokenizer = load_model()
     app.state.queue = asyncio.Queue(maxsize=20)
     app.state.worker = asyncio.create_task(worker(app))
     yield

async def worker(app):
     while True:
          job = await app.state.queue.get()
         
          try:
               output = await asyncio.to_thread(translate, job["text"], job["target"], app.state.translation_rules, app.state.model, app.state.tokenizer)
               if not job["reply"].done():
                    job["reply"].set_result(output)
          except Exception as error:
               if not job["reply"].done():
                    job["reply"].set_exception(error)
          finally:
               app.state.queue.task_done()
       

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
     print(f"Loading complete! Loading took {counter_end} seconds to complete")
     return model, tokenizer

def translate(text_to_translate, target_lang, rules, model, tokenizer):

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

@server.post("/translate")
async def translation_request(request_body: dict, request: Request):
     q = request_body["q"]
     target = request_body["target"]
    # rules = request.app.state.translation_rules
    # model = request.app.state.model
    # tokenizer = request.app.state.tokenizer
    # output = translate(text_to_translate = q, target_lang=target, rules = rules, model = model, tokenizer = tokenizer)


     reply = asyncio.get_running_loop().create_future()
     job = {
    "text": q,
    "target": target,
    "reply": reply,
     }
     try:
          request.app.state.queue.put_nowait(job)
     except asyncio.QueueFull:
          raise HTTPException(status_code=418, detail="Queue is full") #imateapot
     
     output = await reply
     return {"translatedText": f"{output}"}

@server.get("/languages")
def languages_request():
     # don't ask me, i know this sucks ass, but i wasn't the one who wrote the in-game front
     # it was firstly designed around libretranslate, so, uhm. here you go
     return [ {"code": "ru", "name": "Russian", "targets": ["en"]}, {"code": "en", "name": "English", "targets": ["ru"]}]