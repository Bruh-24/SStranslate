Tiny docker-application for translating SS13 chat, currently supports two languages RU/EN
Made by Bruh24 with love, without AI (used for debug only) [SS13-chat frontend made by maximal08](https://github.com/Alliance-codeBase/PostMeta/pull/238/)
I have no affilation with that PR, I was tasked (by myself, lol) to make a better backend, with a bigger model and a glossary, because apparently Libretranslate wasn't enough for me.
This runs on a CPU, tested with AMD Ryzen 5 8400F, latency per request is 1.5-2 seconds; Ryzen 9 7950x tests are in progress

You need docker and hugging face hub installed and available in path
To run this just paste ``hf download pfnet/plamo-3-translate-0.8b --local-dir models/`` and ``docker compose --build``
The service will run on localhost:3535 (change this in compose if for any reason you want to inference the model on a remote server away from the SS13 hosting)
After you see ``translate-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)`` you can detach from container and run the following:

```
curl -X 'POST' \
  'http://127.0.0.1:3535/translate' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{"q": "Changeling is trying to destroy supermatter, HoS get your ass here and help us!", "source": "en", "target": "ru"}'
```

This will translate whatever's put into q to Russian. You can do it in both ways by changing source and target

``glossary.json`` - you may want to add some SS13 terminology into that, because model struggles with it. It's required to have it defined in both ways like this:

```json
    {"source": "Warden", "target": "Смотритель"},
    {"source": "Смотритель", "target": "Warden"},
```

And by the way, if you for any reason stumble upon this repository, I highly advise you to write your SS13 frontend for chat, because the one currently [available](https://github.com/Alliance-codeBase/PostMeta/pull/238/) could have some refactoring by an better maintainer.