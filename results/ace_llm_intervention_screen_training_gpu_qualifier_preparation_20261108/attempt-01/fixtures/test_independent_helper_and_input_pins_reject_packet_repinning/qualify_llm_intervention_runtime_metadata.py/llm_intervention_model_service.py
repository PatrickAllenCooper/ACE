import json,hashlib
MODEL_NAMES={'model.fake','tokenizer.fake'}
def decode(raw):return json.loads(raw)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
