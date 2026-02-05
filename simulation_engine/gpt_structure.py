import time
import base64
import json
import urllib.request
import urllib.error
from typing import List, Union

from simulation_engine.settings import *

try:
  import openai
except Exception:
  openai = None


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def _ollama_generate(prompt: str,
                     model: str,
                     max_tokens: int,
                     temperature: float,
                     top_p: float,
                     timeout: int) -> str:
  payload = {
    "model": model,
    "prompt": prompt,
    "stream": False,
    "options": {
      "temperature": temperature,
      "top_p": top_p,
      "num_predict": max_tokens
    }
  }
  data = json.dumps(payload).encode("utf-8")
  req = urllib.request.Request(
    f"{OLLAMA_API_URL}/api/generate",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
  )
  try:
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      body = resp.read().decode("utf-8", errors="ignore")
    result = json.loads(body)
    return result.get("response", "")
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def gpt_request(prompt: str,
                model: str = "gpt-4o",
                max_tokens: int = 1500) -> str:
  """Make a request to an LLM backend (Ollama by default)."""
  if USE_OLLAMA:
    ollama_model = OLLAMA_MODEL if OLLAMA_MODEL else model
    return _ollama_generate(
      prompt,
      ollama_model,
      max_tokens=OLLAMA_MAX_TOKENS if OLLAMA_MAX_TOKENS else max_tokens,
      temperature=OLLAMA_TEMPERATURE,
      top_p=OLLAMA_TOP_P,
      timeout=OLLAMA_TIMEOUT
    )

  if not USE_OPENAI:
    return "GENERATION ERROR: No LLM backend enabled."

  if openai is None:
    return "GENERATION ERROR: OpenAI package not installed."

  if model == "o1-preview":
    try:
      client = openai.OpenAI(api_key=OPENAI_API_KEY)
      response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
      )
      return response.choices[0].message.content
    except Exception as e:
      return f"GENERATION ERROR: {str(e)}"

  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def gpt4_vision(messages: List[dict], max_tokens: int = 1500) -> str:
  """Make a request to OpenAI's GPT-4 Vision model."""
  if USE_OLLAMA:
    return "GENERATION ERROR: Vision requests are not supported with Ollama in this project."
  if openai is None:
    return "GENERATION ERROR: OpenAI package not installed."
  try:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=messages,
      max_tokens=max_tokens,
      temperature=0.7
    )
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-4o", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 1500,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  if file_attachment and file_type:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    messages = [{"role": "user", "content": prompt}]

    if file_type.lower() == 'image':
      if USE_OLLAMA:
        response = "GENERATION ERROR: Vision requests are not supported with Ollama in this project."
        prompt = generate_prompt(prompt_input, prompt_lib_file)
        if func_clean_up:
          response = func_clean_up(response, prompt=prompt)
        if verbose or DEBUG:
          print_run_prompts(prompt_input, prompt, response)
        return response, prompt, prompt_input, fail_safe
      with open(file_attachment, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
      messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please refer to the attached image."},
            {"type": "image_url", "image_url": 
              {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
      })
      response = gpt4_vision(messages, max_tokens)

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      response = gpt_request(prompt, gpt_version, max_tokens)

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    for i in range(repeat):
      response = gpt_request(prompt, model=gpt_version)
      if response != "GENERATION ERROR":
        break
      time.sleep(2**i)
    else:
      response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def _ollama_embedding(text: str, model: str, timeout: int) -> List[float]:
  payload = {"model": model, "input": text}
  data = json.dumps(payload).encode("utf-8")

  def _post(path: str) -> dict:
    req = urllib.request.Request(
      f"{OLLAMA_API_URL}{path}",
      data=data,
      headers={"Content-Type": "application/json"},
      method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      body = resp.read().decode("utf-8", errors="ignore")
    return json.loads(body)

  # Prefer the newer /api/embed endpoint, fall back to legacy /api/embeddings,
  # then try OpenAI-compatible /v1/embeddings.
  try:
    result = _post("/api/embed")
  except urllib.error.HTTPError as e:
    if e.code != 404:
      raise
    try:
      result = _post("/api/embeddings")
    except urllib.error.HTTPError as e2:
      if e2.code != 404:
        raise
      # OpenAI-compatible embeddings endpoint
      req = urllib.request.Request(
        f"{OLLAMA_API_URL}/v1/embeddings",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
      )
      with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
      result = json.loads(body)

  embedding = result.get("embedding")
  if not embedding and isinstance(result.get("embeddings"), list) and result["embeddings"]:
    embedding = result["embeddings"][0]
  if not embedding and isinstance(result.get("data"), list) and result["data"]:
    embedding = result["data"][0].get("embedding")
  if not embedding:
    raise RuntimeError("Ollama did not return an embedding. Check model name and endpoint.")
  return embedding


def get_text_embedding(text: str,
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using the configured backend."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  text = text.replace("\n", " ").strip()
  if USE_OLLAMA:
    embed_model = OLLAMA_EMBEDDING_MODEL if OLLAMA_EMBEDDING_MODEL else model
    try:
      return _ollama_embedding(text, embed_model, OLLAMA_TIMEOUT)
    except Exception as e:
      raise RuntimeError(f"Ollama embedding error: {e}") from e

  if not USE_OPENAI:
    raise RuntimeError("No embedding backend enabled.")
  if openai is None:
    raise RuntimeError("OpenAI package not installed.")

  response = openai.embeddings.create(
    input=[text], model=model).data[0].embedding
  return response





