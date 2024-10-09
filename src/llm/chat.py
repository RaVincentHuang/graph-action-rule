import os
import openai
client = openai.OpenAI()
import backoff

completion_tokens = prompt_tokens = 0

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

# def gpt_pro(prompt, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None):
#     # messages = [{"role": "user", "content": prompt}]
#     system_message = {"role": "system", "content": "There is no need to output process and thinking explanation, and the output operation formula can be strictly in accordance with the prompt. No need for preamble or closing, just need to answer according to the requirements, without any explanation for any exceptional cases in your response. Also, there is no need to include assumptions."}
#     user_message = {"role": "user", "content": prompt}
#     messages = [user_message, system_message]
#     return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    
    # messages = [{"role": "user", "content": prompt}]
    system_message = {"role": "system", "content": "There is no need to output process and thinking explanation, and the output operation formula can be strictly in accordance with the prompt. No need for preamble or closing, just need to answer according to the requirements, without any explanation for any exceptional cases in your response. Also, there is no need to include assumptions."}
    user_message = {"role": "user", "content": prompt}
    messages = [user_message, system_message]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="gpt-4o-mini", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4o-mini"):
    global completion_tokens, prompt_tokens
    cost = 0
    if backend == "gpt-4o-mini":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}
