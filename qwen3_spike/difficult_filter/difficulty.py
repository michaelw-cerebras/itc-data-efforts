import json
import random
import asyncio
import aiofiles
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
import os
import argparse

async def vllm_call(messages, client, model, max_tokens, temperature):
    if isinstance(messages, list):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    elif isinstance(messages, str):
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": messages}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError("messages must be a str or a list of message dicts")
    return response.choices[0].message.content

async def meta_verify_mcq(response_1, response_2, client, model, max_tokens, temperature):
    response_1 = remove_think_section(response_1)
    response_2 = remove_think_section(response_2)

    messages = [
        {"role": "system", "content":
            "You will be given two responses to a multiple-choice question. "
            "Each response includes a justification and a selected answer. "
            "Your task is to determine whether both responses select the same final answer. "
            "Respond with [[1]] if both responses select the same final answer; otherwise [[0]]."
        },
        {"role": "user", "content": f"Response 1: {response_1}\n\nResponse 2: {response_2}"}
    ]

    result = await vllm_call(messages, client, model, max_tokens, temperature)
    return remove_think_section(result)

def remove_think_section(response):
    return response.split("</think>", 1)[1].strip() if "</think>" in response else response.strip()

async def process_example(example, client, semaphore, out_file):
    async with semaphore:
        try:
            attempt = remove_think_section(await vllm_call(
                example["problem"],
                client,
                model="Qwen/Qwen3-8B",
                max_tokens=30000,  # No token limit for generation
                temperature=0.6
            ))

            ground_truth = example["answer"]

            rating = await meta_verify_mcq(
                ground_truth,
                attempt,
                client,
                model="Qwen/Qwen3-8B",
                max_tokens=8192,  # Cap for rating
                temperature=0.6
            )

            result = {
                "problem_id": example["problem_id"],
                "problem": example["problem"],
                "attempt": attempt,
                "ground_truth": ground_truth,
                "rating": rating
            }

            async with aiofiles.open(out_file, mode="a", encoding="utf-8") as f:
                await f.write(json.dumps(result, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Error processing problem_id={example.get('problem_id')}: {e}")

async def main(args):
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    # Proper client setup
    baseline_client = AsyncOpenAI(
        api_key="serving-on-vllm",
        base_url="http://127.0.0.1:8190/v1",
        timeout=None
    )

    with open(args.input_file, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    sampled_data = random.sample(data, args.sample_size)
    sem = asyncio.Semaphore(args.concurrency)

    await tqdm.gather(
        *[process_example(example, baseline_client, sem, args.out_file) for example in sampled_data],
        desc="Running LLM and rating"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, required=True, help="Number of examples to sample")
    parser.add_argument("--out_file", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON")
    parser.add_argument("--concurrency", type=int, required=True, help="Number of concurrent tasks")
    args = parser.parse_args()
    asyncio.run(main(args))
