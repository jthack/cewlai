#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import logging
import warnings
from abc import ABC, abstractmethod

import ollama
import tiktoken
import re


"""
(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\\.)+ matches one or more subdomain segments

Each segment must start and end with alphanumeric characters
Can optionally contain hyphens in the middle
Limited to 63 characters per segment (as per DNS rules)
Must end with a dot

[a-zA-Z]{2,} matches the top-level domain (TLD) with at least 2 characters
"""
DOMAIN_REGEX = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'

"""
Custom model config for whichever model the user selected to use for domain generation. If this dict is empty, the 
code assumes a set of defaults which should work.
"""
CUSTOM_MODEL_CONFIG: dict = {}


class LLM(ABC):
    """This abstract class defines the interface for LLMs. To support an LLM, implement the methods here."""

    @abstractmethod
    def __init__(self, default_model_config: dict):
        """Initializer with a default config dict, this is where you initialize the model"""

        if not default_model_config.get("model_name"):
            raise ValueError("default_model_config must contain model_name")

        self.default_model_config = CUSTOM_MODEL_CONFIG if len(CUSTOM_MODEL_CONFIG) > 0 else default_model_config
        self.model_name = self.default_model_config.pop("model_name")
        self.chat_history = []

    @abstractmethod
    def chat(self, prompt: str):
        """Defines how to send chat messages to the configured LLM"""
        pass


class Gemini(LLM):
    """Implementation for using Google Gemini as the subdomain generator"""

    def __init__(self, default_model_config: dict):
        import google.generativeai as genai
        super().__init__(default_model_config)
        warnings.filterwarnings('ignore')
        logging.getLogger().setLevel(logging.ERROR)
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        except KeyError as e:
            print("No Gemini API key set via a `GEMINI_API_KEY` env var. Please do so before using this model type.",
                  file=sys.stderr)
            sys.exit(1)

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.default_model_config,
        )
        self.chat_session = self.model.start_chat(history=self.chat_history)

    def chat(self, prompt: str):
        return self.chat_session.send_message(prompt).text


class OpenAI(LLM):
    """Implementation for using OpenAI as the subdomain generator"""

    def __init__(self, default_model_config: dict):
        import openai
        super().__init__(default_model_config)
        self.chat_history = [
            {"role": "system", "content": "You are tasked with thinking of similar domains."}
        ]
        try:
            self.model = openai.OpenAI()
        except openai.OpenAIError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    def chat(self, prompt: str):
        from openai import NOT_GIVEN
        self.chat_history.append({"role": "user", "content": prompt})
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=self.chat_history,
            temperature=self.default_model_config.get("temperature", NOT_GIVEN),
            top_p=self.default_model_config.get("top_p", NOT_GIVEN),
            max_tokens=self.default_model_config.get("max_tokens", NOT_GIVEN),
            stream=False,
        )
        self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


class Ollama(LLM):
    """Implementation for using a local Ollama instance as the subdomain generator"""

    def __init__(self, default_model_config: dict):
        super().__init__(default_model_config)

    def chat(self, prompt: str):
        from ollama import chat
        from ollama import ChatResponse
        self.chat_history += [
            {'role': 'user', 'content': prompt},
        ]
        response: ChatResponse = chat(
            model=self.model_name,
            messages=self.chat_history,
            options={
                'temperature': self.default_model_config.get("temperature", None),
                'top_p': self.default_model_config.get("top_p", None),
                'top_k': self.default_model_config.get("top_k", None),
            })
        self.chat_history += [
            {'role': 'assistant', 'content': response.message.content},
        ]
        return response.message.content


def parse_model_config(model_config_path: str):
    """
    Parses a model config file, if it exists, for use configuring the user provided LLM type.
    If no config values exist, opinionated defaults are used. If the user selected to use Ollama,
    an attempt is made to get the list of installed models and pick the first one. If that attempt fails
    an error will be thrown.
    """
    global CUSTOM_MODEL_CONFIG
    if model_config_path == ".cewlai-model-config":
        from pathlib import Path
        home = Path.home()
        model_config_path = home / model_config_path

    try:
        with open(model_config_path) as model_config:
            data = model_config.read()
            try:
               CUSTOM_MODEL_CONFIG = json.loads(data)
               print(f"Using the custom model config values: \n{json.dumps(CUSTOM_MODEL_CONFIG, indent=2)}")
            except json.decoder.JSONDecodeError as e:
                print(f"Could not parse model config file at {model_config_path} because: {e} \n ")
    except OSError as e:
        error_message = os.strerror(e.errno)
        print(f"Could not open model config file at {model_config_path} because: {error_message} \n ")

    if len(CUSTOM_MODEL_CONFIG) == 0:
        print(
            f'No model config values specified. Using opinionated defaults for known model types which can be found in the code.')


def parse_args():
    """
    Parses command-line arguments, providing a style similar to DNSCewl.
    We won't implement all the DNSCewl functionality here; just basic
    demonstration arguments plus the loop count & limit.
    """
    parser = argparse.ArgumentParser(
        description="DNS-like domain generation script with LLM integration."
    )

    # Mimic a few DNSCewl-style arguments
    parser.add_argument("-t", "--target", type=str,
                        help="Specify a single seed domain.")
    parser.add_argument("-tL", "--target-list", type=str,
                        help="Specify a file with seed domains (one per line).")
    parser.add_argument("--loop", type=int, default=1,
                        help="Number of times to call the LLM in sequence.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop once we exceed this many total domains (0 means no limit).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")
    parser.add_argument("--no-repeats", action="store_true",
                        help="Ensure no repeated domain structures in final output.")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file to write results to.")
    parser.add_argument("--force", action="store_true",
                        help="Skip token usage confirmation.")
    parser.add_argument("-m", "--model", type=str,
                        help="Specify a model to use. Supports `gemini`, `openai`,`ollama`",
                        default="gemini")
    parser.add_argument("-c", "--model-config", type=str,
                        help="Location of model config file. Uses ~/.cewlai-model-config by default.",
                        default=".cewlai-model-config")

    return parser.parse_args()


def get_seed_domains(args):
    """
    Retrieves initial seed domains from arguments (either -t, -tL, or stdin).
    """
    seed_domains = set()

    # Check if we have data on stdin
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                seed_domains.add(line)

    # Single target if provided
    if args.target:
        seed_domains.add(args.target.strip())

    # File-based targets
    if args.target_list:
        with open(args.target_list, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    seed_domains.add(line)

    return list(seed_domains)


def generate_new_domains(chat_session, domain_list, verbose=False):
    """
    Given an LLM chat session and a domain list, prompt the LLM to produce new domains.
    We randomize domain_list first, then craft a system prompt to get predicted variations.
    """
    try:
        # Randomize order of domains before sending to the model
        shuffled_domains = domain_list[:]
        random.shuffle(shuffled_domains)

        # Build the system / content prompt
        prompt_text = (
            "Here is a list of domains:\n"
            f"{', '.join(shuffled_domains)}\n\n"
            "It's your job to output unique new domains that are likely to exist "
            "based on variations or predictive patterns you see in the existing list. "
            "In your output, none of the domains should repeat. "
            "Please output them one domain per line. If they only pass in a root domain, still output potential subdomains."
        )

        if verbose:
            print("[DEBUG] Prompt to LLM:")
            print(prompt_text)
            print()

        raw_output = chat_session.chat(prompt=prompt_text).strip()

        new_candidates = re.findall(DOMAIN_REGEX, raw_output)
        return new_candidates

    except Exception as e:
        print(f"\n[!] Error during domain generation: {str(e)}", file=sys.stderr)
        return []


def estimate_tokens(domain_list):
    """
    Provides an accurate token count using tiktoken.
    Returns truncated domain list and token count.
    """
    MAX_TOKENS = 100000
    enc = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding as approximation

    # Build the prompt template without domains first
    prompt_template = (
        "Here is a list of domains:\n"
        "{domains}\n\n"
        "It's your job to output unique new domains that are likely to exist "
        "based on variations or predictive patterns you see in the existing list. "
        "In your output, none of the domains should repeat. "
        "Please output them one domain per line."
    )

    # Get base token count without domains
    base_tokens = len(enc.encode(prompt_template.format(domains="")))

    # Calculate how many domains we can include
    truncated_domains = []
    current_tokens = base_tokens

    for domain in domain_list:
        domain_tokens = len(enc.encode(domain + ", "))
        if current_tokens + domain_tokens > MAX_TOKENS:
            break
        truncated_domains.append(domain)
        current_tokens += domain_tokens

    # Calculate final token count with actual domains
    final_prompt = prompt_template.format(domains=", ".join(truncated_domains))
    total_tokens = len(enc.encode(final_prompt))

    return truncated_domains, total_tokens


def main():
    args = parse_args()

    parse_model_config(args.model_config)

    match (str(args.model).lower()):
        case "gemini":
            print("Using model " + args.model)
            llm_model = Gemini(
                {
                    "model_name": "gemini-1.5-flash",
                    "temperature": 1,
                    "top_p": 0.95,
                    "max_output_tokens": 8192,
                }
            )
        case "openai":
            print("Using model " + args.model)
            llm_model = OpenAI(
                {
                    "model_name": "gpt-4o",
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_tokens": 8192,
                    "response_mime_type": "text/plain",
                }
            )
        case "ollama":
            print("Using model " + args.model)
            ollama_models = ollama.list()
            if len(ollama_models.models) == 0:
                print("No Ollama models found. Please install one before using this model type.", file=sys.stderr)
                sys.exit(1)
            else:
                print(f"Defaulting to Ollama model: {ollama_models.models[0].model}")
            llm_model = Ollama(
                {
                    "model_name": ollama_models.models[0].model,
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
        case _:
            print("Model type must be either 'gemini','openai', or 'ollama'", file=sys.stderr)
            sys.exit(1)

    # Get initial domain list and check if using stdin
    using_stdin = not sys.stdin.isatty()
    seed_domains = get_seed_domains(args)
    if not seed_domains:
        print("[!] No seed domains provided. Use -t, -tL, or pipe domains to stdin.", file=sys.stderr)
        sys.exit(1)

    # Get token-truncated domain list and count
    seed_domains, estimated_tokens = estimate_tokens(seed_domains)
    estimated_total = estimated_tokens * args.loop

    if args.verbose:
        if len(seed_domains) < len(get_seed_domains(args)):
            print(f"\n[!] Input truncated to {len(seed_domains)} domains to stay under token limit")

    # Skip confirmation if using --force or stdin
    if not (args.force or using_stdin):
        print(f"\nEstimated token usage:")
        print(f"* Per iteration: ~{estimated_tokens} tokens")
        print(f"* Total for {args.loop} loops: ~{estimated_total} tokens")
        response = input("\nContinue? [y/N] ").lower()
        if response != 'y':
            print("Aborting.")
            sys.exit(0)
    elif args.verbose:
        print(f"\nEstimated token usage:")
        print(f"* Per iteration: ~{estimated_tokens} tokens")
        print(f"* Total for {args.loop} loops: ~{estimated_total} tokens")

    # We store all domains in a global set to avoid duplicates across loops
    all_domains = set(seed_domains)
    # Keep track of original domains to exclude from output
    original_domains = set(seed_domains)

    print("\nGenerating domains... This may take a minute or two depending on the number of iterations.")

    # Loop for the specified number of times
    for i in range(args.loop):
        if args.verbose:
            print(f"\n[+] LLM Generation Loop {i + 1}/{args.loop}...")

        new_domains = generate_new_domains(llm_model, list(all_domains), verbose=args.verbose)

        if args.no_repeats:
            # Filter out anything we already have
            new_domains = [d for d in new_domains if d not in all_domains]

        # Add them to our global set
        before_count = len(all_domains)
        for dom in new_domains:
            all_domains.add(dom)
        after_count = len(all_domains)

        if args.verbose:
            print(f"[DEBUG] LLM suggested {len(new_domains)} new domain(s). "
                  f"{after_count - before_count} were added (others were duplicates?).")

        # If we have a limit, check it now
        if 0 < args.limit <= len(all_domains):
            if args.verbose:
                print(f"[!] Reached limit of {args.limit} domains.")
            break


    # Get only the new domains (excluding original seed domains)
    new_domains = sorted(all_domains - original_domains)

    # Output handling
    if args.output:
        with open(args.output, 'w') as f:
            for dom in new_domains:
                f.write(f"{dom}\n")
        print(f"\nResults written to: {args.output}")
    else:
        print("\n=== New Generated Domains ===")
        for dom in new_domains:
            print(dom)

    if args.verbose:
        print(f"\n[DEBUG] Original domains: {len(original_domains)}")
        print(f"[DEBUG] New domains generated: {len(new_domains)}")
        print(f"[DEBUG] Total domains processed: {len(all_domains)}")

if __name__ == "__main__":
    main()
