#!/usr/bin/env python3

import argparse
import os
import random
import sys
import logging
import warnings
import tiktoken

import google.generativeai as genai

def configure_llm():
    """
    Configures Google Gemini Flash 1.5 (or any other Generative AI model).
    Adjust generation_config to match your needs.
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    return model.start_chat(history=[])

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
    parser.add_argument("-t", "--target", 
                        help="Specify a single seed domain.")
    parser.add_argument("-tL", "--target-list", 
                        help="Specify a file with seed domains (one per line).")
    parser.add_argument("--loop", type=int, default=1,
                        help="Number of times to call the LLM in sequence.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop once we exceed this many total domains (0 means no limit).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output.")
    parser.add_argument("--no-repeats", action="store_true",
                        help="Ensure no repeated domain structures in final output.")
    parser.add_argument("-o", "--output",
                        help="Output file to write results to.")
    parser.add_argument("--force", action="store_true",
                        help="Skip token usage confirmation.")
    
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
            "Please output them one domain per line. Only output the domains, no other text. No paths. No protocols. Output should be a list of domains."
        )

        if verbose:
            print("[DEBUG] Prompt to LLM:")
            print(prompt_text)
            print()

        response = chat_session.send_message(prompt_text)
        raw_output = response.text.strip()

        # Split the LLM output on newlines to guess domain lines
        new_candidates = [line.strip() for line in raw_output.split("\n") if line.strip()]

        # Basic quick filter to remove items that are not domain-like.
        filtered_candidates = []
        for dom in new_candidates:
            # Example heuristic: must contain at least one dot
            if "." in dom:
                filtered_candidates.append(dom)

        return filtered_candidates

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

    # Prepare the LLM chat session
    chat_session = configure_llm()

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
            print(f"\n[+] LLM Generation Loop {i+1}/{args.loop}...")

        new_domains = generate_new_domains(chat_session, list(all_domains), verbose=args.verbose)
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
        if args.limit > 0 and len(all_domains) >= args.limit:
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
