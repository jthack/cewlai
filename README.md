# CewlAI

CewlAI is a domain generation tool that uses Google's Gemini AI to create potential domain variations based on seed domains. It's inspired by tools like CeWL but focuses on domain name pattern recognition and generation.

## Features

- Generate domain variations using AI pattern recognition
- Support for single domain or list of domains as input
- Control token usage and iteration count
- Output results to file or console
- Duplicate prevention
- Domain count limiting
- Verbose mode for debugging

## Prerequisites

- Python 3.x
- Google API key for Gemini AI

## Installation

1. Clone the repository:
   git clone https://github.com/jthack/cewlai.git
   cd cewlai

2. Install required packages:
   pip install google-generativeai

3. Set up your Google API key:
   export GEMINI_API_KEY='your-api-key-here'

## Usage

Basic usage:
python main.py -t example.com

Using a list of domains:
python main.py -tL domains.txt

Common options:
python main.py -tL domains.txt --loop 3 --limit 1000 -o output.txt

### Arguments

- -t, --target: Specify a single seed domain
- -tL, --target-list: Input file containing seed domains (one per line)
- --loop: Number of AI generation iterations (default: 1)
- --limit: Maximum number of domains to generate (0 = unlimited)
- -o, --output: Write results to specified file
- -v, --verbose: Enable verbose output
- --no-repeats: Prevent duplicate domains across iterations
- --force: Skip token usage confirmation

## Examples

Generate domains based on a single target:
python main.py -t example.com -o results.txt

Generate domains from a list with multiple iterations:
python main.py -tL company_domains.txt --loop 3 --limit 1000 -o generated_domains.txt

Verbose output with no repeats:
python main.py -t example.com -v --no-repeats

## Output

The tool will generate new domains based on patterns it recognizes in your seed domains. Output can be directed to:
- Console (default)
- File (using -o option)

Only newly generated domains are shown in the output (seed domains are excluded).

## Advanced Usage

### Token Management

The tool estimates token usage before running to help manage API costs. You'll see an estimation like:

Estimated token usage:
* Per iteration: ~150 tokens
* Total for 3 loops: ~450 tokens
Continue? [y/N]

Use the --force flag to skip this confirmation.

### Input File Format

When using -tL, your input file should contain one domain per line:
example.com
subdomain.example.com
another-example.com

### Output Format

The output is a simple list of generated domains, one per line:
api.example.com
dev.example.com
staging.example.com
test.example.com

### Verbose Output

Using -v provides detailed information about the generation process:
[+] LLM Generation Loop 1/3...
[DEBUG] LLM suggested 50 new domain(s). 45 were added (others were duplicates?)
[DEBUG] Original domains: 10
[DEBUG] New domains generated: 45
[DEBUG] Total domains processed: 55

## How It Works

1. Seed Collection: The tool takes your input domains as seeds
2. AI Analysis: Gemini AI analyzes patterns in the seed domains
3. Generation: New domains are generated based on recognized patterns
4. Filtering: Results are filtered to remove duplicates and invalid formats
5. Output: Unique, new domains are presented in the specified format

Remember that this tool is meant for legitimate security testing and research purposes only. 