# CewlAI

<img width="758" alt="Screenshot 2025-01-24 at 9 51 33 PM" src="https://github.com/user-attachments/assets/a01d3ad7-733b-4822-b1e4-42b931194276" />

CewlAI is a domain generation tool that uses Google's Gemini AI to create potential domain variations based on seed domains. It's inspired by dnscewl which was inspired by CeWL, but this tool focuses on domain name pattern recognition and generation.

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
   `git clone https://github.com/jthack/cewlai.git`
   `cd cewlai`

2. Install required packages:
   `pip install -r requirements.txt`

3. Set up your Google API key:
   `export GEMINI_API_KEY='your-api-key-here'`

4. Set up OpenAI API key
   `export OPENAI_API_KEY='your-api-key-here'`

5. (Optional) Set up White Rabbit Neo API key:
   - Get your API key from [Kindo AI](https://app.kindo.ai/)
   `export KINDO_API_KEY='your-api-key-here'`

## Input Methods

The tool supports multiple ways to provide seed domains:

1. Single domain via command line:
   ```
   python main.py -t example.com
   ```

2. List of domains from a file:
   ```
   python main.py -tL domains.txt
   ```

3. Domains from stdin (pipe or redirect):
   ```
   cat domains.txt | python main.py
   # or
   echo "example.com" | python main.py
   ```

Note: When using stdin, the token usage confirmation is automatically skipped.


## Token Management

The tool automatically manages token usage to stay within API limits:

- Input is automatically truncated if it exceeds 100,000 tokens
- Use `-v` flag to see when truncation occurs
- Token usage estimates are shown before processing begins
- Use `--force` to skip the token usage confirmation prompt

Example output with truncation:

```
   $ cat large_domain_list.txt | python main.py -v
   
   [!] Input truncated to 15423 domains to stay under token limit
   
   Estimated token usage:
   * Per iteration: ~98750 tokens
   * Total for 1 loops: ~98750 tokens
   
   Continue? [y/N]
```

## Usage

Basic usage:
python main.py -t example.com

Using a list of domains:
python main.py -tL domains.txt

Common options:
python main.py -tL domains.txt --loop 3 --limit 1000 -o output.txt

### Arguments

```
-t, --target: Specify a single seed domain
-tL, --target-list: Input file containing seed domains (one per line)
--loop: Number of AI generation iterations (default: 1)
--limit: Maximum number of domains to generate (0 = unlimited)
-o, --output: Write results to specified file
-v, --verbose: Enable verbose output
--no-repeats: Prevent duplicate domains across iterations
--force: Skip token usage confirmation
--openai: Use OpenAI API instead of Gemini
-m, --model: Specify model to use (gemini, openai, whiterabbitneo, ollama)
```

## Models

### Gemini (Default)
The default model uses Google's Gemini AI. Requires `GEMINI_API_KEY` to be set.

### OpenAI
Use OpenAI's models by specifying `-m openai`. Requires `OPENAI_API_KEY` to be set.

### White Rabbit Neo
Use White Rabbit Neo by specifying `-m whiterabbitneo`. This model is powered by Kindo AI's WhiteRabbitNeo-33B-DeepSeekCoder model. Requires `KINDO_API_KEY` to be set.

Example usage:
```bash
export KINDO_API_KEY='your-api-key-here'
python main.py -m whiterabbitneo -t example.com
```

### Ollama
Use local Ollama models by specifying `-m ollama`. Requires Ollama to be installed with at least one model.

## Examples

Main use case (unix-way):
`cat domains.txt | python main.py`

Generate domains based on a single target:
`python main.py -t example.com -o results.txt`

Generate domains from a list with multiple iterations:
`python main.py -tL company_domains.txt --loop 3 --limit 1000 -o generated_domains.txt`

Verbose output with no repeats:
`python main.py -t example.com -v --no-repeats`

Using White Rabbit Neo with verbose output:
`python main.py -m whiterabbitneo -t example.com -v`

## Output

The tool will generate new domains based on patterns it recognizes in your seed domains. Output can be directed to:
- Console (default)
- File (using -o option)

Only newly generated domains are shown in the output (seed domains are excluded).

## Advanced Usage

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
```
[+] LLM Generation Loop 1/3...  
[DEBUG] LLM suggested 50 new domain(s). 45 were added (others were duplicates?)  
[DEBUG] Original domains: 10    
[DEBUG] New domains generated: 45  
[DEBUG] Total domains processed: 55  
```

## How It Works

1. Seed Collection: The tool takes your input domains as seeds
2. AI Analysis: The selected model (Gemini, OpenAI, White Rabbit Neo, or Ollama) analyzes patterns in the seed domains
3. Generation: New domains are generated based on recognized patterns
4. Filtering: Results are filtered to remove duplicates and invalid formats
5. Output: Unique, new domains are presented in the specified format

Remember that this tool is meant for legitimate security testing and research purposes only.