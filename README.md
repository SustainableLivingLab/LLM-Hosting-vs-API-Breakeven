# LLM Cost Breakeven Calculator

A Streamlit application that helps you determine the most cost-effective approach between self-hosting a Large Language Model (LLM) or using a third-party API service.

## Overview

This calculator allows you to compare the costs of:
- Self-hosting an LLM on your own infrastructure
- Using a commercial LLM API service (like OpenAI, Anthropic, etc.)

By inputting your specific costs and usage patterns, the application calculates the exact breakeven point where both options cost the same, helping you make data-driven decisions for your AI infrastructure.

## Features

- **Self-hosting cost calculation**: Input monthly costs and time period
- **API token pricing**: Configure both input and output token costs
- **Token ratio customization**: Set the typical input:output token ratio for your use case
- **Visual breakeven analysis**: Interactive graph showing cost comparison
- **Detailed cost breakdown**: Table showing costs at different usage levels
- **Export capabilities**: Download cost comparison data as CSV

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-cost-breakeven-calculator.git
   cd llm-cost-breakeven-calculator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter your monthly hosting cost and the time period for analysis
2. Input the API token costs (per million tokens) for both input and output
3. Set your typical input:output token ratio
4. View the breakeven point and cost comparison visualization
5. Use the detailed table to see costs at different usage levels
6. Download the data for further analysis if needed

## Example

- If your monthly hosting cost is $1000 over a year
- API costs are $1.00 per million input tokens and $2.00 per million output tokens
- Your typical usage pattern is 1:3 (input:output tokens)
- The calculator will show you exactly how many tokens you need to process before self-hosting becomes more economical
