import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set page configuration
st.set_page_config(page_title="LLM Hosting vs API Breakeven Calculator", layout="wide")

# App title and description
st.title("LLM Hosting vs API Breakeven Calculator")
st.markdown("""
This app calculates the breakeven point between self-hosting an LLM and using an API service.
Enter your hosting costs, API token costs, and usage patterns to see when self-hosting becomes cost-effective.
""")

# Create two columns for input
col1, col2 = st.columns(2)

# Self-hosting inputs
with col1:
    st.header("Self-Hosting Costs")
    hosting_cost = st.number_input("Monthly Hosting Cost ($)", min_value=0.0, value=1000.0, step=100.0)
    start_date = st.date_input("Start Date", datetime.date.today())
    end_date = st.date_input("End Date", datetime.date.today() + datetime.timedelta(days=365))
    
    # Calculate total days and convert monthly cost to daily then total
    total_days = (end_date - start_date).days
    if total_days <= 0:
        st.error("End date must be after start date")
        total_days = 1
    
    daily_hosting_cost = hosting_cost / 30  # Approximate daily cost
    total_hosting_cost = daily_hosting_cost * total_days

    st.info(f"Total Hosting Cost for {total_days} days: ${total_hosting_cost:.2f}")

# API costs inputs
with col2:
    st.header("API Costs")
    input_token_cost = st.number_input("Input Token Cost ($ per 1M tokens)", min_value=0.0, value=1.0, step=0.1)
    output_token_cost = st.number_input("Output Token Cost ($ per 1M tokens)", min_value=0.0, value=2.0, step=0.1)
    
    # Convert costs to per-token
    input_token_cost_per_token = input_token_cost / 1_000_000
    output_token_cost_per_token = output_token_cost / 1_000_000
    
    # Token ratio
    st.subheader("Token Ratio (Input:Output)")
    input_ratio = st.number_input("Input Ratio", min_value=1, value=1, step=1)
    output_ratio = st.number_input("Output Ratio", min_value=1, value=3, step=1)
    
    # Calculate average cost per request based on the ratio
    total_ratio = input_ratio + output_ratio
    avg_input_tokens_per_request = input_ratio / total_ratio
    avg_output_tokens_per_request = output_ratio / total_ratio
    
    st.info(f"Token Ratio: {input_ratio}:{output_ratio}")

# Calculate breakeven
total_cost_per_token = (input_token_cost_per_token * avg_input_tokens_per_request) + \
                       (output_token_cost_per_token * avg_output_tokens_per_request)

if total_cost_per_token > 0:
    breakeven_tokens = total_hosting_cost / total_cost_per_token
    breakeven_tokens_millions = breakeven_tokens / 1_000_000
else:
    breakeven_tokens = float('inf')
    breakeven_tokens_millions = float('inf')

# Display results
st.header("Breakeven Analysis")
st.subheader("Results")

col_results1, col_results2 = st.columns(2)

with col_results1:
    st.metric("Breakeven Point (Tokens)", f"{breakeven_tokens:,.0f}")
    st.metric("Breakeven Point (Million Tokens)", f"{breakeven_tokens_millions:.2f}M")
    
    # Calculate daily tokens needed
    if total_days > 0:
        daily_tokens_needed = breakeven_tokens / total_days
        st.metric("Daily Tokens Needed", f"{daily_tokens_needed:,.0f}")
    
with col_results2:
    st.metric("Total Hosting Cost", f"${total_hosting_cost:.2f}")
    
    # Example costs at different usage levels
    example_tokens = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
    for tokens in example_tokens:
        api_cost = tokens * total_cost_per_token
        st.metric(f"API Cost for {tokens/1_000_000:.0f}M tokens", f"${api_cost:.2f}")

# Create visualization
st.header("Cost Comparison Visualization")

# Generate token volume range
max_tokens_to_show = max(breakeven_tokens * 2, 10_000_000)  # Show at least up to 10M tokens
token_volumes = np.linspace(0, max_tokens_to_show, 100)
api_costs = [vol * total_cost_per_token for vol in token_volumes]
hosting_costs = [total_hosting_cost for _ in token_volumes]

# Convert to millions for better readability
token_volumes_millions = token_volumes / 1_000_000

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(token_volumes_millions, api_costs, label="API Cost", linewidth=2)
ax.plot(token_volumes_millions, hosting_costs, label="Hosting Cost", linewidth=2)

# Mark the breakeven point
if not np.isinf(breakeven_tokens_millions):
    breakeven_cost = total_hosting_cost
    ax.scatter([breakeven_tokens_millions], [breakeven_cost], color='red', s=100, zorder=5)
    ax.annotate(f'Breakeven: {breakeven_tokens_millions:.2f}M tokens',
                xy=(breakeven_tokens_millions, breakeven_cost),
                xytext=(breakeven_tokens_millions + max_tokens_to_show/1_000_000/20, 
                        breakeven_cost + total_hosting_cost/10),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'))

# Format y-axis as dollars
def currency_formatter(x, pos):
    return f'${x:,.0f}'

ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

# Add labels and legend
ax.set_xlabel('Token Volume (Millions)')
ax.set_ylabel('Cost ($)')
ax.set_title('API vs. Self-Hosting Cost Comparison')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# Additional information
st.header("Detailed Analysis")

# Create a table showing costs at different token volumes
token_ranges = [
    1_000, 10_000, 100_000, 
    1_000_000, 5_000_000, 10_000_000, 
    50_000_000, 100_000_000, 500_000_000, 
    1_000_000_000
]

data = []
for tokens in token_ranges:
    api_cost = tokens * total_cost_per_token
    diff = total_hosting_cost - api_cost
    cheaper = "API" if api_cost < total_hosting_cost else "Self-hosting" if api_cost > total_hosting_cost else "Equal"
    data.append({
        "Token Volume": f"{tokens:,}",
        "API Cost": f"${api_cost:.2f}",
        "Hosting Cost": f"${total_hosting_cost:.2f}",
        "Difference": f"${abs(diff):.2f}",
        "Cheaper Option": cheaper
    })

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

# Add a download button for the comparison data
csv = df.to_csv(index=False)
st.download_button(
    label="Download Comparison Data as CSV",
    data=csv,
    file_name="llm_cost_comparison.csv",
    mime="text/csv",
)

# Show the formula used
st.header("Calculation Details")
st.markdown("""
### Formulas Used:
- **Total Hosting Cost** = Monthly Hosting Cost ÷ 30 × Number of Days
- **Input Token Cost per Token** = Input Token Cost per 1M ÷ 1,000,000
- **Output Token Cost per Token** = Output Token Cost per 1M ÷ 1,000,000
- **Average Cost per Token** = (Input Token Cost × Input Ratio Percentage) + (Output Token Cost × Output Ratio Percentage)
- **Breakeven Point** = Total Hosting Cost ÷ Average Cost per Token

### Understanding the Results:
- If your expected token volume is below the breakeven point, using the API is more cost-effective.
- If your expected token volume is above the breakeven point, self-hosting is more cost-effective.
- The daily tokens needed shows how many tokens you would need to process each day to reach the breakeven point during your specified time period.
""")