import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import math

# Set page configuration
st.set_page_config(page_title="LLM Hosting vs API Breakeven Calculator", layout="wide")

# App title and description
st.title("LLM Hosting vs API Breakeven Calculator")
st.markdown("""
This app calculates the breakeven point between self-hosting an LLM and using an API service.
Select your model, API provider, and hosting options to see when self-hosting becomes cost-effective.
""")

# Load data from CSV files
@st.cache_data
def load_data():
    api_pricing = pd.read_csv('data/api_pricing.csv')
    groq_data = pd.read_csv('data/groq.csv')
    runpod_data = pd.read_csv('data/runpod.csv')
    return api_pricing, groq_data, runpod_data

try:
    api_pricing, groq_data, runpod_data = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.error("Please ensure the data files are in the correct location: data/api_pricing.csv, data/groq.csv, data/runpod.csv")
    data_loaded = False

if data_loaded:
    # Combine API pricing data from both sources
    api_providers = api_pricing.copy()
    api_providers['Source'] = 'Official API'
    
    # Rename columns in groq_data to match api_pricing
    groq_formatted = groq_data.rename(columns={
        'AI Model': 'Model',
        'Input Token Price (Per Million Tokens)': 'Input/1M Tokens',
        'Output Token Price (Per Million Tokens)': 'Output/1M Tokens'
    })
    groq_formatted['Provider'] = 'Groq'
    groq_formatted['Context'] = 'Varies'  # Add a placeholder for context
    groq_formatted['Source'] = 'Groq'
    
    # Ensure numeric columns are properly formatted
    groq_formatted['Input/1M Tokens'] = groq_formatted['Input/1M Tokens'].astype(str).str.replace('$', '').astype(float)
    groq_formatted['Output/1M Tokens'] = groq_formatted['Output/1M Tokens'].astype(str).str.replace('$', '').astype(float)
    api_providers['Input/1M Tokens'] = api_providers['Input/1M Tokens'].astype(str).str.replace('$', '').astype(float)
    api_providers['Output/1M Tokens'] = api_providers['Output/1M Tokens'].astype(str).str.replace('$', '').astype(float)
    
    # Combine the dataframes
    combined_api = pd.concat([api_providers, groq_formatted[api_providers.columns]], ignore_index=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["API Selection", "Self-Hosting Configuration", "Breakeven Analysis"])
    
    with tab1:
        st.header("Select API Provider and Model")
        
        # Provider selection
        provider_list = sorted(combined_api['Provider'].unique())
        selected_provider = st.selectbox("Select API Provider", provider_list)
        
        # Filter models by provider
        provider_models = combined_api[combined_api['Provider'] == selected_provider]
        selected_model = st.selectbox("Select Model", provider_models['Model'])
        
        # Get pricing for selected model
        model_pricing = provider_models[provider_models['Model'] == selected_model].iloc[0]
        
        # Display selected model information
        st.subheader("Selected Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", selected_model)
            st.metric("Provider", selected_provider)
        with col2:
            st.metric("Input Cost per 1M Tokens", f"${model_pricing['Input/1M Tokens']}")
            st.metric("Output Cost per 1M Tokens", f"${model_pricing['Output/1M Tokens']}")
        with col3:
            st.metric("Context Length", model_pricing['Context'])
            
        # Token ratio for API calls
        st.subheader("Token Ratio (Input:Output)")
        col1, col2 = st.columns(2)
        with col1:
            input_ratio = st.number_input("Input Ratio", min_value=1, value=1, step=1, key="api_input_ratio")
        with col2:
            output_ratio = st.number_input("Output Ratio", min_value=1, value=3, step=1, key="api_output_ratio")
        
        # Calculate API costs
        input_token_cost = model_pricing['Input/1M Tokens']
        output_token_cost = model_pricing['Output/1M Tokens']
        
        # Convert costs to per-token
        input_token_cost_per_token = input_token_cost / 1_000_000
        output_token_cost_per_token = output_token_cost / 1_000_000
        
        # Calculate average cost per request based on the ratio
        total_ratio = input_ratio + output_ratio
        avg_input_tokens_per_request = input_ratio / total_ratio
        avg_output_tokens_per_request = output_ratio / total_ratio
        
        # Calculate weighted cost per token
        api_cost_per_token = (input_token_cost_per_token * avg_input_tokens_per_request) + \
                           (output_token_cost_per_token * avg_output_tokens_per_request)
        
        st.metric("Effective Cost per Token", f"${api_cost_per_token:.10f}")
        st.metric("Cost per 1M Tokens", f"${api_cost_per_token * 1_000_000:.2f}")
    
    with tab2:
        st.header("Self-Hosting Configuration")
        
        # LLM Model Selection for self-hosting
        st.subheader("Select Model to Self-Host")
        
        # Model family selection (This would be expanded with actual model families)
        model_families = ["Llama 3", "Llama 3.1", "Mistral", "Mixtral", "Phi-3", "Other"]
        selected_family = st.selectbox("Model Family", model_families)
        
        # Parameter sizes
        param_sizes = [3, 7, 8, 13, 14, 32, 34, 70, 72]
        param_size = st.selectbox("Parameter Size (billions)", param_sizes)
        
        # Quantization options
        quant_options = [
            {"name": "FP16 (16-bit)", "bits": 16},
            {"name": "FP8 (8-bit)", "bits": 8},
            {"name": "Int4 (4-bit)", "bits": 4}
        ]
        quant_names = [q["name"] for q in quant_options]
        selected_quant = st.selectbox("Quantization", quant_names)
        selected_quant_bits = [q["bits"] for q in quant_options if q["name"] == selected_quant][0]
        
        # Calculate required VRAM
        def calculate_vram(params_billions, bits, overhead=1.2):
            params = params_billions * 1_000_000_000
            bytes_per_param = 4  # 4 bytes for FP32 reference
            compression_ratio = 32 / bits  # How much compression we get
            
            vram_bytes = (params * bytes_per_param) / compression_ratio * overhead
            vram_gb = vram_bytes / (1024 * 1024 * 1024)
            return vram_gb
        
        required_vram = calculate_vram(param_size, selected_quant_bits)
        st.metric("Required VRAM (GB)", f"{required_vram:.2f}")
        
        # Find suitable GPUs from RunPod
        st.subheader("Available GPU Options")
        
        # Prepare RunPod data
        runpod_data['VRAM'] = pd.to_numeric(runpod_data['VRAM'].str.replace(' GB', ''))
        runpod_data['Cost/hr'] = pd.to_numeric(runpod_data['Cost/hr'].str.replace('$', ''))
        runpod_data['Minimum Cost/hr'] = pd.to_numeric(runpod_data['Minimum Cost/hr'].str.replace('$', ''))
        
        # Sort by cost
        runpod_data = runpod_data.sort_values('Cost/hr')
        
        # Find suitable GPUs
        suitable_gpus = runpod_data[runpod_data['VRAM'] >= required_vram]
        
        if not suitable_gpus.empty:
            st.write("Single GPU options that can host this model:")
            st.dataframe(suitable_gpus)
        else:
            st.warning("No single GPU has enough VRAM. Let's check multi-GPU options.")
        
        # Multi-GPU options
        st.subheader("Multi-GPU Options")
        
        # Function to find optimal GPU combinations
        def find_gpu_combinations(required_vram, gpu_data, max_gpus=4):
            combinations = []
            
            for num_gpus in range(2, max_gpus + 1):
                # For each GPU type
                for _, gpu in gpu_data.iterrows():
                    if gpu['VRAM'] * num_gpus >= required_vram:
                        combinations.append({
                            'GPU': gpu['GPU Name'],
                            'Count': num_gpus,
                            'Total VRAM': gpu['VRAM'] * num_gpus,
                            'Cost/hr each': gpu['Cost/hr'],
                            'Total Cost/hr': gpu['Cost/hr'] * num_gpus,
                            'Utilization': required_vram / (gpu['VRAM'] * num_gpus) * 100
                        })
            
            # Sort by total cost
            combinations_df = pd.DataFrame(combinations)
            if not combinations_df.empty:
                combinations_df = combinations_df.sort_values('Total Cost/hr')
            
            return combinations_df
        
        gpu_combinations = find_gpu_combinations(required_vram, runpod_data)
        
        if not gpu_combinations.empty:
            st.write("Multi-GPU options:")
            st.dataframe(gpu_combinations)
        elif suitable_gpus.empty:
            st.error("No suitable GPU configuration found for this model with the given quantization.")
            st.info("Consider using a higher level of quantization to reduce VRAM requirements.")
        
        # Select hosting option
        st.subheader("Select Hosting Option")
        
        # Combine single and multi-GPU options for selection
        all_options = []
        
        if not suitable_gpus.empty:
            for _, gpu in suitable_gpus.iterrows():
                all_options.append({
                    'Option': f"1x {gpu['GPU Name']} ({gpu['VRAM']} GB)",
                    'Cost/hr': gpu['Cost/hr'],
                    'VRAM': gpu['VRAM'],
                    'Utilization': required_vram / gpu['VRAM'] * 100
                })
        
        if not gpu_combinations.empty:
            for _, combo in gpu_combinations.iterrows():
                all_options.append({
                    'Option': f"{int(combo['Count'])}x {combo['GPU']} ({combo['Total VRAM']} GB total)",
                    'Cost/hr': combo['Total Cost/hr'],
                    'VRAM': combo['Total VRAM'],
                    'Utilization': combo['Utilization']
                })
        
        # Create DataFrame for options
        options_df = pd.DataFrame(all_options)
        
        if not options_df.empty:
            # Format the dataframe for display
            display_df = options_df.copy()
            display_df['Utilization'] = display_df['Utilization'].apply(lambda x: f"{x:.1f}%")
            display_df['Cost/hr'] = display_df['Cost/hr'].apply(lambda x: f"${x:.2f}")
            
            # Selection
            option_list = options_df['Option'].tolist()
            selected_option = st.selectbox("Select GPU Configuration", option_list)
            
            # Get the selected option details
            selected_config = options_df[options_df['Option'] == selected_option].iloc[0]
            hourly_cost = selected_config['Cost/hr']
            
            # Time period for hosting
            st.subheader("Hosting Time Period")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", datetime.date.today())
            with col2:
                end_date = st.date_input("End Date", datetime.date.today() + datetime.timedelta(days=30))
            
            # Calculate total days and cost
            total_days = (end_date - start_date).days
            if total_days <= 0:
                st.error("End date must be after start date")
                total_days = 1
            
            daily_hosting_cost = hourly_cost * 24  # Cost per day
            total_hosting_cost = daily_hosting_cost * total_days
            
            st.metric("Hourly Hosting Cost", f"${hourly_cost:.2f}")
            st.metric("Daily Hosting Cost", f"${daily_hosting_cost:.2f}")
            st.metric("Total Hosting Cost", f"${total_hosting_cost:.2f}")
            
        else:
            st.error("No suitable hosting options available for this model configuration.")
            total_hosting_cost = 0
    
    with tab3:
        st.header("Breakeven Analysis")
        
        if 'total_hosting_cost' in locals() and total_hosting_cost > 0 and 'api_cost_per_token' in locals():
            # Calculate breakeven point
            if api_cost_per_token > 0:
                breakeven_tokens = total_hosting_cost / api_cost_per_token
                breakeven_tokens_millions = breakeven_tokens / 1_000_000
            else:
                breakeven_tokens = float('inf')
                breakeven_tokens_millions = float('inf')
            
            # Display results
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
                st.metric("API Provider", selected_provider)
                st.metric("API Model", selected_model)
                
            # Create visualization
            st.subheader("Cost Comparison Visualization")
            
            # Generate token volume range
            max_tokens_to_show = max(breakeven_tokens * 2, 10_000_000)  # Show at least up to 10M tokens
            token_volumes = np.linspace(0, max_tokens_to_show, 100)
            api_costs = [vol * api_cost_per_token for vol in token_volumes]
            hosting_costs = [total_hosting_cost for _ in token_volumes]
            
            # Convert to millions for better readability
            token_volumes_millions = token_volumes / 1_000_000
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(token_volumes_millions, api_costs, label=f"{selected_provider} API Cost", linewidth=2)
            ax.plot(token_volumes_millions, hosting_costs, label="Self-Hosting Cost", linewidth=2)
            
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
                api_cost = tokens * api_cost_per_token
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
            
            # Return on Investment Analysis
            st.subheader("Return on Investment Analysis")
            
            # Calculate how long it takes to recoup costs at different daily usage levels
            daily_usage_levels = [
                100_000,  # 100K tokens/day
                1_000_000,  # 1M tokens/day
                10_000_000,  # 10M tokens/day
                100_000_000  # 100M tokens/day
            ]
            
            roi_data = []
            for daily_usage in daily_usage_levels:
                daily_api_cost = daily_usage * api_cost_per_token
                days_to_breakeven = total_hosting_cost / daily_api_cost if daily_api_cost > 0 else float('inf')
                
                roi_data.append({
                    "Daily Usage": f"{daily_usage:,} tokens",
                    "Daily API Cost": f"${daily_api_cost:.2f}",
                    "Days to Breakeven": f"{days_to_breakeven:.1f}" if not np.isinf(days_to_breakeven) else "Never",
                    "Months to Breakeven": f"{days_to_breakeven/30:.1f}" if not np.isinf(days_to_breakeven) else "Never"
                })
            
            st.dataframe(pd.DataFrame(roi_data), use_container_width=True)
            
        else:
            st.warning("Please complete the API Selection and Self-Hosting Configuration tabs first.")
else:
    st.warning("Please check that your data files are correctly formatted and located in the data directory.")