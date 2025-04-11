import streamlit as st  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np  
from datetime import datetime, timedelta  
import io  
from matplotlib.colors import ListedColormap  

# Define the function to convert Excel date numbers to datetime  
def excel_date_to_datetime(excel_date):  
    return datetime(1899, 12, 30) + timedelta(days=int(excel_date))  
 
st.title("Mod Entry Time Analysis")  
st.write("Drag and drop your CSV file to visualize performance by day of week and time.")
st.write("FOMC and CPI days are automatically filtered out of the analysis.")  
  
# File uploader widget for drag and drop  
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])  
  
if uploaded_file is not None:  
    try:  
        # Read the CSV file into a DataFrame  
        df = pd.read_csv(uploaded_file)  
        st.success("File uploaded successfully!")

        # Only include rows where 'Legs' contains 'STO'
        df = df[df['Legs'].str.contains('STO', na=False)] 

        # Load special days
        df_special = pd.read_csv('Special days.csv', encoding='UTF-8-SIG')  
        special_dates = [excel_date_to_datetime(date) for date in df_special['Date']]
        #df_special  
       
        # Convert 'Date Opened' to datetime 
        df['Date Opened'] = pd.to_datetime(df['Date Opened'])  
        
        # Filter out CPI and FOMC days
        # if st.checkbox("Remove CPI and FOMC Days"):
        df = df[~df['Date Opened'].isin(special_dates)]

        # Optionally display the raw data  
        if st.checkbox("Show raw data"):  
            st.dataframe(df.head())  
        
        # Add filtering based on 'Strategy' column:  
        # Get the sorted unique strategies for the multiselect widget  
        unique_strategies = sorted(df['Strategy'].unique())  
        selected_strategies = st.multiselect('Select Strategies', unique_strategies, default=unique_strategies)  
        
        # Filter the DataFrame based on selected strategies  
        df_filtered = df[df['Strategy'].isin(selected_strategies)]  
        
        # Extract day of week and hour from 'Date Opened'  
        df_filtered['Day of Week'] = df_filtered['Date Opened'].dt.day_name()  
        df_filtered['Hour'] = df_filtered['Date Opened'].dt.hour  
        
        # Create a custom order for days of the week  
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  
        
        # Filter for only the last 365 days  
        cutoff_date_365d = pd.Timestamp.now() - pd.Timedelta(days=365)  
        df_365d = df_filtered[df_filtered['Date Opened'] >= cutoff_date_365d]  
        
        # Filter for only the last 90 days  
        cutoff_date_90d = pd.Timestamp.now() - pd.Timedelta(days=90)  
        df_90d = df_filtered[df_filtered['Date Opened'] >= cutoff_date_90d]  
        
        # Radio button to select metric  
        metric = st.radio('Select Metric', ['P/L', 'PCR'])  
        
        # Add radio button for mask selection
        mask_type = st.radio('Select Mask Type', ['Highlight Above Average', 'Highlight Positive Values'])
        
        if len(df_365d) > 0:  
            # Create pivot tables based on selected metric  
            if metric == 'P/L':  
                pivot_365d = pd.pivot_table(df_365d, values='P/L', index='Day of Week', columns='Hour', aggfunc='mean')  
                pivot_90d = pd.pivot_table(df_90d, values='P/L', index='Day of Week', columns='Hour', aggfunc='mean') if len(df_90d) > 0 else None  
            else:  # PCR  
                # Calculate PCR for each trade  
                df_365d['PCR'] = df_365d['P/L'] / df_365d['Premium Collected']  
                pivot_365d = pd.pivot_table(df_365d, values='PCR', index='Day of Week', columns='Hour', aggfunc='mean')  
                
                if len(df_90d) > 0:  
                    df_90d['PCR'] = df_90d['P/L'] / df_90d['Premium Collected']  
                    pivot_90d = pd.pivot_table(df_90d, values='PCR', index='Day of Week', columns='Hour', aggfunc='mean')  
                else:  
                    pivot_90d = None  
            
            # Reorder the index to match the custom day order  
            pivot_365d = pivot_365d.reindex(day_order)  
            if pivot_90d is not None:  
                pivot_90d = pivot_90d.reindex(day_order)  
            
            # Create a figure and axes for the heatmap  
            fig, ax = plt.subplots(figsize=(12, 8))  
            
            # Set up the plot style
            plt.style.use('default')
            plt.rcParams['font.family'] = ['Inter']
            plt.rcParams['font.sans-serif'] = ['Inter']
            
            # Create a mask for the heatmap based on selection
            if mask_type == 'Highlight Above Average':
                # Original mask: highlight cells above column average
                mask = pivot_365d.gt(pivot_365d.mean())
                # Create a custom colormap (white for below average, green for above average)  
                colors = ['white', '#24EB84']  
            else:  # 'Highlight Positive Values'
                # New mask: highlight positive values
                mask = pivot_365d > 0
                # Create a custom colormap (white for negative/zero, red for positive)  
                colors = ['white', '#EB3424']  
            
            cmap = ListedColormap(colors)  
            
            # Plot the heatmap with the mask  
            sns.heatmap(pivot_365d, cmap=cmap, mask=~mask, annot=True, fmt='.2f', cbar=False, ax=ax)  
            
            # Set title and labels with proper styling
            ax.set_title(f'365-Day {metric} by Day and Hour', fontsize=20, fontweight='semibold', color='#171717', pad=15)
            ax.set_xlabel('Hour of Day', fontsize=16, fontweight='medium', color='#171717', labelpad=10)
            ax.set_ylabel('Day of Week', fontsize=16, fontweight='medium', color='#171717', labelpad=10)
            
            # Style the tick labels
            ax.tick_params(axis='both', which='major', labelsize=14, colors='#171717')
            
            # Set spines and grid
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            ax.set_axisbelow(True)
            
            # Save the figure to a BytesIO object  
            buf = io.BytesIO()  
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)  
            buf.seek(0)  
            
            # Display the heatmap in Streamlit  
            st.pyplot(fig)  
            
            # Add a download button for the heatmap  
            st.download_button(  
                label="Download Heatmap",  
                data=buf,  
                file_name=f"heatmap_{metric}.png",  
                mime="image/png"  
            )  
            
            # Display the pivot table data  
            st.subheader(f"365-Day {metric} Data")  
            st.dataframe(pivot_365d)  
            
            # Display 90-day data if available  
            if pivot_90d is not None and len(pivot_90d) > 0:  
                st.subheader(f"90-Day {metric} Data")  
                st.dataframe(pivot_90d)  
                
                # Create a figure and axes for the 90-day heatmap  
                fig2, ax2 = plt.subplots(figsize=(12, 8))  
                
                # Set up the plot style
                plt.style.use('default')
                plt.rcParams['font.family'] = ['Inter']
                plt.rcParams['font.sans-serif'] = ['Inter']
                
                # Create a mask for the 90-day heatmap based on selection
                if mask_type == 'Highlight Above Average':
                    # Original mask: highlight cells above column average
                    mask_90d = pivot_90d.gt(pivot_90d.mean())
                    # Use the same green colormap as for 365d
                    colors_90d = ['white', '#24EB84']
                else:  # 'Highlight Positive Values'
                    # New mask: highlight positive values
                    mask_90d = pivot_90d > 0
                    # Use red for positive values
                    colors_90d = ['white', '#EB3424']
                
                cmap_90d = ListedColormap(colors_90d)
                
                # Plot the 90-day heatmap with the mask  
                sns.heatmap(pivot_90d, cmap=cmap_90d, mask=~mask_90d, annot=True, fmt='.2f', cbar=False, ax=ax2)  
                
                # Set title and labels with proper styling
                ax2.set_title(f'90-Day {metric} by Day and Hour', fontsize=20, fontweight='semibold', color='#171717', pad=15)
                ax2.set_xlabel('Hour of Day', fontsize=16, fontweight='medium', color='#171717', labelpad=10)
                ax2.set_ylabel('Day of Week', fontsize=16, fontweight='medium', color='#171717', labelpad=10)
                
                # Style the tick labels
                ax2.tick_params(axis='both', which='major', labelsize=14, colors='#171717')
                
                # Set spines and grid
                for spine in ax2.spines.values():
                    spine.set_visible(False)
                
                ax2.set_axisbelow(True)
                
                # Display the 90-day heatmap  
                st.pyplot(fig2)  
                
                # Add a download button for the 90-day heatmap  
                buf2 = io.BytesIO()  
                plt.savefig(buf2, format='png', bbox_inches='tight', dpi=300)  
                buf2.seek(0)  
                
                st.download_button(  
                    label="Download 90-Day Heatmap",  
                    data=buf2,  
                    file_name=f"heatmap_90d_{metric}.png",  
                    mime="image/png"  
                )  
            
            # Create a table of recent performance by time of day  
            if len(df_90d) > 0:  
                # Group by hour and calculate metrics  
                if metric == 'P/L':  
                    times_df = df_90d.groupby('Hour')['P/L'].agg(['mean', 'count']).reset_index()  
                    times_df.columns = ['Hour', 'Avg P/L', 'Trade Count']  
                    times_df = times_df.sort_values('Avg P/L', ascending=False)  
                else:  # PCR  
                    times_df = df_90d.groupby('Hour')['PCR'].agg(['mean', 'count']).reset_index()  
                    times_df.columns = ['Hour', 'Avg PCR', 'Trade Count']  
                    times_df = times_df.sort_values('Avg PCR', ascending=False)  
                
                # Display the table  
                st.title("Trending Entry Times") 
                st.dataframe(times_df)        
            
            else:  
                st.warning("No data found within the last 90 days. Cannot create recent performance table.")  
                  
    except Exception as e:  
        st.error("Error processing file: " + str(e))  
else:  
    st.info("Please upload a CSV file to begin the analysis.")  
  
# Sidebar with additional information  
with st.sidebar:  
    st.header("About This App")  
    st.write(  
        "This application analyzes OO backtest trade logs to visualize performance patterns "  
        "by day of the week and time of day.\n\n"  
        "The heatmap displays both overall historical performance (365d) and recent performance (90d), "  
        f"with metrics computed based on your selection (P/L or PCR).\n\n"  
        "This was designed for simple 0DTE credit selling strategies."
    )  
      
    # Add explanation for Metrics  
    st.markdown("**P/L (Profit / Loss)** is simply the average $ profit or loss "
                "for each entry time and day of the week.\n\n")  
    st.markdown("**PCR (Premium Capture Rate)** is calculated as:")  
    st.latex(r"\text{PCR} = \left(\frac{P/L}{Premium Collected}\right)")  
    st.markdown("Each individual trade is calculated first, and the resulting PCR's are averaged in the table.")  
      
    st.header("Instructions")  
    st.write(  
        "1. Upload your CSV file using the drag and drop area above.\n"  
        "2. Select which metric to visualize (P/L or PCR).\n"  
        "3. Select the mask type (highlight above average or positive values).\n"  
        "4. The app processes the data, generating two pivot tables (365d and 90d) and combines them.\n"  
        "5. The heatmap highlights cells based on your mask selection.\n"  
        "6. You can download the heatmap as a PNG file."  
    )
