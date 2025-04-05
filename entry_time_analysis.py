import streamlit as st  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import numpy as np  
from datetime import datetime, timedelta  
import io  
from matplotlib.colors import ListedColormap  
  
st.title("Entry Time Analysis")  
st.write("Drag and drop your CSV file to visualize performance by day of week and time.")  
  
# File uploader widget for drag and drop  
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])  
  
if uploaded_file is not None:  
    try:  
        # Read the CSV file into a DataFrame  
        df = pd.read_csv(uploaded_file)  
        st.success("File uploaded successfully!")  
          
        # Optionally display the raw data  
        if st.checkbox("Show raw data"):  
            st.dataframe(df.head())  
          
        # Only include rows where 'Legs' contains 'STO'
        df = df[df['Legs'].str.contains('STO', na=False)] 

        # Add filtering based on 'Strategy' column:  
        # Get the sorted unique strategies for the multiselect widget  
        unique_strategies = sorted(df['Strategy'].unique())  
        selected_strategies = st.multiselect('Select Strategy(s) to include', unique_strategies, default=unique_strategies)  
        # Filter DataFrame based on selected strategies  
        df = df[df['Strategy'].isin(selected_strategies)]  
      
        # Convert 'Date Opened' to datetime and create 'Day of Week' column  
        df['Date Opened'] = pd.to_datetime(df['Date Opened'])  
        df['Day of Week'] = df['Date Opened'].dt.day_name()
           
        # Calculate PCR  
        df['PCR'] = (df['P/L'] / df['Premium'])
          
        # Add a radio button to select between P/L and Normalized P/L  
        metric_option = st.radio(  
            "Select metric to visualize:",  
            ["P/L", "PCR"],  
            horizontal=True  
        )  
          
        # Use the selected metric for analysis  
        value_column = metric_option  
          
        # Create the overall pivot table: index = 'Time Opened', columns = day of week (Monday to Friday)  
        pivot_table = df.pivot_table(index='Time Opened', columns='Day of Week',   
                                     values=value_column, aggfunc='mean')  
        rounded_pivot_table = pivot_table.round(3)  
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']  
        # Filter columns that exist in the data  
        existing_days = [day for day in ordered_days if day in rounded_pivot_table.columns]  
        rounded_pivot_table = rounded_pivot_table[existing_days]  
        # Create a second pivot table for entries within the last 90 days  
        ninety_days_ago = datetime.now() - timedelta(days=90)
        recent_df = df[df['Date Opened'] >= ninety_days_ago]  

        if not recent_df.empty:  
            recent_pivot_table = recent_df.pivot_table(index='Time Opened', columns='Day of Week',   
                                                      values=value_column, aggfunc='mean')  
            recent_rounded_pivot_table = recent_pivot_table.round(3)  
            # Filter columns that exist in the recent data  
            recent_existing_days = [day for day in ordered_days if day in recent_rounded_pivot_table.columns]  
            recent_rounded_pivot_table = recent_rounded_pivot_table[recent_existing_days]  
              
            # Rename columns to distinguish between overall and recent  
            overall_table = rounded_pivot_table.copy()  
            overall_table.columns = ['Overall ' + col for col in overall_table.columns]  
              
            recent_table = recent_rounded_pivot_table.copy()  
            recent_table.columns = ['Recent ' + col for col in recent_table.columns]  
              
            # Join the two tables  
            combined_table = overall_table.join(recent_table, how='outer')  
              
            # Reorder columns to group Overall and Recent for each day together  
            ordered_columns = []  
            for day in existing_days:  
                if 'Overall ' + day in combined_table.columns:  
                    ordered_columns.append('Overall ' + day)  
                if 'Recent ' + day in combined_table.columns:  
                    ordered_columns.append('Recent ' + day)  
              
            combined_table = combined_table[ordered_columns]  
              
            # Create a mask for values above the overall dataset average  
            overall_avg = overall_table.values.mean()  
            
            # Create a mask starting with 0 (default)  
            mask = np.zeros(combined_table.shape, dtype=int)         
            
            # Mark cells as 1 if they're above the overall average  
            mask = np.where(combined_table.values > overall_avg, 1, mask)  
            
            # Mark cells as 2 if both the overall and recent values exceed their averages
            # Process column pairs (1&2, 3&4, 5&6, etc.)  
            
            # Note: Python uses 0-based indexing, so columns 1&2 are at indices 0&1  
            for i in range(0, combined_table.shape[1], 2):  
                # Check if we have a complete pair (to handle odd number of columns)  
                if i + 1 < combined_table.shape[1]:  
                # For each row, check if both columns in the pair are above average  
                    for row in range(combined_table.shape[0]):  
                        if (combined_table.iloc[row, i] > overall_avg and   
                            combined_table.iloc[row, i+1] > overall_avg):  
                            # Set both columns in this row to 2  
                            mask[row, i] = 2  
                            mask[row, i+1] = 2  

            # Convert the numpy array back to a DataFrame with the same index and columns as combined_table  
            mask = pd.DataFrame(mask, index=combined_table.index, columns=combined_table.columns)  
            

            # Create a custom colormap: white for 0, lightgreen for 1, green for 2  
            cmap = ListedColormap(['white', 'lightgreen', 'green'])  

            # Format the data based on the metric option chosen
            data_format = '.2%' if metric_option == 'PCR' else '.0f'
              
            # Create a heatmap from the combined table  
            plt.figure(figsize=(14, 10))  
            ax = sns.heatmap(mask, annot=combined_table, fmt=data_format, cmap=cmap,  
                             cbar=False, vmin=0, vmax=2, linewidths=0.5)  
            ax.xaxis.tick_top()  
            ax.xaxis.set_label_position('top')  
            plt.xticks(rotation=45, ha='left')  
            plt.title(f"365d vs 90d {metric_option} Heatmap", fontsize=20, pad=15)  
            plt.xlabel("Day of Week", fontsize=16, labelpad=10)  
            plt.ylabel("Time Opened", fontsize=16, labelpad=10)  
            plt.tight_layout()  
              
            # Render the plot in Streamlit  
            st.pyplot(plt.gcf())  
              
            # Optionally, allow download of the figure as PNG  
            buf = io.BytesIO()  
            plt.savefig(buf, format="png", dpi=300)  
            buf.seek(0)  
            st.download_button(f"Download {metric_option} Heatmap", data=buf,   
                              file_name=f"trading_{metric_option.lower().replace('/', '_')}_heatmap.png",   
                              mime="image/png")
            
            # Now, extract times by day of week where mask value is 2
            results = {}

            # Process each day of the week
            for day in ordered_days:
                # Get columns for this day (both 365d and 90d)
                day_cols = [col for col in mask.columns if day in col]
                
                # Find rows where any column for this day has mask value 2
                mask_value_2_times = []
                for time in mask.index:
                    # Check if any column for this day has mask value 2 for this time
                    if any(mask.loc[time, col] == 2 for col in day_cols):
                        mask_value_2_times.append(time)
                
                # Store the results
                results[day] = mask_value_2_times

            # Create a formatted table with days as columns and times underneath
            print("\
            Times by day of week where mask value is 2:")
            # Find the maximum number of times for any day
            max_times = max(len(times) for times in results.values())

            # Create a table with days as columns and times as rows
            table_data = []
            for i in range(max_times):
                row = []
                for day in ordered_days:
                    if i < len(results[day]):
                        row.append(results[day][i])
                    else:
                        row.append("")
                table_data.append(row)

            # Print the table with tabulate
            #print(tabulate(table_data, headers=days_of_week, tablefmt="grid"))

            # Alternative: Create a DataFrame for better formatting
            times_df = pd.DataFrame(columns=ordered_days)
            for day in ordered_days:
                times_df[day] = pd.Series(results[day] + [""] * (max_times - len(results[day])))

            #print("\
            #DataFrame format:")
            #st.write(times_df.to_string(index=False))
            st.title("Optimal Entry Times") 
            times_df        
        
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
        "3. The app processes the data, generating two pivot tables (365d and 90d) and combines them.\n"  
        "4. The heatmap highlights cells in green if they are above the column average.\n"  
        "5. You can download the heatmap as a PNG file."  
    )  
