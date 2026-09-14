import asyncio
import os
from agents import Agent, Runner, trace

from pydantic import BaseModel
from openai import OpenAI
from loguru import logger
from utils import utils
from tools.prompts import make_prompt, CONTEXT_PROMPT
from tools.output import WellTestContext, ZonalTestMemory, WellTestInterpretation, AnomalyInsights
from tools.tools import save_test_memory, load_well_memory_data
from dotenv import load_dotenv
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import streamlit_shadcn_ui as ui
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


base_path = os.path.dirname(__file__)
repo_path = os.path.dirname(base_path)
mem_file_path = os.path.join(base_path, "mem.txt")
well_test_data_path = os.path.join(base_path, "data", "RMO_Agentic AI_train_test.xlsx")
env_file_path = os.path.join(repo_path, ".env")
st.set_page_config(layout="wide")

#Load environment variables
try:
    # Access nested values
    openai_key = st.secrets["api_keys"]["openai"]
    os.environ["OPENAI_API_KEY"] = openai_key
except:
    load_dotenv(dotenv_path=env_file_path)


#Prompts
ANOMALY_DETECTOR_PROMPT, MEMORY_SAVER_PROMPT, INTERPRETATOR_PROMPT = make_prompt(threshold=0.1, file_path = mem_file_path)


# Agent team
anomaly_detection_agent = Agent(
    name="wt_anomaly_detector",
    instructions=ANOMALY_DETECTOR_PROMPT,
    output_type=AnomalyInsights,
)

MEMORY_SAVER_AGENT = Agent(
    name="WT_memory_saver",
    instructions=MEMORY_SAVER_PROMPT,
    output_type=ZonalTestMemory,
    tools=[save_test_memory],
)

INTERPRETATOR_AGENT = Agent(
    name="wt_interpretator",
    instructions=INTERPRETATOR_PROMPT,
    output_type=WellTestInterpretation,
)


#Function to run the agents
async def main():
    openai_ready = bool(os.getenv("OPENAI_API_KEY"))
    memory_data = []
    memory_error = None

    try:
        memory_data = load_well_memory_data(mem_file_path)
    except Exception as e:
        logger.error(f"Error loading memory data: {e}")
        memory_error = e

    #Sidebar
    with st.sidebar:
        st.title("WellWatch")
        st.subheader("Oil Well Test Monitoring with Agentic AI")
        st.divider()
        st.caption("Workflow")
        if openai_ready:
            st.success("OpenAI API key loaded")
        else:
            st.warning("OpenAI API key missing")

        chosen_well = ui.select("Well", options=['cheetah-90', 'cheetah-20', 'cheetah-10'], key="well_select")

        if chosen_well:
            logger.info("Starting to load data")
            df_raw = utils.load_transform_welltest_data(well_test_data_path, well_name=chosen_well, threshold=0.1)
            #Compute the dca 
            _ , popt = utils.fit_hyperbolic_decline_curve(df_raw, 'Date', 'WTOil')
            df_raw = utils.dca_forecast(df_raw, 'Date', popt)
            #Select key columns
            df = df_raw[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
                    'Z2BHP', 'Z3BHP', 'mean_bhp', 'log_diff_oil',
                    'log_diff_liq', 'log_diff_thp', 'log_diff_wct', 'log_diff_z1bhp',
                    'log_diff_z2bhp', 'log_diff_z3bhp', 'zone1_status', 'zone2_status',
                    'zone3_status','dca_rate']]
        
            df['Date'] = df['Date'].astype(str)
            #Dates list
            dates = df['Date'].unique().tolist()
            dates.sort()

            logger.info(f"Data loaded for well {chosen_well} with {len(df)} rows")
            st.success(f"{len(df)} well tests loaded")
            st.caption(f"{dates[0]} to {dates[-1]}")

        selected_date = ui.select("Well test date", options=dates, key="date_select")
        st.divider()
        agentic_ai_button = ui.button("Run AI Agents", key="agentic_ai", disabled=not openai_ready)

        if not openai_ready:
            st.caption("Add OPENAI_API_KEY to .env to run the agents.")

    if selected_date:
        #select the data up until the selected date
        logger.info(f"Filtering data for date {selected_date}")
        # Filter the dataframe to include only rows with the selected dateand all rows before it
        df = df.loc[df['Date'] <= selected_date]
        logger.info(f"Done filtering data for date {selected_date}")
        df_aggrid = df.copy().round(0)
        df_aggrid['Date'] = pd.to_datetime(df_aggrid['Date'])
        df_aggrid = df_aggrid[['Date', 'WellName', 'WTLIQ', 'WTOil', 'WTTHP', 'WTWCT', 'Z1BHP',
            'Z2BHP', 'Z3BHP']]

        selected_test = df.tail(1).iloc[0]
        oil_delta = selected_test['WTOil'] - selected_test['dca_rate']
        liquid_delta = df['WTLIQ'].iloc[-1] - df['WTLIQ'].iloc[-2] if len(df) > 1 else None
        oil_metric_delta = f"{oil_delta:,.0f} vs DCA"
        liquid_metric_delta = f"{liquid_delta:,.0f} vs prior" if liquid_delta is not None else None

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Oil rate", f"{selected_test['WTOil']:,.0f}", oil_metric_delta)
        kpi_cols[1].metric("Liquid rate", f"{selected_test['WTLIQ']:,.0f}", liquid_metric_delta)
        kpi_cols[2].metric("Water cut", f"{selected_test['WTWCT']:.2f}")
        kpi_cols[3].metric("THP", f"{selected_test['WTTHP']:,.0f}")
        kpi_cols[4].metric("Avg BHP", f"{selected_test['mean_bhp']:,.0f}")

        st.subheader("Well test chart")
        fig = go.Figure()

        for column in ['WTLIQ', 'WTOil', 'WTWCT', 'Z1BHP','Z2BHP', 'Z3BHP']:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[column],
                mode='lines',
                name=column
            ))

        fig.update_layout(
            title="Well test data: WT OIL, WT WCT, WT Liq, Z1 BHP, Z2 BHP & Z3 BHP",
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Columns",
            height=500
        )

        st.plotly_chart(fig, width='stretch')

        st.subheader("Well test datatable")
        gb = GridOptionsBuilder.from_dataframe(df_aggrid)

        cellstyle_jscode = JsCode("""
        function(params) {
            if (params.value < 5000) {
                return {
                    'color': 'white',
                    'backgroundColor': 'salmon'
                }
            } else {
                return {
                    'color': 'black',
                    'backgroundColor': 'white'
                }
            }
        };
        """)

        columns_to_style = ['Z1BHP','Z2BHP', 'Z3BHP']
        for col in columns_to_style:
            gb.configure_column(col, cellStyle=cellstyle_jscode)

        grid_options = gb.build()
        grid_return = AgGrid(df_aggrid, gridOptions=grid_options, editable=True, allow_unsafe_jscode=True, height=300, fit_columns_on_grid_load=True)

        with st.expander("Memory history", expanded=False):
            if memory_error:
                st.warning("Memory file could not be loaded.")
            elif memory_data:
                memory_df = pd.DataFrame([item.model_dump() for item in memory_data])
                memory_df = memory_df.loc[memory_df['WellName'] == chosen_well]

                if memory_df.empty:
                    st.info(f"No saved memory for {chosen_well} yet.")
                else:
                    st.dataframe(memory_df, width='stretch', hide_index=True)
            else:
                st.info("No saved memory yet.")


    #add image of agentic flow
    image_path = os.path.join(base_path, "agentic_ai.png")
    st.image(image_path)
    st.divider()


    if agentic_ai_button:
        # Run the agents in a deterministic story flow
        df.drop(['dca_rate'], axis=1, inplace=True)
        df = df.tail(1)
        df_iterator = df.iterrows()
        idx, serie = next(df_iterator)
        df = serie.to_dict()

        #Define the context
        context = CONTEXT_PROMPT

        well_test_input = WellTestContext(**df)
        logger.info(f"well test input {well_test_input}")
        banner = st.empty()
        st.subheader("Agentic AI review")

        with trace("Deterministic story flow"):
            workflow_status = st.status("Starting agent workflow", expanded=True)

            # Run the anomaly detection agent
            logger.info(f"The anomaly detector agent is at work...")
            workflow_status.write("Anomaly detector is reviewing the selected well test")
            with st.spinner("Anomaly detector is reviewing the selected well test..."):
                result_anomaly = await Runner.run(anomaly_detection_agent, input=f' Here are the well test data {well_test_input.model_dump()}' , context=context)
            logger.info(f"The anomaly detector agent has completed its work...")
            workflow_status.write("Anomaly detector complete")

            with st.container(border=True):
                st.subheader("Anomaly detector agent")
                st.write(str(result_anomaly.final_output.Short_summary))

            # Run the memory saver agent
            logger.info(f"Now the memory savor agent is at work...")
            workflow_status.write("Memory saver is recording the agent assessment")
            with st.spinner("Memory saver is recording the assessment..."):
                result_memory = await Runner.run(MEMORY_SAVER_AGENT, input=f' Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result is {result_anomaly.final_output}' , context=context)
            logger.info(f"The memory savor agent has completed its work...")
            workflow_status.write("Memory saver complete")

            with st.container(border=True):
                st.subheader("Memory saver agent")
                st.markdown(f"**Anomaly:** {result_memory.final_output.Anomaly}")
                st.markdown(f"**Anomaly Type:** {result_memory.final_output.AnomalyType}")
                st.markdown(f"**Zone Status:** Z1 {result_memory.final_output.Z1Status}, Z2 {result_memory.final_output.Z2Status}, Z3 {result_memory.final_output.Z3Status}")

            # Run the interpretator agent
            logger.info(f"Now the insights interpreter agent is at work...")
            workflow_status.write("Interpreter is turning the assessment into an engineering action")
            with st.spinner("Interpreter is preparing the engineering recommendation..."):
                result_interpretator = await Runner.run(INTERPRETATOR_AGENT, 
                                                        input=f'Here are the well test data {well_test_input.model_dump()} and this is what the anomaly analysis result for this well test is {result_anomaly.final_output} - The memory data has past welltest {memory_data}', context=context)
            logger.info(f"The insights interpreter agent has completed its work...")
            workflow_status.write("Interpreter complete")
            workflow_status.update(label="Agent workflow complete", state="complete", expanded=False)

            if result_memory.final_output.Anomaly:
                banner.error(f"Anomaly detected: {result_memory.final_output.AnomalyType}. {result_interpretator.final_output.EngineerAction}")
            else:
                banner.success(f"No anomaly detected. {result_interpretator.final_output.EngineerAction}")
            
            with st.container(border=True):
                st.subheader("Insights Interpretation agent")
                st.markdown(f"**Zonal Config:** {result_interpretator.final_output.ZonalConfiguration}")
                st.markdown(f"**Interpretation:** {result_interpretator.final_output.Interpretation}")
                st.markdown(f"**Engineer Action:** {result_interpretator.final_output.EngineerAction}")
                st.markdown(f"**Insights Summary:** {result_interpretator.final_output.InsightsSummary}")
            
            with open(mem_file_path, 'r') as f:
                logger.info(f"This has been saved in the memory file: {f.read()}")
            
            
  
if __name__ == "__main__":
    asyncio.run(main())