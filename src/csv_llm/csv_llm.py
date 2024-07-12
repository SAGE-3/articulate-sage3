import pandas as pd
import openai
import re
import json
import ast
from .utils import Utils
from .chart_info import *
from collections import Counter
import datetime

class LLM:
    def __init__(self, client, llm_profile, samples_path=None, context_path=None, iterations=1, iter_self_reflection=0):
        self.client = client
        self.llm_profile = llm_profile
        self.iterations = iterations
        self.iter_self_reflection = iter_self_reflection
        self.csv_context = None
        self.csv_last_limited_attribute_scope = None
        self.csv_headers = None
        self.samples = None

        if samples_path:
            self.samples = self.__load_samples__(samples_path)
            # print(json.dumps(self.samples, indent=2))

        if context_path:
            # self.csv_context = Utils.calculate_metrics(context_path)
            self.csv_headers = list(Utils.get_headers(context_path))
            # print(self.csv_headers)
            # print(json.dumps(self.csv_context, indent=2))


    def __load_samples__(self, path):
        df = pd.read_excel(path)
        df.iloc[:, 0] = df.iloc[:, 0].str.strip()

        print(df.head)

        samples = []

        # Prompt Engineering Variant
        for index, row in df.iterrows():
            # if index % 2 == 1: # Only Implicit
            prompt = row.iloc[0]
            response = row.iloc[1]
            samples.append({"role": "user", "content": prompt})
            samples.append({"role": "assistant", "content": response})
                # print(prompt)
        return samples

    def __chat_wrapper__(self, messages):
        completion = self.client.chat.completions.create(
        model=self.llm_profile["model"],
        messages=messages,
        temperature=self.llm_profile["temperature"]
        )

        output = completion.choices[0].message.content
        return output, completion


    def __message_builder__(self, system, user_prompt, csv_headers=None, csv_context=None, few_shot_context=None, chart_type=None, attributes=None, conversational_context=None):#**kwargs): #system, user_prompt, csv_headers_only=False):

        messages = [{"role": "system", "content": system}]
        
        if few_shot_context:
            # print("using few shot samples")
            messages += few_shot_context
        
        # User Msg Builder
        user_message = []

        if csv_context:
            # print(csv_context)
            # print("using csv context")
            user_message.append(f"# This is additional context: \n{csv_context}")
            # user_message.append(f"This is the CSV decomposed: {csv_context}")
            
        if conversational_context:
            # print("using conversational context")
            user_message.append(f"# This is additional context from the conversation: \n{conversational_context}")
            # user_message.append(f"This is the CSV decomposed: {csv_context}")

        if csv_headers:
            # print("using csv headers", csv_headers)
            user_message.append(f"# These are the attributes names/ headers in the dataset: \n{csv_headers}")

        if chart_type:
            # print("using recommended chart type(s)", chart_type)
            user_message.append(f"# These are the recommended chart type(s): \n{chart_type}")

        if attributes:
            # print("using recommened attributes", attributes)
            user_message.append(f"# These are the attributes: \n{attributes}")

        if len(user_message) != 0:
            user_message.append(f"# This is the user's prompt: \n```{user_prompt}```")
        else:
            user_message.append(f"{user_prompt}")

        messages += [{"role": "user", "content": "\n\n".join(user_message)}]

        return messages


    # Filters
    
    def __results_no_filter__(self, output):
        return output
    
    def __results_filter_dates__(self, output):
        # Find text within the double curly braces
        matches = re.findall(r'\{.*?\}', output)
        if matches:
            try:
                # Get the content of the last match, assuming that is the intended result
                result = matches[-1]
                # Convert the string to a dictionary
                result_dict = json.loads(result)
                return result_dict
            except Exception as e:
                print(e, "Error occurred in results_filter_dates")
        return None

    def __results_filter__(self, output):
        # Find all text between the outermost brackets
        matches = re.findall(r'\[(.*)\]', ' '.join(output.split()))
        results = []
        if matches:
            try:
                # Handle the inner contents of the match
                cleaned_str = matches[-1]

                # Manually parse the elements, assuming top-level split by commas not within nested brackets
                items = []
                depth = 0
                start_idx = 0
                # Manually split by commas accounting for nested structures
                for i, char in enumerate(cleaned_str):
                    if char == '[':
                        depth += 1
                    elif char == ']':
                        depth -= 1
                    elif char == ',' and depth == 0:
                        items.append(cleaned_str[start_idx:i].strip())
                        start_idx = i + 1
                # Append the last item
                items.append(cleaned_str[start_idx:].strip())

                # Clean up items to remove any stray quotes and handle special characters
                results = [re.sub(r'([\"\'].*?[\"\'])\s*<\s*(\d+)', r'\1 < \2', item).strip('\"') for item in items]

            except Exception as e:
                print(e, "Error occurred in results_filter")
        return results

    def __results_filter_charts__(self, output):
        matches = re.findall(r'\[(.*?)\]', ' '.join(output.split()))
        results = []
        if matches:
            results = json.loads(f"""[{matches[-1].replace("'",'"').replace('“','"').replace('”','"')}]""")
            results = list({result.split("->")[-1].strip() for result in results})
        return results


    def __results_filter_sql__(self, output):
        try:
            matches = re.findall(r'\`\`\`(.*?)\`\`\`', ' '.join(output.split()))
            match = matches[-1] if matches[-1][:3].lower() != "sql" else matches[-1][3:]
            match = match.replace("\\n", " ")
            return match
        except:
            return ""

    # class PromptTemplates:
    def __base_prompt__(self, user_prompt, system ):
        output_history = []
        results_history = []
        messages = [{"role": "system", "content": system}]

        output, completion = self.__chat_wrapper__(messages=messages)
        output_history.append(output)

        if output:
            return output, output_history
        else:
            return "", output_history
        
        self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt), filter_method=self.__results_filter_dates__)

    # class PromptTemplates:
    def __base_prompt_with_self_reflection__(self, user_prompt, system, messages, filter_method):
        output_history = []
        results_history = []

        output, completion = self.__chat_wrapper__(messages=messages)
        output_history.append(output)

        results_history.append(filter_method(output))

        for i in range(self.iter_self_reflection):
            output, completion, early_stop = self.__prompt_self_reflection__(user_prompt, system, output_history)
            output_history.append(output)
            results_history.append(filter_method(output))

            if early_stop:
                break
            # results_history.append(filter_method(output))

        # print(results_history)
        results = [result for result in results_history if result]
        if results:
            return results[-1], output_history
        else:
            return "", output_history

    def __prompt_self_reflection__(self, user_prompt, system, history):
        output, completion = self.__chat_wrapper__(messages=
            [{"role": "system", "content": system}] + 
            [{"role": "assistant", "content": json.dumps(line)} for line in history] +
            [{"role": "user", "content": f"This is my prompt: {user_prompt} Did you follow the tasks?  Think about it carefully.  Does your solution satisfy the requirements?  Did you forget to complete some parts?  If you believe you have the correct solution explain your reasoning and then say \"ACCOMPLISHED\""}])

        early_stop = "ACCOMPLISHED" in output
        # results = self.__results_filter__(output)
        return output, completion, early_stop


    def __base_prompt_with_consistency__(self, messages, filter_method):
        completions = []
        outputs = []
        predictions = []

        for i in range(self.iterations):
            output, completion = self.__chat_wrapper__(messages=messages)

            completions.append(completion)
            outputs.append(output)

            predictions += filter_method(output)

        frequencies = dict(Counter(predictions))
        predictions = list(set(predictions))
        return predictions, frequencies, outputs

    # PROMPTS

    def prompt_charts(self, user_prompt, attributes=None):
        system = """Given the following decision tree:



    Distribution -> One Variable -> Few Data Points -> Column Histogram
    Distribution -> Two or more variables -> Scatter Chart
    Distribution -> single variable -> Boxplot


    Comparison -> Over Time -> Many Periods -> Non-Cyclical Data -> Line Chart
    Comparison -> Over Time -> Few Periods -> Many Categories -> Line Chart

    Composition -> Static -> Sample Share of Total -> Pie Chart

    Classify the user's prompt. 
    If a user explicitly asks for a chart, pick the chart that they asked for.
    For example, if the user asks for a pie chart, show them a pie chart.
    
    Do not disagree with the user's request. 
    Choose the chart that the user asks for.

    Provide your answer at the end in this format ["Pie Chart"]


    """

        return self.__base_prompt_with_consistency__(
            messages=self.__message_builder__(system, user_prompt, few_shot_context=self.samples, csv_context=self.__limit_csv_scope__(attributes, self.csv_context), attributes=attributes),
            filter_method=self.__results_filter_charts__)



    def prompt_charts_via_chart_info(self, user_prompt, attributes):
        # print(json.dumps(chart_info_filter(["Car", "Price"])[0], indent=2))
        
        charts_details, charts_decision_tree = chart_info_filter(attributes)
        # print("available Charts", list(set(charts_details)))
        system = f"""# Task
You are a visualization expert with decades of experience. Choose the best chart from the limited list of choices provided based on the user's prompt (and potentially extra information from the user).
It is important to choose the appropriate graph, not based on graph popularity, but rather on appropriateness infered from the analyzed dataset.
    Do not disagree with the user's request. 
    Choose the chart that the user asks for.
    If the user explicitly asks for a chart type, do not pick a chart type that you think would be better. 
    Pick the chart type that the user asks for.
    
# Charts
    ['Boxplot', 'Scatter Chart', 'Column Histogram', 'Pie Chart', 'Line Histogram', 'Line Chart']



    
# Action
Take a deep breath, think it through, assume you are the user and imagine their intent, write your reasoning.
    Provide your answer at the end in this format ["Pie Chart"]

    """

        return self.__base_prompt_with_consistency__(
            messages=self.__message_builder__(system, user_prompt, few_shot_context=self.samples, attributes=attributes),
            filter_method=self.__results_filter_charts__) + (list(set(charts_details)),)


    def prompt_attributes(self, user_prompt, attributes):

        system = """# Task
Given user prompt extract all plausible attributes.  Attributes are more than likely associated with the column names on tabular data.  
If you are given a list of headers or attributes, your answer must be a subset of them and contain informaton about transitions.
Only include necessarty attributes in the visualization. 
At most, pick up to 3 variables.
Pick attributes that are common among multiple datasets.

# Action
Take a deep breath, think it through, assume you are the user and imagine their intent, write your reasoning.
Then provide your answer at the end in this format ["Answer 1","Answer 2",...]."""

        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt, csv_headers=attributes),
            filter_method=self.__results_filter__)

    def prompt_transformations(self, user_prompt, attributes):
        system = f"""# Task
You are an expert at deciphering the tasks users want to accomplish with data. Your expertise lies in identifying specific transformations required to fulfill those tasks. The transformations you're adept at detecting are:

None -> ["none"] or [""]
filter -> ["Air temperature_ sensor 1" > 22]
filter -> ["Soil moisture_ sensor 1" > 22, "Soil moisture_ sensor 1" < 30]
filter -> ["Station_name" == "Nuuanu Res", "Soil moisture_ sensor 1" != 30]
Now, take a moment to put yourself in the shoes of the user. Consider their intent behind the query and deduce the necessary transformation(s) to achieve their goal. After careful consideration, provide your answer in the format of ["Answer 1","Answer 2",...]..
You do not need to filter the data by date. 

Example:
User Query: "Show me the stations that have air temperature greater than 75"

User Intent Analysis:
The user wants to filter the dataset to display only those rows where the air temperature exceeds 75 degrees for the month of February.

Transformation Deduction:
["Air temperature, sensor 1" > 75]
Note: Do not filter the data by date

"""

        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt,attributes=attributes, csv_headers=attributes),  filter_method=self.__results_filter__)

    def prompt_proactive_reiterate(self, user_prompt, conversational_context):
        # print(user_prompt, conversational_context)
        #You are a data scientist and a data visualization expert.
        system = f"""You are a proactive data visualization expert.
        Your task is to proactively generate a useful chart for the user based on the context of the current conversation.
        {conversational_context}
        
        {user_prompt}
        Take a deep breath, think it through, assume you are the user and imagine their intent.  
        Response with a prompt that will create a useful visualization for the user.
        
        The charts you are able to generate are box plot, scatter chart, line chart, histogram, and pie chart.
        Try to generate a chart that the user has not yet seen but is still relevant to the context.
        Do not generate the same chart that the user has already seen. 
        Try to pick different variables or chart types. 
        You can also explore different stations near the selected stations. 
        Or you can choose to show an individual station.

        Note that the data is only available from January 1st, 2024 to July 1st, 2024. 
        Be very precise. Do not respond back with anything other than the rewritten prompt. 
        Do not include quotation marks, just return a prompt.

        """
        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt),  filter_method=self.__results_no_filter__)

    def prompt_reiterate(self, user_prompt, conversational_context):
        # print(user_prompt, conversational_context)
        #You are a data scientist and a data visualization expert.
        system = """You are a data scientist and a data visualization expert.
# Task
Reword the user's prompt using precise and specific language that disambiguates.  Do note that the user's request pertains to visualization and graph generation.

# Action
Take a deep breath, think it through, assume you are the user and imagine their intent.  Then provide the rewritten prompt.

Note that the data is only available from January 1st, 2024 to July 1st, 2024. 
The only data attributes that you may use are date, rainfall, temperature, solar radiation, wind speed, and soil moisture.
Be very precise. Do not respond back with anything other than the rewritten prompt. 
Do not include quotation marks, just rewrite the prompt

"""
        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt),  filter_method=self.__results_no_filter__)
        # output, completion = self.__chat_wrapper__(messages=self.__message_builder__(system, user_prompt,csv_headers=self.csv_headers,  conversational_context=conversational_context))
        # return output

    def prompt_extract_intent(self, user_prompt):
        system = """You are a data scientist and a data visualization expert.
# Task
You must extract the intent from the user's promp.  Use adequate reasoning and assumptions to clarify and disabiguate the user's intent.  Do note that the user's request typically pertains to visualization and graph generation.

# Action
Take a deep breath, think it through, assume you are the user and imagine their intent.  Then explain your reasoning using precise language.
"""

        output, completion = self.__chat_wrapper__(messages=self.__message_builder__(system, user_prompt))
        return output
    #        Station 0605, PowerlineTrail is located on latitude: 22.113562 and longitude -159.438786

    def prompt_select_stations(self, user_prompt, data_statistics):
        system = f""" You are tasked to select station IDs that fit the user's criteria.
        These stations will be used to answer the user's query on visualizing Hawaii's climate data.
        
        Here are a list of stations:
        Station 0603, LowerLimahuli (Station 1) is located on latitude 22.219805 and longitude -159.575195
        Station 0602, CommonGround (Station 2) is located on latitude 22.197439 and longitude -159.42123
        Station 0601, Waipa (Station 3) is located on latitude 22.202647 and longitude -159.518833
        Station 0521, Kaala (Station 4) is located on latitude 21.50675 and longitude -158.144861
        Station 0502, NuuanuRes1 (Station 5) is located on latitude 21.339124 and longitude -157.836859
        Station 0501, Lyon (Station 6) is located on latitude 21.333 and longitude -157.8025
        Station 0412, Honolimaloo (Station 7) is located on latitude 21.131411 and longitude -156.758626
        Station 0287, Mamalahoa (Station 8) is located on latitude 19.80258 and longitude -155.850629
        Station 0286, Palamanui (Station 9) is located on latitude 19.73756 and longitude -155.99582
        Station 0283, Laupahoehoe (Station 10) is located on latitude 19.93217 and longitude -155.29129
        Station 0282, Spencer (Station 11) is located on latitude 19.96358 and longitude -155.25014
        Station 0281, IPIF (Station 12) is located on latitude 19.69748 and longitude -155.0954
        Station 0252, Lalamilo (Station 13) is located on latitude 20.019528 and longitude -155.677085
        Station 0251, Kehena (Station 14) is located on latitude 20.122834 and longitude -155.749329
        Station 0243, KonaHema (Station 15) is located on latitude 19.2068247 and longitude -155.8109802
        Station 0242, KaiauluPuuWaawaa (Station 16) is located on latitude 19.77241 and longitude -155.83118
        Station 0241, Keahuolu (Station 17) is located on latitude 19.668669 and longitude -155.957474
        Station 0231, Kaiholena (Station 18) is located on latitude 19.16877 and longitude -155.57035
        Station 0213, Piihonua (Station 19) is located on latitude 19.7064059 and longitude -155.1873737
        Station 0212, Kulaimano (Station 20) is located on latitude 19.834341 and longitude -155.122435
        Station 0211, Kanakaleonui (Station 21) is located on latitude 19.845036 and longitude -155.362586
        Station 0202, Keaau (Station 22) is located on latitude 19.6061748 and longitude -155.0515231
        Station 0201, Nahuku (Station 23) is located on latitude 19.415215 and longitude -155.238394
        Station 0154, Waikamoi (Station 24) is located on latitude 20.773636 and longitude -156.222311
        Station 0153, Summit (Station 25) is located on latitude 20.710361 and longitude -156.25675
        Station 0151, ParkHQ (Station 26) is located on latitude 20.759806 and longitude -156.248167
        Station 0144, Kahikinui (Station 27) is located on latitude 20.633945 and longitude -156.273889
        Station 0131, LahainaWTP (Station 28) is located on latitude 20.89072 and longitude -156.65493
        Station 0121, Lipoa (Station 29) is located on latitude 20.7458333 and longitude -156.4305556
        Station 0119, KulaAg (Station 30) is located on latitude 20.757889 and longitude -156.320028
        Station 0118, Pulehu (Station 31) is located on latitude 20.79532 and longitude -156.35991
        Station 0116, Keokea (Station 32) is located on latitude 20.7067 and longitude -156.3554
        Station 0115, Piiholo (Station 33) is located on latitude 20.8415 and longitude -156.2948
        
        Here is some statistics on the data
        {data_statistics}
        
        The maximum number of stations that you are able to select is 5.
        DO NOT select more than 5 stations at a time.
        Be very consice with your answers. 
        Do not pick more stations than is necessary. 
        If the user picks a certain island, only choose stations on that island. 

        Take a deep breath. Think it through. Imagine you are the user and imagine their intent. 
        Provide your answer only in the format  ["0502", "0521", ...]
        Explain your reasoning.
        Try to respond with at least 1 station.
        """
        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt),  filter_method=self.__results_filter__)
        
    def prompt_select_dates(self, user_prompt):
        system = """ You are a date expert selector.
        You will be given a query from a user and you will select the appropriate dates to filter the data by. 
        The dataset only has data since 2024-01-01T00:00:00.000000Z, so picking dates past this will not work.
        
        Here are some examples of what a user might say and how you will respond. 
        
        Query: Can you show me rainfall for the prior week?
        Answer: {"startDate": "2024-03-25T00:00:00.000000Z", "endDate": "2024-04-01T00:00:00.000000Z"}
        
        Query: what is the temperature for January?
        Answer: {"startDate": "2024-01-01T00:00:00.000000Z", "endDate": "2024-02-01T00:00:00.000000Z"}
        
        Query: Does rainfall and soil temperature have a correlation?
        Answer: {"startDate": "2024-01-01T00:00:00.000000Z", "endDate": "2024-06-01T00:00:00.000000Z"}
        
        If the user does not explicitly say which stations to choose from, you may select all of the stations
        (EX: ["0502", "0501", "0521"])

        Take a deep breath. Think it through. Imagine you are the user and imagine their intent. 
        You are only strictly to answer in the format {"startDate": "2024-01-01T00:00:00.000000Z", "endDate": "2024-07-01T00:00:00.000000Z"}
        If you are unsure, just pick the entire dataset. EX: {"startDate": "2024-01-01T00:00:00.000000Z", "endDate": "today's date"}
        

        """
        system += f"        For extra context, todays date is {datetime.datetime.now()}"
        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt), filter_method=self.__results_filter_dates__)
        
    def prompt_summarize_reasoning(self, user_prompt, chart_type_reasoning, attribute_reasoning, date_reasoning, station_reasoning, transformation_reasoning):
        system = f""" You are an expert summarizer.
        You have just created chart(s) for the user based on a prompt.
        Here is the user's prompt: {user_prompt}
        
        Here is some information on the decisions that you have made when creating this chart.
        
        Choosing the chart type:
        {chart_type_reasoning}
        
        Choosing the attributes:
        {attribute_reasoning}
       
        Choosing the dates
        {date_reasoning}
        
        Choosing the stations:
        {station_reasoning}
        
        Choosing the transformations
        {transformation_reasoning}
        
        You now need to respond back to the user. 
        Say a sentence or two that you think will help the user.
        
        Do not write more information than is necessary.
        
        """
        return self.__base_prompt__(user_prompt, system)
