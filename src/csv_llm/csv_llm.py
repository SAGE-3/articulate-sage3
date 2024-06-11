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
            self.csv_context = Utils.calculate_metrics(context_path)
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
            print("using few shot samples")
            messages += few_shot_context
        
        # User Msg Builder
        user_message = []

        if csv_context:
            print(csv_context)
            print("using csv context")
            user_message.append(f"# This is additional context: \n{csv_context}")
            # user_message.append(f"This is the CSV decomposed: {csv_context}")
            
        if conversational_context:
            print("using conversational context")
            user_message.append(f"# This is additional context from the conversation: \n{conversational_context}")
            # user_message.append(f"This is the CSV decomposed: {csv_context}")

        if csv_headers:
            print("using csv headers", csv_headers)
            user_message.append(f"# These are the attributes names/ headers in the dataset: \n{csv_headers}")

        if chart_type:
            print("using recommended chart type(s)", chart_type)
            user_message.append(f"# These are the recommended chart type(s): \n{chart_type}")

        if attributes:
            print("using recommened attributes", attributes)
            user_message.append(f"# These are the attributes: \n{attributes}")

        if len(user_message) != 0:
            user_message.append(f"# This is the user's prompt: \n```{user_prompt}```")
        else:
            user_message.append(f"{user_prompt}")

        messages += [{"role": "user", "content": "\n\n".join(user_message)}]

        return messages


    # Filters

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

        print(results_history)
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
    Distribution -> One Variable -> Many Data Points -> Line Histogram
    Distribution -> Two Variables -> Scatter Chart
    Distribution -> Three Variables -> 3D Area Chart

    Relationship -> Two Variable -> e Chart
    Relationship -> Three Variable -> Bubble Chart

    Comparison -> Among Items -> Two Variables Per Item -> Variable Width Column Chart
    Comparison -> Among Items -> One Variables Per Item -> Many Categories -> Table
    Comparison -> Among Items -> One Variables Per Item -> Few Categories -> Many Items -> Bar Chart
    Comparison -> Among Items -> One Variables Per Item -> Few Categories -> Few Items -> Column Chart
    Comparison -> Over Time -> Many Periods -> Cyclical Data -> Circular Area Chart
    Comparison -> Over Time -> Many Periods -> Non-Cyclical Data -> Line Chart
    Comparison -> Over Time -> Few Periods -> Single or Few Categories -> Column Chart
    Comparison -> Over Time -> Few Periods -> Many Categories -> Line Chart

    Composition -> Changing Over Time -> Few Periods -> Only Relative Differences Matter -> Stacked 100% Column Chart
    Composition -> Changing Over Time -> Few Periods -> Relative and Absolute Differences Matter -> Stacked Column Chart
    Composition -> Changing Over Time -> Many Periods -> Only Relative Differences Matter -> Stacked 100% Area Chart
    Composition -> Changing Over Time -> Many Periods-> Relative and Absolute Differences Matter -> Stacked Area Chart
    Composition -> Static -> Sample Share of Total -> Pie Chart
    Composition -> Static -> Accumulation or Subtraction to Total -> Waterfall Chart
    Composition -> Static -> Components of Components -> Stacked 100% Column Chart  with SubComponents



    Classify the user's prompt. Multiple solutions are possible and encouraged.
    Provide your answer at the end in this format ["Answer 1","Answer 2",...]

    """

        return self.__base_prompt_with_consistency__(
            messages=self.__message_builder__(system, user_prompt, few_shot_context=self.samples, csv_context=self.__limit_csv_scope__(attributes, self.csv_context), attributes=attributes),
            filter_method=self.__results_filter_charts__)

    def prompt_attribute_positioning(self, user_prompt, attributes=None):
        system = """Given the following chart Requirements

    Column Chart -> [xAxis, yAxis, label (optional)]
    Bar Chart -> [xAxis, yAxis]
    Scatter Chart -> [xAxis, yAxis]
    Circular Area Chart -> [indicator, value, label (optional)] where indicators would be names around the chart, value would be the value at each indicator, and label is an optional name that would represent multiple traces
    Line chart -> [xAxis, yAxis, label (optional)]
    Column Histogram -> [xAxis, yAxis]
    


    Classify the user's prompt. Multiple solutions are possible and encouraged.
    Provide your answer at the end in this format ["Answer 1","Answer 2",...]

    """

    def prompt_charts_via_chart_info(self, user_prompt, attributes):
        # print(json.dumps(chart_info_filter(["Car", "Price"])[0], indent=2))
        

        charts_details, charts_decision_tree = chart_info_filter(attributes)
        print("available Charts", list(set(charts_details)))
        system = f"""# Task
You are a visualization expert with ten years of experience. Choose the best chart from the limited Charts list choices provided based on the user's prompt (and potentially extra information from the user).
It is important to choose the appropriate graph, not based on graph popularity, but rather on appropriateness infered from the analyzed dataset.  Don't be afraid to be creative and select a chart type that will impress the user.

# Charts
{list(set(charts_details))}

# Action
Take a deep breath, think it through, assume you are the user and imagine their intent, write your reasoning.
Provide your answer at the end in this format ["Answer 1"]

    """

        return self.__base_prompt_with_consistency__(
            messages=self.__message_builder__(system, user_prompt, few_shot_context=self.samples, attributes=attributes),
            filter_method=self.__results_filter_charts__) + (list(set(charts_details)),)


    def prompt_attributes(self, user_prompt, attributes):
        assert self.csv_headers, "Context Required"

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

Example:
User Query: "Show me the stations that have air temperature greater than 75"

User Intent Analysis:
The user wants to filter the dataset to display only those rows where the air temperature exceeds 75 degrees for the month of February.

Transformation Deduction:
["Air temperature_ sensor 1" > 75, "Date" > 2024-01-31T00:15:00Z, "Date" < 2024-03-01T00:15:00Z, ]

For extra context, todays date is {datetime.datetime.now()}
"""

        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt,attributes=attributes),  filter_method=self.__results_filter__)


    def prompt_reiterate(self, user_prompt, conversational_context):
        #You are a data scientist and a data visualization expert.
        system = """You are a data scientist and a data visualization expert.
# Task
Reword the user's prompt using precise and specific language that disambiguates.  You encouraged to make assumptions about the users intent given adequate evidence.  Do note that the user's request pertains to visualization and graph generation.

# Action
Take a deep breath, think it through, assume you are the user and imagine their intent.  Then provide the rewritten prompt.
"""
        output, completion = self.__chat_wrapper__(messages=self.__message_builder__(system, user_prompt,csv_headers=self.csv_headers,  conversational_context=conversational_context))
        return output

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

    def prompt_select_stations(self, user_prompt):
        system = """ You are an expert Hawaii database manager. You are tasked to select station IDs that fit the user's criteria.
        
        Here are a list of stations:
        Station 0244, Kealakekua is located on latitude: 19.505859 and longitude -155.762429
        Station 0541, KaaawaMakai is located on latitude: 21.540352 and longitude -157.845041
        Station 0155, EMIBaseYard is located on latitude: 20.890997 and longitude -156.212316
        Station 0204, Pahoa is located on latitude: 19.501169 and longitude -154.943731
        Station 0621, LawaiNTGB is located on latitude: 21.905295 and longitude -159.510395
        Station 0604, UpperLimahuli is located on latitude: 22.18454 and longitude -159.582659
        Station 0603, LowerLimahuli is located on latitude: 22.219805 and longitude -159.575195
        Station 0602, CommonGround is located on latitude: 22.197439 and longitude -159.42123
        Station 0601, Waipa is located on latitude: 22.202647 and longitude -159.518833
        Station 0521, Kaala is located on latitude: 21.50675 and longitude -158.144861
        Station 0520, KaalaRepeater is located on latitude: 21.508229 and longitude -158.144075
        Station 0506, Kalawahine is located on latitude: 21.34522 and longitude -157.80556
        Station 0505, Napuumaia is located on latitude: 21.35489 and longitude -157.83063
        Station 0504, Waiolani is located on latitude: 21.33756 and longitude -157.84092
        Station 0503, UpperWaiolani is located on latitude: 21.34666 and longitude -157.83641
        Station 0502, NuuanuRes1 is located on latitude: 21.339124 and longitude -157.836859
        Station 0501, Lyon is located on latitude: 21.333 and longitude -157.8025
        Station 0431, Anapuka is located on latitude: 21.215984 and longitude -157.241786
        Station 0412, Honolimaloo is located on latitude: 21.131411 and longitude -156.758626
        Station 0411, Keopukaloa is located on latitude: 21.145283 and longitude -156.729459
        Station 0288, PuuWaawaa is located on latitude: 19.725393 and longitude -155.873917
        Station 0287, Mamalahoa is located on latitude: 19.80258 and longitude -155.850629
        Station 0286, Palamanui is located on latitude: 19.73756 and longitude -155.99582
        Station 0283, Laupahoehoe is located on latitude: 19.93217 and longitude -155.29129
        Station 0282, Spencer is located on latitude: 19.96358 and longitude -155.25014
        Station 0281, IPIF is located on latitude: 19.69748 and longitude -155.0954
        Station 0252, Lalamilo is located on latitude: 20.019528 and longitude -155.677085
        Station 0251, Kehena is located on latitude: 20.122834 and longitude -155.749329
        Station 0243, KonaHema is located on latitude: 19.2068247 and longitude -155.8109802
        Station 0242, KaiauluPuuWaawaa is located on latitude: 19.77241 and longitude -155.83118
        Station 0241, Keahuolu is located on latitude: 19.668669 and longitude -155.957474
        Station 0231, Kaiholena is located on latitude: 19.16877 and longitude -155.57035
        Station 0213, Piihonua is located on latitude: 19.7064059 and longitude -155.1873737
        Station 0212, Kulaimano is located on latitude: 19.834341 and longitude -155.122435
        Station 0211, Kanakaleonui is located on latitude: 19.845036 and longitude -155.362586
        Station 0203, Olaa is located on latitude: 19.478399 and longitude -155.214969
        Station 0202, Keaau is located on latitude: 19.6061748 and longitude -155.0515231
        Station 0201, Nahuku is located on latitude: 19.415215 and longitude -155.238394
        Station 0165, Hamoa is located on latitude: 20.719497 and longitude -156.002356
        Station 0164, BigBog is located on latitude: 20.726514 and longitude -156.092308
        Station 0162, Treeline is located on latitude: 20.734315 and longitude -156.123354
        Station 0161, PohakuPalaha is located on latitude: 20.730574 and longitude -156.141316
        Station 0154, Waikamoi is located on latitude: 20.773636 and longitude -156.222311
        Station 0153, Summit is located on latitude: 20.710361 and longitude -156.25675
        Station 0152, NeneNest is located on latitude: 20.738194 and longitude -156.245833
        Station 0151, ParkHQ is located on latitude: 20.759806 and longitude -156.248167
        Station 0145, UpperKahikinui is located on latitude: 20.644215 and longitude -156.284703
        Station 0144, Kahikinui is located on latitude: 20.633945 and longitude -156.273889
        Station 0143, Nakula is located on latitude: 20.674726 and longitude -156.233409
        Station 0141, Auwahi is located on latitude: 20.644222 and longitude -156.342056
        Station 0131, LahainaWTP is located on latitude: 20.89072 and longitude -156.65493
        Station 0121, Lipoa is located on latitude: 20.7458333 and longitude -156.4305556
        Station 0119, KulaAg is located on latitude: 20.757889 and longitude -156.320028
        Station 0118, Pulehu is located on latitude: 20.79532 and longitude -156.35991
        Station 0116, Keokea is located on latitude: 20.7067 and longitude -156.3554
        Station 0115, Piiholo is located on latitude: 20.8415 and longitude -156.2948

        Only pick 3 stations at most.

        Take a deep breath. Think it through. Imagine you are the user and imagine their intent. 
        Provide your answer only in the format ["Answer 1", "Answer 2", ...]
        Only give me the station id numbers. For example ["0541", "0431", ...]
        """
        return self.__base_prompt_with_self_reflection__(user_prompt, system,
            messages=self.__message_builder__(system, user_prompt),  filter_method=self.__results_filter__)
        



