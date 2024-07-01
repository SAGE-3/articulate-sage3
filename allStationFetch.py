@app.post("/processText")
async def process_text(request: TextRequest):
    print(request)
    # required_fields(["prompt"], data)
    start =time.time()

    client_groq = Groq(api_key = os.getenv('GROQ_API_KEY'))

    dataset = os.path.join(__location__,"datasets/hcdpDataReduced.csv")
    few_shot = os.path.join(__location__, "data/finetune_train_articulate.xlsx")


    # llm_re = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 1})
    # llm_base = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 0}, context_path=dataset, iter_self_reflection=4)

    llm_re = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1})
    llm_base = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 0}, context_path=dataset)
    llm_transform = csv_llm.LLM(client, {"model": "gpt-4-turbo-preview", "temperature": 0}, context_path=dataset)
    llm = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1}, context_path=dataset)

    user_prompt = request.prompt #"""show me car brands vs price, sort by decending prices"""
    conversational_context = request.context
    # prediction = command_model.predict(user_prompt)
    # if (prediction == 0):
    #   print("Explicit Command")
    # elif (prediction ==1):
    #   print("Daily conversation")
    # elif (prediction == 2):
    #   print("Implicit command")
    # return {}
    print("****Starting Reiteration****")

    user_prompt_modified = llm_re.prompt_reiterate(user_prompt, conversational_context)
    print(user_prompt_modified)       
    print()
    
    print("****Starting Extracting Stations****")
    stations, station_reasoning = llm_base.prompt_select_stations(user_prompt_modified)
    print(stations)
    print(station_reasoning)
    
    
    
    print("****Starting for loop****")
    print()
    station_info = {}
    headers = {"Authorization": "Bearer 71c5efcd8cfe303f2795e51f01d19c6"}
    for idx,id in enumerate(stations):
        if idx >= 3:
            break
        r = requests.get(f"https://api.hcdp.ikewai.org/mesonet/getVariables?station_id={id}", headers=headers)
        variables = r.json()
        available_variable_names = []
        available_variable_ids = []
        for var in variables:
            available_variable_names.append(var['var_name'].replace(",",""))
            available_variable_ids.append(var['var_id'])
        available_variable_names.append("Date")
        available_variable_ids.append('Date')
        station_info[id] = {'available_variable_names': available_variable_names, 'available_variable_ids': available_variable_ids}
    station_chart_info = {}
    for id in station_info.keys():
        print("****Starting Attribute Selection****")
        print()
        print('available variable names ****', station_info[id]['available_variable_names'])
        chosen_attributes_names, attrib_reasoning = llm_base.prompt_attributes(user_prompt_modified, station_info[id]['available_variable_names'])
        chosen_attribute_ids = []
        for attr in chosen_attributes_names:
            print(attr, "--------------------", available_variable_names)
            # Check if the attribute exists in the available variables
            if attr in available_variable_names:
                index = station_info[id]['available_variable_names'].index(attr)
                print(index)
                # Check if index is valid
                if index != -1:
                    chosen_attribute_ids.append(station_info[id]['available_variable_ids'][index])
                else:
                    break
            else:
                break
        print(chosen_attribute_ids)
        print(attrib_reasoning)
        print()
        print("****Starting Transformation Selection****")
        print()
        transformations, trans_reasoning = llm_transform.prompt_transformations(user_prompt_modified, chosen_attributes_names)
        print(trans_reasoning)
        print("Transformations:", transformations)
        print()
        print("****Starting Chart Selection****")
        print()
        chartType, chart_frequencies, chart_reasoning, chart_scope = llm.prompt_charts_via_chart_info(user_prompt_modified, chosen_attributes_names)
        print(chart_reasoning)
        print(chartType)
        print(chart_frequencies)
        print(chart_scope)
        station_chart_info[id] = {'attributes': chosen_attribute_ids, 'transformations': transformations, 'chartType': chartType, 'available_attribute_info': station_info[id]} #TODO check if values exist
    # print("****Starting Extracting Date Ranges****")
    # stations, station_reasoning = llm_transform.prompt_select_date_range(user_prompt_modified)
    # print(stations)
    print(station_reasoning)





    # print(trans_reasoning)
    # print(transformations)
    end = time.time()
    
    print("Total time elapsed:", end-start)
    
    # attributes.append("Date")
    return {
        'station_chart_info': station_chart_info,
            # "attributes": {attribute: {"Data Type": column["Data Type"]} for attribute in attributes for column in csv_analysis if attribute == column["Column Name"]}, 
            # "csv_uuid": csv_uuid_ext.split(".")[0], 
            "debug": {
                "context": conversational_context,
                "query": user_prompt,
                "reiteration": user_prompt_modified, 
                "time": end-start,
                # "attributes": attrib_reasoning, 
                # "transformations": transformations,
                # "charts": chart_reasoning,
                # "charts_frequency": chart_frequencies,
                # "charts_scope": chart_scope,
                },
            # "analysis": csv_analysis
            }

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
        




#THIS IS THE FUNCTION THAT WORKS. PROCESS AS COMMAND AND START GENERATING A CHART
@app.post("/processText")
async def process_text(request: TextRequest):
    print(request)
    # required_fields(["prompt"], data)
    start =time.time()

    client_groq = Groq(api_key = os.getenv('GROQ_API_KEY'))

    dataset = os.path.join(__location__,"datasets/common_vars.csv")
    few_shot = os.path.join(__location__, "data/finetune_train_articulate.xlsx")


    # llm_re = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 1})
    # llm_base = csv_llm.LLM(client, {"model": "gpt-3.5-turbo-0125", "temperature": 0}, context_path=dataset, iter_self_reflection=4)

    llm_re = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1, "dataset": dataset})
    llm_base = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 0,"dataset": dataset})
    llm_transform = csv_llm.LLM(client, {"model": "gpt-4-turbo-preview", "temperature": 0,"dataset": dataset} )
    llm = csv_llm.LLM(client_groq, {"model": "llama3-70b-8192", "temperature": 1, "dataset": dataset})

    conversational_context = request.context
    user_prompt = request.prompt #"""show me car brands vs price, sort by decending prices"""
    prediction = command_model.predict(user_prompt)
    if (prediction == 0):
      user_prompt_modified = llm_re.prompt_reiterate(user_prompt, "")
      print(user_prompt_modified)       
      print()
    
    elif (prediction ==1):
      print("Daily conversation")
      return {}
    elif (prediction == 2):
      user_prompt_modified = llm_re.prompt_reiterate(user_prompt, conversational_context)
      print(user_prompt_modified)       
      print()
      
    print("****Generating Chart****")
    # print("****Starting Reiteration****")


    # print("****Starting Extracting Stations****")
    stations, station_reasoning = llm_base.prompt_select_stations(user_prompt_modified)
    # print(stations)
    # print(station_reasoning)
    
    # print("****Starting Extracting Stations****")
    dates, dates_reasoning = llm_base.prompt_select_dates(user_prompt_modified)
    print(dates)
    print(dates_reasoning)
    
    
    
    # print("****Starting for loop****")
    # print()
    station_info = {}
    headers = {"Authorization": "Bearer 71c5efcd8cfe303f2795e51f01d19c6"}
    for idx,id in enumerate(stations):
        with open(f'./datasets/stationVariables/stationVariables.json') as f:
          variables = json.load(f)
        # r = requests.get(f"https://api.hcdp.ikewai.org/mesonet/getVariables?station_id={id}", headers=headers)
        # variables = r.json()
        available_variable_names = []
        available_variable_ids = []
        for var in variables:
            available_variable_names.append(var['var_name'].replace(",",""))
            available_variable_ids.append(var['var_id'])
        available_variable_names.append("Date")
        available_variable_ids.append('Date')
        station_info[id] = {'available_variable_names': available_variable_names, 'available_variable_ids': available_variable_ids}
    station_chart_info = {}
    for id in station_info.keys():
        # print("****Starting Attribute Selection****")
        # print()
        # print('available variable names ****', station_info[id]['available_variable_names'])
        available_variable_names = station_info[id]['available_variable_names']
        available_variable_ids = station_info[id]['available_variable_ids']
        chosen_attributes_names, attrib_reasoning = llm_base.prompt_attributes(user_prompt_modified, available_variable_names)
        chosen_attribute_ids = []
        for attr in chosen_attributes_names:
            print(attr, "--------------------", available_variable_names)
            # Check if the attribute exists in the available variables
            if attr in available_variable_names:
                index = available_variable_names.index(attr)
                print(index)
                # Check if index is valid
                if index != -1:
                    chosen_attribute_ids.append(available_variable_ids[index])
                else:
                    break
            else:
                break
        # print(chosen_attribute_ids)
        # print(attrib_reasoning)
        # print()
        # print("****Starting Transformation Selection****")
        # print()
        transformations, trans_reasoning = llm_transform.prompt_transformations(user_prompt_modified, chosen_attributes_names)
        # print(trans_reasoning)
        # print("Transformations:", transformations)
        # print()
        # print("****Starting Chart Selection****")
        # print()
        chartType, chart_frequencies, chart_reasoning, chart_scope = llm.prompt_charts_via_chart_info(user_prompt_modified, chosen_attributes_names)
        # print(chart_reasoning)
        # print(chartType)
        # print(chart_frequencies)
        # print(chart_scope)
        station_chart_info[id] = {'attributes': chosen_attribute_ids, 'transformations': transformations, 'chartType': chartType, 'available_attribute_info': station_info[id]} #TODO check if values exist
    # print("****Starting Extracting Date Ranges****")
    # stations, station_reasoning = llm_transform.prompt_select_date_range(user_prompt_modified)
    # print(stations)
    # print(station_reasoning)

    print(f"************Generated a {station_chart_info}**************")



    # print(trans_reasoning)
    # print(transformations)
    end = time.time()
    
    print("Total time elapsed:", end-start)
    
    # attributes.append("Date")
    return {
        'station_chart_info': station_chart_info,
            # "attributes": {attribute: {"Data Type": column["Data Type"]} for attribute in attributes for column in csv_analysis if attribute == column["Column Name"]}, 
            # "csv_uuid": csv_uuid_ext.split(".")[0], 
            "debug": {
                "context": conversational_context,
                "query": user_prompt,
                "reiteration": user_prompt_modified, 
                "time": end-start,
                # "attributes": attrib_reasoning, 
                # "transformations": transformations,
                # "charts": chart_reasoning,
                # "charts_frequency": chart_frequencies,
                # "charts_scope": chart_scope,
                },
            # "analysis": csv_analysis
            }