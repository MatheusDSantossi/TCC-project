def pre_process_prompt_grammatical_fix_prompt(text, business_name):

    prompt = f"""
        Correct grammatical and spelling errors in the following text so it becomes understandable to a human reader. Normalize product names, utensils, or any other items to a standard form. Consider that this text is provided by a customer from the store {business_name}, and use this context to make the best possible corrections.

        Keep in mind that words might appear abbreviated, such as "INSET SBP AERO," which means "INSETICIDA SBP AEROSSOL." Research thoroughly what each word might mean, whether abbreviated or not, considering that they are most likely retail products found in any supermarket. Use this assumption to help identify and correct the words accurately.
        Also keep in mind that might appear a brand so don't fix words that you are not sure if it's not a brand, for example, "Biscoito treloso" which "Treloso" is a brand.

        Here is the text to be corrected:
        
        Text:
        {text}

        Maintain the original meaning of the text and correct only spelling and grammatical errors. 
        
        For example:

        "quero um cafezinho" → "Quero um café"
        "Que dea é heij" → "Que dia é hoje"
        "Que horas se" → "Que horas são"
        "Sarveti" → "Sorvete"
        "Seai" → "Saia"
        
        Pay close attention to text nuances, identifying and correcting errors. Here are more examples of corrections:

        "Quanto custa o bisc e o sab" → "Quanto custa o biscoito e o sabão" or "Quanto custa o biscoito e o sabonete"
        "Aendo figa localezado os tomate e a cebole" → "Onde fica localizado os tomates e a cebola"
        "Vocês te jel, deterge e saco" → "Vocês têm gel, detergente e saco"
        
        Be aware that some words might be completely incorrect, abbreviated, or incomplete, such as "refrig." or "q h o mercado abre," which should be corrected to "que horas o mercado abre." Apply this logic to all errors, regardless of how minor.

        Important Rules for Response:

        1. If the text contains retail products or related queries: Correct all spelling, grammatical, and abbreviation errors while maintaining the original meaning. Do not add information or products that are not present in the text. Be as concise as possible.
        
        2. If the text is unrelated to retail products or store-related queries (e.g., "Oi, como você está?", "Que horas o mercado fecha?", "Qual é o seu nome?"): Correct only the grammatical and spelling errors and return the text as it is.
        Return the corrected text in Portuguese, preserving its original intent.
        
        3. Don't return any weird characters like ``` or anything like that, just return the text correctly.
        
        4. Don't modify the original text if you're not sure if the text is really incorrect, just modify when you really find it necessary. Otherwise, just return the original text.
        
        IMPORTANT:
        Only return a simple text fixed or not depending on the requirements above. DON'T return or modify the original text without necessarily, for example, don't return anything like this:
        ```
        "Fixed text"
        ```
    """
    return prompt
    
def pre_process_prompt_grammatical_brand_fix_prompt(text, business_name, brands_in_database):
    prompt = f"""
        Correct the following text for grammatical and spelling errors, ensuring it is clear and understandable to a human reader.
        Additionally, if the text contains any product brands, verify them against the provided brands list and correct any misspellings to the exact brand names from that list.
        If a word might be a brand, do not alter it unless you are certain it is misspelled. In other cases, perform standard grammatical corrections.

        Consider that the text is provided by a customer of the store {business_name} and may include retail product names, utensils, or other related items.
        For example, if the text contains an abbreviated or misspelled term like "INSET SBP AERO", it should be corrected to "INSETICIDA SBP AEROSSOL".
        Similarly, if the text includes a brand (e.g., "Treloso") even in a misspelled form like "teloso", and that brand exists in the provided brands list, it must be corrected to the exact brand name ("Treloso").

        Examples:
        - Input: "biscoite teloso" → Output: "biscoito treloso"   (Assuming "Treloso" is in the brands list)
        - Input: "vocês têm cafi preto?" → Output: "vocês têm café preto?"

        Instructions:
        1. Fix grammatical and spelling errors in the text: 
        - Correct common errors while preserving the original meaning.
        2. Normalize product names, utensils, or any other items to a standard form.
        3. When encountering potential brand names:
        - If the term closely resembles a brand from the list {brands_in_database}, correct it to the exact brand name.
        - Do not change words to a brand form if it is uncertain they refer to a brand.
        4. Maintain all other parts of the text without adding extra information.

        Text to correct:
        "{text}"

        Return the corrected text in Portuguese, with no additional formatting or characters.
        """
    return prompt
    
# Specific prompts to find some possible products
def products_find_in_section_fix_prompt(products_find_in_section, products_find_using_prompt, products_alike, fix_prompt_v1):
    
    prompt = f"""
        Task Description
        Your task is to process the input text, identify a store section, and return the exact list of products assigned to that 
        section.

        Response Rules (STRICT)
        Your response must be one of these two outputs only:

        A Python-style list of product names (if a valid section is identified and has products)

        Example: "Seção 1" → ["MILHO VERDE AO VAPOR LATA", "ACETONA"]
        
        The exact string "Texto RA" (if no valid section or products are found)
        
        Example: "Seção 999" → "Texto RA"
        
        Nothing else is allowed – no explanations, extra text, code blocks, or different formats.

        Processing Steps
        1 Normalize the Input
        This is the text to you normalize:\n 
        "{fix_prompt_v1}".
        \n
        Convert to lowercase and remove accents: "Seção" → "secao".
        Remove extra spaces: "secão 2" → "secao 2".
        
        2 Extract the Section Number
        
        Match sections in this priority order:
         Direct Matches: "Seção 1", "secao1", "seção 1" → Section 1
         Standalone Numbers: "produtos na 3" → Section 3
         Partial Matches: "seç" + nearby number → Section detected
         If no valid section is found, return "Texto RA"

        3 Match Products to the Section
        
        These are the products find in the possible section mentioned on the text: \n
        {products_find_in_section}.
        \n
        Use exact product names, keeping capitalization and punctuation.
        IGNORE {products_alike} unless explicitly instructed.
         STRICT Output Examples
         Valid Outputs (Only two options exist!)

        Input: "Seção 1" → Output: ["MILHO VERDE AO VAPOR LATA", "ACETONA"]
        Input: "produtos na 3" → Output: ["MARG. POTE 3KG", "AGUA SANITARIA 3LT"]
        Input: "Seção 999" (invalid section) → Output: "Texto RA"
        Input: "seção" (no number) → Output: "Texto RA"
         Never return anything other than a product list or "Texto RA"

         Important Rules (DO NOT BREAK THESE)
         DO NOT return an empty list ([]) – use "Texto RA" instead.
         DO NOT add explanations, comments, or code blocks.
         DO NOT generate additional text beyond the valid outputs.

        Your response must be either a valid product list OR "Texto RA" – nothing else.
    """

    return prompt

def products_find_with_price_fix_prompt(products_find_with_price, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:
        Your task is to process the input text, extract the price, and return the exact list of products that match that price.

        Response Rules (STRICT)
        Your response must be one of these two outputs only:

        1 A Python-style list of product names (if the price matches any products)

        Example: "12.00" → ["SARD. COQ TOMATE", "MASSA P/TAPIOCA", "CAFE MOIDO"]
        2 The exact string "Texto RA" (if no matching products are found)
        Example: "9999.9999" → "Texto RA"
        
        Nothing else is allowed – no explanations, extra text, or code blocks.

        Processing Steps
        1 Normalize the Input
        Fix spelling & grammar in this text: \n 
        "{fix_prompt_v1}".\n
        
        Convert numbers to standard decimal format (e.g., "2,50" → "2.50").
        Remove extra spaces and unnecessary characters.
        
        2 Extract & Validate Price
        
        Identify valid price values (e.g., "12.99", "2.50", "9.00").
        
        If no valid price is detected, return "Texto RA".
        If the price is formatted incorrectly (e.g., missing decimals), assume the closest valid price.
        
        3 Match Products to the Price
        
        Retrieve ONLY products priced at {products_find_with_price}.
        Use exact product names, keeping capitalization and punctuation.
        Ignore {products_find_using_prompt} and {products_alike} unless explicitly instructed.
        🔹 STRICT Output Examples
        ✅ Valid Outputs (Only two options exist!)

        Input: "12.00" → Output: ["SARD. COQ TOMATE", "MASSA P/TAPIOCA", "CAFE MOIDO"]
        Input: "5.20" → Output: ["SABAO PDC NEUTRO", "MACARRAO PARAFUSO", "SALG. QUEIJO"]
        Input: "6.00" → Output: ["BISC. DOCE", "INSET. AERO A.TOTAL", "ALGODAO"]
        Input: "9999.9999" → Output: "Texto RA"
        Input: "o produto que eu quero é 2.50" → Output: ["BOLACHA RECHEADA", "SUCO TAMPICO", "CEREAL MATINAL"]
        Input: "tenho algo por 3.99?" → Output: ["LEITE FERMENTADO", "REFRIG. COLA 2L"]
        Input: "qual produto custa 999.99?" → Output: "Texto RA"
        
        Never return anything other than a product list or "Texto RA"

        Important Rules (DO NOT BREAK THESE)
         DO NOT return an empty list ([]) – use "Texto RA" instead.
         DO NOT add explanations, comments, or code blocks.
         DO NOT generate additional text beyond the valid outputs.

        Your response must be either a valid product list OR "Texto RA" – nothing else.
        """
        
    return prompt

def products_find_with_brand_fix_prompt(products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description
        Your task is to extract the brand name from the input, normalize it, and return the exact list of products associated 
        with that brand.

        Response Rules (STRICT)
        Your response must be one of these two outputs only:

        1 A Python-style list of product names (if the brand matches any products)

        Example: "Antarctica" → ["REFRIG. GUARANA PET CACULINHA", "REFRIG. SODA LIMONADA PET"]
        2 The exact string "Texto RA" (if no matching products are found)
        Example: "Marca XYZ" → "Texto RA"
        Nothing else is allowed – no explanations, extra text, or code blocks.

        Processing Steps
        1 Normalize the Input
        Fix spelling & grammar in this text:\n
        "{fix_prompt_v1}".
        \n
        
        Standardize brand name variations, including abbreviations and phonetic matches:
        "Antartic" → "Antarctica"
        "Tng" → "Tang"
        "Evene" → "Even"
        
        2 Extract & Validate Brand Name
        
        Identify valid brand names (e.g., "Antarctica", "Tang", "Even").
        If no recognizable brand name is detected, return "Texto RA".
        
        3 Match Products to the Brand
        
        Retrieve ONLY products associated with {products_find_using_brand}.
        
        Use exact product names, keeping capitalization and punctuation.
        Ignore {products_find_using_prompt} and {products_alike} unless explicitly instructed.
        STRICT Output Examples
        Valid Outputs (Only two options exist!)

        Input: "Antarctica" → Output: ["REFRIG. GUARANA PET CACULINHA", "REFRIG. SODA LIMONADA PET"]
        Input: "Quais produtos tem da marca even" → Output: Quais os produtos tem da marca ["SAB. CHA VERDE/LIMAO", "SAB. EXTRATO FLORES"]
        Input: "Tem a tang" → Output: Tem os ["REFR.PO ABACAXI CXT", "REFR.PO MANGA CXT", "REFR.PO UVA", "REFR.PO GUARANA"]
        Input: "marca desconhecida" → Output: "Texto RA"
        Input: "Coca" → Output: ["REFRIG. COCA-COLA PET 2L", "REFRIG. COCA-COLA LATA 350ML"]
        Input: "Marquinha XYZ" → Output: "Texto RA"
        
        Never return anything other than a product list or "Texto RA"

        Important Rules (DO NOT BREAK THESE)
         DO NOT return an empty list ([]) – use "Texto RA" instead.
         DO NOT add explanations, comments, or code blocks.
         DO NOT generate additional text beyond the valid outputs.

        Your response must be either a valid product list OR "Texto RA" – nothing else.
        """
        
    return prompt

def products_find_with_metric_or_quantity_fix_prompt(products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
            Task Description
            Your task is to normalize quantity and metric unit formats in the input and return the exact list of matching 
            products based on the provided product lists.

            Response Rules (STRICT)
            Your response must be one of these two outputs only:

            1 A Python-style list of product names (if the input matches any products)

            Example: "1 X 24 250G" → ["CAFE PILAO ALMOFADA", "CAFE PALACIO MOIDO 250G"]
            
            2 The exact string "Texto RA" (if no matching products are found)
            Example: "999G" → "Texto RA"
            
            Nothing else is allowed – no explanations, extra text, or code blocks.

            Processing Steps
            1 Normalize the Input
            
            Fix spelling & grammar in this text:\n
            "{fix_prompt_v1}".
            \n
            
            Convert quantity and metric formats into standardized forms:
            
            "12X500G" → "1 X 12 500G"
            "1X24X250G" → "1 X 24 250G"
            "1 x 6 x 350 ml" → "1 X 6 350ML"
            
            2 Extract & Validate Quantity/Metric Information
            
            Identify valid metric values (e.g., "250G", "500ML", "350ML", "1 X 12 500G").
            Match input against {products_find_using_metric_or_quantity}.
            If no recognizable quantity or metric unit is detected, return "Texto RA".
            
            3 Match Products to the Quantity/Metric
            
            Retrieve ONLY products associated with {products_find_using_metric_or_quantity}.
            Use exact product names, keeping capitalization and punctuation.
            Ignore {products_find_using_prompt} and {products_alike} unless explicitly instructed.
            
            STRICT Output Examples
            Valid Outputs (Only two options exist!)

            Exact Quantity Matches
            
            Input: "1 X 24 250G" → Output: ["CAFE PILAO ALMOFADA", "CAFE PALACIO MOIDO 250G"]
            Input: "12X500G" → Output: ["REFRIG. PEPSI COLA PET 2 EM 1"]
            Input: "Algum produto com essa quantidade 26X600" → Output: Algum produto com essa quantidade ["CAFE PALACIO MOIDO 250G", "CAFE PALACIO MOIDO 500G"]
            General Metric Matches
            Input: "Quais produtos vocês possuem de 250G" → Output: Quais produtos vocês possuem de ["REFRIG. GUARANA PET CACULINHA"]
            Input: "1 X 6 350ML" → Output: ["CAFE PALACIO MOIDO 250G", "CAFE PALACIO MOIDO 500G"]
            Invalid Inputs (No Matches)
            Input: "999G" → Output: "Texto RA"
            Input: "5X10000ML" → Output: "Texto RA"
        """
        
    return prompt

def products_find_with_synonyms(products_find_using_synonyms, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
            Task Description
            Your task is to normalize synonyms in the input text and return the exact list of matching products based on alternative product descriptions.

            Response Rules (STRICT)
            Your response must be one of these two outputs only:

            1 A Python-style list of product names (if the input matches any products via synonyms)

            Example: "Refrigerante de cola" → ["COCA-COLA LATA 350ML", "PEPSI PET 2L"]
            
            2 The exact string "Texto RA" (if no matching products are found)
            Example: "Bebida espacial" → "Texto RA"
            
            Nothing else is allowed – no explanations, extra text, or code blocks.

            Processing Steps
            1 Normalize the Input
            Fix spelling & grammar in this text: \n
            
            "{fix_prompt_v1}".
            
            \n
            Convert common product synonyms into standard forms:
            "Refrigerante de cola" → "Coca-Cola, Pepsi"
            "Pasta de dente" → "Creme dental"
            "Feijão preto" → "Feijão comum"
            
            2 Identify & Match Synonyms
            Identify terms in the input that correspond to known synonyms.
            Match the normalized term against {products_find_using_synonyms}.
            If no valid synonym is detected, return "Texto RA".
            
            3 Retrieve Products Based on Synonyms
            Retrieve ONLY products associated with {products_find_using_synonyms}.
            
            Use exact product names, keeping capitalization and punctuation.
            
            Ignore {products_find_using_prompt} and {products_alike} unless explicitly instructed.
                        
            STRICT Output Examples
            Valid Outputs (Only two options exist!)

            Exact Synonym Matches
            Input: "Refresco de cola" → Output: ["COCA-COLA LATA 350ML", "PEPSI PET 2L"]
            Input: "Pasta de dente" → Output: ["COLGATE CREME DENTAL 90G", "ORAL-B CREME DENTAL 70G"]
            Input: "Sabão em pó" → Output: ["OMO LAVAGEM PERFEITA 1.6KG", "TIXAN YPÊ 2KG"]
            General Synonym Matches
            Input: "Biscoito recheado" → Output: ["OREO 90G", "TRAQUINAS CHOCOLATE"]
            Input: "Água sem gás" → Output: ["CRYSTAL SEM GÁS 500ML", "MINALBA NATURAL 1.5L"]
            Invalid Inputs (No Matches)
            Input: "Bebida espacial" → Output: "Texto RA"
            Input: "Pó de dragão" → Output: "Texto RA"
        """
        
    return prompt

def products_match_all_conditions(products_find_in_section, products_find_with_price, products_find_using_brand, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
        prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Products with the same price:
        {products_find_with_price}
        
        3. Products and brand with the same brand mentioned by the client:
        {products_find_using_brand}
        
        4. Products with the same metric unity or quantity description:
        {products_find_using_metric_or_quantity}
        
        5. Possible Similar Products:
        {products_find_using_prompt}

        6. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
        return prompt

def products_match_section_price_and_brand(products_find_in_section, products_find_with_price, products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Products with the same price:
        {products_find_with_price}
        
        3. Products and brand with the same brand mentioned by the client:
        {products_find_using_brand}
        
        4. Possible Similar Products:
        {products_find_using_prompt}

        5. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
    
def products_match_section_and_brand(products_find_in_section, products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Products and brand with the same brand mentioned by the client:
        {products_find_using_brand}
        
        3. Possible Similar Products:
        {products_find_using_prompt}

        4. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
    
def products_match_section_and_price(products_find_in_section, products_find_with_price, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Products with the same price:
        {products_find_with_price}
        
        3. Possible Similar Products:
        {products_find_using_prompt}

        4. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
    
def products_match_section_and_metric_or_quantity(products_find_in_section, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Products with the same metric unity or quantity description:
        {products_find_using_metric_or_quantity}
        
        3. Possible Similar Products:
        {products_find_using_prompt}

        4. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
    
def products_match_price_brand_and_metric(products_find_with_price, products_find_using_brand,  products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products with the same metric unity or quantity description:
        {products_find_using_metric_or_quantity}
        
        2. Products with the same price:
        {products_find_with_price}
        
        3. Products and brand with the same brand mentioned by the client:
        {products_find_using_brand}
        
        4. Possible Similar Products:
        {products_find_using_prompt}

        5. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt

def products_match_price_and_brand(products_find_with_price, products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products with the same price:
        {products_find_with_price}
        
        2. Products and brand with the same brand mentioned by the client:
        {products_find_using_brand}
        
        3. Possible Similar Products:
        {products_find_using_prompt}

        4. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
  
def products_match_price_and_quantity_or_metric(products_find_with_price, products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products with the same price:
        {products_find_with_price}
        
        2. Products with the same metric unity or quantity description:
        {products_find_using_metric_or_quantity}
        
        3. Possible Similar Products:
        {products_find_using_prompt}

        4. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        Input: "Seção 2" -> Output: Texto RA
        Input: "Quero saber sobre a seção 3" -> Output: Texto RA
        
        DON'T RETURN any different text just consider the examples above.   
        """
    return prompt
  
  
def products_find_general_fix_prompt(products_alike, products_find_using_prompt, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.
        Consider that you're talking directly with the client don't return any weird answer.

        Provided Information:

        1. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:
        
        1.Product Name Matching Rules:

        Match input terms to products in the list based on exact name, abbreviations, partial matches, or phonetic similarity.
        Variations like "cafe," "CAFÉ," or "Cafe Pilao" should be normalized to "CAFE PILAO ALMOFADA" or another appropriate match from the list.
        
        2. Handling Partial or Ambiguous Matches:

        If the input refers to a category or group (e.g., "café" with no additional qualifiers), prioritize the most generic product from the list (e.g., "CAFE PALACIO MOIDO 250G").
        
        3.Text Not Containing Products:

        If the text contains no identifiable product names or similar terms, return "Texto RA."
        
        4. Output Format:

        If a product is identified, list the product(s) in square brackets with their exact names.
        If multiple products fit, list all relevant matches, separated by commas.
        
        Example:
        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        
        5. Contextual Normalization:

        Consider context and section information when normalizing. For example:
        If "250G" appears, prioritize matching with products of the same size.
        If no context (e.g., size or section) is given, use a default or generic match.
        
        6. Specific Cases:

        Abbreviations: Normalize terms like "refri" or "refrigerante" to specific product names like "REFRIG. PEPSI COLA PET 2 EM 1."
        Plurals or variants: "cafés" -> "CAFE PILAO ALMOFADA."
        
        Examples of Substitutions:

        Input: "cafe" -> Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G].
        Input: "pepsi" -> Output: [REFRIG. PEPSI COLA PET 2 EM 1].
        Input: "quanto custa o refrigerante" -> Output: quanto custa o [REFRIG. GUARANA PET CACULINHA].
        Input: "cafe palacio" -> Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G].
        
        DON'T RETURN any different text just consider the examples above. 
        
        """
    return prompt
