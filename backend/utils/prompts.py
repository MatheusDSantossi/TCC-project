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
    """
    return prompt
    
def products_find_in_section_fix_prompt(products_find_in_section, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
    Task Description:

    You are a store section normalizer and product matcher. Your task is to correct any grammatical and orthographic errors in the input text and return a list of products associated with the specified store section based on the provided product lists. 
    Your response **must not** be empty. If no products are found, return "Texto RA."

    Provided Information:

    1. Products in the specified section:
    {products_find_in_section}

    2. Possible Similar Products:
    {products_find_using_prompt}

    3. Standard Product List (Exact Names and Sections):
    {products_alike}

    Text to Normalize:
    "{fix_prompt_v1}"

    Normalization Instructions:

    1. **Section Matching Rules:**
    - Correct any spelling or formatting mistakes in the input.
    - Match the input term (store section) against the provided lists using **exact matches, partial matches, or numeric identifiers**.
    - Variations like "Seção 1" and "1" should be treated as the same section.

    2. **Return Products for the Section:**
    - Identify and return **all** products that belong to the specified store section.
    - If multiple products match, list them all, separated by commas, within square brackets, using their **exact names** from the list.

    3. **Handling Ambiguity:**
    - If the input only specifies a section number without additional details, return **all** products associated with that section.
    - If no recognizable section or product is found, return "Texto RA" instead of an empty list.

    4. **Output Format:**
    - The output **must** be a Python-style list in square brackets `[ ]`, containing the exact product names.
    - **Do not include any extra text, code block formatting, or symbols.** Your response **must only contain the product list**.
    - **Do not return an empty list (`[]`).** If no products are found, return `"Texto RA"`.

    Examples:
    - Input: "Seção 1" → Output: [MILHO VERDE AO VAPOR LATA, ACETONA, MASSA P/ TAPIOCA]
    - Input: "Seção 3" → Output: [MARG. POTE 3KG, AGUA SANITARIA 3LT, ACUCAR DEMERARA]
    - Input: "Seção 2" → Output: [VITARESCO, BISC. CREAM CRACKER, ALCOOL TRAD, GEL FIXADOR STYLO AMARELO]
    - Input: "Seção 999" → Output: "Texto RA"

    **Important:**  
    - Do not return any different text, additional comments, additional caracters as ```, or code formatting anything like that.  
    - Always return a list or `"Texto RA"`. An empty list (`[]`) is **not allowed**.  
    """

    return prompt

def products_find_with_price_fix_prompt(products_find_with_price, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a product price normalizer and matcher. Your goal is to correct any grammatical and orthographic errors in the 
        provided text and return a list of products that match the specified price from the given product lists. You must ensure 
        no relevant products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.

        Provided Information:

        1. Products with the specified price:
        {products_find_with_price}

        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Prices):
        {products_alike}

        Text to Normalize:
        "{fix_prompt_v1}"

        Normalization Instructions:

        1. **Price Matching Rules:**
        - Correct any spelling or formatting mistakes in the input.
        - Match the input price exactly with the provided product list.
        - Ensure that all products listed at the specified price are included.

        2. **Returning Products for the Price:**
        - Identify and return all products that match the given price.
        - If multiple products match, list them all, separated by commas, within square brackets, using their exact names.

        3. **Handling Ambiguity:**
        - If the input price does not match any available products, return "Texto RA."
        - If the input price is formatted incorrectly (e.g., missing decimals), assume the closest valid price.

        4. **Output Format:**
        - The output **must** be a Python-style list in square brackets `[ ]`, containing the exact product names.
        - **Do not include any extra text, code block formatting, or additional symbols.**
        - If no products match, return `"Texto RA"` instead of an empty list.

        Examples:
        - Input: "12.00" → Output: [SARD. COQ TOMATE, MASSA P/TAPIOCA, CAFE MOIDO]
        - Input: "5.20" → Output: [SABAO PDC NEUTRO, MACARRAO PARAFUSO, SALG. QUEIJO]
        - Input: "9.00" → Output: [FRALDA DESC. G, DESINF.]
        - Input: "6.00" → Output: [BISC. DOCE, INSET. AERO A.TOTAL, ALGODAO]
        - Input: "9999.9999" → Output: "Texto RA"

        Important:
        - Do not return any different text, additional comments, or formatting.
        - Always return a list or "Texto RA." An empty list (`[]`) is not allowed.
        """
        
    return prompt

def products_find_with_brand_fix_prompt(products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a product brand normalizer and matcher. Your objective is to correct any grammatical and orthographic errors in the input text and then identify and return all products associated with the specified brand, based on the provided product lists. Do not return "Texto RA" unless no products match the input.

        Provided Information:

        1. Products with the same brand mentioned by the client:
        {products_find_using_brand}
        
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
        {products_alike}
        
        Text to Normalize:
        "{fix_prompt_v1}"
        
        Normalization Instructions:
        
        1.Brand Matching Rules:

        Correct any spelling or formatting mistakes in the input.
        
        Match the input term (the brand name) against the provided lists using exact matches, abbreviations, partial matches, or phonetic similarity. For example, variations like "antarctica" and "antartic" should be treated as the same brand.
        
        2.Return Products for the Brand:

        Identify and return all products from the Standard Product List that belong to the matched brand.
        
        If multiple products match, list them all, separated by commas, within square brackets, using their exact names from the list.
        
        3.Handling Ambiguity:

        If the input only specifies a brand without additional details, return all products associated with that brand.
        
        If no recognizable brand or product is found, return "Texto RA."
        
        4.Output Format:

        The output must be exactly a Python-style list in square brackets containing the exact product names.
        
        Do not include any extra text, code block formatting, or additional symbols. Your response should contain only the product list.
        
        Examples:
        
        Input: "antarctica/antartic" → Output: [REFRIG. GUARANA PET CACULINHA, REFRIG. SODA LIMONADA PET]
        Input: "even" → Output: [SAB. CHA VERDE/LIMAO, SAB. EXTRATO FLORES]
        Input: "tang" → Output: [REFR.PO ABACAXI CXT, REFR.PO MANGA CXT, REFR.PO UVA, REFR.PO GUARANA]
        
        Important:
        - Do not return any different text, additional comments, or formatting.
        - Always return a list or "Texto RA." An empty list (`[]`) is not allowed.
        """
        
    return prompt

def products_find_with_metric_or_quantity_fix_prompt(products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
            Task Description:

            You are a product quantity and metric unit normalizer. Your goal is to correct any grammatical and orthographic errors in the provided text while ensuring that product names match exactly with the given product lists. You must ensure no relevant products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.

            Provided Information:

            1. Products with the same metric unit or quantity description:
            {products_find_using_metric_or_quantity}

            2. Possible Similar Products:
            {products_find_using_prompt}

            3. Standard Product List (Exact Names and Metrics):
            {products_alike}

            Text to Normalize:
            "{fix_prompt_v1}"

            Normalization Instructions:

            1. **Metric and Quantity Matching Rules:**
            - Correct any spelling or formatting mistakes in the input.
            - Match the input term (e.g., weight, volume, package size) to the provided product list based on:
                - Exact numeric values (e.g., "250G" → "CAFE PALACIO MOIDO 250G")
                - Multipack quantities (e.g., "1 X 12 500G" → "REFRIG. PEPSI COLA PET 2 EM 1")
                - Abbreviations and variations (e.g., "350ML" → "CAFE PALACIO MOIDO 250G")

            2. **Handling Partial or Ambiguous Matches:**
            - If the input includes a size without a specific product (e.g., "250G"), return the most relevant products with that size.
            - If multiple products match, list them all in square brackets, separated by commas.
            - If no valid metric or quantity is found, return "Texto RA."

            3. **Output Format:**
            - The response must be a Python-style list in square brackets `[ ]`, containing the exact product names.
            - No extra text, explanations, or symbols should be included.
            - If no products match, return `"Texto RA"` instead of an empty list.

            Examples:
            - Input: "1 X 24 250G" → Output: [CAFE PILAO ALMOFADA, CAFE PALACIO MOIDO 250G]
            - Input: "1 X 12 500G" → Output: [REFRIG. PEPSI COLA PET 2 EM 1]
            - Input: "250G" → Output: [REFRIG. GUARANA PET CACULINHA]
            - Input: "1 X 6 350ML" → Output: [CAFE PALACIO MOIDO 250G, CAFE PALACIO MOIDO 500G]
            - Input: "999G" → Output: "Texto RA"

            Important:
            - Do not return any different text or formatting.
            - Always return a valid product list or "Texto RA."
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
