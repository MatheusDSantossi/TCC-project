def pre_process_prompt_grammatical_fix_prompt(text, business_name):

    prompt = f"""
        Correct grammatical and spelling errors in the following text so it becomes understandable to a human reader. Normalize product names, utensils, or any other items to a standard form. Consider that this text is provided by a customer from the store {business_name}, and use this context to make the best possible corrections.

        Keep in mind that words might appear abbreviated, such as "INSET SBP AERO," which means "INSETICIDA SBP AEROSSOL." Research thoroughly what each word might mean, whether abbreviated or not, considering that they are most likely retail products found in any supermarket. Use this assumption to help identify and correct the words accurately.

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

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input. Consider that you're talking directly with the client don't return any weird answer.
    
        Provided Information:

        1. Products in the same section:
        {products_find_in_section}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
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

def products_find_with_price_fix_prompt(products_find_with_price, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.
        Consider that you're talking directly with the client don't return any weird answer.

        Provided Information:

        1. Products with the same price:
        {products_find_with_price}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
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
        
        DON'T RETURN any different text just consider the examples above. 
        
        """
        
    return prompt

def products_find_with_brand_fix_prompt(products_find_using_brand, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.
        Consider that you're talking directly with the client don't return any weird answer.

        Provided Information:

        1. Products with the same brand:
        {products_find_using_brand}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
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
        
        DON'T RETURN any different text just consider the examples above. 
        
        """
        
    return prompt

def products_find_with_metric_or_quantity_fix_prompt(products_find_using_metric_or_quantity, products_find_using_prompt, products_alike, fix_prompt_v1):
    prompt = f"""
        Task Description:

        You are a text normalizer and corrector. Your goal is to correct any grammatical and orthographic errors in the provided
        text while standardizing product names to match the exact names from the given product lists. You must ensure no relevant
        products are overlooked and avoid returning "Texto RA" unless absolutely no products match the input.
        Consider that you're talking directly with the client don't return any weird answer.

        Provided Information:

        1. Products with the same metric unity or quantity description:
        {products_find_using_metric_or_quantity}
        
        2. Possible Similar Products:
        {products_find_using_prompt}

        3. Standard Product List (Exact Names and Sections):
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
