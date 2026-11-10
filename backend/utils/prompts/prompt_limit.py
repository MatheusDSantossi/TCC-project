from ..loaders import get_cached_products_df

def for_dividing_prompt_limit(business):
    products_list_2 = []
    
    products_df = get_cached_products_df(business)
    
    for p in products_df.to_dict(orient='records'):
        # prompt += p['Nome']
        products_list_2.append(p['Nome'])
        
    max_prompt_size = 425
    
    for i in range(0, max_prompt_size):
        prompt+= products_list_2[i]
