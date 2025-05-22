embed_model = 'models/text-embedding-004'
def embed_fn(find_section, genai, conn, title, secao="", preco_venda=0, unidade_medida=""):
    
    section = find_section(title, conn)
    
    if section == None:
    
        return genai.embed_content(model=embed_model,
                                content=f'{title}',  # I understood that this have to be the most alike the userp prompt
                                title=title,
                                task_type="RETRIEVAL_DOCUMENT")["embedding"]  
    
    return genai.embed_content(model=embed_model,
                               content=f'{title} e seção {section}',  # I understood that this have to be the most alike the userp prompt
                               title=title,
                               task_type="RETRIEVAL_DOCUMENT")["embedding"]
    
    # return genai.embed_content(model=embed_model,
    #                            content=f'O produto {title}',  # I understood that this have to be the most alike the userp prompt
    #                            title=title,
    #                            task_type="RETRIEVAL_DOCUMENT")["embedding"]
