from typing import Any

_conversations: dict[str,list[dict[str,str]]]={}

def get_history(conversation_id:str|None)->list[dict[str,str]]:
    if not conversation_id:
        return []
    return list(_conversations.get(conversation_id,[]))

def add_message(conversation_id:str,role:str,content:str)->None:
    _conversations.setdefault(conversation_id,[]).append({"role":role,"content":content})

def add_exchange(conversation_id:str,query:str,answer:str)->None:
    add_message(conversation_id,"user",query)
    add_message(conversation_id,"assistant",answer)