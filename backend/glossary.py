"""Local vocabulary hints, never automatic legal/factual transcript replacement."""
import json
from pathlib import Path
import re

CATALOG=Path(__file__).parent/'resources'/'real_estate.json'


def catalog():
    return json.loads(CATALOG.read_text(encoding='utf-8'))


def normalize(terms,limit=200):
    result=[]
    for term in terms:
        term=term.strip()
        if not term:continue
        if len(term)>50 or any(ord(c)<32 or ord(c)==127 for c in term):
            raise ValueError('每個專有名詞最多 50 字，不能包含控制字元。')
        if term not in result:result.append(term)
    if len(result)>limit:
        raise ValueError('常用詞最多 200 個。')
    return result


def split(text):
    return normalize(re.split(r'[\n\r,，、;；]+',text))


def load_custom(data):
    path=Path(data)/'custom-glossary.json'
    return normalize(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else []


def save_custom(data,terms):
    terms=normalize(terms)
    folder=Path(data);folder.mkdir(parents=True,exist_ok=True)
    temp=folder/'custom-glossary.part'
    temp.write_text(json.dumps(terms,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(folder/'custom-glossary.json')
    return terms


def terms_for(domain,custom):
    terms=list(custom)
    if domain=='real_estate':
        data=catalog()
        terms.extend(data.get('priority_terms',[]))
        terms.extend(term for category in data['categories'] for term in category['terms'])
    elif domain!='general':
        raise ValueError('不支援的領域詞庫。')
    return normalize(terms,limit=1000)


def hints(terms):
    # Bound context length and prioritize the user's custom words.
    chosen=[];length=0
    for term in terms:
        if length+len(term)+1>400:break
        chosen.append(term);length+=len(term)+1
    return '、'.join(chosen)
