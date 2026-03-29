
json_data = open('C:/MetaShift/data/compact.json','r',encoding='utf-8').read()

part1 = open('C:/MetaShift/html_part1.txt','r',encoding='utf-8').read()
part2 = open('C:/MetaShift/html_part2.txt','r',encoding='utf-8').read()

html = part1 + json_data + part2

with open('C:/MetaShift/metashift.html','w',encoding='utf-8') as f:
    f.write(html)

import re
div_opens = len(re.findall(r'<div[\s>]', html))
div_closes = html.count('</div>')
script_start = html.find('<script>')
script_end = html.rfind('</' + 'script>')
sc = html[script_start+8:script_end]
bad = sc.lower().count('</' + 'script')
print(f"Written {len(html)} chars")
print(f"Divs: open={div_opens} close={div_closes} {'OK' if div_opens==div_closes else 'MISMATCH'}")
print(f"Bad script tags in JS: {bad} {'OK' if bad==0 else 'BAD'}")
