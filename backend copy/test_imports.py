import chardet

with open('.env', 'rb') as f:
    result = chardet.detect(f.read())
print(result['encoding'])
