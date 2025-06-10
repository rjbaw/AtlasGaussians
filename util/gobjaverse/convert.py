import json, pathlib, sys

inp  = pathlib.Path("train.txt")   
outp = pathlib.Path("train.json")  

data = [line.strip()               
        for line in inp.read_text().splitlines()
        if line.strip()]           

outp.write_text(json.dumps(data))  
print(f"wrote {outp} with {len(data)} items")


inp  = pathlib.Path("test.txt")   
outp = pathlib.Path("test.json")  

data = [line.strip()               
        for line in inp.read_text().splitlines()
        if line.strip()]           

outp.write_text(json.dumps(data))  
print(f"wrote {outp} with {len(data)} items")
