path="data/caltech-101"
import os

dir=os.open(path, os.O_RDONLY)
least=int(1e9)
for file in os.listdir(dir):
    var=os.open(f"{path}/{file}", os.O_RDONLY)
    count=sum(1 for _ in os.listdir(var))
    least=min(least,count)
    least_class=file
    print(file ,"=> ",count)
print("Least value:", least)
print("Class with least value:", least_class)
 