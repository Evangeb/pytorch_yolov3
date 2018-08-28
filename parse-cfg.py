




file = open("yolov3.cfg","r")
layer = []
for line in file:
    print(line)
    if line.startswith('['):
         line = line.strip('[]\n')
         layer.append(line)



print(layer)
file.close()
